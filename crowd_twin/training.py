from __future__ import annotations

import json
import os
import random
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import save_resolved_config
from .contracts import integration_contract
from .data.dataset import UnifiedCrowdDataset, resize_targets_like
from .models.factory import build_model
from .venue import VenueGraph, node_static_features
from .venue import zone_masks as rasterize_zone_masks


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def dataset_registry(path: str | Path = "configs/datasets.yaml") -> dict[str, dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["datasets"]


def build_dataset(
    config: dict[str, Any],
    split: str,
    dataset_names: Iterable[str],
    augment: bool,
    registry_path: str | Path = "configs/datasets.yaml",
) -> UnifiedCrowdDataset:
    registry = dataset_registry(registry_path)
    manifests: list[Path] = []
    roots: dict[str, str] = {}
    manifest_root = Path(config["data"]["manifest_root"])
    sensor_config = config.get("sensors", {})
    normalization_path = sensor_config.get("normalization")
    sensor_normalization = (
        yaml.safe_load(Path(normalization_path).read_text(encoding="utf-8"))
        if normalization_path and Path(normalization_path).is_file()
        else {}
    )
    venue_graph = None
    semantic_masks = None
    venue_adjacency = None
    venue_static = None
    venue_path = config.get("architecture", {}).get("venue_graph")
    if venue_path and Path(venue_path).is_file():
        venue_graph = VenueGraph.load(venue_path)
        semantic_masks = rasterize_zone_masks(
            venue_graph, *tuple(config["data"]["image_size"])
        )
        venue_adjacency = np.asarray(venue_graph.adjacency, dtype=bool)
        venue_static = node_static_features(
            venue_graph,
            feature_dim=int(config["model"].get("static_node_features", 4)),
        )
    for name in dataset_names:
        manifest = manifest_root / name / f"{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Missing normalized manifest {manifest}. Run scripts/prepare_dataset.py first."
            )
        manifests.append(manifest)
        roots[name] = registry[name]["root"]
        if config["data"].get("enforce_leakage_audit", True):
            audit_path = manifest_root / name / "leakage_report.json"
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"Missing leakage audit {audit_path}. Re-run scripts/prepare_dataset.py."
                )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not audit.get("clean", False):
                raise RuntimeError(f"Leakage audit failed for {name}: {audit_path}")
    return UnifiedCrowdDataset(
        manifests=manifests,
        dataset_roots=roots,
        context_frames=int(config["data"]["context_frames"]),
        future_offsets=config["data"]["future_offsets"],
        image_size=tuple(config["data"]["image_size"]),
        zone_grid=tuple(config["data"]["zone_grid"]),
        gaussian_sigma=float(config["data"]["gaussian_sigma"]),
        risk_density_bins=config["data"].get("risk_density_bins", [1.0, 3.0, 6.0]),
        sensor_features=config.get("sensors", {}).get("features", []),
        sensor_normalization=sensor_normalization,
        semantic_zone_masks=semantic_masks,
        venue_adjacency=venue_adjacency,
        venue_node_static=venue_static,
        augment=augment,
        augmentation=config["data"].get("augmentation", {}),
    )


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    density: torch.Tensor
    count: torch.Tensor
    localization: torch.Tensor
    zone_state: torch.Tensor
    forecast: torch.Tensor
    risk: torch.Tensor
    hazard: torch.Tensor
    consistency: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in asdict(self).items()
        }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> LossBreakdown:
    density_target = resize_targets_like(batch["density_target"], outputs["density"])
    localization_target = functional.interpolate(
        batch["localization_target"].flatten(0, 1),
        size=outputs["localization_logits"].shape[-2:],
        mode="nearest",
    ).reshape_as(outputs["localization_logits"])

    density_loss = functional.mse_loss(outputs["density"], density_target)
    predicted_counts = outputs["density"].flatten(2).sum(dim=-1)
    target_counts = batch["density_target"].flatten(2).sum(dim=-1)
    count_loss = functional.l1_loss(predicted_counts, target_counts)
    localization_loss = functional.binary_cross_entropy_with_logits(
        outputs["localization_logits"], localization_target
    )
    zone_state_loss = functional.smooth_l1_loss(
        outputs["current_zone_state"], batch["current_zone_state"]
    )

    future_prediction = outputs["future_zone_state"]
    future_target = batch["future_zone_state"]
    future_mask = (
        batch["future_mask"][..., None, None]
        * batch["future_feature_mask"][:, None, None, :]
    )
    masked_forecast = functional.smooth_l1_loss(
        future_prediction, future_target, reduction="none"
    )
    expanded_mask = future_mask.expand_as(masked_forecast)
    forecast_loss = (masked_forecast * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)

    valid_risk = batch["risk_label"] >= 0
    if valid_risk.any():
        global_risk_logits = outputs["risk_logits"].mean(dim=1)
        risk_loss = functional.cross_entropy(
            global_risk_logits[valid_risk], batch["risk_label"][valid_risk]
        )
    else:
        risk_loss = density_loss.new_zeros(())

    hazard_target = batch.get("hazard_label")
    if hazard_target is None:
        hazard_target = (batch["risk_label"] >= 2).to(
            outputs["hazard_probability"].dtype
        )
    hazard_prediction = outputs["hazard_probability"].mean(dim=1)
    hazard_loss = functional.binary_cross_entropy(
        hazard_prediction.clamp(1e-6, 1 - 1e-6),
        hazard_target.to(hazard_prediction),
    )

    density_count = outputs["density"].flatten(2).sum(dim=-1)
    point_mass = torch.sigmoid(outputs["localization_logits"]).flatten(2).sum(dim=-1)
    consistency_loss = functional.smooth_l1_loss(
        torch.log1p(density_count), torch.log1p(point_mass)
    )
    total = (
        weights["density"] * density_loss
        + weights["count"] * count_loss
        + weights["localization"] * localization_loss
        + weights["zone_state"] * zone_state_loss
        + weights["forecast"] * forecast_loss
        + weights["risk"] * risk_loss
        + weights.get("hazard", 0.25) * hazard_loss
        + weights["consistency"] * consistency_loss
    )
    return LossBreakdown(
        total=total,
        density=density_loss,
        count=count_loss,
        localization=localization_loss,
        zone_state=zone_state_loss,
        forecast=forecast_loss,
        risk=risk_loss,
        hazard=hazard_loss,
        consistency=consistency_loss,
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
    source_datasets: list[str],
    scaler: Any | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    venue = None
    venue_path = config.get("architecture", {}).get("venue_graph")
    if venue_path and Path(venue_path).is_file():
        venue = VenueGraph.load(venue_path)
    payload = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_sha256": config.get("_config_sha256"),
        "source_datasets": source_datasets,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "created_at_unix": time.time(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "model_metadata": asdict(model.metadata) if hasattr(model, "metadata") else {},
        "integration_contract": integration_contract(config, venue),
    }
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    if restore_rng and checkpoint.get("rng_state"):
        state = checkpoint["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
    return checkpoint


def dataset_balanced_sampler(
    dataset: UnifiedCrowdDataset, seed: int
) -> WeightedRandomSampler:
    """Give each source dataset equal expected sampling mass per epoch."""
    sample_datasets = [
        dataset.groups[group_id][end].dataset for group_id, end in dataset.index
    ]
    counts = {name: sample_datasets.count(name) for name in set(sample_datasets)}
    weights = torch.tensor(
        [1.0 / counts[name] for name in sample_datasets], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    scaler: Any | None = None,
    mixed_precision: bool = False,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    accumulation_steps = max(1, accumulation_steps)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        for key in (
            "frames",
            "flow",
            "sensors",
            "sensor_mask",
            "density_target",
            "localization_target",
            "current_zone_counts",
            "current_zone_state",
            "future_zone_counts",
            "future_zone_state",
            "future_feature_mask",
            "future_mask",
            "risk_label",
            "hazard_label",
        ):
            batch[key] = batch[key].to(device)
        for key in ("zone_masks", "adjacency", "node_static"):
            if key in batch:
                batch[key] = batch[key].to(device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            outputs = model(
                batch["frames"],
                batch["flow"],
                sensors=batch["sensors"],
                sensor_mask=batch["sensor_mask"],
                adjacency=batch.get("adjacency"),
                node_static=batch.get("node_static"),
                zone_masks=batch.get("zone_masks"),
            )
            losses = compute_loss(outputs, batch, weights)
        if training:
            scaled_loss = losses.total / accumulation_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for key, value in losses.detached().items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(config: dict[str, Any], registry_path: str | Path = "configs/datasets.yaml") -> Path:
    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    train_names = list(config["data"]["train_datasets"])
    val_names = list(config["data"]["validation_datasets"])
    train_dataset = build_dataset(config, "train", train_names, True, registry_path)
    val_dataset = build_dataset(config, "val", val_names, False, registry_path)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    balanced = bool(config["data"].get("balance_datasets", True)) and len(train_names) > 1
    sampler = dataset_balanced_sampler(train_dataset, int(config["seed"])) if balanced else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    amp_enabled = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"])
    )
    start_epoch = 0
    best_metric = float("inf")
    resume = config["training"].get("resume")
    if resume:
        checkpoint = load_checkpoint(
            resume, device, model, optimizer, scaler, restore_rng=True
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    history_path = output_dir / "history.jsonl"
    patience = 0
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            config["training"]["loss_weights"],
            optimizer,
            float(config["training"]["gradient_clip"]),
            scaler,
            amp_enabled,
            int(config["training"].get("gradient_accumulation_steps", 1)),
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                device,
                config["training"]["loss_weights"],
                None,
                0.0,
                None,
                amp_enabled,
                1,
            )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "temporal_backend": model.metadata.temporal_backend,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            best_metric,
            train_names,
            scaler,
        )
        if val_metrics["total"] < best_metric:
            best_metric = val_metrics["total"]
            patience = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_metric,
                train_names,
                scaler,
            )
        else:
            patience += 1
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} backend={model.metadata.temporal_backend}"
        )
        if patience >= int(config["training"]["early_stopping_patience"]):
            break
    return output_dir / "best.pt"
