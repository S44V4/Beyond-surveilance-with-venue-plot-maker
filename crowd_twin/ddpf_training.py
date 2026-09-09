from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from crowd_twin.config import load_config
from crowd_twin.models.perception import DDPFNet
from crowd_twin.training import build_dataset, dataset_balanced_sampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mass_preserving_resize(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = functional.interpolate(target, size=size, mode="area")
    source_mass = target.flatten(1).sum(dim=1)
    target_mass = resized.flatten(1).sum(dim=1).clamp_min(1e-8)
    return resized * (source_mass / target_mass).view(-1, 1, 1, 1)


def compute_ddpf_loss(
    output: dict[str, torch.Tensor],
    density_target: torch.Tensor,
    localization_target: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    target_density = _mass_preserving_resize(density_target, output["density"].shape[-2:])
    target_localization = functional.interpolate(
        localization_target, size=output["localization_logits"].shape[-2:], mode="nearest"
    )
    density = functional.mse_loss(output["density"], target_density)
    predicted_count = output["density"].flatten(1).sum(dim=1)
    true_count = density_target.flatten(1).sum(dim=1)
    count = functional.smooth_l1_loss(predicted_count, true_count)
    localization = functional.binary_cross_entropy_with_logits(
        output["localization_logits"], target_localization
    )
    total = (
        float(weights.get("density", 1.0)) * density
        + float(weights.get("count", 0.1)) * count
        + float(weights.get("localization", 0.5)) * localization
    )
    return {"total": total, "density": density, "count": count, "localization": localization}


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    frames = batch["frames"].to(device)
    density = batch["density_target"].to(device)
    localization = batch["localization_target"].to(device)
    batch_size, time_steps = frames.shape[:2]
    return (
        frames.reshape(batch_size * time_steps, *frames.shape[2:]),
        density.reshape(batch_size * time_steps, *density.shape[2:]),
        localization.reshape(batch_size * time_steps, *localization.shape[2:]),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "density", "count", "localization")}
    samples = 0
    for batch in loader:
        frames, density, localization = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            losses = compute_ddpf_loss(model(frames), density, localization, weights)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        count = frames.shape[0]
        samples += count
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
    stale_epochs: int,
) -> None:
    model_config = {
        "backbone": config["model"]["backbone"],
        "pretrained": bool(config["model"].get("pretrained", False)),
        "export_channels": int(config["model"].get("export_channels", 16)),
    }
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        feature_size = tuple(
            model(torch.zeros(1, 3, *image_size, device=next(model.parameters()).device))[
                "features"
            ].shape[-2:]
        )
    model.train(was_training)
    payload = {
        "format_version": 1,
        "checkpoint_kind": "trained_ddpf",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model_config,
        "image_size": list(image_size),
        "feature_size": list(feature_size),
        "epoch": epoch,
        "best_metric": best_metric,
        "stale_epochs": stale_epochs,
        "source_datasets": list(config["data"]["train_datasets"]),
        "created_at_unix": time.time(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_ddpf(
    config_path: str | Path,
    registry_path: str | Path,
    resume_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    train_names = list(config["data"]["train_datasets"])
    validation_names = list(config["data"]["validation_datasets"])
    train_data = build_dataset(config, "train", train_names, True, registry_path)
    validation_data = build_dataset(config, "val", validation_names, False, registry_path)
    for split_name, dataset in (("training", train_data), ("validation", validation_data)):
        if not any(record.points for record in dataset.records):
            raise RuntimeError(
                f"DDPF {split_name} manifest contains no point annotations. "
                "Re-run dataset preparation and verify the annotation adapter before training."
            )
    balanced = bool(config["data"].get("balance_datasets", True)) and len(train_names) > 1
    sampler = dataset_balanced_sampler(train_data, int(config["seed"])) if balanced else None
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["data"].get("num_workers", 0)),
    }
    train_loader = DataLoader(train_data, shuffle=sampler is None, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    requested = config["training"].get("device")
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {
        "backbone": config["model"]["backbone"],
        "pretrained": bool(config["model"].get("pretrained", False)),
        "export_channels": int(config["model"].get("export_channels", 16)),
    }
    model = DDPFNet(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = 0
    best = float("inf")
    stale = 0
    resume = str(resume_path) if resume_path is not None else config["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        stale = int(checkpoint.get("stale_epochs", 0))
    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model, train_loader, device, config["training"]["loss_weights"], optimizer,
            scaler, amp, float(config["training"].get("gradient_clip", 1.0)),
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, config["training"]["loss_weights"],
            mixed_precision=amp,
        )
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            stale = 0
            _save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, config, epoch, best, stale
            )
        else:
            stale += 1
        _save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, config, epoch, best, stale
        )
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.5f} "
            f"val={validation_metrics['total']:.5f} device={device}"
        )
        if stale >= int(config["training"].get("early_stopping_patience", 8)):
            break
    return output_dir / "best.pt"


@torch.inference_mode()
def evaluate_ddpf(
    config_path: str | Path,
    checkpoint_path: str | Path,
    registry_path: str | Path,
    split: str = "test",
) -> dict[str, float]:
    config = load_config(config_path)
    names = list(config["data"].get(f"{split}_datasets", config["data"]["validation_datasets"]))
    dataset = build_dataset(config, split, names, False, registry_path)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPFNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    true_positive = false_positive = false_negative = 0.0
    for batch in loader:
        frames, density, localization = _move_batch(batch, device)
        output = model(frames)
        predicted_count = output["density"].flatten(1).sum(dim=1)
        true_count = density.flatten(1).sum(dim=1)
        errors = predicted_count - true_count
        absolute_errors.extend(errors.abs().cpu().tolist())
        squared_errors.extend(errors.square().cpu().tolist())
        target = functional.interpolate(
            localization, size=output["localization_logits"].shape[-2:], mode="nearest"
        ) > 0.5
        predicted = torch.sigmoid(output["localization_logits"]) > 0.5
        true_positive += float((predicted & target).sum())
        false_positive += float((predicted & ~target).sum())
        false_negative += float((~predicted & target).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    metrics = {
        "count_mae": float(np.mean(absolute_errors)),
        "count_rmse": float(np.sqrt(np.mean(squared_errors))),
        "localization_precision": precision,
        "localization_recall": recall,
        "localization_f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }
    output = Path(config["output_dir"]) / f"evaluation_{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
