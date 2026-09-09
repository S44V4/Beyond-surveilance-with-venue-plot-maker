from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional
from torch.utils.data import DataLoader

from crowd_twin.contracts import GRAPH_HANDOFF_SCHEMA_VERSION, STATE_FEATURES
from crowd_twin.data.strfe import CachedDDPFWindowDataset
from crowd_twin.models.strfe import STRFE, PersistenceForecast, RecurrentForecastBaseline
from crowd_twin.venue import VenueGraph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_strfe_dataset(
    config: dict[str, Any], split: str, venue: VenueGraph, augment: bool = False
) -> CachedDDPFWindowDataset:
    data = config["data"]
    calibration = None
    if data.get("risk_calibration"):
        calibration_path = Path(data["risk_calibration"])
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return CachedDDPFWindowDataset(
        manifest=data["manifest"],
        split=split,  # type: ignore[arg-type]
        context_frames=int(data["context_frames"]),
        horizons_seconds=config["model"]["horizons_seconds"],
        expected_venue_id=venue.venue_id,
        expected_zone_ids=venue.zone_ids,
        risk_calibration=calibration,
        augment=augment,
        horizontal_flip_probability=float(
            data.get("augmentation", {}).get("horizontal_flip_probability", 0.0)
        ),
    )


def build_strfe_model(config: dict[str, Any]) -> STRFE:
    model = config["model"]
    return STRFE(
        input_channels=int(model["input_channels"]),
        feature_dim=int(model["feature_dim"]),
        horizons_seconds=model["horizons_seconds"],
        sensor_features=config.get("sensors", {}).get("features", []),
        temporal_layers=int(model.get("temporal_layers", 1)),
        dropout=float(model.get("dropout", 0.1)),
        use_mamba=bool(model.get("use_mamba", False)),
        use_motion=bool(model.get("components", {}).get("motion", True)),
        use_short_term=bool(model.get("components", {}).get("short_term", True)),
        use_long_term=bool(model.get("components", {}).get("long_term", True)),
        use_sensors=bool(model.get("components", {}).get("sensors", True)),
    )


def _batch_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device=device, dtype=torch.float32)
        for key in (
            "visual_features",
            "density",
            "localization_logits",
            "flow",
            "zone_masks",
            "sensors",
            "sensor_mask",
        )
    }


def _scaled_state_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = tuple(range(target.ndim - 1))
    scale = target.detach().abs().amax(dim=dimensions).clamp_min(1.0)
    per_feature = functional.smooth_l1_loss(
        prediction / scale, target / scale, reduction="none"
    ).mean(dim=dimensions)
    return per_feature.mean(), per_feature


def evaluate_strfe(
    model: STRFE,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    recurrent_baseline_epochs: int = 0,
    risk_threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    predicted = []
    targets = []
    current_predicted = []
    current_targets = []
    histories = []
    elapsed = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs = _batch_inputs(batch, device)
            started = time.perf_counter()
            output = model(**inputs)
            elapsed += time.perf_counter() - started
            predicted.append(output["future_zone_state"].cpu())
            targets.append(batch["future_zone_state"].float())
            current_predicted.append(output["current_zone_state"].cpu())
            current_targets.append(batch["current_zone_state"].float())
            histories.append(batch["zone_state_history"].float())
    future_prediction = torch.cat(predicted)
    future_target = torch.cat(targets)
    current_prediction = torch.cat(current_predicted)
    current_target = torch.cat(current_targets)
    history = torch.cat(histories)
    persistence = PersistenceForecast(future_target.shape[1])(current_target)

    def metrics(values: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
        error = values - truth
        absolute = error.abs()
        risk_pred = values[..., STATE_FEATURES.index("risk")] >= risk_threshold
        risk_true = truth[..., STATE_FEATURES.index("risk")] >= risk_threshold
        true_positive = int((risk_pred & risk_true).sum())
        false_positive = int((risk_pred & ~risk_true).sum())
        false_negative = int((~risk_pred & risk_true).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return {
            "count_mae": float(absolute[..., 0].mean()),
            "count_rmse": float(torch.sqrt((error[..., 0] ** 2).mean())),
            "density_mae": float(absolute[..., 1].mean()),
            "density_rmse": float(torch.sqrt((error[..., 1] ** 2).mean())),
            "state_mae": {
                name: float(absolute[..., index].mean())
                for index, name in enumerate(STATE_FEATURES)
            },
            "risk_precision": precision,
            "risk_recall": recall,
            "risk_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "horizon_count_mae": [
                float(absolute[:, index, :, 0].mean()) for index in range(absolute.shape[1])
            ],
        }

    result = {
        "samples": len(future_target),
        "runtime_ms_per_sequence": elapsed / max(len(future_target), 1) * 1000.0,
        "current": metrics(current_prediction[:, None], current_target[:, None]),
        "forecast": metrics(future_prediction, future_target),
        "persistence_baseline": metrics(persistence, future_target),
    }
    if recurrent_baseline_epochs > 0:
        recurrent = RecurrentForecastBaseline(
            hidden_dim=max(16, model.feature_dim // 2), horizons=future_target.shape[1]
        ).to(device)
        optimizer = torch.optim.AdamW(recurrent.parameters(), lr=1e-3)
        recurrent.train()
        history_device = history.to(device)
        target_device = future_target.to(device)
        for _ in range(recurrent_baseline_epochs):
            baseline_prediction = recurrent(history_device)
            loss, _ = _scaled_state_loss(baseline_prediction, target_device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        recurrent.eval()
        with torch.inference_mode():
            baseline_prediction = recurrent(history_device).cpu()
        result["recurrent_baseline"] = metrics(baseline_prediction, future_target)
    return result


def train_strfe(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 23037))
    set_seed(seed)
    requested = config.get("training", {}).get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    venue = VenueGraph.load(config["architecture"]["venue_graph"])
    train_dataset = build_strfe_dataset(config, "train", venue, augment=True)
    validation_dataset = build_strfe_dataset(config, "val", venue)
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    model = build_strfe_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "strfe_best.pt"
    start_epoch = 0
    best_loss = float("inf")
    resume = training.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
    patience = int(training.get("early_stopping_patience", 8))
    stale = 0
    history = []
    epochs = int(training.get("epochs", 20))
    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0
        horizon_totals = np.zeros(len(model.horizons), dtype=np.float64)
        for batch in train_loader:
            inputs = _batch_inputs(batch, device)
            current_target = batch["current_zone_state"].to(device).float()
            future_target = batch["future_zone_state"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(**inputs)
                current_loss, _ = _scaled_state_loss(
                    prediction["current_zone_state"], current_target
                )
                horizon_losses = []
                for index in range(len(model.horizons)):
                    loss, _ = _scaled_state_loss(
                        prediction["future_zone_state"][:, index], future_target[:, index]
                    )
                    horizon_losses.append(loss)
                loss = current_loss + torch.stack(horizon_losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            horizon_totals += np.asarray([float(value.detach()) for value in horizon_losses])

        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for batch in validation_loader:
                inputs = _batch_inputs(batch, device)
                prediction = model(**inputs)
                current_loss, _ = _scaled_state_loss(
                    prediction["current_zone_state"],
                    batch["current_zone_state"].to(device).float(),
                )
                future_loss, _ = _scaled_state_loss(
                    prediction["future_zone_state"],
                    batch["future_zone_state"].to(device).float(),
                )
                validation_total += float(current_loss + future_loss)
        train_loss = train_total / max(len(train_loader), 1)
        validation_loss = validation_total / max(len(validation_loader), 1)
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "horizon_train_loss": (horizon_totals / max(len(train_loader), 1)).tolist(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record))
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale = 0
            sources = [sequence.metadata.source for sequence in train_dataset.sequences]
            checkpoint_kind = (
                "synthetic_integration_fixture"
                if sources and all(source.startswith("synthetic") for source in sources)
                else "dataset_trained"
            )
            checkpoint = {
                "format_version": 1,
                "checkpoint_kind": checkpoint_kind,
                "warning": (
                    "Synthetic STRFE integration checkpoint; not a research result."
                    if checkpoint_kind == "synthetic_integration_fixture"
                    else None
                ),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": {
                    "input_channels": model.input_channels,
                    "feature_dim": model.feature_dim,
                    "horizons_seconds": list(model.horizons),
                    "sensor_features": list(model.sensor_features),
                    "temporal_layers": int(config["model"].get("temporal_layers", 1)),
                    "dropout": float(config["model"].get("dropout", 0.1)),
                    "use_mamba": bool(config["model"].get("use_mamba", False)),
                    "use_motion": model.use_motion,
                    "use_short_term": model.use_short_term,
                    "use_long_term": model.use_long_term,
                    "use_sensors": model.use_sensors,
                },
                "model_metadata": model.metadata.to_dict(),
                "integration_contract": {
                    "schema_version": GRAPH_HANDOFF_SCHEMA_VERSION,
                    "venue_id": venue.venue_id,
                    "zone_ids": list(venue.zone_ids),
                    "state_features": list(STATE_FEATURES),
                    "feature_dim": model.feature_dim,
                    "horizons_seconds": list(model.horizons),
                    "temporal_unit": str(config["data"]["temporal_unit"]),
                    "risk_calibration_sha256": (
                        _manifest_sha256(config["data"]["risk_calibration"])
                        if config["data"].get("risk_calibration")
                        else None
                    ),
                },
                "epoch": epoch,
                "best_validation_loss": best_loss,
                "seed": seed,
                "manifest": str(Path(config["data"]["manifest"]).resolve()),
                "manifest_sha256": _manifest_sha256(config["data"]["manifest"]),
                "resolved_config": {
                    key: value for key, value in config.items() if not key.startswith("_")
                },
            }
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}),
        encoding="utf-8",
    )
    return {"checkpoint": str(best_path), "best_validation_loss": best_loss, "history": history}


def load_strfe_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[STRFE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("STRFE checkpoint has no model_config")
    model = STRFE(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
