from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from .evaluation import MetricAccumulator
from .models.factory import build_model
from .training import build_dataset, load_checkpoint, seed_everything


def _degrade(batch: dict[str, Any], name: str) -> dict[str, Any]:
    frames = batch["frames"]
    if name == "gaussian_noise":
        batch["frames"] = (frames + torch.randn_like(frames) * 0.05).clamp(0, 1)
    elif name == "gaussian_blur":
        shape = frames.shape
        batch["frames"] = functional.avg_pool2d(
            frames.flatten(0, 1), kernel_size=5, stride=1, padding=2
        ).reshape(shape)
    elif name == "low_light":
        batch["frames"] = (frames * 0.3).pow(1.4)
    elif name == "compression":
        batch["frames"] = torch.round(frames * 31.0) / 31.0
    elif name == "occlusion":
        height, width = frames.shape[-2:]
        batch["frames"][..., height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0
    elif name == "flow_dropout":
        batch["flow"].zero_()
    elif name == "sensor_dropout":
        batch["sensors"].zero_()
        batch["sensor_mask"].zero_()
    else:
        raise ValueError(f"Unknown degradation {name!r}")
    return batch


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    risk_classes: int,
    localization_threshold: float,
    degradation: str | None = None,
) -> dict[str, Any]:
    metrics = MetricAccumulator(risk_classes, localization_threshold)
    with torch.inference_mode():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            if degradation:
                batch = _degrade(batch, degradation)
            started = time.perf_counter()
            outputs = model(
                batch["frames"],
                batch["flow"],
                sensors=batch["sensors"],
                sensor_mask=batch["sensor_mask"],
                adjacency=batch.get("adjacency"),
                node_static=batch.get("node_static"),
                zone_masks=batch.get("zone_masks"),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics.update(outputs, batch, (time.perf_counter() - started) * 1000)
    return metrics.compute()


def robustness_test(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    dataset_name: str,
    output_path: str | Path,
    degradations: list[str] | None = None,
    registry_path: str | Path = "configs/datasets.yaml",
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    resolved = checkpoint["config"]
    model = build_model(resolved).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dataset = build_dataset(resolved, "test", [dataset_name], False, registry_path)
    loader = DataLoader(
        dataset,
        batch_size=int(resolved["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(resolved["data"]["num_workers"]),
    )
    risk_classes = int(resolved["model"]["risk_classes"])
    threshold = float(resolved.get("evaluation", {}).get("localization_threshold", 0.35))
    clean = _evaluate(model, loader, device, risk_classes, threshold)
    names = degradations or list(
        resolved.get("evaluation", {}).get("robustness_degradations", [])
    )
    shifted = {
        name: _evaluate(model, loader, device, risk_classes, threshold, name)
        for name in names
    }
    for values in shifted.values():
        values["delta_count_mae"] = values["count_mae"] - clean["count_mae"]
        if clean["risk_accuracy"] is not None and values["risk_accuracy"] is not None:
            values["delta_risk_accuracy"] = values["risk_accuracy"] - clean["risk_accuracy"]
    payload = {"dataset": dataset_name, "clean": clean, "degradations": shifted}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
