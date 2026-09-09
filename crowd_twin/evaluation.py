from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import resize_targets_like
from .models.factory import build_model
from .training import build_dataset, load_checkpoint, seed_everything


class MetricAccumulator:
    def __init__(self, risk_classes: int = 4, localization_threshold: float = 0.35) -> None:
        self.count_errors: list[float] = []
        self.forecast_errors: list[float] = []
        self.risk_predictions: list[int] = []
        self.risk_targets: list[int] = []
        self.risk_probabilities: list[list[float]] = []
        self.latencies_ms: list[float] = []
        self.risk_classes = risk_classes
        self.localization_threshold = localization_threshold
        self.localization_tp = 0
        self.localization_fp = 0
        self.localization_fn = 0

    def update(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any], latency_ms: float
    ) -> None:
        prediction = outputs["density"][:, -1].flatten(1).sum(dim=1)
        target = batch["density_target"][:, -1].flatten(1).sum(dim=1)
        self.count_errors.extend((prediction - target).detach().abs().cpu().tolist())
        future_mask = batch["future_mask"][..., None]
        future_error = (
            (outputs["future_zone_state"][..., 0] - batch["future_zone_counts"]).abs()
            * future_mask
        )
        denominator = (
            future_mask.sum(dim=(1, 2)) * future_error.shape[-1]
        ).clamp_min(1.0)
        per_sample = future_error.sum(dim=(1, 2)) / denominator
        self.forecast_errors.extend(per_sample.detach().cpu().tolist())
        valid = batch["risk_label"] >= 0
        if valid.any():
            prediction = outputs["risk_logits"].mean(dim=1).argmax(dim=-1)
            probabilities = torch.softmax(outputs["risk_logits"].mean(dim=1), dim=-1)
            self.risk_predictions.extend(prediction[valid].detach().cpu().tolist())
            self.risk_targets.extend(batch["risk_label"][valid].detach().cpu().tolist())
            self.risk_probabilities.extend(probabilities[valid].detach().cpu().tolist())
        localization_target = torch.nn.functional.interpolate(
            batch["localization_target"][:, -1],
            size=outputs["localization_logits"].shape[-2:],
            mode="nearest",
        ).bool()
        localization_prediction = (
            torch.sigmoid(outputs["localization_logits"][:, -1])
            >= self.localization_threshold
        )
        self.localization_tp += int((localization_prediction & localization_target).sum())
        self.localization_fp += int((localization_prediction & ~localization_target).sum())
        self.localization_fn += int((~localization_prediction & localization_target).sum())
        self.latencies_ms.append(latency_ms)

    def compute(self) -> dict[str, float | int | None]:
        errors = np.asarray(self.count_errors, dtype=np.float64)
        forecast = np.asarray(self.forecast_errors, dtype=np.float64)
        risk_accuracy = None
        if self.risk_targets:
            risk_accuracy = float(
                np.mean(np.asarray(self.risk_predictions) == np.asarray(self.risk_targets))
            )
        precision = self.localization_tp / max(self.localization_tp + self.localization_fp, 1)
        recall = self.localization_tp / max(self.localization_tp + self.localization_fn, 1)
        classification: dict[str, Any] | None = None
        if self.risk_targets:
            from sklearn.metrics import classification_report

            classification = classification_report(
                self.risk_targets,
                self.risk_predictions,
                labels=list(range(self.risk_classes)),
                output_dict=True,
                zero_division=0,
            )
        return {
            "samples": len(errors),
            "count_mae": float(errors.mean()) if len(errors) else math.nan,
            "count_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
            "forecast_mae": float(forecast.mean()) if len(forecast) else math.nan,
            "risk_accuracy": risk_accuracy,
            "risk_per_class": classification,
            "localization_precision": precision,
            "localization_recall": recall,
            "localization_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "latency_ms_mean": float(np.mean(self.latencies_ms)) if self.latencies_ms else math.nan,
            "latency_ms_p95": float(np.percentile(self.latencies_ms, 95))
            if self.latencies_ms
            else math.nan,
        }


def save_classification_artifacts(
    accumulator: MetricAccumulator, output_dir: str | Path, dataset_name: str
) -> list[str]:
    """Save confusion matrix, one-vs-rest ROC/PR curves, and raw prediction table."""
    if not accumulator.risk_targets:
        return []
    import csv

    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        auc,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    output = Path(output_dir) / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    labels = list(range(accumulator.risk_classes))
    targets = np.asarray(accumulator.risk_targets)
    predictions = np.asarray(accumulator.risk_predictions)
    probabilities = np.asarray(accumulator.risk_probabilities)
    matrix = confusion_matrix(targets, predictions, labels=labels)
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()

    fig, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for class_index in labels:
        binary = (targets == class_index).astype(np.uint8)
        if binary.min() == binary.max():
            continue
        false_positive, true_positive, _ = roc_curve(binary, probabilities[:, class_index])
        precision, recall, _ = precision_recall_curve(
            binary, probabilities[:, class_index]
        )
        roc_axis.plot(false_positive, true_positive, label=f"class {class_index} AUC={auc(false_positive, true_positive):.3f}")
        pr_axis.plot(recall, precision, label=f"class {class_index} AUC={auc(recall, precision):.3f}")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall")
    roc_axis.legend(fontsize=7)
    pr_axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "roc_pr_curves.png", dpi=160)
    plt.close(fig)
    with (output / "risk_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "prediction", *[f"probability_{i}" for i in labels]])
        writer.writerows(
            [target, prediction, *probability]
            for target, prediction, probability in zip(targets, predictions, probabilities)
        )
    return [str(path.resolve()) for path in sorted(output.iterdir())]


def evaluate_dataset(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    dataset_name: str,
    split: str = "test",
    registry_path: str | Path = "configs/datasets.yaml",
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint["config"]
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dataset = build_dataset(checkpoint_config, split, [dataset_name], False, registry_path)
    loader = DataLoader(
        dataset,
        batch_size=int(checkpoint_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(checkpoint_config["data"]["num_workers"]),
    )
    metrics = MetricAccumulator(
        int(checkpoint_config["model"]["risk_classes"]),
        float(checkpoint_config.get("evaluation", {}).get("localization_threshold", 0.35)),
    )
    with torch.inference_mode():
        for batch in loader:
            for key in (
                "frames",
                "flow",
                "sensors",
                "sensor_mask",
                "density_target",
                "localization_target",
                "future_zone_counts",
                "future_mask",
                "risk_label",
            ):
                batch[key] = batch[key].to(device)
            for key in ("zone_masks", "adjacency", "node_static"):
                if key in batch:
                    batch[key] = batch[key].to(device)
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
            latency_ms = (time.perf_counter() - started) * 1000
            batch["density_target"] = resize_targets_like(
                batch["density_target"], outputs["density"]
            )
            metrics.update(outputs, batch, latency_ms)
    artifacts = (
        save_classification_artifacts(metrics, artifacts_dir, dataset_name)
        if artifacts_dir is not None
        else []
    )
    result = {
        "dataset": dataset_name,
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source_datasets": checkpoint.get("source_datasets", []),
        "temporal_backend": checkpoint.get("model_metadata", {}).get("temporal_backend"),
        "metrics": metrics.compute(),
        "artifacts": artifacts,
    }
    return result


def evaluate_matrix(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    datasets: list[str],
    output_path: str | Path,
    registry_path: str | Path = "configs/datasets.yaml",
) -> dict[str, Any]:
    artifacts_dir = Path(output_path).with_suffix("").parent / "evaluation_artifacts"
    results = {
        dataset: evaluate_dataset(
            config,
            checkpoint_path,
            dataset,
            registry_path=registry_path,
            artifacts_dir=artifacts_dir,
        )
        for dataset in datasets
    }
    payload = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_specific": results,
        "cross_dataset": {
            name: result
            for name, result in results.items()
            if name not in result["source_datasets"]
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
