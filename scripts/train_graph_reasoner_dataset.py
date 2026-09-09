from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from crowd_twin.config import load_config
from crowd_twin.contracts import GRAPH_HANDOFF_SCHEMA_VERSION, STATE_FEATURES
from crowd_twin.models.graph_reasoner import GraphRiskReasoner
from crowd_twin.strfe_training import build_strfe_dataset, load_strfe_checkpoint
from crowd_twin.venue import VenueGraph, node_static_features


def _strfe_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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


def _scaled_state_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(target.ndim - 1))
    scale = target.detach().abs().amax(dim=dimensions).clamp_min(1.0)
    return functional.smooth_l1_loss(prediction / scale, target / scale)


def _risk_labels(state: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    thresholds = thresholds.to(state.device)
    return torch.bucketize(state[..., STATE_FEATURES.index("risk")].contiguous(), thresholds)


def _fit_risk_thresholds(dataset: Any, quantiles: list[float]) -> torch.Tensor:
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("risk_class_quantiles must contain values between zero and one")
    if sorted(quantiles) != quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError("risk_class_quantiles must be unique and increasing")
    values = np.concatenate(
        [
            sequence.zone_state[..., STATE_FEATURES.index("risk")].reshape(-1)
            for sequence in dataset.sequences
            if sequence.zone_state is not None
        ]
    )
    thresholds = np.quantile(values, quantiles).astype(np.float32)
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("training risk distribution cannot produce distinct class thresholds")
    return torch.from_numpy(thresholds)


def _run_epoch(
    graph: GraphRiskReasoner,
    strfe: torch.nn.Module,
    loader: DataLoader,
    adjacency: torch.Tensor,
    static: torch.Tensor,
    device: torch.device,
    risk_thresholds: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
    detailed: bool = False,
    risk_labels: list[str] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    graph.train(training)
    totals = {"loss": 0.0, "count_mae": 0.0, "risk_correct": 0.0, "risk_total": 0.0}
    batches = 0
    confusion = torch.zeros(
        (graph.risk_classes, graph.risk_classes), dtype=torch.int64
    )
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            temporal = strfe(**_strfe_inputs(batch, device))
        current_target = batch["current_zone_state"].to(device).float()
        future_target = batch["future_zone_state"].to(device).float()
        outputs = graph(
            current_zone_features=temporal["current_zone_features"],
            future_zone_features=temporal["future_zone_features"],
            adjacency=adjacency,
            node_static=static,
            current_state_prior=temporal["current_zone_state"],
            future_state_prior=temporal["future_zone_state"],
        )
        current_labels = _risk_labels(current_target, risk_thresholds)
        future_labels = _risk_labels(future_target, risk_thresholds)
        state_loss = _scaled_state_loss(outputs["current_zone_state"], current_target)
        forecast_loss = _scaled_state_loss(outputs["future_zone_state"], future_target)
        risk_loss = functional.cross_entropy(
            outputs["risk_logits"].reshape(-1, graph.risk_classes), current_labels.reshape(-1)
        )
        future_risk_loss = functional.cross_entropy(
            outputs["future_risk_logits"].reshape(-1, graph.risk_classes),
            future_labels.reshape(-1),
        )
        hazard_target = (current_labels >= 2).float()
        hazard_loss = functional.binary_cross_entropy(
            outputs["hazard_probability"].clamp(1e-6, 1 - 1e-6), hazard_target
        )
        loss = state_loss + forecast_loss + 0.25 * (risk_loss + future_risk_loss) + 0.1 * hazard_loss
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(graph.parameters(), gradient_clip)
            optimizer.step()
        predicted_labels = outputs["risk_logits"].argmax(dim=-1)
        flat_truth = current_labels.detach().cpu().reshape(-1)
        flat_prediction = predicted_labels.detach().cpu().reshape(-1)
        confusion += torch.bincount(
            flat_truth * graph.risk_classes + flat_prediction,
            minlength=graph.risk_classes**2,
        ).reshape(graph.risk_classes, graph.risk_classes)
        totals["loss"] += float(loss.detach())
        totals["count_mae"] += float(
            (outputs["current_zone_state"][..., 0] - current_target[..., 0])
            .abs()
            .mean()
            .detach()
        )
        totals["risk_correct"] += float((predicted_labels == current_labels).sum())
        totals["risk_total"] += float(current_labels.numel())
        batches += 1
    result: dict[str, Any] = {
        "loss": totals["loss"] / max(batches, 1),
        "count_mae": totals["count_mae"] / max(batches, 1),
        "risk_accuracy": totals["risk_correct"] / max(totals["risk_total"], 1.0),
    }
    if detailed:
        labels = risk_labels or [f"class_{index}" for index in range(graph.risk_classes)]
        per_class = {}
        for index, label in enumerate(labels):
            true_positive = int(confusion[index, index])
            false_positive = int(confusion[:, index].sum()) - true_positive
            false_negative = int(confusion[index, :].sum()) - true_positive
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            per_class[label] = {
                "support": int(confusion[index, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        result["confusion_matrix"] = confusion.tolist()
        result["per_class"] = per_class
        result["macro_f1"] = float(
            np.mean([metrics["f1"] for metrics in per_class.values()])
        )
    return result


def main() -> None:
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(
        description="Train Graph Transformer on real DroneCrowd STRFE windows"
    )
    parser.add_argument("--config", default="configs/strfe_dronecrowd_research.yaml")
    parser.add_argument("--strfe-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/graph_reasoner_dronecrowd_research/best.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=23037)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    requested = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    config = load_config(args.config)
    graph_config = config["graph_reasoner"]
    graph_model_config = graph_config["model"]
    graph_training_config = graph_config["training"]
    epochs = int(
        args.epochs if args.epochs is not None else graph_training_config["epochs"]
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else graph_training_config["batch_size"]
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else graph_training_config["learning_rate"]
    )
    patience = int(
        args.patience
        if args.patience is not None
        else graph_training_config["early_stopping_patience"]
    )
    venue = VenueGraph.load(config["architecture"]["venue_graph"])
    strfe, _strfe_checkpoint = load_strfe_checkpoint(args.strfe_checkpoint, device)
    strfe.requires_grad_(False)
    datasets = {
        split: build_strfe_dataset(config, split, venue) for split in ("train", "val", "test")
    }
    risk_quantiles = [float(value) for value in config["evaluation"]["risk_class_quantiles"]]
    risk_labels = [str(value) for value in graph_model_config["risk_class_labels"]]
    if len(risk_labels) != len(risk_quantiles) + 1 or len(set(risk_labels)) != len(risk_labels):
        raise ValueError("risk_class_labels must be unique and one longer than risk quantiles")
    risk_thresholds = _fit_risk_thresholds(datasets["train"], risk_quantiles).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model_config = {
        "feature_dim": strfe.feature_dim,
        "hidden_dim": int(graph_model_config["hidden_dim"]),
        "static_feature_dim": int(graph_model_config["static_feature_dim"]),
        "graph_heads": int(graph_model_config["attention_heads"]),
        "graph_layers": int(graph_model_config["layers"]),
        "risk_classes": len(risk_quantiles) + 1,
        "horizons_seconds": list(strfe.horizons),
        "dropout": float(graph_model_config["dropout"]),
    }
    graph = GraphRiskReasoner(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=learning_rate,
        weight_decay=float(graph_training_config["weight_decay"]),
    )
    adjacency = torch.tensor(venue.adjacency, dtype=torch.bool, device=device)
    static = torch.from_numpy(node_static_features(venue, feature_dim=5)).to(device)
    best = float("inf")
    stale = 0
    history = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        train = _run_epoch(
            graph,
            strfe,
            loaders["train"],
            adjacency,
            static,
            device,
            risk_thresholds,
            optimizer,
            float(graph_training_config["gradient_clip"]),
        )
        with torch.inference_mode():
            validation = _run_epoch(
                graph, strfe, loaders["val"], adjacency, static, device, risk_thresholds
            )
        record = {"epoch": epoch + 1, "train": train, "validation": validation}
        history.append(record)
        print(json.dumps(record))
        if validation["loss"] < best:
            best = validation["loss"]
            stale = 0
            checkpoint = {
                "format_version": 1,
                "checkpoint_kind": "dataset_trained_proxy_supervision",
                "warning": (
                    "Risk labels are reproducible density/congestion proxies, not human-labelled emergencies."
                ),
                "model_state": graph.state_dict(),
                "model_config": model_config,
                "model_metadata": graph.metadata.to_dict(),
                "integration_contract": {
                    "schema_version": GRAPH_HANDOFF_SCHEMA_VERSION,
                    "venue_id": venue.venue_id,
                    "zone_ids": list(venue.zone_ids),
                    "state_features": list(STATE_FEATURES),
                    "feature_dim": strfe.feature_dim,
                    "horizons_seconds": list(strfe.horizons),
                    "static_feature_dim": 5,
                    "temporal_unit": str(config["data"]["temporal_unit"]),
                },
                "risk_labels": risk_labels,
                "risk_class_thresholds": risk_thresholds.detach().cpu().tolist(),
                "risk_class_quantiles": risk_quantiles,
                "risk_score_semantics": "expected_ordinal_pressure_class",
                "epoch": epoch,
                "best_validation_loss": best,
                "source_datasets": ["DroneCrowd"],
                "source_strfe_checkpoint": str(Path(args.strfe_checkpoint).resolve()),
                "source_strfe_sha256": hashlib.sha256(
                    Path(args.strfe_checkpoint).read_bytes()
                ).hexdigest(),
                "created_at_unix": time.time(),
                "seed": args.seed,
            }
            temporary = output.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    graph.load_state_dict(checkpoint["model_state"], strict=True)
    with torch.inference_mode():
        test = _run_epoch(
            graph,
            strfe,
            loaders["test"],
            adjacency,
            static,
            device,
            risk_thresholds,
            detailed=True,
            risk_labels=risk_labels,
        )
    result = {
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "test": test,
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "risk_supervision": "density_congestion_proxy",
        "risk_class_thresholds": risk_thresholds.detach().cpu().tolist(),
        "risk_class_quantiles": risk_quantiles,
        "history": history,
    }
    (output.parent / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    confusion_array = np.asarray(test["confusion_matrix"], dtype=np.int64)
    np.savetxt(output.parent / "risk_confusion_matrix.csv", confusion_array, fmt="%d", delimiter=",")
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(confusion_array, cmap="Blues")
        axis.set_xticks(range(len(risk_labels)), risk_labels, rotation=35, ha="right")
        axis.set_yticks(range(len(risk_labels)), risk_labels)
        axis.set_xlabel("Predicted proxy class")
        axis.set_ylabel("Target proxy class")
        axis.set_title("Graph reasoner test confusion matrix")
        for row in range(len(risk_labels)):
            for column in range(len(risk_labels)):
                axis.text(column, row, str(confusion_array[row, column]), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output.parent / "risk_confusion_matrix.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
