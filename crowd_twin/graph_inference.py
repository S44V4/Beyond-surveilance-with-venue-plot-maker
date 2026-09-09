from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .alerts import build_model_alerts
from .contracts import STATE_FEATURES, GraphHandoff
from .models.graph_reasoner import GraphRiskReasoner
from .venue import VenueGraph


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GraphHandoffPredictor:
    """Strict checkpoint-backed inference over an exported STRFE handoff."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        venue_path: str | Path,
        device: str | None = None,
    ) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.venue_path = Path(venue_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.venue = VenueGraph.load(self.venue_path)
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        model_config = self.checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise TypeError("Graph checkpoint has no model_config")
        contract = self.checkpoint.get("integration_contract")
        if not isinstance(contract, dict):
            raise TypeError("Graph checkpoint has no integration_contract")
        if contract.get("venue_id") != self.venue.venue_id:
            raise ValueError("Graph checkpoint venue_id does not match the loaded venue")
        if tuple(contract.get("zone_ids", ())) != self.venue.zone_ids:
            raise ValueError("Graph checkpoint zone order does not match the loaded venue")
        if tuple(contract.get("state_features", ())) != STATE_FEATURES:
            raise ValueError("Graph checkpoint state-feature order is incompatible")
        self.temporal_unit = str(contract.get("temporal_unit", "seconds"))
        labels = self.checkpoint.get("risk_labels", ("stable", "watch", "high", "critical"))
        self.risk_labels = tuple(str(value) for value in labels)
        if len(self.risk_labels) != int(model_config["risk_classes"]):
            raise ValueError("Graph checkpoint risk labels do not match risk_classes")
        self.model = GraphRiskReasoner(**model_config).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"], strict=True)
        self.model.eval()
        self.contract = contract
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)

    @torch.inference_mode()
    def predict(self, handoff: GraphHandoff) -> dict[str, Any]:
        started = time.perf_counter()
        handoff.validate(
            venue=self.venue,
            expected_feature_dim=int(self.contract["feature_dim"]),
            expected_horizons=self.contract["horizons_seconds"],
        )
        if handoff.metadata.temporal_unit != self.temporal_unit:
            raise ValueError("Graph handoff temporal unit does not match the checkpoint")
        tensors = handoff.tensors(
            self.venue,
            static_feature_dim=int(self.contract["static_feature_dim"]),
            device=self.device,
        )
        outputs = self.model(**tensors)
        states = outputs["current_zone_state"][0].cpu().numpy()
        future_states = outputs["future_zone_state"][0].cpu().numpy()
        probabilities = torch.softmax(outputs["risk_logits"][0], dim=-1).cpu().numpy()
        future_probabilities = torch.softmax(
            outputs["future_risk_logits"][0], dim=-1
        ).cpu().numpy()
        attention = outputs["graph_attention"][0].mean(dim=0).cpu().numpy()
        confidence = outputs["risk_confidence"][0].cpu().numpy()
        hazard = outputs["hazard_probability"][0].cpu().numpy()
        class_scale = np.linspace(0.0, 1.0, probabilities.shape[-1], dtype=np.float32)
        risk_scores = probabilities @ class_scale
        future_scores = future_probabilities @ class_scale
        alert_class_index = max(1, len(self.risk_labels) // 2)
        x_values = sorted({round(zone.centroid[0], 6) for zone in self.venue.zones})
        y_values = sorted({round(zone.centroid[1], 6) for zone in self.venue.zones})

        zones: list[dict[str, Any]] = []
        for index, zone in enumerate(self.venue.zones):
            predicted_class = int(probabilities[index].argmax())
            strongest = np.argsort(attention[index])[::-1][:3]
            state = {
                name: float(states[index, feature_index])
                for feature_index, name in enumerate(STATE_FEATURES)
            }
            zones.append(
                {
                    "id": zone.zone_id,
                    "name": zone.label,
                    "polygon": zone.polygon,
                    "row": min(
                        range(len(y_values)), key=lambda item: abs(y_values[item] - zone.centroid[1])
                    ),
                    "column": min(
                        range(len(x_values)), key=lambda item: abs(x_values[item] - zone.centroid[0])
                    ),
                    **state,
                    "state": state,
                    "risk": float(risk_scores[index] * 100.0),
                    "risk_level": self.risk_labels[predicted_class],
                    "risk_class_index": predicted_class,
                    "risk_class_count": len(self.risk_labels),
                    "confidence": float(confidence[index] * 100.0),
                    "hazard_probability": float(hazard[index] * 100.0),
                    "attention_neighbors": [
                        {
                            "zone": self.venue.zones[int(neighbor)].zone_id,
                            "weight": float(attention[index, neighbor]),
                        }
                        for neighbor in strongest
                    ],
                }
            )
        forecasts = []
        for horizon_index, horizon in enumerate(self.contract["horizons_seconds"]):
            horizon_zones = []
            for index, zone in enumerate(self.venue.zones):
                predicted_class = int(future_probabilities[horizon_index, index].argmax())
                state = {
                    name: float(future_states[horizon_index, index, feature])
                    for feature, name in enumerate(STATE_FEATURES)
                }
                horizon_zones.append(
                    {
                        "id": zone.zone_id,
                        **state,
                        "state": state,
                        "risk": float(future_scores[horizon_index, index] * 100.0),
                        "risk_level": self.risk_labels[predicted_class],
                        "risk_class_index": predicted_class,
                        "risk_class_count": len(self.risk_labels),
                    }
                )
            forecasts.append(
                {
                    "horizon_seconds": int(horizon),
                    "horizon_value": int(horizon),
                    "horizon_unit": handoff.metadata.temporal_unit,
                    "total_count": float(
                        future_states[horizon_index, :, STATE_FEATURES.index("count")].sum()
                    ),
                    "mean_risk": float(future_scores[horizon_index].mean() * 100.0),
                    "max_risk": float(future_scores[horizon_index].max() * 100.0),
                    "zones": horizon_zones,
                }
            )
        global_summary = {
            "estimated_count": float(states[:, STATE_FEATURES.index("count")].sum()),
            "mean_risk": float(risk_scores.mean() * 100.0),
            "max_risk": float(risk_scores.max() * 100.0),
            "unsafe_zones": int((probabilities.argmax(axis=-1) >= alert_class_index).sum()),
            "mean_confidence": float(confidence.mean() * 100.0),
            "mean_hazard_probability": float(hazard.mean() * 100.0),
        }
        result = {
            "run_id": str(uuid.uuid4()),
            "generated_at_unix": time.time(),
            "source": {
                "kind": "strfe_graph_handoff",
                "sequence_id": handoff.metadata.sequence_id,
            },
            "schema_version": handoff.metadata.schema_version,
            "venue_id": self.venue.venue_id,
            "sequence_id": handoff.metadata.sequence_id,
            "zone_ids": list(self.venue.zone_ids),
            "state_features": list(STATE_FEATURES),
            "horizons_seconds": list(self.contract["horizons_seconds"]),
            "temporal_unit": handoff.metadata.temporal_unit,
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
                "epoch": self.checkpoint.get("epochs"),
                "source_datasets": self.checkpoint.get("source_datasets", []),
                "kind": self.checkpoint.get("checkpoint_kind", "unknown"),
                "training_loss": self.checkpoint.get("training_loss"),
            },
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "architecture": "STRFE handoff -> Dynamic Crowd Graph -> Graph Transformer",
            "venue": {
                "venue_id": self.venue.venue_id,
                "source": self.venue.source,
                "adaptive": True,
            },
            "sensors": {
                "provided": [],
                "coverage": (
                    float(np.asarray(handoff.sensor_coverage).reshape(-1)[0] * 100.0)
                    if handoff.sensor_coverage is not None
                    else 0.0
                ),
            },
            "global": global_summary,
            "zones": zones,
            "forecasts": forecasts,
            "maps": {},
            "attention_matrix": attention.tolist(),
            "risk_provenance": {
                "supervision": self.checkpoint.get("warning"),
                "class_labels": list(self.risk_labels),
                "class_thresholds": self.checkpoint.get("risk_class_thresholds"),
                "class_quantiles": self.checkpoint.get("risk_class_quantiles"),
                "score_semantics": self.checkpoint.get(
                    "risk_score_semantics", "expected_ordinal_class"
                ),
            },
        }
        result["alerts"] = build_model_alerts(zones, forecasts, global_summary)
        return result


def save_graph_prediction_visualizations(
    prediction: dict[str, Any], venue: VenueGraph, output_dir: str | Path
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    risk_by_id = {zone["id"]: zone["risk"] for zone in prediction["zones"]}
    figure, axis = plt.subplots(figsize=(10, 6))
    color_map = plt.get_cmap("RdYlGn_r")
    for zone in venue.zones:
        risk = risk_by_id[zone.zone_id]
        patch = Polygon(
            np.asarray(zone.polygon),
            closed=True,
            facecolor=color_map(np.clip(risk / 100.0, 0, 1)),
            edgecolor="#17201c",
            linewidth=1.5,
        )
        axis.add_patch(patch)
        x, y = zone.centroid
        axis.text(x, y, f"{zone.zone_id}\n{risk:.0f}", ha="center", va="center", fontsize=8)
    axis.set(
        xlim=(0, venue.width),
        ylim=(venue.height, 0),
        title="Current ordinal crowd-pressure index",
    )
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    risk_path = output / "venue_risk.png"
    figure.savefig(risk_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    attention = np.asarray(prediction["attention_matrix"], dtype=np.float32)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(attention, cmap="viridis", vmin=0.0, vmax=max(float(attention.max()), 1e-6))
    axis.set_xticks(range(len(venue.zones)), venue.zone_ids, rotation=45, ha="right")
    axis.set_yticks(range(len(venue.zones)), venue.zone_ids)
    axis.set(xlabel="Attended zone", ylabel="Query zone", title="Graph attention")
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    attention_path = output / "graph_attention.png"
    figure.savefig(attention_path, dpi=160)
    plt.close(figure)

    prediction_path = output / "prediction.json"
    prediction_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    return [risk_path, attention_path, prediction_path]
