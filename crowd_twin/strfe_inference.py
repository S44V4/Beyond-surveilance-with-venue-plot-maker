from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crowd_twin.config import load_config
from crowd_twin.data.strfe import CachedDDPFSequence
from crowd_twin.graph_inference import GraphHandoffPredictor
from crowd_twin.strfe_training import load_strfe_checkpoint
from crowd_twin.venue import VenueGraph


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class STRFEGraphPredictor:
    """Checkpoint-gated cached-DDPF -> STRFE -> Graph Transformer inference."""

    def __init__(
        self,
        config_path: str | Path,
        strfe_checkpoint_path: str | Path,
        graph_checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.config = load_config(config_path)
        self.context_frames = int(self.config["data"]["context_frames"])
        self.venue_path = Path(self.config["architecture"]["venue_graph"])
        self.venue = VenueGraph.load(self.venue_path)
        self.strfe_checkpoint_path = Path(strfe_checkpoint_path).resolve()
        self.strfe, self.strfe_checkpoint = load_strfe_checkpoint(
            self.strfe_checkpoint_path, self.device
        )
        self.graph = GraphHandoffPredictor(
            graph_checkpoint_path, self.venue_path, device=requested
        )
        contract = self.strfe_checkpoint.get("integration_contract")
        if not isinstance(contract, dict):
            raise TypeError("STRFE checkpoint has no integration_contract")
        expected = {
            "venue_id": self.venue.venue_id,
            "zone_ids": list(self.venue.zone_ids),
            "feature_dim": int(self.graph.contract["feature_dim"]),
            "horizons_seconds": list(self.graph.contract["horizons_seconds"]),
            "temporal_unit": str(self.config["data"].get("temporal_unit", "seconds")),
        }
        for key, value in expected.items():
            actual = contract.get(key)
            if (
                key == "temporal_unit"
                and actual is None
                and self.strfe_checkpoint.get("checkpoint_kind")
                == "synthetic_integration_fixture"
            ):
                actual = "seconds"
            if actual != value:
                raise ValueError(f"STRFE-to-Graph checkpoint mismatch for {key}")
        self.strfe_sha256 = _sha256(self.strfe_checkpoint_path)

    @torch.inference_mode()
    def predict_path(self, cache_path: str | Path) -> dict[str, Any]:
        return self.predict_sequence(CachedDDPFSequence.load(cache_path))

    @torch.inference_mode()
    def predict_sequence(self, sequence: CachedDDPFSequence) -> dict[str, Any]:
        if sequence.metadata.venue_id != self.venue.venue_id:
            raise ValueError("cached DDPF venue does not match the configured venue")
        if sequence.metadata.zone_ids != self.venue.zone_ids:
            raise ValueError("cached DDPF zone ordering does not match the venue")
        if len(sequence.metadata.frame_indices) < self.context_frames:
            raise ValueError("cached DDPF sequence is shorter than STRFE context_frames")
        if sequence.metadata.feature_channels != self.strfe.input_channels:
            raise ValueError("cached DDPF feature channels do not match the STRFE checkpoint")
        if sequence.metadata.sensor_features and (
            sequence.metadata.sensor_features != self.strfe.sensor_features
        ):
            raise ValueError("cached DDPF sensor feature order does not match STRFE")
        context = slice(max(0, len(sequence.metadata.frame_indices) - self.context_frames), None)
        sensor_dim = len(self.strfe.sensor_features)
        sensors = (
            sequence.sensors[context]
            if sequence.sensors is not None and sequence.sensors.shape[1] == sensor_dim
            else np.zeros((self.context_frames, sensor_dim), dtype=np.float32)
        )
        sensor_mask = (
            sequence.sensor_mask[context]
            if sequence.sensor_mask is not None and sequence.sensor_mask.shape[1] == sensor_dim
            else np.zeros_like(sensors)
        )

        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(self.device)

        output = self.strfe(
            visual_features=tensor(sequence.visual_features[context]),
            density=tensor(sequence.density[context]),
            localization_logits=tensor(sequence.localization_logits[context]),
            flow=tensor(sequence.flow[context]),
            zone_masks=tensor(sequence.zone_masks)[0],
            sensors=tensor(sensors),
            sensor_mask=tensor(sensor_mask),
        )
        handoff = self.strfe.to_graph_handoff(
            output,
            venue_id=self.venue.venue_id,
            sequence_id=sequence.metadata.sequence_id,
            zone_ids=self.venue.zone_ids,
            temporal_unit=str(self.config["data"].get("temporal_unit", "seconds")),
            normalization={
                "visual_features": "frozen_ddpf_export",
                "flow": sequence.metadata.flow_convention,
                "sensor_config": self.config.get("sensors", {}).get("normalization"),
            },
        )
        prediction = self.graph.predict(handoff)
        prediction["pipeline"] = {
            "input": "cached_ddpf_sequence",
            "input_source": sequence.metadata.source,
            "sequence_id": sequence.metadata.sequence_id,
            "strfe_checkpoint_sha256": self.strfe_sha256,
            "strfe_checkpoint_kind": self.strfe_checkpoint.get("checkpoint_kind", "unknown"),
            "graph_checkpoint_sha256": self.graph.checkpoint_sha256,
            "graph_checkpoint_kind": self.graph.checkpoint.get("checkpoint_kind", "unknown"),
        }
        return prediction
