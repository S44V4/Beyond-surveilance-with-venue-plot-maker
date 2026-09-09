from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .venue import VenueGraph, node_static_features

GRAPH_HANDOFF_SCHEMA_VERSION = "1.0"
STATE_FEATURES: tuple[str, ...] = (
    "count",
    "density",
    "velocity",
    "acceleration",
    "divergence",
    "congestion",
    "risk",
)


@dataclass(frozen=True, slots=True)
class GraphHandoffMetadata:
    """Versioned boundary between STRFE and graph intelligence.

    The zone ordering and feature ordering are part of the data, not informal
    conventions. A consumer must reject a payload that does not match its
    venue/checkpoint instead of silently permuting graph nodes.
    """

    schema_version: str
    venue_id: str
    sequence_id: str
    zone_ids: tuple[str, ...]
    horizons_seconds: tuple[int, ...]
    feature_dim: int
    temporal_unit: str = "seconds"
    state_features: tuple[str, ...] = STATE_FEATURES
    normalization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphHandoffMetadata:
        return cls(
            schema_version=str(payload["schema_version"]),
            venue_id=str(payload["venue_id"]),
            sequence_id=str(payload["sequence_id"]),
            zone_ids=tuple(str(value) for value in payload["zone_ids"]),
            horizons_seconds=tuple(int(value) for value in payload["horizons_seconds"]),
            feature_dim=int(payload["feature_dim"]),
            temporal_unit=str(payload.get("temporal_unit", "seconds")),
            state_features=tuple(str(value) for value in payload.get("state_features", STATE_FEATURES)),
            normalization=dict(payload.get("normalization", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}

    def validate(self) -> None:
        if self.schema_version != GRAPH_HANDOFF_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported graph handoff schema {self.schema_version!r}; "
                f"expected {GRAPH_HANDOFF_SCHEMA_VERSION!r}"
            )
        if not self.venue_id or not self.sequence_id:
            raise ValueError("venue_id and sequence_id must be non-empty")
        if not self.zone_ids or len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("zone_ids must be a non-empty ordered list of unique identifiers")
        if not self.horizons_seconds or any(value <= 0 for value in self.horizons_seconds):
            raise ValueError("horizons_seconds must contain positive values")
        if tuple(sorted(self.horizons_seconds)) != self.horizons_seconds:
            raise ValueError("horizons_seconds must be strictly ordered")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.temporal_unit not in {"seconds", "normalized_frame_steps"}:
            raise ValueError("temporal_unit must be seconds or normalized_frame_steps")
        if self.state_features != STATE_FEATURES:
            raise ValueError(
                f"state_features order mismatch: expected {list(STATE_FEATURES)}, "
                f"received {list(self.state_features)}"
            )


@dataclass(slots=True)
class GraphHandoff:
    metadata: GraphHandoffMetadata
    current_zone_features: np.ndarray
    future_zone_features: np.ndarray
    future_zone_state: np.ndarray
    current_zone_state: np.ndarray | None = None
    sensor_coverage: np.ndarray | None = None

    def validate(
        self,
        venue: VenueGraph | None = None,
        expected_feature_dim: int | None = None,
        expected_horizons: Sequence[int] | None = None,
    ) -> None:
        self.metadata.validate()
        current = np.asarray(self.current_zone_features)
        future = np.asarray(self.future_zone_features)
        future_state = np.asarray(self.future_zone_state)
        if current.ndim not in (2, 3):
            raise ValueError("current_zone_features must be [N,D] or [B,N,D]")
        if future.ndim not in (3, 4):
            raise ValueError("future_zone_features must be [H,N,D] or [B,H,N,D]")
        if future_state.ndim not in (3, 4):
            raise ValueError("future_zone_state must be [H,N,7] or [B,H,N,7]")

        current_batched = current[None] if current.ndim == 2 else current
        future_batched = future[None] if future.ndim == 3 else future
        state_batched = future_state[None] if future_state.ndim == 3 else future_state
        batch, nodes, feature_dim = current_batched.shape
        expected_shape = (batch, len(self.metadata.horizons_seconds), nodes, feature_dim)
        if future_batched.shape != expected_shape:
            raise ValueError(
                f"future_zone_features shape {future_batched.shape} does not match {expected_shape}"
            )
        expected_state_shape = (batch, len(self.metadata.horizons_seconds), nodes, len(STATE_FEATURES))
        if state_batched.shape != expected_state_shape:
            raise ValueError(
                f"future_zone_state shape {state_batched.shape} does not match {expected_state_shape}"
            )
        if nodes != len(self.metadata.zone_ids):
            raise ValueError("Tensor node dimension does not match zone_ids")
        if feature_dim != self.metadata.feature_dim:
            raise ValueError("Tensor feature dimension does not match metadata.feature_dim")
        if not np.isfinite(current_batched).all() or not np.isfinite(future_batched).all():
            raise ValueError("Graph handoff features contain NaN or infinity")
        if not np.isfinite(state_batched).all():
            raise ValueError("Graph handoff state contains NaN or infinity")
        if self.current_zone_state is not None:
            state = np.asarray(self.current_zone_state)
            state = state[None] if state.ndim == 2 else state
            if state.shape != (batch, nodes, len(STATE_FEATURES)):
                raise ValueError("current_zone_state must be [N,7] or [B,N,7]")
            if not np.isfinite(state).all():
                raise ValueError("current_zone_state contains NaN or infinity")
        if self.sensor_coverage is not None:
            coverage = np.asarray(self.sensor_coverage, dtype=np.float32).reshape(-1)
            if coverage.shape != (batch,):
                raise ValueError("sensor_coverage must contain one value per batch item")
            if not np.isfinite(coverage).all() or ((coverage < 0) | (coverage > 1)).any():
                raise ValueError("sensor_coverage must contain finite values between zero and one")

        if venue is not None:
            venue.validate()
            if self.metadata.venue_id != venue.venue_id:
                raise ValueError(
                    f"Venue mismatch: handoff={self.metadata.venue_id!r}, venue={venue.venue_id!r}"
                )
            if self.metadata.zone_ids != venue.zone_ids:
                raise ValueError(
                    "Zone-order mismatch between STRFE handoff and venue graph; "
                    "the payload cannot be consumed safely"
                )
        if expected_feature_dim is not None and feature_dim != expected_feature_dim:
            raise ValueError(
                f"Feature-dimension mismatch: handoff={feature_dim}, checkpoint={expected_feature_dim}"
            )
        if expected_horizons is not None and self.metadata.horizons_seconds != tuple(expected_horizons):
            raise ValueError(
                "Forecast-horizon mismatch between STRFE handoff and graph checkpoint"
            )

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "current_zone_features": np.asarray(self.current_zone_features, dtype=np.float32),
            "future_zone_features": np.asarray(self.future_zone_features, dtype=np.float32),
            "future_zone_state": np.asarray(self.future_zone_state, dtype=np.float32),
            "metadata_json": np.asarray(json.dumps(self.metadata.to_dict(), sort_keys=True)),
        }
        if self.current_zone_state is not None:
            arrays["current_zone_state"] = np.asarray(self.current_zone_state, dtype=np.float32)
        if self.sensor_coverage is not None:
            arrays["sensor_coverage"] = np.asarray(self.sensor_coverage, dtype=np.float32)
        return arrays

    def save(self, path: str | Path) -> tuple[Path, Path]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        np.savez_compressed(output, **self.arrays())
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(json.dumps(self.metadata.to_dict(), indent=2), encoding="utf-8")
        return output, metadata_path

    @classmethod
    def _from_archive(cls, archive: Any) -> GraphHandoff:
        if "metadata_json" not in archive.files:
            raise ValueError("Graph handoff archive has no embedded metadata_json")
        metadata_raw = archive["metadata_json"].item()
        metadata = GraphHandoffMetadata.from_dict(json.loads(str(metadata_raw)))
        handoff = cls(
            metadata=metadata,
            current_zone_features=np.asarray(archive["current_zone_features"], dtype=np.float32),
            future_zone_features=np.asarray(archive["future_zone_features"], dtype=np.float32),
            future_zone_state=np.asarray(archive["future_zone_state"], dtype=np.float32),
            current_zone_state=(
                np.asarray(archive["current_zone_state"], dtype=np.float32)
                if "current_zone_state" in archive.files
                else None
            ),
            sensor_coverage=(
                np.asarray(archive["sensor_coverage"], dtype=np.float32)
                if "sensor_coverage" in archive.files
                else None
            ),
        )
        handoff.validate()
        return handoff

    @classmethod
    def load(cls, path: str | Path) -> GraphHandoff:
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls._from_archive(archive)

    @classmethod
    def load_bytes(cls, content: bytes) -> GraphHandoff:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            return cls._from_archive(archive)

    def tensors(
        self, venue: VenueGraph, static_feature_dim: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        self.validate(venue=venue)

        def batched(values: np.ndarray, unbatched_rank: int) -> torch.Tensor:
            tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
            if tensor.ndim == unbatched_rank:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device)

        payload = {
            "current_zone_features": batched(self.current_zone_features, 2),
            "future_zone_features": batched(self.future_zone_features, 3),
            "future_state_prior": batched(self.future_zone_state, 3),
            "adjacency": torch.tensor(venue.adjacency, dtype=torch.bool, device=device),
            "node_static": torch.from_numpy(
                node_static_features(venue, feature_dim=static_feature_dim)
            ).to(device),
        }
        if self.current_zone_state is not None:
            payload["current_state_prior"] = batched(self.current_zone_state, 2)
        return payload


def integration_contract(
    config: Mapping[str, Any], venue: VenueGraph | None
) -> dict[str, Any]:
    model = config["model"]
    architecture = config.get("architecture", {})
    return {
        "schema_version": str(
            architecture.get("integration_schema_version", GRAPH_HANDOFF_SCHEMA_VERSION)
        ),
        "venue_id": venue.venue_id if venue is not None else None,
        "zone_ids": list(venue.zone_ids) if venue is not None else None,
        "state_features": list(STATE_FEATURES),
        "feature_dim": int(model["feature_dim"]),
        "static_feature_dim": int(model.get("static_node_features", 4)),
        "horizons_seconds": [int(value) for value in model["horizons_seconds"]],
        "temporal_unit": str(config.get("data", {}).get("temporal_unit", "seconds")),
    }


def validate_contract(
    contract: Mapping[str, Any], config: Mapping[str, Any], venue: VenueGraph | None
) -> None:
    expected = integration_contract(config, venue)
    for key in (
        "schema_version",
        "venue_id",
        "zone_ids",
        "state_features",
        "feature_dim",
        "static_feature_dim",
        "horizons_seconds",
        "temporal_unit",
    ):
        if contract.get(key) != expected[key]:
            raise ValueError(
                f"Checkpoint integration contract mismatch for {key}: "
                f"checkpoint={contract.get(key)!r}, runtime={expected[key]!r}"
            )
