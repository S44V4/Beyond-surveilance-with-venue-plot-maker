from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from crowd_twin.contracts import GRAPH_HANDOFF_SCHEMA_VERSION, STATE_FEATURES

Split = Literal["train", "val", "test"]
FLOW_CONVENTION = "farneback_dx_dy_magnitude_divergence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CachedDDPFMetadata:
    """Metadata for one frozen DDPF sequence consumed by STRFE."""

    schema_version: str
    sequence_id: str
    source_sequence: str
    split: Split
    venue_id: str
    zone_ids: tuple[str, ...]
    fps: float
    frame_indices: tuple[int, ...]
    timestamps_ms: tuple[float, ...]
    feature_channels: int
    sensor_features: tuple[str, ...] = ()
    flow_convention: str = FLOW_CONVENTION
    source: str = "cached_ddpf"
    augmentation_provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CachedDDPFMetadata:
        return cls(
            schema_version=str(payload["schema_version"]),
            sequence_id=str(payload["sequence_id"]),
            source_sequence=str(payload.get("source_sequence", payload["sequence_id"])),
            split=str(payload["split"]),  # type: ignore[arg-type]
            venue_id=str(payload["venue_id"]),
            zone_ids=tuple(str(value) for value in payload["zone_ids"]),
            fps=float(payload["fps"]),
            frame_indices=tuple(int(value) for value in payload["frame_indices"]),
            timestamps_ms=tuple(float(value) for value in payload["timestamps_ms"]),
            feature_channels=int(payload["feature_channels"]),
            sensor_features=tuple(str(value) for value in payload.get("sensor_features", ())),
            flow_convention=str(payload.get("flow_convention", FLOW_CONVENTION)),
            source=str(payload.get("source", "cached_ddpf")),
            augmentation_provenance=dict(payload.get("augmentation_provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("zone_ids", "frame_indices", "timestamps_ms", "sensor_features"):
            payload[key] = list(payload[key])
        return payload

    def validate(self) -> None:
        if self.schema_version != GRAPH_HANDOFF_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cached-DDPF schema {self.schema_version!r}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not self.sequence_id or not self.source_sequence or not self.venue_id:
            raise ValueError("sequence_id, source_sequence, and venue_id are required")
        if not self.zone_ids or len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("zone_ids must be ordered and unique")
        if self.fps <= 0 or self.feature_channels <= 0:
            raise ValueError("fps and feature_channels must be positive")
        if self.flow_convention != FLOW_CONVENTION:
            raise ValueError(f"Unsupported flow convention {self.flow_convention!r}")
        if len(self.frame_indices) != len(self.timestamps_ms) or not self.frame_indices:
            raise ValueError("frame_indices and timestamps_ms must be equal non-empty sequences")
        if any(right <= left for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise ValueError("frame_indices must be strictly increasing")
        if any(right <= left for left, right in zip(self.timestamps_ms, self.timestamps_ms[1:])):
            raise ValueError("timestamps_ms must be strictly increasing")


@dataclass(slots=True)
class CachedDDPFSequence:
    metadata: CachedDDPFMetadata
    visual_features: np.ndarray
    density: np.ndarray
    localization_logits: np.ndarray
    flow: np.ndarray
    zone_masks: np.ndarray
    zone_state: np.ndarray | None = None
    sensors: np.ndarray | None = None
    sensor_mask: np.ndarray | None = None

    def validate(self, time_gap_tolerance: float = 0.25) -> None:
        self.metadata.validate()
        visual = np.asarray(self.visual_features)
        density = np.asarray(self.density)
        localization = np.asarray(self.localization_logits)
        flow = np.asarray(self.flow)
        masks = np.asarray(self.zone_masks)
        time = len(self.metadata.frame_indices)
        if visual.ndim != 4 or visual.shape[:2] != (time, self.metadata.feature_channels):
            raise ValueError("visual_features must be [T,C,Hf,Wf]")
        if density.shape != (time, 1, visual.shape[-2], visual.shape[-1]):
            raise ValueError("density must be [T,1,Hf,Wf] and align with visual_features")
        if localization.shape != density.shape:
            raise ValueError("localization_logits must match density")
        if flow.shape != (time, 4, visual.shape[-2], visual.shape[-1]):
            raise ValueError("flow must be [T,4,Hf,Wf]")
        if masks.shape != (len(self.metadata.zone_ids), visual.shape[-2], visual.shape[-1]):
            raise ValueError("zone_masks must be [N,Hf,Wf] in metadata zone order")
        if (masks.sum(axis=(1, 2)) <= 0).any():
            raise ValueError("every zone mask must contain pixels")
        if self.zone_state is not None and np.asarray(self.zone_state).shape != (
            time,
            len(self.metadata.zone_ids),
            len(STATE_FEATURES),
        ):
            raise ValueError("zone_state must be [T,N,7]")
        if self.sensors is not None:
            expected = (time, len(self.metadata.sensor_features))
            if np.asarray(self.sensors).shape != expected:
                raise ValueError("sensors must be [T,S]")
            if self.sensor_mask is None or np.asarray(self.sensor_mask).shape != expected:
                raise ValueError("sensor_mask must match sensors")
        arrays = [visual, density, localization, flow, masks]
        if self.zone_state is not None:
            arrays.append(np.asarray(self.zone_state))
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("cached DDPF sequence contains NaN or infinity")
        if time > 1:
            expected_gap = 1000.0 / self.metadata.fps
            gaps = np.diff(np.asarray(self.metadata.timestamps_ms))
            if np.max(np.abs(gaps - expected_gap)) > expected_gap * time_gap_tolerance:
                raise ValueError("sequence contains invalid timestamp gaps")

    def arrays(self) -> dict[str, np.ndarray]:
        payload = {
            "metadata_json": np.asarray(json.dumps(self.metadata.to_dict(), sort_keys=True)),
            "visual_features": np.asarray(self.visual_features, dtype=np.float32),
            "density": np.asarray(self.density, dtype=np.float32),
            "localization_logits": np.asarray(self.localization_logits, dtype=np.float32),
            "flow": np.asarray(self.flow, dtype=np.float32),
            "zone_masks": np.asarray(self.zone_masks, dtype=np.float32),
        }
        if self.zone_state is not None:
            payload["zone_state"] = np.asarray(self.zone_state, dtype=np.float32)
        if self.sensors is not None:
            payload["sensors"] = np.asarray(self.sensors, dtype=np.float32)
            payload["sensor_mask"] = np.asarray(self.sensor_mask, dtype=np.float32)
        return payload

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.arrays())
        return output

    @classmethod
    def load(cls, path: str | Path) -> CachedDDPFSequence:
        with np.load(Path(path), allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError("cached DDPF archive has no metadata_json")
            metadata = CachedDDPFMetadata.from_dict(json.loads(str(archive["metadata_json"].item())))
            sequence = cls(
                metadata=metadata,
                visual_features=np.asarray(archive["visual_features"], dtype=np.float32),
                density=np.asarray(archive["density"], dtype=np.float32),
                localization_logits=np.asarray(archive["localization_logits"], dtype=np.float32),
                flow=np.asarray(archive["flow"], dtype=np.float32),
                zone_masks=np.asarray(archive["zone_masks"], dtype=np.float32),
                zone_state=(
                    np.asarray(archive["zone_state"], dtype=np.float32)
                    if "zone_state" in archive.files
                    else None
                ),
                sensors=(
                    np.asarray(archive["sensors"], dtype=np.float32)
                    if "sensors" in archive.files
                    else None
                ),
                sensor_mask=(
                    np.asarray(archive["sensor_mask"], dtype=np.float32)
                    if "sensor_mask" in archive.files
                    else None
                ),
            )
        sequence.validate()
        return sequence


def _perception_state_components(
    density: np.ndarray, flow: np.ndarray, masks: np.ndarray
) -> dict[str, np.ndarray]:
    """Extract measurable zone components without assigning risk semantics."""

    density_array = np.asarray(density, dtype=np.float32)
    flow_array = np.asarray(flow, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=np.float32) > 0.5
    areas = mask_array.sum(axis=(1, 2)).clip(min=1).astype(np.float32)
    counts = np.stack(
        [
            np.asarray(
                [np.clip(frame[0], 0, None)[mask].sum() for mask in mask_array],
                dtype=np.float32,
            )
            for frame in density_array
        ]
    )
    normalized_density = counts / areas[None] * 100.0
    velocity = np.stack(
        [
            np.asarray([frame[2][mask].mean() for mask in mask_array], dtype=np.float32)
            for frame in flow_array
        ]
    )
    divergence = np.stack(
        [
            np.asarray([frame[3][mask].mean() for mask in mask_array], dtype=np.float32)
            for frame in flow_array
        ]
    )
    acceleration = np.diff(velocity, axis=0, prepend=np.zeros_like(velocity[:1]))
    convergence = np.clip(-divergence, 0.0, None)
    return {
        "count": counts,
        "density": normalized_density,
        "velocity": velocity,
        "acceleration": acceleration,
        "divergence": divergence,
        "convergence": convergence,
    }


def fit_risk_proxy_calibration(
    sequences: Sequence[CachedDDPFSequence],
    *,
    scale_quantile: float,
    component_weights: dict[str, float],
) -> dict[str, Any]:
    """Fit pressure-proxy scales on training sequences only."""

    if not 0.5 <= scale_quantile < 1.0:
        raise ValueError("scale_quantile must be in [0.5, 1.0)")
    names = ("density", "motion", "convergence")
    weights = {name: float(component_weights.get(name, 0.0)) for name in names}
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("risk-proxy component weights must have a positive sum")
    weights = {name: value / weight_total for name, value in weights.items()}
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    source_sequences = []
    for sequence in sequences:
        if sequence.metadata.split != "train":
            raise ValueError("risk-proxy calibration may use training sequences only")
        components = _perception_state_components(
            sequence.density, sequence.flow, sequence.zone_masks
        )
        values["density"].append(components["density"].reshape(-1))
        values["motion"].append(components["velocity"].reshape(-1))
        values["convergence"].append(components["convergence"].reshape(-1))
        source_sequences.append(sequence.metadata.source_sequence)
    if not source_sequences:
        raise ValueError("at least one training sequence is required for calibration")
    scales = {
        name: max(float(np.quantile(np.concatenate(rows), scale_quantile)), 1e-6)
        for name, rows in values.items()
    }
    return {
        "kind": "training_split_quantile_pressure_proxy",
        "scale_quantile": scale_quantile,
        "component_scales": scales,
        "component_weights": weights,
        "source_split": "train",
        "source_sequences": sorted(source_sequences),
    }


def zone_states_from_perception(
    density: np.ndarray,
    flow: np.ndarray,
    masks: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Build measurable seven-field state targets from cached perception.

    Density is count per 100 feature-map cells, not people per square metre.
    Risk is a documented density/motion/convergence proxy when no human-labelled
    emergency annotation exists.
    """
    components = _perception_state_components(density, flow, masks)
    scales = calibration.get("component_scales")
    weights = calibration.get("component_weights")
    if not isinstance(scales, dict) or not isinstance(weights, dict):
        raise TypeError("risk calibration requires component_scales and component_weights")
    names = ("density", "motion", "convergence")
    if any(float(scales.get(name, 0.0)) <= 0 for name in names):
        raise ValueError("all risk-proxy component scales must be positive")
    weight_values = np.asarray([float(weights.get(name, 0.0)) for name in names], dtype=np.float32)
    if float(weight_values.sum()) <= 0:
        raise ValueError("risk-proxy component weights must have a positive sum")
    weight_values /= weight_values.sum()
    congestion = np.clip(components["density"] / float(scales["density"]), 0.0, 1.0)
    motion_pressure = np.clip(components["velocity"] / float(scales["motion"]), 0.0, 1.0)
    convergence_pressure = np.clip(
        components["convergence"] / float(scales["convergence"]), 0.0, 1.0
    )
    risk = np.clip(
        weight_values[0] * congestion
        + weight_values[1] * motion_pressure
        + weight_values[2] * convergence_pressure,
        0.0,
        1.0,
    )
    # STATE_FEATURES is the authoritative boundary order. Congestion is index
    # five and the pressure proxy is index six.
    return np.stack(
        [
            components["count"],
            components["density"],
            components["velocity"],
            components["acceleration"],
            components["divergence"],
            congestion,
            risk,
        ],
        axis=-1,
    ).astype(np.float32)


def write_strfe_manifest(paths: Sequence[str | Path], output: str | Path) -> Path:
    destination = Path(output).resolve()
    records = []
    for value in paths:
        path = Path(value).resolve()
        sequence = CachedDDPFSequence.load(path)
        try:
            stored_path = str(path.relative_to(destination.parent))
        except ValueError:
            stored_path = str(path)
        records.append(
            {
                "path": stored_path,
                "sha256": _sha256(path),
                "sequence_id": sequence.metadata.sequence_id,
                "source_sequence": sequence.metadata.source_sequence,
                "split": sequence.metadata.split,
                "venue_id": sequence.metadata.venue_id,
                "frames": len(sequence.metadata.frame_indices),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": "1.0", "records": records}, indent=2))
    return destination


def audit_strfe_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    sequence_splits: dict[str, set[str]] = {}
    digest_splits: dict[str, set[str]] = {}
    missing = []
    for record in records:
        sequence_splits.setdefault(str(record["source_sequence"]), set()).add(str(record["split"]))
        digest_splits.setdefault(str(record["sha256"]), set()).add(str(record["split"]))
        record_path = Path(record["path"])
        record_path = record_path if record_path.is_absolute() else manifest_path.parent / record_path
        if not record_path.is_file():
            missing.append(str(record["path"]))
    overlap = sorted(key for key, values in sequence_splits.items() if len(values) > 1)
    duplicates = sorted(key for key, values in digest_splits.items() if len(values) > 1)
    report = {
        "records": len(records),
        "sequence_overlap": overlap,
        "duplicate_cache_across_splits": duplicates,
        "missing_cache": missing,
    }
    report["clean"] = not (overlap or duplicates or missing)
    return report


class CachedDDPFWindowDataset(Dataset[dict[str, Any]]):
    """Leakage-safe fixed temporal windows over frozen DDPF sequence caches."""

    def __init__(
        self,
        manifest: str | Path,
        split: Split,
        context_frames: int,
        horizons_seconds: Sequence[int],
        expected_venue_id: str | None = None,
        expected_zone_ids: Sequence[str] | None = None,
        risk_calibration: dict[str, Any] | None = None,
        augment: bool = False,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        report = audit_strfe_manifest(manifest)
        if not report["clean"]:
            raise ValueError(f"STRFE manifest failed leakage audit: {report}")
        manifest_path = Path(manifest).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.context_frames = int(context_frames)
        self.horizons_seconds = tuple(int(value) for value in horizons_seconds)
        self.augment = augment
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.risk_calibration = risk_calibration
        if self.context_frames <= 0 or not self.horizons_seconds:
            raise ValueError("context_frames and horizons_seconds must be positive")
        self.sequences: list[CachedDDPFSequence] = []
        self.index: list[tuple[int, int, tuple[int, ...]]] = []
        for record in payload.get("records", []):
            if record["split"] != split:
                continue
            record_path = Path(record["path"])
            record_path = record_path if record_path.is_absolute() else manifest_path.parent / record_path
            sequence = CachedDDPFSequence.load(record_path)
            if sequence.zone_state is None:
                # Zone targets depend only on the immutable cached perception
                # arrays. Compute them once per sequence instead of once per
                # overlapping temporal window.
                if not isinstance(self.risk_calibration, dict):
                    raise ValueError(
                        "a training-split risk calibration is required when zone_state is absent"
                    )
                sequence.zone_state = zone_states_from_perception(
                    sequence.density,
                    sequence.flow,
                    sequence.zone_masks,
                    self.risk_calibration,
                )
            if expected_venue_id is not None and sequence.metadata.venue_id != expected_venue_id:
                raise ValueError("cached DDPF venue_id does not match STRFE configuration")
            if expected_zone_ids is not None and sequence.metadata.zone_ids != tuple(expected_zone_ids):
                raise ValueError("cached DDPF zone order does not match venue graph")
            sequence_index = len(self.sequences)
            self.sequences.append(sequence)
            offsets = tuple(round(value * sequence.metadata.fps) for value in self.horizons_seconds)
            maximum = max(offsets)
            for current in range(self.context_frames - 1, len(sequence.metadata.frame_indices) - maximum):
                self.index.append((sequence_index, current, offsets))
        if not self.index:
            raise ValueError(f"No complete {split} temporal windows were found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sequence_index, current, offsets = self.index[item]
        sequence = self.sequences[sequence_index]
        start = current - self.context_frames + 1
        context = slice(start, current + 1)
        target_indices = [current + offset for offset in offsets]
        zone_state = sequence.zone_state
        if zone_state is None:
            raise RuntimeError("cached sequence zone_state was not initialized")
        visual = sequence.visual_features[context].copy()
        density = sequence.density[context].copy()
        localization = sequence.localization_logits[context].copy()
        flow = sequence.flow[context].copy()
        masks = sequence.zone_masks.copy()
        flipped = self.augment and random.random() < self.horizontal_flip_probability
        if flipped:
            visual = np.ascontiguousarray(visual[..., ::-1])
            density = np.ascontiguousarray(density[..., ::-1])
            localization = np.ascontiguousarray(localization[..., ::-1])
            flow = np.ascontiguousarray(flow[..., ::-1])
            flow[:, 0] *= -1.0
            masks = np.ascontiguousarray(masks[..., ::-1])
        sensor_count = len(sequence.metadata.sensor_features)
        sensors = (
            sequence.sensors[context].copy()
            if sequence.sensors is not None
            else np.zeros((self.context_frames, sensor_count), dtype=np.float32)
        )
        sensor_mask = (
            sequence.sensor_mask[context].copy()
            if sequence.sensor_mask is not None
            else np.zeros_like(sensors)
        )
        return {
            "visual_features": torch.from_numpy(visual),
            "density": torch.from_numpy(density),
            "localization_logits": torch.from_numpy(localization),
            "flow": torch.from_numpy(flow),
            "zone_masks": torch.from_numpy(masks),
            "sensors": torch.from_numpy(sensors),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "current_zone_state": torch.from_numpy(zone_state[current].copy()),
            "zone_state_history": torch.from_numpy(zone_state[context].copy()),
            "future_zone_state": torch.from_numpy(zone_state[target_indices].copy()),
            "sequence_id": sequence.metadata.sequence_id,
            "venue_id": sequence.metadata.venue_id,
            "zone_ids": sequence.metadata.zone_ids,
            "frame_index": sequence.metadata.frame_indices[current],
            "timestamp_ms": sequence.metadata.timestamps_ms[current],
            "augmentation": "horizontal_flip" if flipped else "none",
        }


def optical_flow_sequence(frames_rgb: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Reproducible four-channel Farneback flow aligned to DDPF feature resolution."""
    frames = np.asarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames_rgb must be [T,H,W,3]")
    output = [np.zeros((4, *output_size), dtype=np.float32)]
    for index in range(1, len(frames)):
        previous = cv2.cvtColor(frames[index - 1], cv2.COLOR_RGB2GRAY)
        current = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        field = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        dx, dy = field[..., 0], field[..., 1]
        magnitude = np.sqrt(dx * dx + dy * dy)
        divergence = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
        channels = np.stack([dx, dy, magnitude, divergence])
        resized = np.stack(
            [cv2.resize(channel, output_size[::-1], interpolation=cv2.INTER_LINEAR) for channel in channels]
        ).astype(np.float32)
        resized[0] *= output_size[1] / frames.shape[2]
        resized[1] *= output_size[0] / frames.shape[1]
        output.append(resized)
    return np.stack(output)
