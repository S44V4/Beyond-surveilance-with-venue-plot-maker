from __future__ import annotations

import base64
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from .alerts import build_model_alerts
from .config import load_config
from .contracts import STATE_FEATURES, validate_contract
from .data.dataset import flow_features, zone_means
from .data.schema import read_jsonl
from .models.digital_twin import InterventionBatch, LearnedDigitalTwin
from .models.layers import grid_adjacency
from .models.network import GraphCrowdTwin
from .training import load_checkpoint
from .venue import (
    VenueGraph,
    aggregate_perception_to_zones,
    node_static_features,
    zone_masks,
)


class CheckpointUnavailable(RuntimeError):
    """Raised when an endpoint requiring learned weights has no checkpoint."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_data_url(values: np.ndarray, color_map: int = cv2.COLORMAP_INFERNO) -> str:
    values = np.nan_to_num(values.astype(np.float32))
    low, high = float(values.min()), float(values.max())
    normalized = np.zeros_like(values, dtype=np.uint8)
    if high > low:
        normalized = np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, color_map)
    ok, encoded = cv2.imencode(".png", colored)
    if not ok:
        raise RuntimeError("Could not encode inference heatmap")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _bgr_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode inference image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _venue_risk_map(
    graph: VenueGraph, risk_scores: np.ndarray, attention: np.ndarray
) -> str:
    canvas = np.full((graph.height, graph.width, 3), (26, 33, 29), dtype=np.uint8)
    overlay = canvas.copy()
    palette = cv2.applyColorMap(
        np.clip(risk_scores, 0, 100).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO
    )[:, 0]
    for index, zone in enumerate(graph.zones):
        polygon = np.asarray(zone.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], tuple(int(value) for value in palette[index]))
        cv2.polylines(overlay, [polygon], True, (232, 239, 234), 2, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
    for source, zone in enumerate(graph.zones):
        candidates = np.argsort(attention[source])[::-1]
        target = next((int(value) for value in candidates if int(value) != source), None)
        if target is not None and attention[source, target] > 0:
            start = tuple(round(value) for value in zone.centroid)
            end = tuple(round(value) for value in graph.zones[target].centroid)
            width = max(1, round(float(attention[source, target]) * 8))
            cv2.line(canvas, start, end, (252, 170, 106), width, cv2.LINE_AA)
    for index, zone in enumerate(graph.zones):
        x, y = (round(value) for value in zone.centroid)
        cv2.putText(
            canvas,
            f"{zone.zone_id} {risk_scores[index]:.0f}%",
            (x - 34, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return _bgr_data_url(canvas)


def _zone_label(index: int, columns: int) -> str:
    return f"{chr(65 + index // columns)}{index % columns + 1}"


def _risk_level(probabilities: np.ndarray) -> str:
    labels = ("stable", "watch", "high", "critical")
    index = int(np.argmax(probabilities))
    return labels[min(index, len(labels) - 1)]


def _resize_rgb(frame_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def _sample_video(path: Path, sample_fps: float, maximum: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The uploaded file could not be decoded as a video")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or sample_fps)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, round(source_fps / max(sample_fps, 0.1)))
    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < maximum:
        ok, frame = capture.read()
        if not ok:
            break
        if index % interval == 0:
            frames.append(frame)
        index += 1
    capture.release()
    if not frames:
        raise ValueError("The uploaded video contains no decodable frames")
    return frames, {
        "source_fps": source_fps,
        "source_frames": source_frames,
        "sampled_frames": len(frames),
        "duration_seconds": source_frames / source_fps if source_frames and source_fps else None,
    }


class CrowdTwinPredictor:
    """Owns verified model weights and converts media into dashboard-ready evidence."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config_path = Path(
            config_path or os.getenv("CROWD_TWIN_CONFIG", "configs/base.yaml")
        ).resolve()
        self.checkpoint_path = Path(
            checkpoint_path or os.getenv("CROWD_TWIN_CHECKPOINT", "checkpoints/best.pt")
        ).resolve()
        requested = device or os.getenv("CROWD_TWIN_DEVICE", "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.model: GraphCrowdTwin | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.config = load_config(self.config_path)
        self.checkpoint_sha256: str | None = None
        self.load_error: str | None = None
        self.contract_status = "not_loaded"
        self.venue_graph: VenueGraph | None = None
        venue_path = self.config.get("architecture", {}).get("venue_graph")
        if venue_path and Path(venue_path).is_file():
            self.venue_graph = VenueGraph.load(venue_path)
        self._load()

    @property
    def ready(self) -> bool:
        checkpoint_ready = self.model is not None and self.checkpoint is not None
        venue_required = bool(
            self.config.get("architecture", {}).get(
                "require_venue_graph_for_deployment", False
            )
        )
        return checkpoint_ready and (self.venue_graph is not None or not venue_required)

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            self.load_error = f"Checkpoint not found: {self.checkpoint_path}"
            return
        try:
            raw = load_checkpoint(self.checkpoint_path, self.device)
            checkpoint_config = raw.get("config")
            if not isinstance(checkpoint_config, dict):
                raise TypeError("Checkpoint has no resolved training config")
            model = GraphCrowdTwin(checkpoint_config).to(self.device)
            model.load_state_dict(raw["model_state"], strict=True)
            model.eval()
            self.model = model
            self.checkpoint = raw
            self.config = checkpoint_config
            venue_path = self.config.get("architecture", {}).get("venue_graph")
            self.venue_graph = (
                VenueGraph.load(venue_path)
                if venue_path and Path(venue_path).is_file()
                else None
            )
            self.checkpoint_sha256 = _file_sha256(self.checkpoint_path)
            contract = raw.get("integration_contract")
            contract_required = bool(
                self.config.get("architecture", {}).get(
                    "require_integration_contract", True
                )
            )
            if not isinstance(contract, dict):
                if contract_required:
                    raise ValueError(
                        "Checkpoint has no versioned integration_contract"
                    )
                self.contract_status = "legacy_unverified"
            else:
                validate_contract(contract, self.config, self.venue_graph)
                self.contract_status = "verified"
            if (
                self.config.get("architecture", {}).get(
                    "require_venue_graph_for_deployment", False
                )
                and self.venue_graph is None
            ):
                self.load_error = f"Required venue graph not found: {venue_path}"
        except Exception as exc:  # noqa: BLE001 - preserve loader diagnostic, never random weights
            self.model = None
            self.checkpoint = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def require_ready(self) -> GraphCrowdTwin:
        if not self.ready:
            raise CheckpointUnavailable(
                self.load_error
                or "A verified trained checkpoint is required before inference can run"
            )
        assert self.model is not None
        return self.model

    def metadata(self) -> dict[str, Any]:
        model_metadata = self.checkpoint.get("model_metadata", {}) if self.checkpoint else {}
        return {
            "ready": self.ready,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "load_error": self.load_error,
            "device": str(self.device),
            "epoch": self.checkpoint.get("epoch") if self.checkpoint else None,
            "validation_metric": self.checkpoint.get("best_metric") if self.checkpoint else None,
            "source_datasets": self.checkpoint.get("source_datasets", []) if self.checkpoint else [],
            "git_commit": self.checkpoint.get("git_commit") if self.checkpoint else None,
            "created_at_unix": self.checkpoint.get("created_at_unix") if self.checkpoint else None,
            "integration_contract_status": self.contract_status,
            "integration_contract": (
                self.checkpoint.get("integration_contract") if self.checkpoint else None
            ),
            "architecture": (
                "AVIM semantic venue + DDPF-Net + multimodal STRFE "
                "+ dynamic Graph Transformer"
            ),
            "architecture_version": model_metadata.get(
                "architecture", self.config.get("architecture", {}).get("version")
            ),
            "venue_graph": self.venue_graph.to_dict() if self.venue_graph else None,
            "temporal_backend": model_metadata.get("temporal_backend"),
            "horizons_seconds": model_metadata.get(
                "horizons_seconds", self.config["model"]["horizons_seconds"]
            ),
            "zone_grid": model_metadata.get("zone_grid", self.config["data"]["zone_grid"]),
            "state_features": model_metadata.get("state_features", list(STATE_FEATURES)),
            "feature_dim": model_metadata.get("feature_dim", self.config["model"]["feature_dim"]),
        }

    def _tensorize(self, frames_bgr: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        context = int(self.config["data"]["context_frames"])
        image_size = tuple(self.config["data"]["image_size"])
        selected = frames_bgr[-context:]
        if len(selected) < context:
            selected = [selected[0]] * (context - len(selected)) + selected
        resized = [_resize_rgb(frame, image_size) for frame in selected]
        flows = [np.zeros((4, *image_size), dtype=np.float32)]
        flows.extend(flow_features(resized[index - 1], resized[index]) for index in range(1, context))
        frame_array = np.stack(resized).astype(np.float32) / 255.0
        flow_array = np.stack(flows).astype(np.float32)
        frames = torch.from_numpy(frame_array).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        flow = torch.from_numpy(flow_array).unsqueeze(0).to(self.device)
        return frames, flow, flow_array

    def _sensor_inputs(
        self, context: int, readings: dict[str, float] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = list(self.config.get("sensors", {}).get("features", []))
        values = np.zeros((1, context, len(features)), dtype=np.float32)
        mask = np.zeros_like(values)
        normalization_path = self.config.get("sensors", {}).get("normalization")
        normalization = (
            yaml.safe_load(Path(normalization_path).read_text(encoding="utf-8"))
            if normalization_path and Path(normalization_path).is_file()
            else {}
        )
        for index, feature in enumerate(features):
            if readings is None or readings.get(feature) is None:
                continue
            statistics = normalization.get(feature, {})
            mean = float(statistics.get("mean", 0.0))
            std = max(float(statistics.get("std", 1.0)), 1e-8)
            values[:, :, index] = (float(readings[feature]) - mean) / std
            mask[:, :, index] = 1.0
        return torch.from_numpy(values).to(self.device), torch.from_numpy(mask).to(self.device)

    @torch.inference_mode()
    def _analyze(
        self,
        frames_bgr: list[np.ndarray],
        source: dict[str, Any],
        sensor_readings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        model = self.require_ready()
        frames, flow, flow_array = self._tensorize(frames_bgr)
        final_flow = flow_array[-1]
        sensors, sensor_mask = self._sensor_inputs(frames.shape[1], sensor_readings)
        adjacency = None
        static = None
        masks = None
        if self.venue_graph is not None:
            adjacency = torch.tensor(
                self.venue_graph.adjacency, dtype=torch.bool, device=self.device
            )
            static = torch.from_numpy(
                node_static_features(
                    self.venue_graph, feature_dim=model.static_feature_dim
                )
            ).to(self.device)
            masks = torch.from_numpy(
                zone_masks(self.venue_graph, frames.shape[-2], frames.shape[-1])
            ).to(self.device)
        started = time.perf_counter()
        outputs = model(
            frames,
            flow,
            sensors=sensors,
            sensor_mask=sensor_mask,
            adjacency=adjacency,
            node_static=static,
            zone_masks=masks,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        states = outputs["current_zone_state"][0].cpu().numpy()
        risk_probs = torch.softmax(outputs["risk_logits"][0], dim=-1).cpu().numpy()
        confidence = outputs["risk_confidence"][0].cpu().numpy()
        hazard = outputs["hazard_probability"][0].cpu().numpy()
        future_states = outputs["future_zone_state"][0].cpu().numpy()
        future_probs = torch.softmax(outputs["future_risk_logits"][0], dim=-1).cpu().numpy()
        attention = outputs["graph_attention"][0, -1].mean(dim=0).cpu().numpy()
        density = outputs["density"][0, -1, 0].cpu().numpy()
        localization = torch.sigmoid(outputs["localization_logits"][0, -1, 0]).cpu().numpy()

        perception_aggregate = None
        observed_velocity = None
        observed_acceleration = None
        observed_divergence = None
        if self.venue_graph is not None and masks is not None:
            mask_array = masks.cpu().numpy()
            perception_aggregate = aggregate_perception_to_zones(
                density,
                localization,
                mask_array,
                float(self.config.get("evaluation", {}).get("localization_threshold", 0.35)),
            )
            observed_velocity = zone_means(
                final_flow[2], tuple(self.config["data"]["zone_grid"]), mask_array
            )
            previous_velocity = zone_means(
                flow_array[-2, 2] if len(flow_array) > 1 else np.zeros_like(final_flow[2]),
                tuple(self.config["data"]["zone_grid"]),
                mask_array,
            )
            observed_acceleration = observed_velocity - previous_velocity
            observed_divergence = zone_means(
                final_flow[3], tuple(self.config["data"]["zone_grid"]), mask_array
            )

        _rows, columns = tuple(self.config["data"]["zone_grid"])
        zone_ids = (
            [zone.zone_id for zone in self.venue_graph.zones]
            if self.venue_graph is not None
            else [_zone_label(index, columns) for index in range(len(states))]
        )
        zone_names = (
            [zone.label for zone in self.venue_graph.zones]
            if self.venue_graph is not None
            else zone_ids
        )
        classes = risk_probs.shape[-1]
        class_scale = np.arange(classes, dtype=np.float32) / max(classes - 1, 1) * 100
        risk_scores = (risk_probs * class_scale).sum(axis=-1)
        future_scores = (future_probs * class_scale).sum(axis=-1)
        zones: list[dict[str, Any]] = []
        for index, state in enumerate(states):
            score = float(risk_scores[index])
            top_neighbors = np.argsort(attention[index])[::-1][:3]
            observed_count = (
                float(perception_aggregate["count"][index])
                if perception_aggregate is not None
                else float(state[0])
            )
            observed_density = (
                float(perception_aggregate["density"][index])
                if perception_aggregate is not None
                else float(state[1])
            )
            zones.append(
                {
                    "id": zone_ids[index],
                    "name": zone_names[index],
                    "row": index // columns,
                    "column": index % columns,
                    "polygon": (
                        self.venue_graph.zones[index].polygon
                        if self.venue_graph is not None
                        else None
                    ),
                    "count": observed_count,
                    "density": observed_density,
                    "velocity": (
                        float(observed_velocity[index])
                        if observed_velocity is not None
                        else float(state[2])
                    ),
                    "acceleration": (
                        float(observed_acceleration[index])
                        if observed_acceleration is not None
                        else float(state[3])
                    ),
                    "divergence": (
                        float(observed_divergence[index])
                        if observed_divergence is not None
                        else float(state[4])
                    ),
                    "congestion": float(state[5]),
                    "risk": score,
                    "risk_level": _risk_level(risk_probs[index]),
                    "confidence": float(confidence[index] * 100),
                    "hazard_probability": float(hazard[index] * 100),
                    "attention_neighbors": [
                        {
                            "zone": zone_ids[int(neighbor)],
                            "weight": float(attention[index, neighbor]),
                        }
                        for neighbor in top_neighbors
                    ],
                    "measurement_source": (
                        "ddpf_density_and_optical_flow"
                        if perception_aggregate is not None
                        else "learned_zone_state"
                    ),
                }
            )

        forecasts = []
        for horizon_index, horizon in enumerate(model.horizons):
            per_zone = [
                {
                    "id": zone_ids[index],
                    "count": float(future_states[horizon_index, index, 0]),
                    "density": float(future_states[horizon_index, index, 1]),
                    "risk": float(future_scores[horizon_index, index]),
                }
                for index in range(len(zones))
            ]
            forecasts.append(
                {
                    "horizon_seconds": int(horizon),
                    "total_count": float(sum(zone["count"] for zone in per_zone)),
                    "mean_risk": float(np.mean(future_scores[horizon_index])),
                    "max_risk": float(np.max(future_scores[horizon_index])),
                    "zones": per_zone,
                }
            )

        mean_risk = float(np.mean(risk_scores))
        global_summary = {
            "estimated_count": float(sum(zone["count"] for zone in zones)),
            "mean_risk": mean_risk,
            "max_risk": float(np.max(risk_scores)),
            "unsafe_zones": int(
                (
                    risk_probs.argmax(axis=-1)
                    >= max(1, risk_probs.shape[-1] // 2)
                ).sum()
            ),
            "mean_confidence": float(np.mean(confidence) * 100),
            "mean_hazard_probability": float(hazard.mean() * 100),
        }
        result = {
            "run_id": str(uuid.uuid4()),
            "generated_at_unix": time.time(),
            "source": source,
            "checkpoint": {
                "sha256": self.checkpoint_sha256,
                "epoch": self.checkpoint.get("epoch") if self.checkpoint else None,
                "source_datasets": self.checkpoint.get("source_datasets", []) if self.checkpoint else [],
            },
            "latency_ms": elapsed_ms,
            "architecture": model.metadata.architecture,
            "venue": {
                "venue_id": self.venue_graph.venue_id,
                "source": self.venue_graph.source,
                "adaptive": True,
            } if self.venue_graph else {"adaptive": False, "source": "configured_grid"},
            "sensors": {
                "provided": sorted(sensor_readings or {}),
                "coverage": float(outputs["sensor_coverage"][0].cpu() * 100),
            },
            "global": global_summary,
            "zones": zones,
            "forecasts": forecasts,
            "maps": {
                "density": _png_data_url(density, cv2.COLORMAP_INFERNO),
                "localization": _png_data_url(localization, cv2.COLORMAP_VIRIDIS),
                "flow": _png_data_url(final_flow[2], cv2.COLORMAP_TURBO),
                "attention": _png_data_url(attention, cv2.COLORMAP_VIRIDIS),
                **(
                    {"risk": _venue_risk_map(self.venue_graph, risk_scores, attention)}
                    if self.venue_graph is not None
                    else {}
                ),
            },
        }
        result["alerts"] = build_model_alerts(zones, forecasts, global_summary)
        return result

    def analyze_image_bytes(
        self,
        content: bytes,
        filename: str,
        sensor_readings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        encoded = np.frombuffer(content, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The uploaded file could not be decoded as an image")
        return self._analyze(
            [frame],
            {"kind": "image", "filename": filename, "width": frame.shape[1], "height": frame.shape[0]},
            sensor_readings,
        )

    def analyze_video_path(
        self,
        path: Path,
        filename: str,
        sensor_readings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        settings = self.config["inference"]
        frames, metadata = _sample_video(
            path, float(settings["sample_fps"]), int(settings["max_frames"])
        )
        metadata.update({"kind": "video", "filename": filename})
        return self._analyze(frames, metadata, sensor_readings)


class DigitalTwinPredictor:
    """Checkpoint-gated learned scenario runner."""

    def __init__(self, device: str = "cpu", checkpoint_path: str | Path | None = None) -> None:
        self.device = torch.device(device)
        configured_path = checkpoint_path or os.getenv("CROWD_TWIN_DIGITAL_TWIN_CHECKPOINT")
        self.path: Path | None = Path(configured_path).resolve() if configured_path else None
        self.model: LearnedDigitalTwin | None = None
        self.error: str | None = None
        if self.path is None:
            self.error = "Required runtime setting CROWD_TWIN_DIGITAL_TWIN_CHECKPOINT is not configured"
            return
        if not self.path.is_file():
            self.error = f"Digital-twin checkpoint not found: {self.path}"
            return
        try:
            checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
            config = checkpoint.get("model_config", {})
            self.model = LearnedDigitalTwin(**config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001 - checkpoint errors are surfaced as readiness state
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @torch.inference_mode()
    def simulate(
        self,
        payload: dict[str, Any],
        grid: tuple[int, int],
        adjacency: list[list[bool]] | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise CheckpointUnavailable(self.error or "A trained digital-twin checkpoint is required")
        state = torch.tensor(payload["current_zone_state"], dtype=torch.float32, device=self.device)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        nodes = state.shape[1]

        def vector(name: str, default: float = 0.0) -> torch.Tensor:
            value = payload.get(name, [default] * nodes)
            tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
            return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

        redirects = torch.tensor(
            payload.get("redirect_matrix", np.zeros((nodes, nodes)).tolist()),
            dtype=torch.float32,
            device=self.device,
        )
        if redirects.ndim == 2:
            redirects = redirects.unsqueeze(0)
        intervention = InterventionBatch(
            exit_capacity_delta=vector("exit_capacity_delta"),
            redirect_matrix=redirects,
            blocked_zones=vector("blocked_zones").bool(),
            external_inflow=vector("external_inflow"),
        )
        result = self.model(
            state,
            intervention,
            (
                torch.tensor(adjacency, dtype=torch.bool, device=self.device)
                if adjacency is not None
                else grid_adjacency(*grid).to(self.device)
            ),
            int(payload.get("steps", 10)),
        )
        trajectory = result["trajectory"][0].cpu().tolist()
        uncertainty = result["uncertainty"][0].cpu().tolist()
        return {
            "trajectory": trajectory,
            "uncertainty": uncertainty,
            "steps": len(trajectory),
            "checkpoint_sha256": _file_sha256(self.path),
        }


def dataset_statuses(
    registry_path: str | Path = "configs/datasets.yaml",
    manifest_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    registry_file = Path(registry_path)
    registry = yaml.safe_load(registry_file.read_text(encoding="utf-8"))["datasets"]
    root = Path(
        manifest_root or os.getenv("CROWD_TWIN_MANIFEST_ROOT", "data/manifests")
    ).resolve()
    statuses = []
    for name, item in registry.items():
        dataset_manifest_root = root / name
        manifests = {
            split: dataset_manifest_root / f"{split}.jsonl"
            for split in ("train", "val", "test")
        }
        counts: dict[str, int] = {}
        for split, path in manifests.items():
            if path.is_file():
                # The status endpoint only needs manifest cardinality. Parsing
                # tens of thousands of full annotations here delayed the
                # dashboard readiness request by many seconds.
                with path.open("r", encoding="utf-8") as handle:
                    counts[split] = sum(1 for line in handle if line.strip())
            else:
                counts[split] = 0
        leakage_path = dataset_manifest_root / "leakage_report.json"
        leakage: dict[str, Any] | None = None
        if leakage_path.is_file():
            import json

            leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        available = any(counts.values()) and leakage is not None and bool(leakage.get("clean"))
        statuses.append(
            {
                "id": name,
                "adapter": item["adapter"],
                "official_split": item["official_split"],
                "root": item["root"],
                "manifest_counts": counts,
                "leakage": leakage,
                "available": available,
                "status": "ready" if available else "not_prepared",
            }
        )
    return statuses
