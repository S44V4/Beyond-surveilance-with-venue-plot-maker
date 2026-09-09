from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from crowd_twin.contracts import (
    GRAPH_HANDOFF_SCHEMA_VERSION,
    STATE_FEATURES,
    GraphHandoff,
    GraphHandoffMetadata,
)

from .layers import ConvLSTMCell, TemporalStateSpaceBlock


@dataclass(frozen=True, slots=True)
class STRFEMetadata:
    schema_version: str
    input_channels: int
    feature_dim: int
    horizons_seconds: tuple[int, ...]
    sensor_features: tuple[str, ...]
    temporal_backend: str
    state_features: tuple[str, ...] = STATE_FEATURES
    flow_convention: str = "farneback_dx_dy_magnitude_divergence"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


class STRFE(nn.Module):
    """Spatiotemporal feature reasoning over frozen DDPF outputs.

    Inputs:
        visual_features: [B,T,C,Hf,Wf]
        density/localization_logits: [B,T,1,Hf,Wf]
        flow: [B,T,4,Hf,Wf]
        zone_masks: [N,Hf,Wf] or [B,N,Hf,Wf]
        sensors/sensor_mask: optional [B,T,S]

    Outputs are the strict graph boundary: current features [B,N,D], future
    features [B,H,N,D], current state [B,N,7], and future state [B,H,N,7].
    """

    def __init__(
        self,
        input_channels: int,
        feature_dim: int,
        horizons_seconds: Sequence[int],
        sensor_features: Sequence[str] = (),
        temporal_layers: int = 1,
        dropout: float = 0.1,
        use_mamba: bool = False,
        use_motion: bool = True,
        use_short_term: bool = True,
        use_long_term: bool = True,
        use_sensors: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or feature_dim <= 0:
            raise ValueError("input_channels and feature_dim must be positive")
        if feature_dim % 4:
            raise ValueError("feature_dim must be divisible by four")
        self.input_channels = int(input_channels)
        self.feature_dim = int(feature_dim)
        self.horizons = tuple(int(value) for value in horizons_seconds)
        if not self.horizons or tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("horizons_seconds must be a non-empty ordered sequence")
        self.sensor_features = tuple(str(value) for value in sensor_features)
        self.sensor_dim = len(self.sensor_features)
        self.use_motion = bool(use_motion)
        self.use_short_term = bool(use_short_term)
        self.use_long_term = bool(use_long_term)
        self.use_sensors = bool(use_sensors)

        self.visual_projection = nn.Sequential(
            nn.Conv2d(self.input_channels, self.feature_dim, 1),
            nn.GELU(),
            nn.GroupNorm(4, self.feature_dim),
        )
        self.perception_projection = nn.Sequential(
            nn.Conv2d(2, self.feature_dim, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, self.feature_dim),
        )
        self.motion_projection = nn.Sequential(
            nn.Conv2d(4, self.feature_dim, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, self.feature_dim),
        )
        self.short_term = ConvLSTMCell(self.feature_dim, self.feature_dim)
        self.long_term = TemporalStateSpaceBlock(
            self.feature_dim, int(temporal_layers), float(dropout), bool(use_mamba)
        )
        sensor_input = max(self.sensor_dim, 1) * 2
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_input, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.feature_dim),
        )
        self.current_state_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, len(STATE_FEATURES)),
        )
        self.horizon_embeddings = nn.Parameter(
            torch.randn(len(self.horizons), self.feature_dim) * 0.02
        )
        self.future_feature_decoder = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.future_state_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, len(STATE_FEATURES)),
        )
        self.metadata = STRFEMetadata(
            schema_version=GRAPH_HANDOFF_SCHEMA_VERSION,
            input_channels=self.input_channels,
            feature_dim=self.feature_dim,
            horizons_seconds=self.horizons,
            sensor_features=self.sensor_features,
            temporal_backend=self.long_term.backend,
        )

    @staticmethod
    def _pool(feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 3:
            masks = masks.unsqueeze(0).expand(feature_map.shape[0], -1, -1, -1)
        if masks.ndim != 4 or masks.shape[0] != feature_map.shape[0]:
            raise ValueError("zone_masks must be [N,H,W] or [B,N,H,W]")
        masks = functional.interpolate(
            masks.to(device=feature_map.device, dtype=feature_map.dtype),
            size=feature_map.shape[-2:],
            mode="nearest",
        )
        normalized = masks.flatten(2)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bnp,bcp->bnc", normalized, feature_map.flatten(2))

    @staticmethod
    def _constrain_state(state: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                functional.softplus(state[..., 0]),
                functional.softplus(state[..., 1]),
                functional.softplus(state[..., 2]),
                state[..., 3],
                state[..., 4],
                functional.softplus(state[..., 5]),
                torch.sigmoid(state[..., 6]),
            ],
            dim=-1,
        )

    @staticmethod
    def _constrain_residual(current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply a learned temporal delta while preserving a zero-delta baseline."""
        risk_logit = torch.logit(current[..., 6].clamp(1e-5, 1 - 1e-5))
        return torch.stack(
            [
                functional.relu(current[..., 0] + delta[..., 0]),
                functional.relu(current[..., 1] + delta[..., 1]),
                functional.relu(current[..., 2] + delta[..., 2]),
                current[..., 3] + delta[..., 3],
                current[..., 4] + delta[..., 4],
                functional.relu(current[..., 5] + delta[..., 5]),
                torch.sigmoid(risk_logit + delta[..., 6]),
            ],
            dim=-1,
        )

    def _sensor_embedding(
        self,
        batch: int,
        time: int,
        sensors: torch.Tensor | None,
        sensor_mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sensor_dim == 0:
            values = reference.new_zeros(batch, time, 1)
            mask = reference.new_zeros(batch, time, 1)
        elif sensors is None:
            values = reference.new_zeros(batch, time, self.sensor_dim)
            mask = torch.zeros_like(values)
        else:
            if sensors.shape != (batch, time, self.sensor_dim):
                raise ValueError("sensors must be [B,T,S] in configured feature order")
            values = sensors.to(reference)
            mask = torch.ones_like(values) if sensor_mask is None else sensor_mask.to(reference)
            if mask.shape != values.shape:
                raise ValueError("sensor_mask must match sensors")
        coverage = mask.mean(dim=(1, 2))
        embedding = self.sensor_projection(torch.cat([values * mask, mask], dim=-1))
        if not self.use_sensors:
            embedding = torch.zeros_like(embedding)
            coverage = torch.zeros_like(coverage)
        return embedding, coverage

    def forward(
        self,
        visual_features: torch.Tensor,
        density: torch.Tensor,
        localization_logits: torch.Tensor,
        flow: torch.Tensor,
        zone_masks: torch.Tensor,
        sensors: torch.Tensor | None = None,
        sensor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if visual_features.ndim != 5:
            raise ValueError("visual_features must be [B,T,C,Hf,Wf]")
        batch, time, channels, height, width = visual_features.shape
        if channels != self.input_channels:
            raise ValueError("visual feature channels do not match the STRFE checkpoint")
        expected_single = (batch, time, 1, height, width)
        if density.shape != expected_single or localization_logits.shape != expected_single:
            raise ValueError("density and localization_logits must be [B,T,1,Hf,Wf]")
        if flow.shape != (batch, time, 4, height, width):
            raise ValueError("flow must be [B,T,4,Hf,Wf]")

        flat_visual = visual_features.reshape(batch * time, channels, height, width)
        visual_maps = self.visual_projection(flat_visual).reshape(
            batch, time, self.feature_dim, height, width
        )
        perception_input = torch.cat([density, torch.sigmoid(localization_logits)], dim=2)
        perception_maps = self.perception_projection(
            perception_input.reshape(batch * time, 2, height, width)
        ).reshape(batch, time, self.feature_dim, height, width)
        motion_maps = self.motion_projection(
            flow.reshape(batch * time, 4, height, width)
        ).reshape(batch, time, self.feature_dim, height, width)

        state: tuple[torch.Tensor, torch.Tensor] | None = None
        short_nodes = []
        for step in range(time):
            motion = motion_maps[:, step] if self.use_motion else torch.zeros_like(motion_maps[:, step])
            fused_map = visual_maps[:, step] + perception_maps[:, step] + motion
            state = self.short_term(fused_map, state)
            short_map = state[0] if self.use_short_term else fused_map
            short_nodes.append(self._pool(short_map, zone_masks))
        short_sequence = torch.stack(short_nodes, dim=1)
        nodes = short_sequence.shape[2]
        temporal_input = short_sequence.permute(0, 2, 1, 3).reshape(
            batch * nodes, time, self.feature_dim
        )
        long_nodes = (
            self.long_term(temporal_input)[:, -1].reshape(batch, nodes, self.feature_dim)
            if self.use_long_term
            else short_sequence[:, -1]
        )
        sensor_sequence, sensor_coverage = self._sensor_embedding(
            batch, time, sensors, sensor_mask, visual_features
        )
        sensor_latest = sensor_sequence[:, -1, None].expand(-1, nodes, -1)
        current_features = self.fusion(
            torch.cat([short_sequence[:, -1], long_nodes, sensor_latest], dim=-1)
        )
        current_state = self._constrain_state(self.current_state_head(current_features))

        future_features = []
        future_states = []
        for embedding in self.horizon_embeddings:
            horizon = embedding.view(1, 1, -1).expand(batch, nodes, -1)
            decoded = current_features + self.future_feature_decoder(
                torch.cat([current_features, horizon], dim=-1)
            )
            future_features.append(decoded)
            future_states.append(
                self._constrain_residual(current_state, self.future_state_head(decoded))
            )
        return {
            "current_zone_features": current_features,
            "future_zone_features": torch.stack(future_features, dim=1),
            "current_zone_state": current_state,
            "future_zone_state": torch.stack(future_states, dim=1),
            "sensor_coverage": sensor_coverage,
            "short_term_sequence": short_sequence,
        }

    def to_graph_handoff(
        self,
        output: dict[str, torch.Tensor],
        venue_id: str,
        sequence_id: str,
        zone_ids: Sequence[str],
        temporal_unit: str = "seconds",
        normalization: dict[str, Any] | None = None,
    ) -> GraphHandoff:
        def array(key: str) -> np.ndarray:
            return output[key].detach().cpu().numpy().astype(np.float32)

        handoff = GraphHandoff(
            metadata=GraphHandoffMetadata(
                schema_version=GRAPH_HANDOFF_SCHEMA_VERSION,
                venue_id=venue_id,
                sequence_id=sequence_id,
                zone_ids=tuple(zone_ids),
                horizons_seconds=self.horizons,
                feature_dim=self.feature_dim,
                temporal_unit=temporal_unit,
                normalization=normalization or {},
            ),
            current_zone_features=array("current_zone_features"),
            future_zone_features=array("future_zone_features"),
            current_zone_state=array("current_zone_state"),
            future_zone_state=array("future_zone_state"),
            sensor_coverage=array("sensor_coverage"),
        )
        handoff.validate()
        return handoff


class PersistenceForecast(nn.Module):
    """Fair non-learning baseline: every horizon equals the current state."""

    def __init__(self, horizons: int) -> None:
        super().__init__()
        self.horizons = int(horizons)

    def forward(self, current_zone_state: torch.Tensor) -> torch.Tensor:
        return current_zone_state[:, None].expand(-1, self.horizons, -1, -1)


class RecurrentForecastBaseline(nn.Module):
    """Simple GRU comparator over zone-state histories."""

    def __init__(self, hidden_dim: int, horizons: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(len(STATE_FEATURES), hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, horizons * len(STATE_FEATURES))
        self.horizons = int(horizons)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        if state_history.ndim != 4:
            raise ValueError("state_history must be [B,T,N,7]")
        batch, time, nodes, features = state_history.shape
        sequence = state_history.permute(0, 2, 1, 3).reshape(batch * nodes, time, features)
        encoded, _ = self.encoder(sequence)
        prediction = self.decoder(encoded[:, -1]).reshape(
            batch, nodes, self.horizons, features
        )
        return prediction.permute(0, 2, 1, 3)
