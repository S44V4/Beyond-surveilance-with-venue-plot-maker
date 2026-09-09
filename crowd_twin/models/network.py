from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .layers import (
    ConvLSTMCell,
    GraphTransformerLayer,
    TemporalStateSpaceBlock,
    grid_adjacency,
)
from .perception import DDPFNet


@dataclass(slots=True)
class ModelMetadata:
    architecture: str
    temporal_backend: str
    horizons_seconds: tuple[int, ...]
    zone_grid: tuple[int, int]
    sensor_features: tuple[str, ...]
    schema_version: str = "1.0"
    feature_dim: int = 0
    static_feature_dim: int = 0
    state_features: tuple[str, ...] = (
        "count",
        "density",
        "velocity",
        "acceleration",
        "divergence",
        "congestion",
        "risk",
    )


class GraphCrowdTwin(nn.Module):
    """Revised DDPF + STRFE + Dynamic Graph Transformer crowd model.

    The class name is retained so existing checkpoints and API imports fail
    explicitly through strict state loading instead of breaking silently.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data_config = config["data"]
        model_config = config["model"]
        sensor_config = config.get("sensors", {})
        self.zone_grid = tuple(data_config["zone_grid"])
        self.horizons = tuple(model_config["horizons_seconds"])
        self.feature_dim = int(model_config["feature_dim"])
        self.sensor_features = tuple(
            sensor_config.get(
                "features", ["temperature", "co2", "smoke", "humidity", "tvoc", "air_quality"]
            )
        )
        self.sensor_dim = len(self.sensor_features)
        self.static_feature_dim = int(model_config.get("static_node_features", 4))
        components = model_config.get("components", {})
        self.use_motion = bool(components.get("motion", True))
        self.use_short_temporal = bool(components.get("short_temporal", True))
        self.use_long_temporal = bool(components.get("long_temporal", True))
        self.use_sensors = bool(components.get("sensors", True))
        self.use_graph_transformer = bool(components.get("graph_transformer", True))
        self.use_static_venue = bool(components.get("static_venue", True))

        # DDPF-Net: shared visual backbone with density and point-localization heads.
        self.perception = DDPFNet(
            backbone=model_config.get("backbone", "convnext_tiny"),
            pretrained=bool(model_config.get("pretrained_backbone", False)),
        )
        backbone_channels = self.perception.backbone.output_channels
        self.visual_projection = nn.Conv2d(backbone_channels, self.feature_dim, 1)
        self.motion_projection = nn.Sequential(
            nn.Conv2d(4, self.feature_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim // 2, self.feature_dim, 1),
        )
        self.cross_attention = nn.MultiheadAttention(
            self.feature_dim,
            int(model_config["graph_heads"]),
            dropout=float(model_config["dropout"]),
            batch_first=True,
        )

        # STRFE short-term branch.
        self.convlstm = ConvLSTMCell(self.feature_dim, self.feature_dim)
        # STRFE long-term branch; Mamba2 is used only when installed.
        self.temporal = TemporalStateSpaceBlock(
            self.feature_dim,
            int(model_config["temporal_layers"]),
            float(model_config["dropout"]),
            bool(model_config.get("use_mamba", True)),
        )
        self.sensor_projection = nn.Sequential(
            nn.Linear(max(self.sensor_dim, 1) * 2, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )
        self.static_projection = nn.Linear(self.static_feature_dim, self.feature_dim)
        self.strfe_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )

        self.graph_transformer = nn.ModuleList(
            [
                GraphTransformerLayer(
                    self.feature_dim,
                    heads=int(model_config["graph_heads"]),
                    dropout=float(model_config["dropout"]),
                )
                for _ in range(int(model_config.get("graph_transformer_layers", 3)))
            ]
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 7),
        )
        risk_classes = int(model_config["risk_classes"])
        self.risk_head = nn.Linear(self.feature_dim, risk_classes)
        self.hazard_head = nn.Linear(self.feature_dim, 1)
        self.horizon_embeddings = nn.Parameter(
            torch.randn(len(self.horizons), self.feature_dim) * 0.02
        )
        self.forecast_decoder = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(model_config["dropout"])),
            nn.Linear(self.feature_dim, 7),
        )
        self.forecast_risk_head = nn.Linear(self.feature_dim * 2, risk_classes)
        self.register_buffer("adjacency", grid_adjacency(*self.zone_grid), persistent=True)
        self.metadata = ModelMetadata(
            architecture="avim-ddpf-strfe-graph-transformer-v2",
            temporal_backend=self.temporal.backend,
            horizons_seconds=self.horizons,
            zone_grid=self.zone_grid,
            sensor_features=self.sensor_features,
            feature_dim=self.feature_dim,
            static_feature_dim=self.static_feature_dim,
        )

    def _zones(
        self, feature_map: torch.Tensor, zone_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        if zone_masks is None:
            pooled = functional.adaptive_avg_pool2d(feature_map, self.zone_grid)
            return pooled.flatten(2).transpose(1, 2)
        masks = zone_masks.to(device=feature_map.device, dtype=feature_map.dtype)
        if masks.ndim == 3:
            masks = masks.unsqueeze(0).expand(feature_map.shape[0], -1, -1, -1)
        if masks.shape[0] != feature_map.shape[0]:
            raise ValueError("zone_masks batch dimension does not match feature maps")
        masks = functional.interpolate(
            masks,
            size=feature_map.shape[-2:],
            mode="nearest",
        )
        flat_masks = masks.flatten(2)
        denominator = flat_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = flat_masks / denominator
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

    def _sensor_embedding(
        self,
        batch: int,
        time: int,
        sensors: torch.Tensor | None,
        sensor_mask: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sensor_dim == 0:
            values = torch.zeros(batch, time, 1, device=device)
            mask = torch.zeros_like(values)
        elif sensors is None:
            values = torch.zeros(batch, time, self.sensor_dim, device=device)
            mask = torch.zeros_like(values)
        else:
            if sensors.shape != (batch, time, self.sensor_dim):
                raise ValueError("sensors must be [batch,time,configured_sensor_features]")
            values = sensors.to(device=device, dtype=torch.float32)
            mask = (
                torch.ones_like(values)
                if sensor_mask is None
                else sensor_mask.to(device=device, dtype=torch.float32)
            )
        embedding = self.sensor_projection(torch.cat([values * mask, mask], dim=-1))
        denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
        coverage = mask.sum(dim=(1, 2)) / denominator.new_full(denominator.shape, time * max(self.sensor_dim, 1))
        return embedding, coverage

    def forward(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
        sensors: torch.Tensor | None = None,
        sensor_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        node_static: torch.Tensor | None = None,
        zone_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or flow.ndim != 5:
            raise ValueError("frames and flow must have [batch,time,channels,height,width]")
        batch, time, channels, height, width = frames.shape
        perception = self.perception(frames.reshape(batch * time, channels, height, width))
        feature_height, feature_width = perception["features"].shape[-2:]
        visual_maps = self.visual_projection(perception["features"]).reshape(
            batch, time, self.feature_dim, feature_height, feature_width
        )
        motion_maps = self.motion_projection(
            functional.interpolate(
                flow.reshape(batch * time, 4, height, width),
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )
        ).reshape(batch, time, self.feature_dim, feature_height, feature_width)

        conv_state: tuple[torch.Tensor, torch.Tensor] | None = None
        short_sequence: list[torch.Tensor] = []
        cross_attention_sequence: list[torch.Tensor] = []
        for step in range(time):
            step_motion = motion_maps[:, step] if self.use_motion else torch.zeros_like(motion_maps[:, step])
            conv_state = self.convlstm(visual_maps[:, step] + step_motion, conv_state)
            visual_tokens = self._zones(visual_maps[:, step], zone_masks)
            motion_tokens = self._zones(step_motion, zone_masks)
            fused_tokens, cross_weights = self.cross_attention(
                visual_tokens, motion_tokens, motion_tokens, need_weights=True
            )
            short = self._zones(conv_state[0], zone_masks) if self.use_short_temporal else visual_tokens
            short_sequence.append(short + fused_tokens)
            cross_attention_sequence.append(cross_weights)
        short_nodes = torch.stack(short_sequence, dim=1)
        node_count = short_nodes.shape[2]
        temporal_input = short_nodes.permute(0, 2, 1, 3).reshape(
            batch * node_count, time, self.feature_dim
        )
        long_nodes = (
            self.temporal(temporal_input)[:, -1].reshape(batch, node_count, self.feature_dim)
            if self.use_long_temporal
            else short_nodes[:, -1]
        )
        sensor_sequence, sensor_coverage = self._sensor_embedding(
            batch, time, sensors, sensor_mask, frames.device
        )
        sensor_latest = sensor_sequence[:, -1, None].expand(-1, node_count, -1)
        if not self.use_sensors:
            sensor_latest = torch.zeros_like(sensor_latest)
        encoded = self.strfe_fusion(
            torch.cat([short_nodes[:, -1], long_nodes, sensor_latest], dim=-1)
        )

        if node_static is None:
            node_static = encoded.new_zeros(batch, node_count, self.static_feature_dim)
        elif node_static.ndim == 2:
            node_static = node_static.unsqueeze(0).expand(batch, -1, -1)
        if node_static.shape != (batch, node_count, self.static_feature_dim):
            raise ValueError("node_static must match [batch,nodes,static_node_features]")
        if self.use_static_venue:
            encoded = encoded + self.static_projection(node_static.to(encoded))

        if adjacency is None:
            if node_count != self.adjacency.shape[0]:
                raise ValueError("An adaptive venue adjacency is required for semantic zones")
            graph_mask = self.adjacency
        else:
            graph_mask = adjacency
        graph_attentions: list[torch.Tensor] = []
        if self.use_graph_transformer:
            for layer in self.graph_transformer:
                encoded, attention = layer(encoded, graph_mask)
                graph_attentions.append(attention)
            final_attention = graph_attentions[-1]
        else:
            identity = torch.eye(node_count, device=encoded.device, dtype=encoded.dtype)
            final_attention = identity[None, None].expand(
                batch, int(self.cross_attention.num_heads), -1, -1
            )

        current_state = self._constrain_state(self.state_head(encoded))
        risk_logits = self.risk_head(encoded)
        risk_probabilities = torch.softmax(risk_logits, dim=-1)
        risk_confidence = 1.0 - (
            -(risk_probabilities.clamp_min(1e-8).log() * risk_probabilities).sum(dim=-1)
            / torch.log(torch.tensor(risk_probabilities.shape[-1], device=frames.device))
        )

        future_states: list[torch.Tensor] = []
        future_risk_logits: list[torch.Tensor] = []
        for horizon_embedding in self.horizon_embeddings:
            horizon = horizon_embedding.view(1, 1, -1).expand(batch, node_count, -1)
            decoder_input = torch.cat([encoded, horizon], dim=-1)
            future_states.append(self._constrain_state(self.forecast_decoder(decoder_input)))
            future_risk_logits.append(self.forecast_risk_head(decoder_input))

        density = perception["density"].reshape(batch, time, 1, feature_height, feature_width)
        localization = perception["localization_logits"].reshape(
            batch, time, 1, feature_height, feature_width
        )
        return {
            "density": density,
            "localization_logits": localization,
            "current_zone_state": current_state,
            "risk_logits": risk_logits,
            "risk_confidence": risk_confidence,
            "hazard_probability": torch.sigmoid(self.hazard_head(encoded)).squeeze(-1),
            "future_zone_state": torch.stack(future_states, dim=1),
            "future_risk_logits": torch.stack(future_risk_logits, dim=1),
            "graph_embeddings": encoded,
            "graph_attention": final_attention[:, None].expand(-1, time, -1, -1, -1),
            "cross_attention": torch.stack(cross_attention_sequence, dim=1),
            "sensor_coverage": sensor_coverage,
        }
