from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from crowd_twin.contracts import GRAPH_HANDOFF_SCHEMA_VERSION, STATE_FEATURES

from .layers import GraphTransformerLayer


@dataclass(frozen=True, slots=True)
class GraphReasonerMetadata:
    schema_version: str
    feature_dim: int
    hidden_dim: int
    static_feature_dim: int
    risk_classes: int
    horizons_seconds: tuple[int, ...]
    state_features: tuple[str, ...] = STATE_FEATURES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


class GraphRiskReasoner(nn.Module):
    """Standalone dynamic-graph and Graph Transformer workstream.

    This model intentionally begins at the frozen STRFE handoff. It lets the
    graph workstream train, test, and integrate without importing or modifying
    the teammate-owned temporal implementation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        static_feature_dim: int,
        graph_heads: int,
        graph_layers: int,
        risk_classes: int,
        horizons_seconds: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % graph_heads:
            raise ValueError("hidden_dim must be divisible by graph_heads")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.static_feature_dim = int(static_feature_dim)
        self.risk_classes = int(risk_classes)
        self.horizons_seconds = tuple(int(value) for value in horizons_seconds)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.static_projection = nn.Linear(self.static_feature_dim, self.hidden_dim)
        self.state_prior_projection = nn.Sequential(
            nn.Linear(len(STATE_FEATURES), self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    self.hidden_dim, heads=graph_heads, dropout=dropout
                )
                for _ in range(int(graph_layers))
            ]
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(STATE_FEATURES)),
        )
        self.risk_head = nn.Linear(self.hidden_dim, self.risk_classes)
        self.hazard_head = nn.Linear(self.hidden_dim, 1)
        self.metadata = GraphReasonerMetadata(
            schema_version=GRAPH_HANDOFF_SCHEMA_VERSION,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            static_feature_dim=self.static_feature_dim,
            risk_classes=self.risk_classes,
            horizons_seconds=self.horizons_seconds,
        )

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

    def _validate(
        self,
        current_features: torch.Tensor,
        future_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_static: torch.Tensor,
        future_state_prior: torch.Tensor | None,
        current_state_prior: torch.Tensor | None,
    ) -> tuple[int, int, int]:
        if current_features.ndim != 3:
            raise ValueError("current_zone_features must be [B,N,D]")
        if future_features.ndim != 4:
            raise ValueError("future_zone_features must be [B,H,N,D]")
        batch, nodes, dimension = current_features.shape
        if dimension != self.feature_dim:
            raise ValueError(
                f"current feature dimension {dimension} does not match checkpoint {self.feature_dim}"
            )
        expected_future = (batch, len(self.horizons_seconds), nodes, self.feature_dim)
        if future_features.shape != expected_future:
            raise ValueError(
                f"future_zone_features shape {tuple(future_features.shape)} "
                f"does not match {expected_future}"
            )
        if adjacency.shape not in ((nodes, nodes), (batch, nodes, nodes)):
            raise ValueError("adjacency must be [N,N] or [B,N,N]")
        if node_static.ndim == 2:
            expected_static = (nodes, self.static_feature_dim)
        else:
            expected_static = (batch, nodes, self.static_feature_dim)
        if tuple(node_static.shape) != expected_static:
            raise ValueError(f"node_static shape must be {expected_static}")
        if future_state_prior is not None and tuple(future_state_prior.shape) != (
            batch,
            len(self.horizons_seconds),
            nodes,
            len(STATE_FEATURES),
        ):
            raise ValueError("future_state_prior must be [B,H,N,7]")
        if current_state_prior is not None and tuple(current_state_prior.shape) != (
            batch,
            nodes,
            len(STATE_FEATURES),
        ):
            raise ValueError("current_state_prior must be [B,N,7]")
        return batch, nodes, dimension

    def _encode(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        static: torch.Tensor,
        state_prior: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.feature_projection(features) + self.static_projection(static)
        if state_prior is not None:
            encoded = encoded + self.state_prior_projection(state_prior)
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            encoded, attention = layer(encoded, adjacency)
            attentions.append(attention)
        if not attentions:
            raise RuntimeError("GraphRiskReasoner requires at least one graph layer")
        return encoded, attentions[-1]

    def forward(
        self,
        current_zone_features: torch.Tensor,
        future_zone_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_static: torch.Tensor,
        future_state_prior: torch.Tensor | None = None,
        current_state_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, _nodes, _ = self._validate(
            current_zone_features,
            future_zone_features,
            adjacency,
            node_static,
            future_state_prior,
            current_state_prior,
        )
        if node_static.ndim == 2:
            static = node_static.unsqueeze(0).expand(batch, -1, -1)
        else:
            static = node_static
        current_encoded, current_attention = self._encode(
            current_zone_features, adjacency, static, current_state_prior
        )
        future_embeddings: list[torch.Tensor] = []
        future_attentions: list[torch.Tensor] = []
        future_states: list[torch.Tensor] = []
        future_risk_logits: list[torch.Tensor] = []
        for horizon_index in range(len(self.horizons_seconds)):
            prior = (
                future_state_prior[:, horizon_index]
                if future_state_prior is not None
                else None
            )
            encoded, attention = self._encode(
                future_zone_features[:, horizon_index], adjacency, static, prior
            )
            future_embeddings.append(encoded)
            future_attentions.append(attention)
            future_states.append(self._constrain_state(self.state_head(encoded)))
            future_risk_logits.append(self.risk_head(encoded))

        current_state = self._constrain_state(self.state_head(current_encoded))
        current_risk_logits = self.risk_head(current_encoded)
        probabilities = torch.softmax(current_risk_logits, dim=-1)
        entropy = -(
            probabilities.clamp_min(1e-8).log() * probabilities
        ).sum(dim=-1)
        normalizer = torch.log(
            torch.tensor(self.risk_classes, device=probabilities.device, dtype=probabilities.dtype)
        )
        return {
            "current_zone_state": current_state,
            "future_zone_state": torch.stack(future_states, dim=1),
            "risk_logits": current_risk_logits,
            "future_risk_logits": torch.stack(future_risk_logits, dim=1),
            "hazard_probability": torch.sigmoid(self.hazard_head(current_encoded)).squeeze(-1),
            "risk_confidence": 1.0 - entropy / normalizer,
            "graph_embeddings": current_encoded,
            "future_graph_embeddings": torch.stack(future_embeddings, dim=1),
            "graph_attention": current_attention,
            "future_graph_attention": torch.stack(future_attentions, dim=1),
        }
