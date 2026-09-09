from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .layers import GraphAttentionLayer


@dataclass(slots=True)
class InterventionBatch:
    exit_capacity_delta: torch.Tensor
    redirect_matrix: torch.Tensor
    blocked_zones: torch.Tensor
    external_inflow: torch.Tensor

    def validate(self, node_count: int) -> None:
        if self.exit_capacity_delta.shape[-1] != node_count:
            raise ValueError("exit_capacity_delta has the wrong node dimension")
        if self.redirect_matrix.shape[-2:] != (node_count, node_count):
            raise ValueError("redirect_matrix must be [batch,nodes,nodes]")
        if self.blocked_zones.shape[-1] != node_count:
            raise ValueError("blocked_zones has the wrong node dimension")
        if self.external_inflow.shape[-1] != node_count:
            raise ValueError("external_inflow has the wrong node dimension")


class LearnedDigitalTwin(nn.Module):
    """Learned counterfactual transition model; no rule-based risk reductions."""

    def __init__(self, state_dim: int = 7, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.intervention_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.graph = GraphAttentionLayer(hidden_dim, hidden_dim, heads=heads)
        self.transition = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.uncertainty = nn.Sequential(nn.Linear(hidden_dim, state_dim), nn.Softplus())

    def forward(
        self,
        state: torch.Tensor,
        intervention: InterventionBatch,
        adjacency: torch.Tensor,
        steps: int,
    ) -> dict[str, torch.Tensor]:
        batch, node_count, _ = state.shape
        intervention.validate(node_count)
        redirect_out = intervention.redirect_matrix.sum(dim=-1)
        redirect_in = intervention.redirect_matrix.sum(dim=-2)
        intervention_features = torch.stack(
            [
                intervention.exit_capacity_delta,
                redirect_in - redirect_out,
                intervention.blocked_zones.float(),
                intervention.external_inflow,
            ],
            dim=-1,
        )
        encoded_intervention = self.intervention_encoder(intervention_features)
        current = state
        trajectories: list[torch.Tensor] = []
        uncertainties: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        effective_adjacency = adjacency.clone()
        if effective_adjacency.ndim == 2:
            effective_adjacency = effective_adjacency.unsqueeze(0).expand(batch, -1, -1)
        blocked = intervention.blocked_zones.bool()
        effective_adjacency = effective_adjacency & ~blocked[:, :, None] & ~blocked[:, None, :]
        effective_adjacency = effective_adjacency | torch.eye(
            node_count, dtype=torch.bool, device=state.device
        )[None]
        for _ in range(steps):
            hidden = self.state_encoder(current) + encoded_intervention
            hidden, attention = self.graph(hidden, effective_adjacency)
            delta = self.transition(hidden)
            current = current + delta
            current = torch.stack(
                [
                    functional.softplus(current[..., 0]),
                    functional.softplus(current[..., 1]),
                    functional.softplus(current[..., 2]),
                    current[..., 3],
                    current[..., 4],
                    functional.softplus(current[..., 5]),
                    torch.sigmoid(current[..., 6]),
                ],
                dim=-1,
            )
            trajectories.append(current)
            uncertainties.append(self.uncertainty(hidden))
            attentions.append(attention)
        return {
            "trajectory": torch.stack(trajectories, dim=1),
            "uncertainty": torch.stack(uncertainties, dim=1),
            "attention": torch.stack(attentions, dim=1),
        }
