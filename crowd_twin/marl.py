from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class CrowdRoutingEnv:
    """Graph-based multi-agent evacuation environment for policy training."""

    def __init__(
        self,
        adjacency: np.ndarray,
        exits: Iterable[int],
        zone_capacity: np.ndarray,
        max_steps: int = 128,
    ) -> None:
        self.adjacency = adjacency.astype(bool)
        self.exits = {int(exit_id) for exit_id in exits}
        self.capacity = zone_capacity.astype(np.float32)
        self.max_steps = max_steps
        self.node_count = len(zone_capacity)
        self.positions = np.empty(0, dtype=np.int64)
        self.step_count = 0

    def reset(self, positions: np.ndarray, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        self.positions = positions.astype(np.int64).copy()
        self.step_count = 0
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        occupancy = np.bincount(self.positions, minlength=self.node_count).astype(np.float32)
        density = occupancy / np.maximum(self.capacity, 1.0)
        return {
            "positions": self.positions.copy(),
            "occupancy": occupancy,
            "density": density,
            "global": np.concatenate([occupancy, density]),
        }

    def action_mask(self) -> np.ndarray:
        masks = np.zeros((len(self.positions), self.node_count), dtype=bool)
        for index, position in enumerate(self.positions):
            masks[index] = self.adjacency[position]
            masks[index, position] = True
        return masks

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, bool, dict[str, float]]:
        masks = self.action_mask()
        valid = masks[np.arange(len(actions)), actions]
        next_positions = np.where(valid, actions, self.positions)
        evacuated = np.array([position in self.exits for position in next_positions])
        active_positions = next_positions[~evacuated]
        occupancy = np.bincount(active_positions, minlength=self.node_count).astype(np.float32)
        density = occupancy / np.maximum(self.capacity, 1.0)
        congestion = np.maximum(density - 1.0, 0.0)
        global_penalty = congestion.sum() + len(active_positions) / max(len(actions), 1)
        rewards = np.where(evacuated, 2.0, -0.01 - global_penalty * 0.05).astype(np.float32)
        rewards[~valid] -= 0.25
        self.positions = active_positions
        self.step_count += 1
        done = len(self.positions) == 0 or self.step_count >= self.max_steps
        return self.observation(), rewards, done, {
            "evacuated": float(evacuated.sum()),
            "remaining": float(len(self.positions)),
            "congestion": float(congestion.sum()),
        }


class MAPPOPolicy(nn.Module):
    def __init__(self, observation_dim: int, node_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.node_count = node_count
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def act(
        self, observations: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observations).masked_fill(~action_mask, -1e9)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), self.critic(observations).squeeze(-1)

    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observations).masked_fill(~action_mask, -1e9)
        distribution = Categorical(logits=logits)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(observations).squeeze(-1),
        )


class HierarchicalGraphMAPPO(nn.Module):
    """Centralized critic with specialized leader, zone, exit and signage actors."""

    roles = ("leader", "zone", "exit", "signage")

    def __init__(
        self,
        graph_embedding_dim: int,
        node_count: int,
        leader_actions: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.encoder = nn.Sequential(
            nn.Linear(graph_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actors = nn.ModuleDict(
            {
                "leader": nn.Linear(hidden_dim, leader_actions),
                "zone": nn.Linear(hidden_dim, node_count),
                "exit": nn.Linear(hidden_dim, node_count),
                "signage": nn.Linear(hidden_dim, node_count),
            }
        )
        self.critic = nn.Sequential(
            nn.Linear(graph_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def distributions(
        self,
        graph_embedding: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Categorical]:
        hidden = self.encoder(graph_embedding)
        distributions: dict[str, Categorical] = {}
        for role, actor in self.actors.items():
            logits = actor(hidden)
            if masks and role in masks:
                logits = logits.masked_fill(~masks[role].bool(), -1e9)
            distributions[role] = Categorical(logits=logits)
        return distributions

    def act(
        self,
        graph_embedding: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        distributions = self.distributions(graph_embedding, masks)
        actions = {role: distribution.sample() for role, distribution in distributions.items()}
        log_probabilities = {
            role: distribution.log_prob(actions[role])
            for role, distribution in distributions.items()
        }
        return actions, log_probabilities, self.critic(graph_embedding).squeeze(-1)


def hierarchical_mappo_loss(
    policy: HierarchicalGraphMAPPO,
    graph_embeddings: torch.Tensor,
    actions: dict[str, torch.Tensor],
    old_log_probabilities: dict[str, torch.Tensor],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    masks: dict[str, torch.Tensor] | None = None,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> MAPPOLoss:
    distributions = policy.distributions(graph_embeddings, masks)
    actor_losses = []
    entropies = []
    for role, distribution in distributions.items():
        log_probability = distribution.log_prob(actions[role])
        ratio = (log_probability - old_log_probabilities[role]).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        actor_losses.append(-torch.minimum(unclipped, clipped).mean())
        entropies.append(distribution.entropy().mean())
    actor_loss = torch.stack(actor_losses).mean()
    entropy = torch.stack(entropies).mean()
    values = policy.critic(graph_embeddings).squeeze(-1)
    critic_loss = (values - returns).square().mean()
    total = actor_loss + value_weight * critic_loss - entropy_weight * entropy
    return MAPPOLoss(total, actor_loss, critic_loss, entropy)


@dataclass(slots=True)
class MAPPOLoss:
    total: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor


def mappo_loss(
    policy: MAPPOPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> MAPPOLoss:
    log_probabilities, entropy, values = policy.evaluate(observations, actions, action_masks)
    ratio = (log_probabilities - old_log_probabilities).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.minimum(unclipped, clipped).mean()
    critic_loss = (values - returns).square().mean()
    entropy_loss = entropy.mean()
    total = actor_loss + value_weight * critic_loss - entropy_weight * entropy_loss
    return MAPPOLoss(total, actor_loss, critic_loss, entropy_loss)
