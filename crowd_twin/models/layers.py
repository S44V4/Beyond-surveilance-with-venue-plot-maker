from __future__ import annotations

import math

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
            hidden = x.new_zeros(shape)
            cell = x.new_zeros(shape)
        else:
            hidden, cell = state
        gates = self.gates(torch.cat([x, hidden], dim=1))
        input_gate, forget_gate, output_gate, proposal = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(proposal)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class GraphAttentionLayer(nn.Module):
    """Multi-head GATv2-style attention without torch-geometric dependencies."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.output = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self, nodes: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, node_count, _ = nodes.shape
        query = self.query(nodes).view(batch, node_count, self.heads, self.head_dim)
        key = self.key(nodes).view(batch, node_count, self.heads, self.head_dim)
        value = self.value(nodes).view(batch, node_count, self.heads, self.head_dim)
        pair = torch.tanh(query[:, :, None] + key[:, None, :])
        logits = torch.einsum("bijhd,hd->bhij", pair, self.attention) / math.sqrt(self.head_dim)
        mask = adjacency.to(dtype=torch.bool, device=nodes.device)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        aggregated = torch.einsum("bhij,bjhd->bihd", weights, value).reshape(
            batch, node_count, -1
        )
        output = self.norm(self.residual(nodes) + self.dropout(self.output(aggregated)))
        return output, weights


class GraphTransformerLayer(nn.Module):
    """Transformer block over venue nodes with a physical-accessibility mask."""

    def __init__(self, dimension: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.norm_output = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, nodes: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = nodes.shape[1]
        mask = adjacency.to(dtype=torch.bool, device=nodes.device)
        if mask.ndim == 3:
            if not torch.equal(mask, mask[:1].expand_as(mask)):
                raise ValueError("Batched graph masks must be identical within a batch")
            mask = mask[0]
        if mask.shape != (node_count, node_count):
            raise ValueError("adjacency does not match the number of graph nodes")
        attended, weights = self.attention(
            nodes,
            nodes,
            nodes,
            attn_mask=~mask,
            need_weights=True,
            average_attn_weights=False,
        )
        nodes = self.norm_attention(nodes + self.dropout(attended))
        nodes = self.norm_output(nodes + self.dropout(self.feed_forward(nodes)))
        return nodes, weights


class TemporalStateSpaceBlock(nn.Module):
    """Uses Mamba2 when installed; otherwise a documented gated GRU fallback."""

    def __init__(self, dimension: int, layers: int, dropout: float, prefer_mamba: bool) -> None:
        super().__init__()
        self.backend = "gru"
        self.block: nn.Module
        if prefer_mamba:
            try:
                from mamba_ssm import Mamba2  # type: ignore[import-not-found]

                self.block = nn.Sequential(
                    *[Mamba2(d_model=dimension, d_state=64, d_conv=4, expand=2) for _ in range(layers)]
                )
                self.backend = "mamba2"
            except (ImportError, RuntimeError):
                self.block = nn.GRU(
                    input_size=dimension,
                    hidden_size=dimension,
                    num_layers=layers,
                    dropout=dropout if layers > 1 else 0.0,
                    batch_first=True,
                )
        else:
            self.block = nn.GRU(
                input_size=dimension,
                hidden_size=dimension,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        result = self.block(sequence)
        return result[0] if isinstance(result, tuple) else result


def grid_adjacency(rows: int, columns: int) -> torch.Tensor:
    node_count = rows * columns
    adjacency = torch.eye(node_count, dtype=torch.bool)
    for row in range(rows):
        for column in range(columns):
            source = row * columns + column
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if 0 <= neighbor_row < rows and 0 <= neighbor_column < columns:
                    target = neighbor_row * columns + neighbor_column
                    adjacency[source, target] = True
    return adjacency
