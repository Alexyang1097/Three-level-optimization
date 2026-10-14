from __future__ import annotations

import torch
from torch import nn

from .data import INPUT_DIM, NUM_TYPES


class CausalCoTTransformer(nn.Module):
    """Small causal Transformer matching the proposed Section 7 architecture."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
        mlp_width: int = 256,
        max_positions: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, hidden_size)
        self.type_embed = nn.Embedding(NUM_TYPES, hidden_size)
        self.pos_embed = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.readout = nn.Linear(hidden_size, INPUT_DIM)

    def forward(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = tokens.shape
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max_positions={self.pos_embed.num_embeddings}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.input_proj(tokens) + self.type_embed(types) + self.pos_embed(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.readout(hidden)


def gather_query_predictions(outputs: torch.Tensor, pred_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_idx = torch.arange(outputs.shape[0], device=outputs.device)
    p1 = outputs[batch_idx, pred_pos[:, 0], :4]
    p2 = outputs[batch_idx, pred_pos[:, 1], :4]
    p3 = outputs[batch_idx, pred_pos[:, 2], :1]
    return p1, p2, p3

