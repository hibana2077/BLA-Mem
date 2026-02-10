from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSequenceModel


class TransformerBaseline(BaseSequenceModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_len: int,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_len = max_len
        self.pooling = pooling

        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, _ = x.shape
        if n > self.max_len:
            raise ValueError(f"seq_len {n} > max_len {self.max_len}. Increase --max-len.")
        pos = torch.arange(n, device=x.device)
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        h = self.encoder(h)
        h = self.norm(h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "last":
            pooled = h[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(pooled)
