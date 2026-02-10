from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSequenceModel


class UltrametricHeatFlowModel(BaseSequenceModel):
    """Hierarchical (ultrametric) heat-flow style mixing via pooling/unpooling.

    Minimal GPU-friendly approximation of exp(-tau L_H) using multilevel averages.
    """

    def __init__(self, input_dim: int, output_dim: int, d_model: int, levels: int) -> None:
        super().__init__()
        self.phi = nn.Linear(input_dim, d_model)
        self.levels = int(levels)

        # mixing gates per level (from fine->coarse and coarse->fine)
        self.alpha = nn.Parameter(torch.zeros(self.levels))
        self.beta = nn.Parameter(torch.zeros(self.levels))

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    @staticmethod
    def _pair_pool(x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D) -> (B,ceil(N/2),D)
        b, n, d = x.shape
        if n % 2 == 1:
            x = torch.cat([x, x[:, -1:, :]], dim=1)
            n = n + 1
        x = x.view(b, n // 2, 2, d).mean(dim=2)
        return x

    @staticmethod
    def _pair_unpool(x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B,M,D) -> (B,2M,D) then crop
        x = x.repeat_interleave(2, dim=1)
        return x[:, :target_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.phi(x)

        # Down: collect multilevel pooled representations
        pools: list[torch.Tensor] = [h0]
        h = h0
        for _ in range(self.levels):
            if h.shape[1] <= 1:
                break
            h = self._pair_pool(h)
            pools.append(h)

        # Up: inject coarse info back to fine with learned mixing
        out = pools[0]
        max_level = len(pools) - 1
        for lvl in range(max_level, 0, -1):
            coarse = pools[lvl]
            fine_len = pools[lvl - 1].shape[1]
            coarse_up = self._pair_unpool(coarse, fine_len)

            a = torch.sigmoid(self.alpha[min(lvl - 1, self.levels - 1)])
            b = torch.sigmoid(self.beta[min(lvl - 1, self.levels - 1)])

            # two-stage mix: smooth then residual
            out = (1.0 - a) * out + a * coarse_up
            out = out + b * (coarse_up - out)

        pooled = self.norm(out.mean(dim=1))
        return self.head(pooled)
