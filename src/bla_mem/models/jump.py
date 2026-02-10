from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSequenceModel


class JumpGeneratorModel(BaseSequenceModel):
    """Dirichlet-form symmetric jump generator on a 1D chain (Toeplitz-by-distance).

    Uses a learned nonnegative distance kernel J_d and applies a few explicit steps
    approximating exp(-tau L) where (Lh)_i = sum_{j!=i} J_{|i-j|} (h_i - h_j).

    For Toeplitz-by-distance, this becomes:
      Lh = (2 * sum_d J_d) * h - conv1d(h, kernel)
    with a symmetric kernel and zero center weight.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        max_dist: int = 64,
        n_steps: int = 8,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.phi = nn.Linear(input_dim, d_model)
        self.max_dist = int(max_dist)
        self.n_steps = int(n_steps)
        self.tau = float(tau)

        # Unconstrained params -> J_d via softplus; add mild 1/d decay for extrapolation.
        self.logits = nn.Parameter(torch.zeros(self.max_dist))

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def _kernel(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (kernel_1d, sumJ)
        d = torch.arange(1, self.max_dist + 1, device=device, dtype=dtype)
        J = F.softplus(self.logits.to(device=device, dtype=dtype)) / d
        sumJ = J.sum()

        # symmetric kernel with zero center weight: [J_D ... J_1, 0, J_1 ... J_D]
        k = torch.cat([J.flip(0), torch.zeros(1, device=device, dtype=dtype), J], dim=0)
        return k, sumJ

    def _laplacian(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,N,D)
        bsz, n, d = h.shape
        k, sumJ = self._kernel(h.device, h.dtype)
        pad = self.max_dist

        # depthwise conv: (B,D,N) conv1d with groups=D
        x = h.transpose(1, 2)  # (B,D,N)
        weight = k.view(1, 1, -1).repeat(d, 1, 1)  # (D,1,K)
        neigh = F.conv1d(x, weight=weight, bias=None, padding=pad, groups=d)  # (B,D,N)
        neigh = neigh.transpose(1, 2)  # (B,N,D)

        return (2.0 * sumJ) * h - neigh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        if h.shape[1] <= 1 or self.max_dist <= 0 or self.n_steps <= 0:
            pooled = self.norm(h.mean(dim=1))
            return self.head(pooled)

        dt = self.tau / float(self.n_steps)
        for _ in range(self.n_steps):
            h = h - dt * self._laplacian(h)

        pooled = self.norm(h.mean(dim=1))
        return self.head(pooled)
