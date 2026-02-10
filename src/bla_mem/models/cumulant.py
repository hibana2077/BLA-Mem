from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSequenceModel


class CumulantScanModel(BaseSequenceModel):
    """Pre-Lie / cumulant-style prefix compression (minimal toy version).

    Keeps a small fixed-size state (k1..kK) and updates it with structured
    additive + elementwise interaction terms.
    """

    def __init__(self, input_dim: int, output_dim: int, d_model: int, k_order: int = 3) -> None:
        super().__init__()
        if k_order < 1 or k_order > 4:
            raise ValueError("k_order should be 1..4 for this toy implementation")
        self.k_order = int(k_order)
        self.d_model = int(d_model)

        self.phi = nn.Linear(input_dim, d_model)

        self.u1 = nn.Linear(d_model, d_model)
        self.u2 = nn.Linear(d_model, d_model)
        self.u3 = nn.Linear(d_model, d_model)
        self.u4 = nn.Linear(d_model, d_model)

        self.mix = nn.Linear(self.k_order * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.phi(x)  # (B,N,D)
        bsz, n, d = u.shape

        k1 = torch.zeros((bsz, d), device=x.device, dtype=x.dtype)
        k2 = torch.zeros_like(k1)
        k3 = torch.zeros_like(k1)
        k4 = torch.zeros_like(k1)

        for t in range(n):
            ut = u[:, t]
            dk1 = torch.tanh(self.u1(ut))
            k1_next = k1 + dk1

            if self.k_order >= 2:
                dk2 = torch.tanh(self.u2(ut)) * k1
                k2_next = k2 + dk2
            else:
                k2_next = k2

            if self.k_order >= 3:
                dk3 = torch.tanh(self.u3(ut)) * k2 + 0.5 * (dk1 * k1)
                k3_next = k3 + dk3
            else:
                k3_next = k3

            if self.k_order >= 4:
                dk4 = torch.tanh(self.u4(ut)) * k3 + (dk1 * k2)
                k4_next = k4 + dk4
            else:
                k4_next = k4

            k1, k2, k3, k4 = k1_next, k2_next, k3_next, k4_next

        states = [k1]
        if self.k_order >= 2:
            states.append(k2)
        if self.k_order >= 3:
            states.append(k3)
        if self.k_order >= 4:
            states.append(k4)

        h = self.mix(torch.cat(states, dim=-1))
        h = self.norm(h)
        return self.head(h)
