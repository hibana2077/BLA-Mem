from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSequenceModel


class MagnusCommutatorModel(BaseSequenceModel):
    """Magnus/commutator evolution over a sequence of learned generators.

    Minimal 2nd-order truncation with O(N) commutator accumulation.
    Uses block-diagonal generators to keep matrix_exp cheap.
    """

    def __init__(self, input_dim: int, output_dim: int, d_model: int, block_size: int) -> None:
        super().__init__()
        if d_model % block_size != 0:
            raise ValueError("d_model must be divisible by block_size")
        self.d_model = d_model
        self.block_size = int(block_size)
        self.n_blocks = d_model // block_size

        self.phi = nn.Linear(input_dim, d_model)
        self.gen = nn.Linear(input_dim, self.n_blocks * self.block_size * self.block_size)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,input)
        bsz, n, _ = x.shape
        h_tokens = self.phi(x)  # (B,N,D)
        h0 = h_tokens.mean(dim=1)  # (B,D)

        A = self.gen(x)
        A = A.view(bsz, n, self.n_blocks, self.block_size, self.block_size)

        # Prefix-sum commutator accumulation: sum_{k} [S_{k-1}, A_k] = sum_{i<k} [A_i, A_k]
        S = torch.zeros((bsz, self.n_blocks, self.block_size, self.block_size), device=x.device, dtype=x.dtype)
        comm = torch.zeros_like(S)
        for t in range(n):
            At = A[:, t]
            comm = comm + (S @ At - At @ S)
            S = S + At

        Omega = S + 0.5 * comm

        # Apply exp(Omega) blockwise to h0
        h0b = h0.view(bsz, self.n_blocks, self.block_size)
        Omega_flat = Omega.view(bsz * self.n_blocks, self.block_size, self.block_size)
        exp_flat = torch.matrix_exp(Omega_flat)
        expm = exp_flat.view(bsz, self.n_blocks, self.block_size, self.block_size)

        out = torch.einsum("bnij,bnj->bni", expm, h0b).reshape(bsz, self.d_model)
        out = self.norm(out)
        return self.head(out)
