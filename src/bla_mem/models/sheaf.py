from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSequenceModel


def _apply_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return F.linear(x, linear.weight, linear.bias)


def _apply_linear_T(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    # If y = x @ W^T, then applying transpose is y @ W.
    return F.linear(x, linear.weight.t(), None)


def conjugate_gradient(
    A_mv,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Solves A x = b for batched tensors; inner product over last dim.
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A_mv(x)
    p = r

    rs_old = (r * r).sum(dim=-1, keepdim=True)
    for _ in range(n_iters):
        Ap = A_mv(p)
        denom = (p * Ap).sum(dim=-1, keepdim=True).clamp_min(eps)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        if torch.max(rs_new).item() < eps:
            break
        beta = rs_new / rs_old.clamp_min(eps)
        p = r + beta * p
        rs_old = rs_new
    return x


class SheafGluingModel(BaseSequenceModel):
    """Minimal sheaf-gluing layer on a 1D chain.

    Implements h = argmin ||h - phi(x)||^2 + lam * sum_e ||R_L h_i - R_R h_{i+1}||^2
    via solving (I + lam L_sheaf) h = b with fixed-iter CG.
    """

    def __init__(self, input_dim: int, output_dim: int, d_model: int, lam: float, cg_iters: int) -> None:
        super().__init__()
        self.phi = nn.Linear(input_dim, d_model)
        self.R_left = nn.Linear(d_model, d_model, bias=False)
        self.R_right = nn.Linear(d_model, d_model, bias=False)
        self.lam = float(lam)
        self.cg_iters = int(cg_iters)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def _laplacian(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, D)
        bsz, n, d = z.shape
        if n <= 1:
            return torch.zeros_like(z)

        z_i = z[:, :-1, :]  # (B, N-1, D)
        z_j = z[:, 1:, :]

        g = _apply_linear(z_i, self.R_left) - _apply_linear(z_j, self.R_right)  # (B, N-1, D)

        out = torch.zeros_like(z)
        out[:, :-1, :] += _apply_linear_T(g, self.R_left)
        out[:, 1:, :] -= _apply_linear_T(g, self.R_right)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, input_dim)
        b = self.phi(x)

        def A_mv(v: torch.Tensor) -> torch.Tensor:
            vv = v.view_as(b)
            y = vv + self.lam * self._laplacian(vv)
            return y.reshape(v.shape)

        b_flat = b.reshape(b.shape[0], -1)
        x0 = b_flat
        h_flat = conjugate_gradient(A_mv=A_mv, b=b_flat, x0=x0, n_iters=self.cg_iters)
        h = h_flat.view_as(b)

        pooled = self.norm(h.mean(dim=1))
        return self.head(pooled)
