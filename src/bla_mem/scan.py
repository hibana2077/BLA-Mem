from __future__ import annotations

from typing import Callable, List, Sequence

import torch


def _levels_like(levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(t) for t in levels]


def hillis_steele_scan_inclusive(
    xs: Sequence[torch.Tensor],
    *,
    merge: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], List[torch.Tensor]],
    dim: int = 1,
) -> List[torch.Tensor]:
    """Inclusive associative scan over a sequence dimension.

    xs is a level list where each tensor is shaped (B, N, D_k) by default.
    This uses Hillis–Steele (O(N log N) work but parallel-friendly).
    """

    levels = [t.clone() for t in xs]
    n = levels[0].shape[dim]

    step = 1
    while step < n:
        # slice: right = x[step:], left = x[:-step]
        right = [t.narrow(dim, step, n - step) for t in levels]
        left = [t.narrow(dim, 0, n - step) for t in levels]
        merged = merge(left, right)

        # Autograd-safe update: do not mutate `levels[...]` in-place.
        updated: List[torch.Tensor] = []
        for k, t in enumerate(levels):
            prefix = t.narrow(dim, 0, step)
            updated.append(torch.cat([prefix, merged[k]], dim=dim))
        levels = updated

        step *= 2

    return levels
