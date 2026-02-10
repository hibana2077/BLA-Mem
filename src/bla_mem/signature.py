from __future__ import annotations

from typing import List, Sequence

import torch

from .tensor_algebra import add_level0, split_signature_flat


def _require_signatory():
    try:
        import signatory  # type: ignore

        return signatory
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional dependency `signatory` (or it failed to import). "
            "Install it in your cloud environment (torch/cuda must match)."
        ) from e


def time_augment(x: torch.Tensor) -> torch.Tensor:
    """Append a time channel in [0,1] along the sequence length."""

    b, t, _ = x.shape
    tt = torch.linspace(0.0, 1.0, steps=t, device=x.device, dtype=x.dtype)
    tt = tt.view(1, t, 1).expand(b, t, 1)
    return torch.cat([x, tt], dim=-1)


def chunk_path(x: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Chunk a path into (B, N, chunk_size, C). Truncates remainder."""

    if x.dim() != 3:
        raise ValueError("x must have shape (B, T, C)")
    b, t, c = x.shape
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    n = t // chunk_size
    t_eff = n * chunk_size
    x = x[:, :t_eff, :]
    return x.view(b, n, chunk_size, c)


def chunk_signature_levels(
    x: torch.Tensor,
    *,
    depth: int,
    chunk_size: int,
    time_aug: bool,
    basepoint: bool = True,
) -> List[torch.Tensor]:
    """Compute per-chunk signatures and return levels with level-0 included.

    Output levels are shaped as:
      levels[k]: (B, N, channels**k), for k=0..depth
    """

    signatory = _require_signatory()

    if time_aug:
        x = time_augment(x)

    b, t, channels = x.shape
    chunks = chunk_path(x, chunk_size=chunk_size)  # (B, N, L, C)
    b, n, l, c = chunks.shape

    flat = chunks.reshape(b * n, l, c)

    if l < 2:
        raise ValueError("chunk_size must be >= 2 to form path increments")

    # Use the first point as the basepoint to avoid a spurious increment from 0.
    # This makes each chunk's signature depend only on within-chunk increments.
    bp = flat[:, 0, :]
    stream = flat[:, 1:, :]

    # signatory.signature returns (B*N, sum_{k=1..depth} c**k)
    sig_flat = signatory.signature(stream, depth=depth, basepoint=bp)

    levels_1_to_m = split_signature_flat(sig_flat, channels=c, depth=depth)
    levels_with0 = add_level0(levels_1_to_m)

    # reshape to (B, N, dim)
    out: List[torch.Tensor] = []
    for lvl in levels_with0:
        out.append(lvl.view(b, n, -1))

    return out
