from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


def _level_sizes(channels: int, depth: int) -> List[int]:
    return [channels**k for k in range(depth + 1)]


def split_signature_flat(flat: torch.Tensor, *, channels: int, depth: int) -> List[torch.Tensor]:
    """Split signatory-style signature output into per-level tensors.

    Expected `flat` shape: (..., sum_{k=1..depth} channels**k)
    Returns levels [1..depth], each with shape (..., channels**k).

    Note: signatory does not include level-0 (the scalar 1) in the returned signature.
    """

    if depth < 1:
        raise ValueError("depth must be >= 1")

    sizes = [channels**k for k in range(1, depth + 1)]
    total = sum(sizes)
    if flat.shape[-1] != total:
        raise ValueError(
            f"Bad flat signature dim: got {flat.shape[-1]}, expected {total} (channels={channels}, depth={depth})."
        )

    out: List[torch.Tensor] = []
    start = 0
    for size in sizes:
        out.append(flat[..., start : start + size])
        start += size
    return out


def flatten_levels(levels: Sequence[torch.Tensor], *, include_level0: bool = False) -> torch.Tensor:
    """Flatten a level list into a single last-dimension vector."""

    if not levels:
        raise ValueError("levels must be non-empty")

    if include_level0:
        to_cat = list(levels)
    else:
        to_cat = list(levels[1:])
    return torch.cat(to_cat, dim=-1) if len(to_cat) > 1 else to_cat[0]


def make_identity_like(levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Return identity element with same batch shape/device/dtype as `levels`."""

    if not levels:
        raise ValueError("levels must be non-empty")

    batch_shape = levels[0].shape[:-1]
    device = levels[0].device
    dtype = levels[0].dtype

    out: List[torch.Tensor] = []
    out.append(torch.ones((*batch_shape, 1), device=device, dtype=dtype))
    for k in range(1, len(levels)):
        out.append(torch.zeros((*batch_shape, levels[k].shape[-1]), device=device, dtype=dtype))
    return out


def add_level0(levels_1_to_m: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if not levels_1_to_m:
        raise ValueError("expected at least level-1")

    batch_shape = levels_1_to_m[0].shape[:-1]
    device = levels_1_to_m[0].device
    dtype = levels_1_to_m[0].dtype
    return [torch.ones((*batch_shape, 1), device=device, dtype=dtype), *list(levels_1_to_m)]


def element_add(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [x + y for x, y in zip(a, b, strict=True)]


def element_sub(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [x - y for x, y in zip(a, b, strict=True)]


def element_scale(a: Sequence[torch.Tensor], s: float) -> List[torch.Tensor]:
    return [x * s for x in a]


def element_mul(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Truncated tensor-algebra multiplication (concatenation product).

    a and b are represented as per-level flattened tensors:
      a[k] has shape (..., channels**k), for k=0..m.

    Product is truncated so that degrees > m are dropped.
    """

    m = len(a) - 1
    if len(b) - 1 != m:
        raise ValueError("a and b must have same max degree")

    out: List[torch.Tensor] = []
    # degree 0
    out.append(a[0] * b[0])

    for k in range(1, m + 1):
        acc = None
        for i in range(0, k + 1):
            j = k - i
            ai = a[i]
            bj = b[j]
            term = torch.einsum("...i,...j->...ij", ai, bj).reshape(*ai.shape[:-1], ai.shape[-1] * bj.shape[-1])
            acc = term if acc is None else (acc + term)
        out.append(acc)

    return out


def element_pow(a: Sequence[torch.Tensor], n: int) -> List[torch.Tensor]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return make_identity_like(a)
    out = list(a)
    for _ in range(n - 1):
        out = element_mul(out, a)
    return out


def element_exp(x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Exponential in truncated tensor algebra.

    Works because the augmentation ideal is nilpotent in the truncated quotient.
    """

    m = len(x) - 1
    out = make_identity_like(x)

    # exp(x) = sum_{n=0..m} x^n / n!
    term = make_identity_like(x)
    for n in range(1, m + 1):
        term = element_mul(term, x)
        # small m (<=4) in MVP, so a tiny Python factorial is fine
        fact = 1
        for k in range(2, n + 1):
            fact *= k
        out = element_add(out, element_scale(term, 1.0 / float(fact)))

    return out


def element_log(g: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Logarithm in truncated tensor algebra.

    Requires g[0] == 1 (identity scalar) and g is group-like (for signatures).
    In the truncated setting, we compute:
      log(1 + u) = sum_{n=1..m} (-1)^{n+1} u^n / n
    where u = g - 1.
    """

    m = len(g) - 1

    # u = g - 1
    one = make_identity_like(g)
    u = element_sub(g, one)

    out = [torch.zeros_like(t) for t in g]
    term = list(u)

    for n in range(1, m + 1):
        coeff = (1.0 / n) * (1.0 if (n % 2 == 1) else -1.0)
        out = element_add(out, element_scale(term, coeff))
        term = element_mul(term, u)

    return out


def bch_merge(log_a: Sequence[torch.Tensor], log_b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """BCH in the truncated tensor-algebra representation via log(exp(a)exp(b))."""

    return element_log(element_mul(element_exp(log_a), element_exp(log_b)))
