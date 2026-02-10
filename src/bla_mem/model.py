from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from .scan import hillis_steele_scan_inclusive
from .signature import chunk_signature_levels
from .tensor_algebra import bch_merge, element_log, flatten_levels


@dataclass(frozen=True)
class BLAMemConfig:
    input_dim: int
    depth: int
    chunk_size: int
    time_aug: bool = True
    prenorm: bool = True
    norm_eps: float = 1e-5
    readout_hidden: int = 256
    readout_pool: str = "last"  # last | mean | attn (over chunk prefixes)
    readout_dropout: float = 0.0
    readout_residual: bool = True
    out_dim: int = 1


class _ResidualReadoutBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, *, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class BLAMem(nn.Module):
    """BLA-Mem MVP: chunk signature -> log -> associative scan via BCH -> readout.

    Representation:
      - Signature is computed by `signatory` per chunk.
      - We convert signature (tensor algebra) to log(signature) via truncated log.
      - Merge is BCH implemented as log(exp(a)exp(b)) in the same truncated algebra.
    """

    def __init__(self, cfg: BLAMemConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.readout_pool not in {"last", "mean", "attn"}:
            raise ValueError("readout_pool must be one of: last, mean, attn")
        if cfg.readout_dropout < 0.0:
            raise ValueError("readout_dropout must be >= 0")

        channels = cfg.input_dim + (1 if cfg.time_aug else 0)
        self.channels = channels

        # Flat dimension of levels 1..m in tensor basis
        self.mem_dim = sum(channels**k for k in range(1, cfg.depth + 1))

        if cfg.norm_eps <= 0:
            raise ValueError("norm_eps must be > 0")

        # PreNorm on each log-signature level (k=1..depth), shaped (B, N, channels**k)
        if cfg.prenorm:
            self.level_norms = nn.ModuleList(
                [nn.LayerNorm(channels**k, eps=cfg.norm_eps) for k in range(1, cfg.depth + 1)]
            )
        else:
            self.level_norms = nn.ModuleList([])

        if cfg.readout_pool == "attn":
            self.pool_query = nn.Parameter(torch.empty(self.mem_dim))
            nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

        # Stable readout: (optional) input LayerNorm + (optional) residual block(s) in hidden space
        self.readout_in_norm = nn.LayerNorm(self.mem_dim, eps=cfg.norm_eps) if cfg.prenorm else nn.Identity()
        self.readout_in = nn.Linear(self.mem_dim, cfg.readout_hidden)
        self.readout_act = nn.GELU()
        self.readout_drop = nn.Dropout(p=cfg.readout_dropout)

        self.readout_hidden_norm = nn.LayerNorm(cfg.readout_hidden, eps=cfg.norm_eps) if cfg.prenorm else nn.Identity()
        self.readout_residual = (
            _ResidualReadoutBlock(cfg.readout_hidden, cfg.readout_hidden, cfg.readout_dropout, eps=cfg.norm_eps)
            if cfg.readout_residual
            else nn.Identity()
        )
        self.readout_out = nn.Linear(cfg.readout_hidden, cfg.out_dim)

    def _log_chunks(self, x: torch.Tensor) -> List[torch.Tensor]:
        sig_levels = chunk_signature_levels(
            x,
            depth=self.cfg.depth,
            chunk_size=self.cfg.chunk_size,
            time_aug=self.cfg.time_aug,
            basepoint=True,
        )
        # log(signature)
        log_levels = element_log(sig_levels)

        if self.cfg.prenorm:
            # Do not normalize level-0 (it is always 0 after log).
            normed: List[torch.Tensor] = [log_levels[0]]
            for ln, lvl in zip(self.level_norms, log_levels[1:], strict=True):
                normed.append(ln(lvl))
            return normed

        return log_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape (B, T, C)")

        log_levels = self._log_chunks(x)  # levels[k]: (B, N, channels**k)

        def merge(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
            return bch_merge(a, b)

        prefix = hillis_steele_scan_inclusive(log_levels, merge=merge, dim=1)

        # Per-chunk prefix representations: (B, N, mem_dim)
        flat_seq = torch.cat([lvl[:, :, :] for lvl in prefix[1:]], dim=-1)

        if self.cfg.readout_pool == "mean":
            pooled = flat_seq.mean(dim=1)
        elif self.cfg.readout_pool == "attn":
            scores = torch.matmul(flat_seq, self.pool_query)  # (B, N)
            weights = torch.softmax(scores, dim=1)
            pooled = (weights.unsqueeze(-1) * flat_seq).sum(dim=1)
        else:
            pooled = flat_seq[:, -1, :]

        h = self.readout_in(self.readout_in_norm(pooled))
        h = self.readout_act(h)
        h = self.readout_drop(h)

        # Residual path in hidden space (PreNorm lives inside the residual block).
        if self.cfg.readout_residual:
            h = self.readout_residual(h)
        else:
            h = self.readout_hidden_norm(h)

        return self.readout_out(h)
