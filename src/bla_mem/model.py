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
    readout_hidden: int = 256
    readout_pool: str = "last"  # last | mean | attn (over chunk prefixes)
    readout_dropout: float = 0.0
    out_dim: int = 1


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

        if cfg.readout_pool == "attn":
            self.pool_query = nn.Parameter(torch.empty(self.mem_dim))
            nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

        self.readout = nn.Sequential(
            nn.Linear(self.mem_dim, cfg.readout_hidden),
            nn.ReLU(),
            nn.Dropout(p=cfg.readout_dropout),
            nn.Linear(cfg.readout_hidden, cfg.out_dim),
        )

    def _log_chunks(self, x: torch.Tensor) -> List[torch.Tensor]:
        sig_levels = chunk_signature_levels(
            x,
            depth=self.cfg.depth,
            chunk_size=self.cfg.chunk_size,
            time_aug=self.cfg.time_aug,
            basepoint=True,
        )
        # log(signature)
        return element_log(sig_levels)

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

        return self.readout(pooled)
