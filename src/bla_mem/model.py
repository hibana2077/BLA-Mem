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

        channels = cfg.input_dim + (1 if cfg.time_aug else 0)
        self.channels = channels

        # Flat dimension of levels 1..m in tensor basis
        self.mem_dim = sum(channels**k for k in range(1, cfg.depth + 1))

        self.readout = nn.Sequential(
            nn.Linear(self.mem_dim, cfg.readout_hidden),
            nn.ReLU(),
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

        # global memory = last prefix element
        global_levels = [lvl[:, -1, :] for lvl in prefix]
        flat = flatten_levels(global_levels, include_level0=False)

        return self.readout(flat)
