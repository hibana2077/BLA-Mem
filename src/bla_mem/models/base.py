from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseSequenceModel(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, input_dim) -> logits/preds: (B, output_dim) or (B,1)."""
        raise NotImplementedError
