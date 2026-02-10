from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


class Task(Protocol):
    name: str
    input_dim: int
    output_dim: int

    def generate_batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def metric(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class ParityTask:
    name: str = "parity"
    input_dim: int = 1
    output_dim: int = 2

    def generate_batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, 1) with bits in {0,1}
        x = torch.randint(0, 2, (batch_size, seq_len, 1), device=device, dtype=torch.long)
        # parity target: sum mod 2
        y = (x.sum(dim=1).squeeze(-1) % 2).to(torch.long)  # (B,)
        return x.float(), y

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets)

    def metric(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        return (pred == targets).float().mean()


@dataclass(frozen=True)
class AddingTask:
    name: str = "adding"
    input_dim: int = 2
    output_dim: int = 1

    def generate_batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard adding problem:
        # feature[0]=value in [0,1], feature[1]=marker in {0,1} with exactly two ones.
        values = torch.rand((batch_size, seq_len, 1), device=device)
        markers = torch.zeros((batch_size, seq_len, 1), device=device)

        # pick two distinct indices per sample
        idx1 = torch.randint(0, seq_len, (batch_size,), device=device)
        idx2 = torch.randint(0, seq_len - 1, (batch_size,), device=device)
        idx2 = idx2 + (idx2 >= idx1).long()

        markers[torch.arange(batch_size, device=device), idx1, 0] = 1.0
        markers[torch.arange(batch_size, device=device), idx2, 0] = 1.0

        x = torch.cat([values, markers], dim=-1)
        y = (values[torch.arange(batch_size, device=device), idx1, 0] + values[torch.arange(batch_size, device=device), idx2, 0]).unsqueeze(-1)
        return x, y

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(preds, targets)

    def metric(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # negative MSE as a "higher is better" metric is awkward; use RMSE with sign.
        rmse = torch.sqrt(nn.functional.mse_loss(preds, targets))
        return -rmse


def get_task(name: str) -> Task:
    name = name.strip().lower()
    if name == "parity":
        return ParityTask()
    if name == "adding":
        return AddingTask()
    raise ValueError(f"Unknown task: {name}")
