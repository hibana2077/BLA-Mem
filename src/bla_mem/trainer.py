from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .tasks import Task
from .utils import EvalResult, format_float


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64

    train_len: int = 128
    test_lens: tuple[int, ...] = (128, 256, 512)

    lr: float = 3e-4
    weight_decay: float = 0.0

    log_every: int = 50
    eval_every: int = 200
    eval_batches: int = 25

    grad_clip: float = 1.0

    # debug/sanity: if True, reuse the same (x,y) batch every step.
    fixed_batch: bool = False


class Trainer:
    def __init__(self, model: nn.Module, task: Task, cfg: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.task = task
        self.cfg = cfg
        self.device = device

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    @torch.no_grad()
    def evaluate(self, seq_len: int) -> EvalResult:
        self.model.eval()
        losses = []
        metrics = []
        for _ in range(self.cfg.eval_batches):
            x, y = self.task.generate_batch(self.cfg.batch_size, seq_len, self.device)
            out = self.model(x)
            loss = self.task.loss(out, y)
            metric = self.task.metric(out, y)
            losses.append(loss.detach())
            metrics.append(metric.detach())
        return EvalResult(loss=float(torch.stack(losses).mean().item()), metric=float(torch.stack(metrics).mean().item()))

    def train(self) -> None:
        self.model.train()

        fixed_x = fixed_y = None
        if self.cfg.fixed_batch:
            fixed_x, fixed_y = self.task.generate_batch(self.cfg.batch_size, self.cfg.train_len, self.device)

        for step in range(1, self.cfg.steps + 1):
            if self.cfg.fixed_batch:
                x, y = fixed_x, fixed_y
            else:
                x, y = self.task.generate_batch(self.cfg.batch_size, self.cfg.train_len, self.device)
            out = self.model(x)
            loss = self.task.loss(out, y)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

            if step % self.cfg.log_every == 0 or step == 1:
                with torch.no_grad():
                    metric = self.task.metric(out, y).detach().item()
                print(
                    f"step={step} loss={format_float(loss.detach().item())} metric={format_float(metric)} train_len={self.cfg.train_len}",
                    flush=True,
                )

            if step % self.cfg.eval_every == 0:
                results = []
                for L in self.cfg.test_lens:
                    r = self.evaluate(L)
                    results.append((L, r))
                msg = " | ".join(
                    [f"L={L} loss={format_float(r.loss)} metric={format_float(r.metric)}" for L, r in results]
                )
                print(f"EVAL step={step} {msg}", flush=True)
                self.model.train()
