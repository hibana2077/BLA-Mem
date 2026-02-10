from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from tqdm import trange

from bla_mem import BLAMem
from bla_mem.model import BLAMemConfig


def make_adding_batch(batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classic adding problem.

    Input: (B, T, 2) = [value, marker]
    Target: sum of the two marked values.
    """

    x = torch.zeros(batch_size, seq_len, 2, device=device)
    values = torch.rand(batch_size, seq_len, device=device)

    idx1 = torch.randint(0, seq_len, (batch_size,), device=device)
    idx2 = torch.randint(0, seq_len, (batch_size,), device=device)

    x[:, :, 0] = values
    x[torch.arange(batch_size, device=device), idx1, 1] = 1.0
    x[torch.arange(batch_size, device=device), idx2, 1] = 1.0

    y = values[torch.arange(batch_size, device=device), idx1] + values[torch.arange(batch_size, device=device), idx2]
    return x, y.unsqueeze(-1)


def make_parity_batch(batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parity of two marked bits (binary classification).

    Input: (B, T, 2) = [bit, marker]
    Target: (bit[idx1] XOR bit[idx2]) in {0,1}.
    """

    x = torch.zeros(batch_size, seq_len, 2, device=device)
    bits = torch.randint(0, 2, (batch_size, seq_len), device=device).float()

    idx1 = torch.randint(0, seq_len, (batch_size,), device=device)
    idx2 = torch.randint(0, seq_len, (batch_size,), device=device)

    x[:, :, 0] = bits
    x[torch.arange(batch_size, device=device), idx1, 1] = 1.0
    x[torch.arange(batch_size, device=device), idx2, 1] = 1.0

    y = (bits[torch.arange(batch_size, device=device), idx1].long() ^ bits[torch.arange(batch_size, device=device), idx2].long())
    return x, y


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        return self.head(h[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["adding", "parity"], required=True)
    p.add_argument("--model", choices=["bla", "gru"], default="bla")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--time-aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("--eval-every", type=int, default=200, help="Run evaluation every N steps (0 to disable)")
    p.add_argument("--eval-batches", type=int, default=20, help="Number of random batches to average over during eval")
    p.add_argument("--patience", type=int, default=0, help="Early stop if no eval improvement for this many evals (0 to disable)")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement in eval metric to reset patience")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--readout-hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seq_len % args.chunk != 0:
        raise ValueError("For this MVP, require --seq-len divisible by --chunk.")

    if args.task == "adding":
        batch_fn = make_adding_batch
        out_dim = 1
        loss_fn = nn.MSELoss()
    else:
        batch_fn = make_parity_batch
        out_dim = 2
        loss_fn = nn.CrossEntropyLoss()

    if args.model == "bla":
        cfg = BLAMemConfig(
            input_dim=2,
            depth=args.depth,
            chunk_size=args.chunk,
            time_aug=args.time_aug,
            readout_hidden=args.readout_hidden,
            out_dim=out_dim,
        )
        model: nn.Module = BLAMem(cfg).to(device)
    else:
        model = GRUBaseline(input_dim=2, hidden=args.readout_hidden, out_dim=out_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    greater_is_better = args.task == "parity"
    best_eval_metric = float("-inf") if greater_is_better else float("inf")
    bad_evals = 0

    def run_eval() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_metric = 0.0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = batch_fn(args.batch, args.seq_len, device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += float(loss.detach())

                if args.task == "adding":
                    total_metric += float(loss.detach())
                else:
                    acc = (pred.argmax(dim=-1) == y).float().mean()
                    total_metric += float(acc.detach())
        model.train()
        return total_loss / args.eval_batches, total_metric / args.eval_batches

    model.train()
    step_iter = range(1, args.steps + 1) if args.no_tqdm else trange(1, args.steps + 1)
    for step in step_iter:
        x, y = batch_fn(args.batch, args.seq_len, device)
        pred = model(x)

        if args.task == "adding":
            loss = loss_fn(pred, y)
            metric = float(loss.detach())
            metric_name = "mse"
        else:
            loss = loss_fn(pred, y)
            acc = (pred.argmax(dim=-1) == y).float().mean()
            metric = float(acc.detach())
            metric_name = "acc"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            print(f"step={step} loss={float(loss.detach()):.4f} {metric_name}={metric:.4f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_loss, eval_metric = run_eval()
            print(f"eval step={step} loss={eval_loss:.4f} {metric_name}={eval_metric:.4f}")

            improved = (
                (eval_metric > best_eval_metric + args.min_delta)
                if greater_is_better
                else (eval_metric < best_eval_metric - args.min_delta)
            )
            if improved:
                best_eval_metric = eval_metric
                bad_evals = 0
            else:
                bad_evals += 1
                if args.patience > 0 and bad_evals >= args.patience:
                    print(
                        f"early_stop step={step} best_{metric_name}={best_eval_metric:.4f} "
                        f"bad_evals={bad_evals}/{args.patience}"
                    )
                    break


if __name__ == "__main__":
    main()
