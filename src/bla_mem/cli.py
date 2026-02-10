from __future__ import annotations

import argparse

import torch

from .models import create_model
from .models.factory import ModelConfig
from .tasks import get_task
from .trainer import TrainConfig, Trainer
from .utils import get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bla-mem", description="Toy experiments: parity/adding with multiple sequence models")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a model on a toy task")
    t.add_argument("--task", type=str, required=True, choices=["parity", "adding"])
    t.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["transformer", "sheaf", "ultrametric", "magnus", "cumulant", "jump"],
    )

    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")

    # training
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--train-len", type=int, default=128)
    t.add_argument("--test-lens", type=int, nargs="+", default=[128, 256, 512])
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--eval-batches", type=int, default=25)
    t.add_argument("--grad-clip", type=float, default=1.0)

    # model shared
    t.add_argument("--d-model", type=int, default=128)

    # transformer
    t.add_argument("--n-heads", type=int, default=4)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--max-len", type=int, default=8192)
    t.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])

    # sheaf
    t.add_argument("--sheaf-lambda", type=float, default=1.0)
    t.add_argument("--cg-iters", type=int, default=20)

    # ultrametric
    t.add_argument("--levels", type=int, default=8)

    # magnus
    t.add_argument("--block-size", type=int, default=8)

    # cumulant
    t.add_argument("--k-order", type=int, default=3)

    # jump
    t.add_argument("--jump-max-dist", type=int, default=64)
    t.add_argument("--jump-steps", type=int, default=8)
    t.add_argument("--jump-tau", type=float, default=0.5)

    return p


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)

    task = get_task(args.task)

    model_cfg = ModelConfig(
        model=args.model,
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        pooling=args.pooling,
        sheaf_lambda=args.sheaf_lambda,
        cg_iters=args.cg_iters,
        levels=args.levels,
        block_size=args.block_size,
        k_order=args.k_order,
        jump_max_dist=args.jump_max_dist,
        jump_steps=args.jump_steps,
        jump_tau=args.jump_tau,
    )
    model = create_model(model_cfg)

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        train_len=args.train_len,
        test_lens=tuple(int(x) for x in args.test_lens),
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        grad_clip=args.grad_clip,
    )

    print(
        f"task={task.name} model={args.model} device={device} steps={train_cfg.steps} batch={train_cfg.batch_size} train_len={train_cfg.train_len} test_lens={train_cfg.test_lens}",
        flush=True,
    )

    trainer = Trainer(model=model, task=task, cfg=train_cfg, device=device)
    trainer.train()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
        return

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
