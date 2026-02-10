from __future__ import annotations

from dataclasses import dataclass

from .base import BaseSequenceModel
from .transformer import TransformerBaseline
from .sheaf import SheafGluingModel
from .ultrametric import UltrametricHeatFlowModel
from .magnus import MagnusCommutatorModel
from .cumulant import CumulantScanModel
from .jump import JumpGeneratorModel


@dataclass(frozen=True)
class ModelConfig:
    model: str
    input_dim: int
    output_dim: int

    d_model: int = 128

    # transformer
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 8192
    pooling: str = "mean"

    # generic
    depth: int = 1

    # sheaf
    sheaf_lambda: float = 1.0
    cg_iters: int = 20

    # ultrametric
    levels: int = 8

    # magnus
    block_size: int = 8

    # cumulant
    k_order: int = 3

    # jump
    jump_max_dist: int = 64
    jump_steps: int = 8
    jump_tau: float = 0.5


def create_model(cfg: ModelConfig) -> BaseSequenceModel:
    name = cfg.model.strip().lower()

    if name == "transformer":
        return TransformerBaseline(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pooling=cfg.pooling,
        )

    if name == "sheaf":
        return SheafGluingModel(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            lam=cfg.sheaf_lambda,
            cg_iters=cfg.cg_iters,
        )

    if name == "ultrametric":
        return UltrametricHeatFlowModel(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            levels=cfg.levels,
        )

    if name == "magnus":
        return MagnusCommutatorModel(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            block_size=cfg.block_size,
        )

    if name in {"cumulant", "prelie"}:
        return CumulantScanModel(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            k_order=cfg.k_order,
        )

    if name in {"jump", "dirichlet"}:
        return JumpGeneratorModel(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            d_model=cfg.d_model,
            max_dist=cfg.jump_max_dist,
            n_steps=cfg.jump_steps,
            tau=cfg.jump_tau,
        )

    raise ValueError(f"Unknown model: {cfg.model}")
