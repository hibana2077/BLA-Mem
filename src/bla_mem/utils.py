from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class EvalResult:
    loss: float
    metric: float


def format_float(x: float) -> str:
    if abs(x) >= 1e3 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.6f}"


def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}
