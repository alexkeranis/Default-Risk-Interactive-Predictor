"""Global determinism. Called at every entry point (training, inference CLI, tests)."""

from __future__ import annotations

import os
import random

import numpy as np

from bnpl_risk.config import SEED


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by torch for deterministic cuBLAS if a GPU is ever used.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG seeded for third-party code
    try:
        import torch
    except ImportError:  # torch is only needed for the neural network family
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
