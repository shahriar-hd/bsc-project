"""
Reproducibility helpers shared by training and preprocessing.

Both pipelines must be deterministic for the thesis results to be
reproducible: preprocessing fixes the subject-aware split, training fixes
weight init and augmentation sampling.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: Seed value applied to every RNG.
        deterministic: If True, force deterministic cuDNN kernels. This costs
            throughput but guarantees identical runs — required when reporting
            ablation results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return  # preprocessing does not need torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker distinctly but reproducibly.

    Without this, every worker inherits the parent's NumPy seed and produces
    identical augmentation streams.
    """
    seed = (int(np.random.get_state()[1][0]) + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
