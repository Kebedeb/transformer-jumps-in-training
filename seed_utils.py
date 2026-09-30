"""Utility functions for setting and managing random seeds."""

from __future__ import annotations
import os
import random
from typing import Optional
import numpy as np
import torch

DEFAULT_SEED = 1234
SEED_ENV_VAR = "PROJECT_SEED"


def get_seed(default: int = DEFAULT_SEED) -> int:
    """Get the random seed from the environment or use a default."""
    raw_value = os.environ.get(SEED_ENV_VAR)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed python, numpy, and torch (CPU and all CUDA devices)."""
    resolved_seed = get_seed() if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    return resolved_seed


def make_rng(seed: Optional[int] = None) -> random.Random:
    """Create a new python RNG with a specific seed."""
    resolved_seed = get_seed() if seed is None else seed
    return random.Random(resolved_seed)


def make_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """CPU torch.Generator for reproducible batch sampling."""
    resolved_seed = get_seed() if seed is None else seed
    return torch.Generator().manual_seed(resolved_seed)