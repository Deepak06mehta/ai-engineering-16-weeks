"""Reproducibility. Call seed_all() at the top of every experiment and record the seed."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = int(os.environ.get("SEED", 1337))


def seed_all(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy and torch. Returns the seed so you can log it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: MPS does not guarantee bit-identical results across runs.
    # Record that fact in the experiment log rather than fighting it.
    return seed
