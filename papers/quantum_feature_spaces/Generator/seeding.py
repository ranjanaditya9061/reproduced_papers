"""Centralised, reproducible seeding for the generation pipeline.

A single :func:`seed_everything` is called at entry so a run is fully determined
by the seeds in the experiment config.  (Each component also seeds itself
explicitly; this is the defensive global net.)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs from a single integer."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)