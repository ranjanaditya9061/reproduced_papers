"""Shared input sampler.

One seeded sampler draws the inputs that *every* teacher labels, so datasets
from different teachers at the same ``(n_features, sample_seed)`` use identical X
points — making the cross-teacher comparison apples-to-apples.

The draw is a single contiguous ``torch.rand`` from a freshly seeded generator,
so it is prefix-stable: ``sample_X(2N, ...)`` reproduces the first ``N`` rows of
``sample_X(N, ...)`` exactly.
"""

from __future__ import annotations

import math

import torch


def sample_X(
    n_samples: int,
    n_features: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw ``(n_samples, n_features)`` inputs uniformly in ``[0, 2π]``."""
    gen = torch.Generator().manual_seed(seed)
    return 2.0 * math.pi * torch.rand(n_samples, n_features, generator=gen, dtype=dtype)