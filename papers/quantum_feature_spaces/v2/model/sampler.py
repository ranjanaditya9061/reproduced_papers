"""The shared input sampler.

One seeded sampler draws the inputs that *every* model sees, so datasets from different models at
the same ``(n_features, sample_seed)`` use identical ``X`` points -- which is what makes the
cross-model comparison apples-to-apples, and what fixes the gauge for the input-Fisher spectra in
:mod:`v2.metrics.distribution` (the same physical coordinates, in the same radian units, for every
model).

The draw is a single contiguous ``torch.rand`` from a freshly seeded generator, so it is
**prefix-stable**: ``sample_X(2N, ...)`` reproduces the first ``N`` rows of ``sample_X(N, ...)``
exactly.  That is what lets :mod:`v2.pipeline.generate` extend a pool incrementally and reuse the
cached prefix byte-for-byte.

``U[0, 2pi]^{n_features}`` is also the distribution the global variance ratio ``G_O`` is taken
over, which is why that statistic reads as the dataset's intrinsic SNR rather than as an arbitrary
sampling choice.

Carried from ``model/sampler.py``.
"""

from __future__ import annotations

import math

import torch


def sample_X(n_samples: int, n_features: int, seed: int,
             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Draw ``(n_samples, n_features)`` inputs uniformly in ``[0, 2pi]``."""
    gen = torch.Generator().manual_seed(int(seed))
    return 2.0 * math.pi * torch.rand(n_samples, int(n_features), generator=gen, dtype=dtype)
