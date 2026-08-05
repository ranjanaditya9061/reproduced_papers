"""The one shared train/test partition.

A pure deterministic function of ``(n, test_fraction, split_seed)`` that **discards nothing**:
margin filtering and class balancing are a downstream diagnostic, never a load-time transform.
Because it is a permutation of all ``n`` rows, the indices address the raw pool order directly, so
a stored score vector or embedding matrix computed on the full pool can be sliced by the same
indices -- which is what keeps the paired ``perm``/``det`` learner arms on identical rows.
"""

from __future__ import annotations

import torch


def split_indices(n: int, *, test_fraction: float = 0.20, split_seed: int = 0):
    """``(train_idx, test_idx)`` over ``range(n)``."""
    perm = torch.randperm(int(n), generator=torch.Generator().manual_seed(int(split_seed)))
    n_test = int(n * float(test_fraction))
    return perm[n_test:], perm[:n_test]
