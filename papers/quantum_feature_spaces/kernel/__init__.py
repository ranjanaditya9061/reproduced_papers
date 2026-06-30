"""Kernel stage: turn a (loaded) feature embedding into a Gram.

    from kernel import RBFGram, gram_from_features, gram_from_cache

    K = RBFGram().gram_from_features(F_train)          # fits gamma on train
    K_te = RBFGram_or_same.gram_from_features(F_train, F_test)   # cross-Gram

The kernel here is a single Gaussian RBF over real features -- it is agnostic to
where the features come from.  Feature generation + on-disk storage live in the
:mod:`embedding` stage; :func:`gram_from_cache` turns a stored embedding blob into
a Gram.  Comparing embeddings on the same ``(X, y)`` -- via accuracy, kernel-target
alignment, or geometric difference -- is the Power-of-Data experiment (Huang et
al. 2021), and lives in :mod:`analyzer`.
"""

from __future__ import annotations

from .cache import gram_from_cache
from .rbf import RBFGram, gram_from_features

__all__ = [
    "RBFGram",
    "gram_from_features",
    "gram_from_cache",
]