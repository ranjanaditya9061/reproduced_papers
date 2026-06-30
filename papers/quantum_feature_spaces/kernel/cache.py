"""Consume a cached embedding -> a Gram (no feature/quantum recompute).

The *generation + storage* of embeddings lives in the :mod:`embedding` stage (its
own config + ``embeddings/`` directory).  This module only knows how to turn a
stored feature matrix back into a Gram: RBF over the stored feature rows.
"""

from __future__ import annotations

import torch

from .rbf import RBFGram


def gram_from_cache(blob: dict, idx_a=None, idx_b=None, gamma="median") -> torch.Tensor:
    """RBF Gram from a cached embedding, with no feature recompute.

    ``blob["data"]`` is the stored feature matrix.  ``idx_a``/``idx_b`` select
    train/test row subsets (``None`` means "all"); ``gamma`` is fitted on the
    ``idx_a`` rows and reused for the ``idx_b`` cross-block.
    """
    data = blob["data"]
    Fa = data if idx_a is None else data[idx_a]
    Fb = None if idx_b is None else data[idx_b]
    return RBFGram(gamma).gram_from_features(Fa, Fb)