"""The one shared loader — a deterministic train/test split over the FULL pool.

Every learner and the analyzer obtains data through :func:`load_split`.  It is a
pure deterministic function of ``(artifact, test_fraction, split_seed)`` and
**discards nothing**: margin filtering / balancing are a separate diagnostic
(:func:`Generator.prepare.prepare`), never applied here.  Because the split is
just a permutation of all ``N`` rows, ``train_idx`` / ``test_idx`` index directly
into the raw pool order — so a stored embedding matrix (computed on the full
``X``) can be sliced by the same indices to train/test a downstream learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .artifact import load_raw
from .prepare import derive_labels


@dataclass
class Split:
    X_train: torch.Tensor
    y_train: torch.Tensor
    soft_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    soft_test: torch.Tensor
    train_idx: torch.Tensor   # indices into the full pool (raw load_raw order)
    test_idx: torch.Tensor
    meta: dict


def load_split(
    artifact_path: str | Path,
    *,
    test_fraction: float = 0.20,
    split_seed: int = 0,
) -> Split:
    """Load an artifact and return a deterministic, seeded train/test split.

    Operates on the full pool (no margin filter, no balancing).
    """
    X, soft, meta = load_raw(artifact_path)
    y = derive_labels(soft)

    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
    n_test = int(n * test_fraction)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    return Split(
        X_train=X[train_idx], y_train=y[train_idx], soft_train=soft[train_idx],
        X_test=X[test_idx], y_test=y[test_idx], soft_test=soft[test_idx],
        train_idx=train_idx, test_idx=test_idx,
        meta=meta,
    )
