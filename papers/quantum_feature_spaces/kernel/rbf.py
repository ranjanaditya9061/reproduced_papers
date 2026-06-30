"""Gaussian RBF Gram over a (loaded) real feature matrix.

The kernel stage is feature-agnostic: it does not know *where* the features came
from (raw angles, Fourier, qubit/photonic projected) -- it only turns a feature
matrix into a Gram.  Feature generation + on-disk storage live in the
:mod:`embedding` stage.

``gamma="median"`` uses the median-pairwise-distance heuristic, fitted on the
first (training) ``gram_from_features`` call and reused for later cross-grams so
train and test kernels are consistent.
"""

from __future__ import annotations

import torch


class RBFGram:
    """``K(x, x') = exp(-gamma ||f(x) - f(x')||^2)`` over precomputed features."""

    def __init__(self, gamma="median"):
        self.gamma = gamma
        self._fitted_gamma: float | None = None

    def _resolve_gamma(self, sq: torch.Tensor) -> float:
        if self.gamma == "median":
            pos = sq[sq > 0]
            if pos.numel() == 0:
                return 1.0
            med = float(pos.median())
            return 1.0 / med if med > 0 else 1.0
        return float(self.gamma)

    def gram_from_features(self, F: torch.Tensor, F2: torch.Tensor | None = None) -> torch.Tensor:
        """RBF Gram from precomputed feature matrices.

        With ``F2=None`` this is the training Gram and fits ``gamma`` (median
        heuristic); a later cross-Gram reuses the fitted ``gamma``.
        """
        if F2 is None:
            sq = torch.cdist(F, F) ** 2
            self._fitted_gamma = self._resolve_gamma(sq)
            return torch.exp(-self._fitted_gamma * sq)
        sq = torch.cdist(F, F2) ** 2
        g = self._fitted_gamma if self._fitted_gamma is not None else self._resolve_gamma(sq)
        return torch.exp(-g * sq)


def gram_from_features(F: torch.Tensor, F2: torch.Tensor | None = None, gamma="median") -> torch.Tensor:
    """One-shot convenience: ``RBFGram(gamma).gram_from_features(F, F2)``."""
    return RBFGram(gamma).gram_from_features(F, F2)