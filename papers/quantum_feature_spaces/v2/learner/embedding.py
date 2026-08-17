"""Embedding-based learner: an explicit feature map, materialised as an ``(N, d)`` matrix, plus a
closed-form ridge readout.

    ridge  -- fixed/random features (fourier_poly/fourier/raw/rff, see :mod:`.features`) + ridge

Solves ``(Z^T Z + alpha I) w = Z^T y`` on an explicit ``Z`` -- the defining property of
"embedding-based" here, contrasted with :mod:`.kernel`'s ``SvrLearner``, which (at its default
``basis="raw"``) never materialises a feature map at all, computing the RBF kernel directly between
raw inputs instead.

**Default basis is ``fourier_poly`` (order=3, degree=2), not plain ``fourier``.**  A ridge readout
on plain ``fourier`` (or any other additive basis in :mod:`.features`) is a sum of per-coordinate
functions, so it cannot represent the *product* structure of a Fock-basis probability (a
permanent/determinant is multilinear in the input columns) -- see :func:`.features.
fourier_features`'s docstring for the measured failure on that basis alone.  ``fourier_poly``
(:func:`.features.fourier_poly_features`) closes most of this gap by expanding the order-3 Fourier
base with degree-2 ``PolynomialFeatures`` cross terms before the ridge solve, so pairwise products
between Fourier features are representable by construction rather than needing a separate ablation
sweep to discover the map is missing them.  ``sweep_degree_grid`` (:mod:`.auto`) remains the tool
for sweeping the full ``(order, degree)`` grid, including higher degrees than the ``2`` used here by
default -- e.g. ``parity`` moves from ``R^2~0.07`` at ``degree=1`` to ``~0.71`` at ``order=2,
degree=3``, competitive with ``svr``/``mlp``; that grid is reported as an ablation over the default,
not as a replacement for it.  A previous correlation-selected adaptive-basis variant
(``topk_fourier``) was tried and removed: on every observable tested it matched plain ``fourier`` at
``order=1`` and never approached the degree-grid's cross-term rows, so it bought nothing over the
fixed grid at the same feature budget.
"""

from __future__ import annotations

import torch

from .base import Learner
from .features import build_features


class RidgeLearner(Learner):
    """Closed-form ridge on a fixed, explicitly materialised feature map.

    Convex and solved exactly, so a failure of this arm is a statement about the *feature map's*
    expressivity, never about an optimiser.  That is what makes it the right control for the paired
    protocol in :mod:`.compare`: the "learner inadequate" row cannot be blamed on training dynamics
    here.  See :class:`~learner.kernel.SvrLearner` for the kernel-based (no materialised feature
    map) alternative, and :func:`~learner.auto.sweep_degree_grid` for the interaction-term sweep.
    """

    name = "ridge"

    def __init__(self, *, basis: str = "fourier_poly", order: int = 3, degree: int = 2,
                 alpha: float = 1e-3, n_components: int = 512, gamma: float = 0.5, seed: int = 0):
        super().__init__(basis=basis, order=order, degree=degree, alpha=alpha,
                         n_components=n_components, gamma=gamma, seed=seed)
        self.w = None
        self.mean = 0.0

    def _phi(self, X: torch.Tensor) -> torch.Tensor:
        h = self.hparams
        Z = build_features(X, h["basis"], order=h["order"], degree=h["degree"],
                           n_components=h["n_components"], gamma=h["gamma"], seed=h["seed"]).double()
        return torch.cat([Z, torch.ones(Z.shape[0], 1, dtype=Z.dtype)], dim=1)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "RidgeLearner":
        Z, yy = self._phi(X), y.double()
        self.mean = float(yy.mean())
        A = Z.T @ Z + float(self.hparams["alpha"]) * torch.eye(Z.shape[1], dtype=Z.dtype)
        self.w = torch.linalg.solve(A, Z.T @ (yy - self.mean))
        self._record_residual(X, y)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (self._phi(X) @ self.w + self.mean).float()
