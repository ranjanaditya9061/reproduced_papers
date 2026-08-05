"""Embedding-based learners: a fixed feature map plus a convex readout.

    ridge  -- Fourier / raw / RFF features + closed-form ridge regression
    svr    -- RBF-SVR on the same features

**The Fourier map here is deliberately a separate implementation from**
:mod:`v2.model.features`, which is *teacher-side* and helps define the labels.  Sharing one
implementation let the learner see the classical models' own encoding, which inflated their scores
and depressed the photonic ones -- measured on ``parity``, ``0.689 -> 0.957`` for the bilinear model
against ``0.785 -> 0.606`` for photonic.  So the duplication is the point: a learner experiment must
not be able to reach the labels' own featurisation.  Keep the two in sync only by intent, never by
import.
"""

from __future__ import annotations

import math

import torch

from .base import Learner


def raw_features(X: torch.Tensor) -> torch.Tensor:
    return X


def fourier_features(X: torch.Tensor, order: int) -> torch.Tensor:
    """``[cos(jx), sin(jx)]_{j=1..order}`` -> ``(N, 2*order*d)``.  Learner-side; see the docstring.

    **Additive across coordinates -- no interaction terms.**  A ridge readout on this map is a sum
    of univariate functions, so it cannot represent the *product* structure of a Fock-basis
    probability (a permanent is multilinear in the columns, so ``p(n)`` carries cross terms in
    ``e^{i x_i}``).  Measured on ``fermion``/``parity``: ``R^2 = -0.03`` at order 3 and ``-0.06`` at
    order 8, against ``+0.79`` for the MLP.  A tensor-product basis would be ``order^{n_f}`` wide,
    so use ``rff`` (interactions implicit, needs ``gamma`` tuning) or the ``mlp`` learner instead.
    This map is a genuine baseline, not a broken one -- but do not read its failure as evidence
    about the labels.
    """
    parts = []
    for j in range(1, int(order) + 1):
        parts.append(torch.cos(j * X))
        parts.append(torch.sin(j * X))
    return torch.cat(parts, dim=1)


def rff_features(X: torch.Tensor, n_components: int, gamma: float, seed: int) -> torch.Tensor:
    """Random Fourier features approximating an RBF kernel: ``sqrt(2/D) cos(Wx + b)``."""
    g = torch.Generator().manual_seed(int(seed))
    d = X.shape[1]
    W = torch.randn(d, int(n_components), generator=g) * math.sqrt(2.0 * gamma)
    b = 2.0 * math.pi * torch.rand(int(n_components), generator=g)
    return math.sqrt(2.0 / int(n_components)) * torch.cos(X @ W + b)


def build_features(X: torch.Tensor, basis: str, *, order: int = 3, n_components: int = 512,
                   gamma: float = 0.5, seed: int = 0) -> torch.Tensor:
    if basis == "raw":
        return raw_features(X)
    if basis == "fourier":
        return fourier_features(X, order)
    if basis == "rff":
        return rff_features(X, n_components, gamma, seed)
    if basis == "combo":
        return torch.cat([X, fourier_features(X, order)], dim=1)
    raise ValueError(f"unknown basis {basis!r}; choose raw | fourier | rff | combo")


class RidgeLearner(Learner):
    """Closed-form ridge on a fixed feature map -- the cheap, deterministic, seed-free baseline.

    Convex and solved exactly, so a failure of this arm is a statement about the *feature map's*
    expressivity, never about an optimiser.  That is what makes it the right control for the paired
    protocol in :mod:`v2.learner.compare`: the "learner inadequate" row cannot be blamed on
    training dynamics here.
    """

    name = "ridge"

    def __init__(self, *, basis: str = "fourier", order: int = 3, alpha: float = 1e-3,
                 n_components: int = 512, gamma: float = 0.5, seed: int = 0):
        super().__init__(basis=basis, order=order, alpha=alpha, n_components=n_components,
                         gamma=gamma, seed=seed)
        self.w = None
        self.mean = 0.0

    def _phi(self, X: torch.Tensor) -> torch.Tensor:
        h = self.hparams
        Z = build_features(X, h["basis"], order=h["order"], n_components=h["n_components"],
                           gamma=h["gamma"], seed=h["seed"]).double()
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


class SvrLearner(Learner):
    """RBF-SVR on the same feature map -- a non-linear readout, for contrast with ridge."""

    name = "svr"

    def __init__(self, *, basis: str = "fourier", order: int = 3, C: float = 10.0,
                 epsilon: float = 0.01, gamma: str | float = "scale", n_components: int = 512,
                 seed: int = 0):
        super().__init__(basis=basis, order=order, C=C, epsilon=epsilon, gamma=gamma,
                         n_components=n_components, seed=seed)
        self.model = None

    def _phi(self, X: torch.Tensor):
        h = self.hparams
        return build_features(X, h["basis"], order=h["order"], n_components=h["n_components"],
                             seed=h["seed"]).numpy()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "SvrLearner":
        from sklearn.svm import SVR
        h = self.hparams
        self.model = SVR(C=h["C"], epsilon=h["epsilon"], gamma=h["gamma"])
        self.model.fit(self._phi(X), y.double().numpy())
        self._record_residual(X, y)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.model.predict(self._phi(X))).float()
