"""Kernel-based learner: RBF-SVR computed directly on raw inputs, no materialised feature map.

Contrasted with :mod:`.embedding`'s ``RidgeLearner``, which solves on an explicit ``(N, d)``
feature matrix: :class:`SvrLearner` at its default ``basis="raw"`` passes ``x``
straight to ``sklearn.svm.SVR(kernel="rbf")``, which computes ``k(x_i, x_j) = exp(-gamma ||x_i -
x_j||^2)`` pairwise between raw inputs -- an implicit, here infinite-dimensional, representation
that is never written down as a matrix. Passing a non-``raw`` basis still works (RBF on top of an
explicit embedding instead of on ``x`` directly) but is then a hybrid, not the clean kernel-based
control.
"""

from __future__ import annotations

import torch

from .base import Learner
from .features import build_features


class SvrLearner(Learner):
    """RBF-SVR -- the kernel-based arm.  See module docstring for the embedding-vs-kernel split."""

    name = "svr"

    def __init__(self, *, basis: str = "raw", order: int = 3, C: float = 10.0,
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
