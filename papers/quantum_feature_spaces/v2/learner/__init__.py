"""Learners: ``(x, soft) -> fitted student``, interchangeable in a sweep.

    ridge  -- Fourier / raw / RFF features + closed-form ridge   (convex, seed-free)
    svr    -- RBF-SVR on the same features
    mlp    -- torch MLP with early stopping on a TRAIN slice

Two things are deliberate rather than incidental:

* **Adjudication is on held-out log-likelihood, not ``R^2``** (:mod:`.base`).  ``R^2`` renormalises
  by the label's own variance, so it is not comparable across arms whose labels have different
  spreads -- and the observable scales here span four orders of magnitude.
* **The learner's Fourier map is a separate implementation from the teacher's**
  (:mod:`.embedding` vs :mod:`v2.model.features`), so a learner experiment cannot reach the labels'
  own featurisation.  Sharing it previously inflated the classical models' scores and depressed the
  photonic ones.

The paired ``perm``/``det`` protocol -- which is what makes any of these numbers interpretable --
lives in :mod:`.compare`.
"""

from __future__ import annotations

from .base import (LEARNERS, Learner, build_learner, evaluate, gaussian_log_likelihood, r2_score)
from .embedding import RidgeLearner, SvrLearner, build_features, fourier_features
from .nn import MlpLearner

__all__ = [
    "LEARNERS", "Learner", "build_learner", "evaluate", "gaussian_log_likelihood", "r2_score",
    "RidgeLearner", "SvrLearner", "MlpLearner", "build_features", "fourier_features",
]
