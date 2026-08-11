"""Learners: ``(x, soft) -> fitted student``, interchangeable in a sweep.  Three families, one per
module:

    embedding-based (:mod:`.embedding`)  -- explicit feature map, closed-form ridge
        ridge
    kernel-based (:mod:`.kernel`)        -- RBF computed directly on raw x, no materialised map
        svr
    nn-based (:mod:`.nn`)                -- learned representation, trained end-to-end
        mlp           -- torch MLP with early stopping on a TRAIN slice

:mod:`.auto` is a thin dispatcher, not a fourth family: ``run_config(cfg_path, observable,
learner_name, **learner_kwargs)`` loads a config's saved dataset, builds ``learner_name`` from the
registry below, fits it, and returns held-out stats -- convenience over ``config -> data -> fit``,
picking no learner on your behalf.  It also has ``sweep_degree_grid``, which adds cross terms via
``PolynomialFeatures`` on top of a Fourier base -- the one tool here that gets past ``ridge``'s
additive-only ceiling; see its docstring for why an adaptive-frequency variant (``topk_fourier``)
was tried and removed instead of kept alongside it.  And ``sweep_heatmap``/``plot_heatmap``: fit
every registered learner on every observable for one config and render the ``R^2`` grid as a PNG,
learners on the y-axis, observables on the x-axis -- optionally with a ``fourier_grid`` row
(``include_degree_grid=True``) reporting ``sweep_degree_grid``'s best cell per observable, so the
interaction-aware ridge sweep sits in the same picture as the plain learners.

Two things are deliberate rather than incidental:

* **Adjudication is on held-out log-likelihood, not ``R^2``** (:mod:`.base`).  ``R^2`` renormalises
  by the label's own variance, so it is not comparable across arms whose labels have different
  spreads -- and the observable scales here span four orders of magnitude.
* **The learner's feature maps are a separate implementation from the teacher's**
  (:mod:`.features` vs :mod:`model.features`), so a learner experiment cannot reach the labels'
  own featurisation.  Sharing it previously inflated the classical models' scores and depressed the
  photonic ones.

The paired ``perm``/``det`` protocol -- which is what makes any of these numbers interpretable --
lives in :mod:`.compare`.
"""

from __future__ import annotations

from .base import (LEARNERS, Learner, build_learner, evaluate, gaussian_log_likelihood, r2_score)
from .embedding import RidgeLearner
from .features import build_features, fourier_features
from .kernel import SvrLearner
from .nn import MlpLearner
from .auto import run_config, sweep_degree_grid, sweep_heatmap, plot_heatmap

__all__ = [
    "LEARNERS", "Learner", "build_learner", "evaluate", "gaussian_log_likelihood", "r2_score",
    "RidgeLearner", "SvrLearner", "MlpLearner",
    "run_config", "sweep_degree_grid", "sweep_heatmap", "plot_heatmap",
    "build_features", "fourier_features",
]
