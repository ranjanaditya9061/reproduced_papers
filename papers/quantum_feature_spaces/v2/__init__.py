"""v2: one pipeline for distribution models, observables, learners and metrics.

Supersedes the two-and-a-half pipelines in the paper root (``model/`` + ``Generator/`` +
``embedding/`` + ``learn/``, and the legacy ``data/`` + ``learner/`` + ``train.py``), which are
left untouched and keep working.  Three structural changes drive the rewrite:

1. **The distribution artifact depends on the input and the circuit ONLY.**  The observable is a
   readout applied afterwards, cached separately, so one boson-sampling run serves every
   observable.  In the old pipeline the observable rode in the dataset hash
   (``Generator/artifact.py`` folded ``hash_spec`` in, and every Fock teacher returned
   ``{"observable": ...}``), so ``parity`` and ``osc`` on an identical circuit and identical
   inputs cost two independent simulations.
2. **Every model emits a distribution over a labelled outcome basis.**  ``X -> probs``, and
   nothing else; the four Fock teachers' duplicated batching / shot-sampling / capture code
   lives once on :class:`v2.model.base.DistributionModel`, and the seven ``spoqc*`` modules
   collapse into one state-prep registry.
3. **The input size is a study invariant** (:data:`v2.config.N_FEATURES`).  Every complexity
   measure in :mod:`v2.metrics` is denominated in it -- the input Fisher matrix is
   ``n_features x n_features`` -- so varying it would make nothing comparable.  Fixing it is
   also what makes the ``(m, k)`` sweep well-posed: the circuit grows while the FIM stays the
   same size, which the old ``n_features = m - 1`` coupling made impossible.

Layout::

    config.py       the ExperimentConfig and the N_FEATURES invariant
    circuit/        circuit + state-prep + encoding builders (shared)
    model/          FROZEN distribution models:  X -> (N, n_outcomes) probs
    observable/     distribution -> score, in three functional shapes
    pipeline/       artifacts and stores (dist artifact + per-observable score cache)
    parametric/     differentiable-in-weights models -- nn learners ONLY, not metrics
    learner/        embedding-based and nn-based learners
    metrics/        TWO SEPARATE analyses: distribution (A) and observable (B)
"""

from __future__ import annotations

from .config import N_FEATURES, ExperimentConfig, load_config

__all__ = ["N_FEATURES", "ExperimentConfig", "load_config"]
