"""One cached-fit primitive, shared by every learner-fitting call site in this repo.

    from learner.cache import cached_fit
    result = cached_fit("configs/photonic.yaml", "parity", "ridge",
                        out_root="datasets", scores_root="scores",
                        n_train=None, graph_density=0.5, split_seed=0, order=3)
    result["r2"], result["train"]["r2"]

**Why this exists.** Before this module, every fit was paid fresh, every call, everywhere --
``learner.auto.run_config``, ``learner.compare.run_arm``, ``eval.r2_vs_gradient.best_r2``,
``eval.resolution_ceiling.fit_best_learner`` each independently called
``build_learner(...).fit(...)`` with no persistence, so the *same* (config, observable, learner,
hyperparameters, split) combination is refit from scratch every time it is asked for, even seconds
apart in the same sweep. ``eval.size_r2_line`` was the one exception, but it cached the *entire*
multi-seed, multi-family sweep result as a single JSON keyed only on ``(observable, learner)`` --
blind to ``n_train``/``graph_density``/``split_seed``/hyperparameter overrides, and useless to any
other caller wanting just one cell.

**What gets cached, per exact ``(artifact, observable, learner_name, learner_kwargs, split_seed,
n_train)`` combination:**

* Predictions on **both** splits, ``y_hat_train`` and ``y_hat_test`` (``(n_train,)``/``(n_test,)``
  float64 tensors) -- enough to recompute any downstream statistic (``r2``, ``rmse``, residual
  histograms, overfitting checks comparing train vs. test fit quality) without ever refitting.
* The readable stats :func:`~learner.base.evaluate` already computes, on both splits.

**Cache key.** Mirrors :mod:`pipeline.score`'s own convention (``scores/<circuit_hash>/<source>/
...``) rather than inventing a new one: the artifact identity (``circuit_hash``/``source``, from
:func:`~pipeline.score.load_dataset`'s returned ``artifact_name`` plus the shots-vs-exact branch)
fixes *which dataset*, and a hash of every argument that can change the fit itself --
``learner_name``, every ``**learner_kwargs`` (sorted, so key order never matters), ``split_seed``,
``n_train``, ``graph_density`` -- fixes *which fit*. This has to include ``**learner_kwargs`` in
full (not just ``learner_name``) because ``svr``'s ``C``/``gamma`` or ``mlp``'s architecture change
what gets fit without changing the observable or the config at all.

Shot count is covered implicitly, not as an explicit key field: a different ``cfg.generation.shots``
changes ``load_dataset``'s returned ``artifact_name`` (:func:`~pipeline.shots.shot_source_tag`
folds the shot count into the source tag), so a 100-shot and a 2000-shot run of the same config
already land under different ``source`` directories and never collide.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _learner_spec_hash(learner_name: str, learner_kwargs: dict, *, split_seed: int | None,
                       n_train: int | None, graph_density: float) -> str:
    """Stable hash over everything that determines the fit, independent of key order."""
    payload = {
        "learner": learner_name,
        "kwargs": {k: learner_kwargs[k] for k in sorted(learner_kwargs)},
        "split_seed": split_seed,
        "n_train": n_train,
        "graph_density": graph_density,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_paths(scores_root: str | Path, artifact_name: str, source: str, observable: str,
                 learner_hash: str) -> tuple[Path, Path]:
    base = Path(scores_root) / artifact_name / source / "learners" / observable
    return base / f"{learner_hash}.pt", base / f"{learner_hash}.json"


#: fp32 bytes/element -- same convention as `model/base.py`'s `FORWARD_ELEMENTS` (~128MB per
#: 33.5M fp32 elements), reused here rather than picked independently.
_BYTES_PER_ELEMENT = 4


def fit_task_bytes(cfg_path: str | Path, *, n_train: int | None = None) -> int:
    """Estimated peak memory of ONE learner fit at ``cfg_path``, for
    :func:`~circuit.spin.parallel_row_map`'s ``bytes_per_task`` -- computed from the requested
    ``n_train`` and the config's ``n_features``, rather than a flat constant.

    **Not** a function of ``(m, k)``/``n_out``: :func:`~pipeline.score.load_dataset` returns
    ``X`` shaped ``(N, n_features)`` and ``soft`` shaped ``(N,)`` -- the huge, ``m``/``k``-scaled
    ``n_out`` quantity belongs to dataset generation/scoring (already paid for and cached on disk
    by the time a learner fit runs), not to what a learner actually holds in memory.
    ``ridge``/``mlp`` scale with ``n_train x n_features`` (plus each learner's own feature
    expansion -- ``ridge``'s Fourier/polynomial basis is ``O(order x n_features)`` to
    ``O((order x n_features)^degree)``, small next to the term below at this repo's typical
    ``n_train``); ``svr``'s kernel Gram matrix is the actual dominant term, ``O(n_train^2)``,
    independent of ``n_features`` entirely. The quadratic term is kept even for a ``ridge``/``mlp``
    call (this function does not know which learner a given task will use), since overestimating
    a non-``svr`` fit's memory only costs a slower run, while undercounting an ``svr`` fit's is the
    actual OOM/thrash this whole gate exists to prevent.

    Only reads the config (a cheap YAML parse, :func:`~config.load_config`, no simulation) --
    cheap enough to call once per task when building a task list, unlike actually loading the
    dataset itself. A config this cannot resolve ``generation.size``/``problem.n_features`` from
    (should not happen for a well-formed config) falls back to a conservative flat estimate rather
    than raising, since a sizing helper failing should never be what aborts a sweep.
    """
    try:
        from config import load_config

        cfg = load_config(cfg_path)
        n_features = int(cfg.problem.n_features)
        n_rows = int(n_train) if n_train else int(cfg.generation.size)
        linear_term = n_rows * n_features
        quadratic_term = n_rows * n_rows              # svr's kernel Gram matrix, the dominant term
        return max(1, linear_term + quadratic_term) * _BYTES_PER_ELEMENT * 2
    except Exception:
        from circuit.spin import _DEFAULT_TASK_BYTES

        return _DEFAULT_TASK_BYTES


def preload_dataset(cfg_path: str | Path, observable: str, *, out_root: str | Path = "datasets",
                    scores_root: str | Path = "scores", graph_density: float = 0.5) -> tuple:
    """``(cfg, X, soft, artifact_name)`` for :func:`cached_fit`'s ``_preloaded`` argument -- call
    this once per (config, observable) and pass the result to every :func:`cached_fit` call for
    that pair, instead of letting each one reload the dataset from disk."""
    from config import load_config
    from pipeline.score import load_dataset

    cfg = load_config(cfg_path)
    X, soft, artifact_name = load_dataset(cfg, observable, out_root=out_root,
                                          scores_root=scores_root, graph_density=graph_density)
    return cfg, X, soft, artifact_name


def cached_fit(cfg_path: str | Path, observable: str, learner_name: str, *,
              out_root: str | Path = "datasets", scores_root: str | Path = "scores",
              n_train: int | None = None, graph_density: float = 0.5,
              split_seed: int | None = None, force: bool = False,
              _preloaded: tuple | None = None,
              **learner_kwargs) -> dict:
    """Fit ``learner_name`` on ``observable`` from ``cfg_path``'s dataset, or load a prior fit's
    saved predictions/stats if the exact same combination was already run.

    Returns the held-out (test-split) stats from :func:`~learner.base.evaluate`, plus ``n_train``/
    ``n_test``/``artifact`` (matching :func:`~learner.auto.run_config`'s existing return shape, so
    callers do not need to change how they read the result), and additionally a ``"train"`` key
    with the same stats computed on the training split.

    ``force=True`` ignores any cached entry and refits, overwriting the cache -- same convention as
    :func:`~pipeline.score.load_soft`'s own ``force`` flag.

    ``_preloaded``, if given, is ``(cfg, X, soft, artifact_name)`` already fetched by the caller --
    lets a caller that is about to fit many (learner, seed) combinations on the same
    (config, observable) pay :func:`~pipeline.score.load_dataset`'s cost once and reuse it across
    every cache lookup/fit, the way :func:`eval.r2_vs_gradient.best_r2` and
    :func:`eval.resolution_ceiling.fit_best_learner` need to (measured: ~12-20s per
    ``load_dataset`` call at ``m=12``, almost entirely I/O, which would otherwise dominate wall
    time by roughly two orders of magnitude across a multi-seed multi-learner sweep). Internal
    parameter, not part of the public per-cell API other callers use.
    """
    from config import load_config
    from pipeline.score import load_dataset
    from pipeline.split import split_indices

    from .base import build_learner, evaluate

    if _preloaded is not None:
        cfg, X, soft, artifact_name = _preloaded
    else:
        cfg = load_config(cfg_path)
        X, soft, artifact_name = load_dataset(cfg, observable, out_root=out_root,
                                              scores_root=scores_root, graph_density=graph_density)
    source = "shots" if cfg.generation.shots else "exact"
    resolved_split_seed = cfg.split.split_seed if split_seed is None else split_seed

    learner_hash = _learner_spec_hash(learner_name, learner_kwargs, split_seed=resolved_split_seed,
                                      n_train=n_train, graph_density=graph_density)
    pt_path, json_path = _cache_paths(scores_root, artifact_name, source, observable, learner_hash)

    if pt_path.exists() and json_path.exists() and not force:
        stats = json.loads(json_path.read_text())
        return stats

    tr, te = split_indices(len(X), test_fraction=cfg.split.test_fraction,
                           split_seed=resolved_split_seed)
    if n_train:
        tr = tr[:int(n_train)]

    model = build_learner(learner_name, **learner_kwargs).fit(X[tr], soft[tr])
    res_test = evaluate(model, X[te], soft[te])
    res_train = evaluate(model, X[tr], soft[tr])

    y_hat_train = model.predict(X[tr]).double()
    y_hat_test = model.predict(X[te]).double()

    result = dict(res_test)
    result.update(n_train=len(tr), n_test=len(te), artifact=artifact_name, train=res_train,
                 label_var=float(soft[te].double().var()))

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"y_hat_train": y_hat_train, "y_hat_test": y_hat_test,
               "train_idx": tr, "test_idx": te}, pt_path)
    json_path.write_text(json.dumps(result, indent=2))

    return result
