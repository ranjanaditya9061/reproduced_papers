"""Cached shot-noisy labels for models with **no native shot sampler**, via multinomial sampling
on the already-generated exact distribution.

    from metrics.multinomial_shots import cached_multinomial_shots
    X, soft, artifact_name = cached_multinomial_shots(cfg, "parity", out_root="datasets",
                                                       scores_root="scores")

**Why this exists.** ``model.supports_shots`` is ``False`` for every model outside ``photonic``/
``fermion`` (``quadratic_fock``, ``mlp_fock``, ``mlp``, ``analytical``) -- :mod:`pipeline.shots`
deliberately has no counts-store path for them (its own module docstring: a ``"multinomial"``
method was left out on purpose, since sampling from a stored ``p`` "requires the full distribution,"
reinstating the dependency the Clifford/MH methods exist to avoid at circuit sizes where the basis
cannot be enumerated at all).  That constraint does not apply to these small-basis classical-control
models -- their exact distribution already exists on disk, so there is nothing to lose by sampling
from it directly with :mod:`metrics.shot_sampler`.

**Two cache layers, mirroring the native counts-store/score split**
(:mod:`pipeline.shots`'s ``.npz`` sequence store vs. :func:`~pipeline.score.load_soft`'s per-
observable ``.pt``):

1. :func:`cached_multinomial_counts` -- the raw draw (``(N, n_out)`` int counts), keyed by
   ``(artifact, shots, shot_seed)`` alone, **no observable involved**.  Cached so that scoring a
   second observable at the same shot budget and seed reuses the SAME realised draw rather than
   silently sampling an independent one -- exactly how the native ``photonic``/``fermion`` path
   shares one ``.npz`` sequence store across every observable scored against it.
2. :func:`cached_multinomial_shots` -- the scored labels (``soft``, one scalar per row), keyed by
   ``(artifact, shots, shot_seed, observable)``, built from #1's cached counts.

**Cached like every other module in this package**, not recomputed per call -- this is what makes
it safe to call from inside a fit loop.  :func:`~pipeline.score.load_dataset` calls
:func:`cached_multinomial_shots` once per ``(config, observable, shots)`` combination that reaches
it (cache-hit on every repeat call, e.g. once per fitted learner and once per reseeded split in a
``best_of_grid_shots``-style sweep) -- without a cache, a shot-budget sweep with ``n_seeds`` splits
and several learners would redraw the multinomial shots and rescore from scratch on every single
fit, and the "seed" axis (meant to vary only the train/test split, never the underlying data --
see :mod:`eval.best_of_grid`'s own docstring on that point) would silently also be redrawing the
data itself each time, contaminating what the seed-averaged R^2 is supposed to measure.

Cache key never involves the learner's own split seed, only ``cfg.seeds.shot_seed`` (the config's
own shot-realisation seed, same convention :func:`pipeline.shots.offset_seed` uses for the native
samplers), so every learner/split-seed combination in a sweep shares one cached draw and one cached
score, exactly matching how the native counts-store path works (one store, many splits read from
it).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

if __package__ in (None, ""):                    # allow `python metrics/multinomial_shots.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"


def _require_no_native_sampler(cfg, out_root: str | Path):
    """``(model, dist, artifact_name, shot_seed, target_shots, source)`` -- the checks and lookups
    shared by both cache layers: rejects a model with a native sampler (use the normal path
    instead) or a config with no shots requested, and requires the exact artifact to already
    exist."""
    from model import build_model
    from pipeline.artifact import exact_path
    from pipeline.distribution import load_dist

    target_shots = int(cfg.generation.shots)
    if target_shots <= 0:
        raise ValueError(f"cfg.generation.shots={target_shots}; this module is for the shots "
                         "branch only")

    model = build_model(cfg)
    if model.supports_shots:
        raise ValueError(
            f"model.kind={cfg.model.kind!r} has a native shot sampler -- use "
            "pipeline.generate.generate_shots + load_dataset instead of this multinomial "
            "fallback, which exists only for models without one")

    path = exact_path(cfg, model, out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run `python -m pipeline.generate --config ...` "
                         "first (the multinomial shots fallback samples from the exact "
                         "distribution, which must already exist)")

    dist = load_dist(path)
    artifact_name = str(dist.meta["hash"])
    shot_seed = int(cfg.seeds.shot_seed)
    source = f"multinomial_s{shot_seed}_n{target_shots}"
    return model, dist, artifact_name, shot_seed, target_shots, source


def _counts_cache_path(scores_root: str | Path, artifact_name: str, source: str) -> Path:
    return Path(scores_root) / artifact_name / source / "multinomial_shots" / "counts.pt"


def cached_multinomial_counts(cfg, *, out_root: str | Path = "datasets",
                              scores_root: str | Path = "scores", force: bool = False) -> torch.Tensor:
    """``(N, n_out)`` int64 shot counts at ``cfg.generation.shots``, cached per
    ``(artifact, shots, shot_seed)`` -- **no observable involved**, so every observable scored at
    the same shot budget and seed shares this one realised draw (see the module docstring).
    """
    from .shot_sampler import sample_shots_from_probs_batch

    _model, dist, artifact_name, shot_seed, target_shots, source = _require_no_native_sampler(
        cfg, out_root)

    cache_path = _counts_cache_path(scores_root, artifact_name, source)
    if cache_path.exists() and not force:
        return torch.load(cache_path, weights_only=True)

    counts = sample_shots_from_probs_batch(dist.probs, shots=target_shots, seed=shot_seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(counts, cache_path)
    return counts


def _soft_cache_paths(scores_root: str | Path, artifact_name: str, source: str,
                      observable: str) -> tuple[Path, Path]:
    base = Path(scores_root) / artifact_name / source / "multinomial_shots" / observable
    return base.with_suffix(".pt"), base.with_suffix(".json")


def cached_multinomial_shots(cfg, name: str, *, out_root: str | Path = "datasets",
                             scores_root: str | Path = "scores", graph_density: float = 0.5,
                             force: bool = False) -> tuple[torch.Tensor, torch.Tensor, str]:
    """``(X, soft, artifact_name)`` at ``cfg.generation.shots``, for a model with
    ``supports_shots == False`` -- raises if ``model.supports_shots`` is ``True`` (use
    :func:`~pipeline.generate.generate_shots` + the normal :func:`~pipeline.score.load_dataset`
    path for those instead) or if the exact distribution has not been generated yet.

    Scores :func:`cached_multinomial_counts`'s cached draw (shared across every observable at this
    shot budget/seed) rather than drawing its own -- see the module docstring for why.

    ``artifact_name`` uses a ``multinomial_s<shot_seed>_n<shots>`` tag -- a distinct namespace
    from :func:`~pipeline.shots.shot_source_tag`'s ``<method>_s<seed>_n<shots>`` (``method`` is
    always ``"clifford"``/``"mh"`` there), so this can never collide with a native-sampler
    artifact even if a model later grows one.
    """
    from model.sampler import sample_X
    from observable import resolve_observable
    from pipeline.score import context_for

    from .shot_sampler import empirical_probs_from_counts

    _model, dist, artifact_name, shot_seed, target_shots, source = _require_no_native_sampler(
        cfg, out_root)

    pt_path, json_path = _soft_cache_paths(scores_root, artifact_name, source, name)
    X = sample_X(int(dist.meta["size"]), int(dist.meta["n_features"]), int(dist.meta["sample_seed"]))
    full_artifact_name = f"{artifact_name}_{source}"

    if pt_path.exists() and json_path.exists() and not force:
        soft = torch.load(pt_path, weights_only=True)
        return X, soft, full_artifact_name

    counts = cached_multinomial_counts(cfg, out_root=out_root, scores_root=scores_root, force=force)
    p_hat = empirical_probs_from_counts(counts).to(dist.probs.dtype)

    ctx = context_for(dist.meta, dist.keys, dist.probs_at_zero.numpy(), graph_density=graph_density)
    obs = resolve_observable(name, ctx)
    soft = obs.score(p_hat).detach()

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soft, pt_path)
    json_path.write_text(json.dumps({"artifact": full_artifact_name, "observable": name,
                                     "shots": target_shots, "shot_seed": shot_seed,
                                     "n": int(soft.shape[0])}))
    return X, soft, full_artifact_name
