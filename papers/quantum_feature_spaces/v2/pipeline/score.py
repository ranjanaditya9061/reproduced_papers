"""Scoring: apply a readout to a saved distribution, cached per observable **and per label source**.

    python -m pipeline.score --config configs/photonic.yaml \
        --observables parity majority prod_parity_consecutive ent osc connected_maxcc
    python -m pipeline.score --config configs/photonic_shots.yaml --from-shots ...

    scores_<circuit_hash>/<source>/<observable_spec_hash>.pt

**This is the stage that makes the rewrite pay off.**  Scoring is one matvec over stored ``probs``,
so a sweep over twenty observables costs one simulation instead of twenty.

``<source>`` is ``"exact"`` or ``"<shot_hash>_b<n_blocks>"``.  It has to be in the path: the same
observable on the same circuit gives a *different* label set per shot realisation and per budget, so
a single file standing for all of them would reintroduce the exact bug this module exists to prevent
-- one cache entry silently representing several different datasets.

The context an observable needs beyond ``probs`` -- ``input_state`` for ``single_output``,
``probs_at_zero`` for ``xent`` -- comes from the *exact* branch, since both are functions of the
circuit alone.  That is also why ``xent`` is the one observable unavailable once the outcome basis is
too large to enumerate: there is no full ``q`` to score against.

Scoring the shots branch goes through :func:`~observable.base.observable_on_keys`, building the
score vector over the outcomes the draw actually **observed**.  A shot draw never enumerates the
basis, so neither does its readout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):                    # allow `python pipeline/score.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pipeline"

import numpy as np
import torch

from config import load_config
from model import build_model
from observable import (DiagonalQuadratic, Expectation, ObservableContext, ProbFunction,
                          Quadratic, observable_on_keys, observable_spec, observable_spec_hash,
                          resolve_observable)
from .artifact import load_meta
from .distribution import load_dist
from .shots import load_shots, shot_source_tag

#: Readout-side knobs accepted by :func:`context_for`; everything else comes from the artifact.
OBSERVABLE_KNOBS = ("graph_density", "graph_seed", "angle_seed", "n_vertices")

EXACT_SOURCE = "exact"


def context_for(meta, keys, reference_probs=None, **knobs) -> ObservableContext:
    """Build the :class:`ObservableContext` a loaded artifact implies.

    ``keys``, ``m``, ``k``, ``input_state``, ``readout_modes`` and the ``q`` reference all come from
    the stored exact branch -- which is what makes an offline re-score exact rather than needing a
    matched-seed model rebuilt by hand.
    """
    bad = set(knobs) - set(OBSERVABLE_KNOBS)
    if bad:
        raise ValueError(f"unknown observable knobs {sorted(bad)}; allowed: {list(OBSERVABLE_KNOBS)}")
    
    return ObservableContext(
        m=int(meta["m"]), k=int(meta["k"]), keys=keys, seed=int(meta["model_seed"]),
        input_state=meta.get("input_state"),
        reference_probs=reference_probs,
        readout_modes=tuple(meta.get("readout_modes") or ()),
        n_features=meta.get("n_features"),
        **knobs,
    )


def load_circuit_hash(dist_path: str | Path) -> str:
    from .artifact import load_meta
    return str(load_meta(dist_path)["hash"])


def score_path(dist_path: str | Path, name: str, ctx: ObservableContext,
               scores_root: str | Path = "scores", source: str = EXACT_SOURCE) -> Path:
    return (Path(scores_root) / load_circuit_hash(dist_path) / source
            / f"{observable_spec_hash(name, ctx)}.pt")


def _score_row(obs, probs_row: torch.Tensor, idx: np.ndarray) -> float:
    """``obs.score()``'s own formula, restricted to one row's observed-key indices ``idx`` into
    ``obs``'s tables (built once, over the dataset-wide observed-key union in ``load_soft``) --
    exact because every family here contributes 0 from any outcome outside a row's own support
    (:func:`observable.base.observable_on_keys`'s docstring:
    ``Expectation``/``Quadratic``/``DiagonalQuadratic`` are homogeneous in ``p``, and every
    implemented :class:`~observable.base.ProbFunction` transform vanishes at 0 --
    ``partial_basis_safe`` already guards the one way this could go wrong).  Mirrors each class's
    own ``score()`` rather than reimplementing it independently, so there is nothing to keep in
    sync by hand.
    """
    p = probs_row
    if isinstance(obs, DiagonalQuadratic):
        return float((p * p * obs.score_vec[idx]).sum())
    if isinstance(obs, Quadratic):
        K = obs.pair_kernel[idx][:, idx]
        return float(p @ K @ p)
    if isinstance(obs, Expectation):
        return float((p * obs.score_vec[idx]).sum())
    if isinstance(obs, ProbFunction):
        t = obs.transform(p.unsqueeze(0)).squeeze(0)
        return float(t.sum() if obs.score_vec is None else (t * obs.score_vec[idx]).sum())
    raise NotImplementedError(f"row-streaming shot scoring not implemented for {type(obs).__name__}")


def score_shots_streaming(obs, seq: np.ndarray, shots: int) -> torch.Tensor:
    """``(N,)`` scores, one row at a time -- the memory-bounded alternative to
    ``pipeline.shots.to_counts(keys, seq)``, which builds a single DENSE ``(N, n_keys)`` matrix
    over the dataset-WIDE observed-key union (not any one row's own support) and blows up
    combinatorially with ``m``: ``10000 x 490309`` int64 is 36.5 GiB at ``m=16``, 1.37 TiB by
    ``m=20`` (PLOTS.md item 7b).

    Each row draws only ``<= shots`` distinct outcomes, so its own local ``(idx, counts)`` (via
    ``np.unique``) is at most ``shots``-long regardless of ``m`` -- this scores that row through
    :func:`_score_row` and lets ``idx``/``counts``/``probs_row`` go out of scope before the next
    row starts, so peak memory is one row's support, never the dataset-wide union.
    """
    N = seq.shape[0]
    out = torch.empty(N, dtype=torch.float32)
    for i in range(N):
        idx, counts = np.unique(seq[i], return_counts=True)
        probs_row = torch.from_numpy(counts.astype(np.float32) / shots)
        out[i] = _score_row(obs, probs_row, idx)
    return out


def load_soft(dist_path: str | Path, name: str, *, scores_root: str | Path = "scores",
              force: bool = False, shots_dir: str | Path | None = None, num_shots: int | Path | None = None,**knobs) -> torch.Tensor:
    """``(N,)`` scores for ``name``; cached on disk, keyed by observable **and** label source.

    ``shots_dir`` scores the finite-shot empirical distribution instead of the exact one.  Note the
    plug-in bias that entails: measured at 100 shots, ``ent`` shifts by ``+0.271`` nats against the
    predicted ``(support-1)/(2S) = 0.275``, and ``osc`` labels correlate only ``0.06`` with the exact
    ones -- ``sin(1/(u+eps))`` sampled on counts quantised to ``1/S`` is not the same functional.
    Expectations are unbiased and survive (``corr ~ 0.76-0.91``).
    """
    
    if shots_dir is None:
        dist = load_dist(dist_path)
        ctx = context_for(dist.meta, dist.keys, dist.probs_at_zero.numpy(), **knobs)
        obs, probs, source = None, None, EXACT_SOURCE
        probs = dist.probs
        out_path = dist_path
        n_rows = len(probs)
        seq = shots = None
    else:
        keys, seq, shot_meta = load_shots(shots_dir, num_shots)
        shots = int(shot_meta["shots"])
        source = shot_source_tag(shot_meta)
        ctx = context_for(shot_meta, keys, **knobs)
        # The shots branch carries its OWN observed basis -- it never enumerated the full one -- so
        # the score vector is built over those keys (O(n_keys) tables only -- see
        # score_shots_streaming for why the (N, n_keys) dense matrix this used to require never
        # gets built).  Equivalent to the dense score on the keys they share (verified to 0.00e+00),
        # and the only form available once the basis is unenumerable.
        obs = observable_on_keys(name, ctx, keys)
        out_path = shots_dir
        n_rows = seq.shape[0]

    out = score_path(out_path, name, ctx, scores_root, source)
    if out.exists() and not force:
        blob = torch.load(out, weights_only=False)
        if len(blob["soft"]) >= n_rows:
            return blob["soft"][:n_rows]

    if shots_dir is None:
        obs = obs if obs is not None else resolve_observable(name, ctx)
        soft = obs.score(probs).detach()
    else:
        soft = score_shots_streaming(obs, seq, shots).detach()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"soft": soft, "spec": observable_spec(name, ctx), "source": source}, out)
    return soft


def load_dataset(cfg, name: str, *, out_root: str | Path = "datasets",
                 scores_root: str | Path = "scores", graph_density: float = 0.5,
                 force: bool = False) -> tuple[torch.Tensor, torch.Tensor, str]:
    """``(X, soft, artifact_name)`` for ``cfg`` -- the exact branch if ``cfg.generation.shots ==
    0``, else the shots branch, matching :func:`main`'s own dispatch.

    The one thing :func:`load_soft` alone does not give a caller that wants to fit a learner:
    ``X``. Neither branch stores it (:mod:`pipeline.distribution`'s docstring: ``sample_X`` is
    prefix-stable and deterministic in ``sample_seed``, so storing ``X`` would just be a second,
    driftable source of truth) -- the exact branch reconstructs it via :class:`Distribution`
    already; this does the equivalent for the shots branch from ``shot_meta``'s own
    ``n_features``/``sample_seed``/``size`` (present on both branches via
    :func:`~pipeline.artifact.hash_fields`/:func:`~pipeline.artifact.save_meta`).

    Exists because :mod:`learner.auto` and :mod:`learner.compare` need "the dataset this config
    names" and, before this function, only ever asked the exact branch for it -- silently wrong
    whenever ``cfg.generation.shots > 0`` (scores would come from :func:`load_soft`'s shots path
    while ``X`` still came from the exact one, a row mismatch if sizes ever differ) and an outright
    ``SystemExit`` past the point where the exact branch cannot be generated at all (the whole
    reason :meth:`model.fermion.FermionModel.shot_counts` exists).

    ``artifact_name`` is the real circuit hash (:func:`load_circuit_hash`'s ``meta["hash"]``, the
    same convention :func:`score_path` already uses for its own cache), joined with the branch's
    own tag on the shots side -- **not** ``path.name``/``sdir.name``. A bare ``path.name`` on the
    exact branch is always the literal constant ``"exact"`` (:data:`~pipeline.artifact.EXACT_DIR`),
    identical for every ``cfg`` regardless of ``m``/``k``/circuit identity, and ``sdir.name`` on the
    shots branch is only the shot tag (``<method>_s<seed>_n<shots>``), which collides across two
    different circuits drawn with the same method/seed/shot count. Both were latent bugs (the field
    was informational-only, cited in printed reports, before anything treated it as a real cache
    key); :func:`~learner.cache.cached_fit` is the first caller to build actual on-disk cache paths
    from it, which is what turned the collision from cosmetic into a real one -- two different
    ``m`` configs silently sharing one cache entry, each fit overwriting the last's, every read
    after the first returning some earlier ``m``'s stale result instead of refitting.
    """
    from model import build_model
    from model.sampler import sample_X

    from .artifact import exact_path
    from .shots import shot_source_tag, shots_path

    model = build_model(cfg)
    if cfg.generation.shots and not model.supports_shots:
        # No native sampler (quadratic_fock, mlp_fock, mlp, analytical): pipeline.shots has no
        # counts-store path for these by design (see its own module docstring on why a
        # multinomial-from-stored-p method was deliberately left out) -- fall back to the cached
        # multinomial-shots metric on the exact distribution instead. Cached (unlike an in-memory
        # draw), so every repeat call from cached_fit across a learner/seed sweep is a cache hit,
        # not a redraw -- see metrics.multinomial_shots's own docstring.
        from metrics.multinomial_shots import cached_multinomial_shots
        return cached_multinomial_shots(cfg, name, out_root=out_root, scores_root=scores_root,
                                        graph_density=graph_density, force=force)
    if cfg.generation.shots:
        sdir = shots_path(cfg, model, root=out_root)
        if not sdir.exists():
            raise SystemExit(f"no shots at {sdir}; set generation.shots > 0 and run "
                             f"pipeline.generate first")
        soft = load_soft(None, name, scores_root=scores_root, force=force, shots_dir=sdir,
                         num_shots=cfg.generation.shots, graph_density=graph_density)
        meta = load_meta(sdir)
        X = sample_X(int(meta["size"]), int(meta["n_features"]), int(meta["sample_seed"]))
        # Tag with the REQUESTED (cropped) shot count, cfg.generation.shots -- not meta["shots"],
        # which is the store's on-disk total and only grows as generate_shots extends it. Reusing
        # meta["shots"] here made every N cropped from the same store (e.g. 100/1000/10000, all
        # cropped from one 50000-shot store) collide on one artifact_name, and therefore on one
        # learner.cache.cached_fit cache entry -- the R^2 for those N's was silently identical
        # because they were reading back the same cached fit rather than a fit on their own crop.
        cropped_meta = {**meta, "shots": int(cfg.generation.shots)}
        artifact_name = f"{meta['hash']}_{shot_source_tag(cropped_meta)}"
        return X, soft, artifact_name
    else:
        path = exact_path(cfg, model, out_root)
        if not path.exists():
            raise SystemExit(f"no artifact at {path}; run pipeline.generate first")
        dist = load_dist(path)
        soft = load_soft(path, name, scores_root=scores_root, force=force,
                         graph_density=graph_density)
        return dist.X, soft, load_circuit_hash(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--graph-density", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from .artifact import exact_path
    from .shots import shots_path
    cfg = load_config(args.config)
    model = build_model(cfg)

    
    sdir = None
    path = None
    if cfg.generation.shots:
        sdir = shots_path(cfg, model)
        if not sdir.exists():
            raise SystemExit(f"no shots at {sdir}; set generation.shots > 0 and re-run generate")
    else:
        path = exact_path(cfg, model, args.out_root)
        if not path.exists():
            raise SystemExit(f"no artifact at {path}; run `python -m pipeline.generate --config "
                                f"{args.config}` first")
        print(f"[score] {path.name}: {'  (FROM SHOTS)' if sdir else '  (exact)'}")
        
    for name in args.observables:
        soft = load_soft(path, name, scores_root=args.scores_root, force=args.force,
                         shots_dir=sdir, graph_density=args.graph_density, num_shots = cfg.generation.shots)
        print(f"  {name:32s} mean={soft.mean():+.5f}  var={soft.var():.5g}")

if __name__ == "__main__":
    main()