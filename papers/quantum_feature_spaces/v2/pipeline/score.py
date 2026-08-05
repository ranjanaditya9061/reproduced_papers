"""Scoring: apply a readout to a saved distribution, cached per observable **and per label source**.

    python -m v2.pipeline.score --config v2/configs/photonic.yaml \
        --observables parity majority prod_parity_consecutive ent osc connected_maxcc
    python -m v2.pipeline.score --config v2/configs/photonic_shots.yaml --from-shots ...

    scores_v2/<circuit_hash>/<source>/<observable_spec_hash>.pt

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

Scoring the shots branch goes through :func:`~v2.observable.base.observable_on_keys`, building the
score vector over the outcomes the draw actually **observed**.  A shot draw never enumerates the
basis, so neither does its readout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):                    # allow `python v2/pipeline/score.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.pipeline"

import torch

from config import load_config
from model import build_model
from observable import (ObservableContext, observable_on_keys, observable_spec,
                          observable_spec_hash, resolve_observable)
from .distribution import Distribution, load_dist
from .shots import load_shot_probs

#: Readout-side knobs accepted by :func:`context_for`; everything else comes from the artifact.
OBSERVABLE_KNOBS = ("graph_density", "graph_seed", "angle_seed", "n_vertices")

EXACT_SOURCE = "exact"


def context_for(dist: Distribution, **knobs) -> ObservableContext:
    """Build the :class:`ObservableContext` a loaded artifact implies.

    ``keys``, ``m``, ``k``, ``input_state``, ``readout_modes`` and the ``q`` reference all come from
    the stored exact branch -- which is what makes an offline re-score exact rather than needing a
    matched-seed model rebuilt by hand.
    """
    bad = set(knobs) - set(OBSERVABLE_KNOBS)
    if bad:
        raise ValueError(f"unknown observable knobs {sorted(bad)}; allowed: {list(OBSERVABLE_KNOBS)}")
    meta = dist.meta
    return ObservableContext(
        m=int(meta["m"]), k=int(meta["k"]), keys=dist.keys, seed=int(meta["model_seed"]),
        input_state=meta.get("input_state"),
        reference_probs=dist.probs_at_zero.numpy(),
        readout_modes=tuple(meta.get("readout_modes") or ()),
        **knobs,
    )


def shot_source_tag(shot_meta: dict) -> str:
    """``<shot_hash>_b<n_blocks>`` -- the realisation *and* the budget, both label-changing."""
    import hashlib
    import json
    spec = {k: shot_meta[k] for k in ("block", "shot_seed", "method")}
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{h}_b{int(shot_meta['n_blocks'])}"


def load_circuit_hash(dist_path: str | Path) -> str:
    from .artifact import load_meta
    return str(load_meta(dist_path)["hash"])


def score_path(dist_path: str | Path, name: str, ctx: ObservableContext,
               scores_root: str | Path = "scores_v2", source: str = EXACT_SOURCE) -> Path:
    return (Path(scores_root) / load_circuit_hash(dist_path) / source
            / f"{observable_spec_hash(name, ctx)}.pt")


def load_soft(dist_path: str | Path, name: str, *, scores_root: str | Path = "scores_v2",
              force: bool = False, dist: Distribution | None = None,
              shots_dir: str | Path | None = None, **knobs) -> torch.Tensor:
    """``(N,)`` scores for ``name``; cached on disk, keyed by observable **and** label source.

    ``shots_dir`` scores the finite-shot empirical distribution instead of the exact one.  Note the
    plug-in bias that entails: measured at 100 shots, ``ent`` shifts by ``+0.271`` nats against the
    predicted ``(support-1)/(2S) = 0.275``, and ``osc`` labels correlate only ``0.06`` with the exact
    ones -- ``sin(1/(u+eps))`` sampled on counts quantised to ``1/S`` is not the same functional.
    Expectations are unbiased and survive (``corr ~ 0.76-0.91``).
    """
    dist = dist if dist is not None else load_dist(dist_path)
    ctx = context_for(dist, **knobs)

    obs, probs, source = None, None, EXACT_SOURCE
    if shots_dir is None:
        probs = dist.probs
    else:
        keys, shot_probs, shot_meta = load_shot_probs(shots_dir)
        source = shot_source_tag(shot_meta)
        # The shots branch carries its OWN observed basis -- it never enumerated the full one -- so
        # the score vector is built over those keys.  Equivalent to the dense score on the keys they
        # share (verified to 0.00e+00), and the only form available once the basis is unenumerable.
        obs = observable_on_keys(name, ctx, keys)
        probs = shot_probs[:len(dist)]

    out = score_path(dist_path, name, ctx, scores_root, source)
    if out.exists() and not force:
        blob = torch.load(out, weights_only=False)
        if len(blob["soft"]) >= len(dist):
            return blob["soft"][:len(dist)]

    obs = obs if obs is not None else resolve_observable(name, ctx)
    soft = obs.score(probs).detach()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"soft": soft, "spec": observable_spec(name, ctx), "source": source}, out)
    return soft


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--out-root", default="datasets_v2")
    ap.add_argument("--shots-root", default="shots_v2")
    ap.add_argument("--scores-root", default="scores_v2")
    ap.add_argument("--from-shots", action="store_true",
                    help="score the finite-shot empirical distribution instead of the exact one")
    ap.add_argument("--graph-density", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from .artifact import artifact_path
    from .shots import shots_path
    cfg = load_config(args.config)
    model = build_model(cfg)
    path = artifact_path(cfg, model, args.out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run `python -m v2.pipeline.generate --config "
                         f"{args.config}` first")

    sdir = None
    if args.from_shots:
        sdir = shots_path(cfg, model, shots_root=args.shots_root)
        if not sdir.exists():
            raise SystemExit(f"no shots at {sdir}; set generation.shots > 0 and re-run generate")

    dist = load_dist(path)
    print(f"[score] {path.name}: {len(dist)} rows x {dist.n_out} outcomes"
          f"{'  (FROM SHOTS)' if sdir else '  (exact)'}")
    for name in args.observables:
        soft = load_soft(path, name, scores_root=args.scores_root, force=args.force, dist=dist,
                         shots_dir=sdir, graph_density=args.graph_density)
        print(f"  {name:32s} mean={soft.mean():+.5f}  var={soft.var():.5g}")


if __name__ == "__main__":
    main()
