"""``Var[O|x]``, cached: the single-shot variance of an observable at each input in a config's pool.

    from metrics.shot_variance import cached_shot_variance
    result = cached_shot_variance("configs/photonic.yaml", "parity")
    result["var"]          # (N,) list, one entry per row of the config's input pool
    result["var_samp"]     # mean over the pool -- E_x[Var[O|x]]

**What this computes, and where the number comes from.**  For :class:`~observable.base.Expectation`
observables (``T(p) = p . v``), ``Var[O|x] = E_n[v(n)^2] - E_n[v(n)]^2`` is the **exact** variance of
a single projective measurement at ``x`` -- draw one outcome ``n ~ p(.|x)``, read off ``v(n)``, and
this is that draw's variance.  For :class:`~observable.base.Quadratic`/:class:`~observable.base.
ProbFunction` observables, there is no single-outcome random variable whose mean is ``T`` (``T`` is
nonlinear in ``p``), so no exact single-shot variance exists; what is returned instead is the
delta-method (multivariate, via the influence function ``psi_n = dT/dp_n``) leading-order asymptotic
variance of the plug-in estimator, ``Var_p(psi) = E_p[psi^2] - E_p[psi]^2`` -- exact as the shot
count ``S -> infinity``, not at finite ``S``.  Both cases are the *same* formula, ``E_p[X^2] -
E_p[X]^2``, with ``X`` either the raw score ``v`` (``Expectation``) or the influence function
``psi`` (everything else) -- see :meth:`observable.base.Observable.effective_variance`'s own
docstring, which this module calls directly rather than reimplementing.

**No shots anywhere in this file.**  Every quantity here is computed from the *exact* stored
distribution (``dist.probs``, or the model's exact ``probs(x)``) -- this is a Chebyshev/delta-method
*ingredient*, not a shot-noise measurement.  A metric that needs actual finite-``S`` shot draws
belongs in a different, explicitly shots-capable module -- do not add a ``shots=`` kwarg here.

**Caching.**  Mirrors :mod:`learner.cache`'s convention: one file per ``(artifact, observable)``
under ``scores_root/<artifact>/<source>/shot_variance/<observable>.json``, holding every row's
``Var[O|x]`` plus the pool mean (``Var_samp``, see :mod:`metrics.circuit_variance` for the
``Var_circ`` half of the ratio). A cache hit skips both the dataset load and the recompute.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python metrics/shot_variance.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"


def _cache_path(scores_root: str | Path, artifact_name: str, source: str, observable: str) -> Path:
    return Path(scores_root) / artifact_name / source / "shot_variance" / f"{observable}.json"


def cached_shot_variance(cfg_path: str | Path, observable: str, *,
                         out_root: str | Path = "datasets", scores_root: str | Path = "scores",
                         graph_density: float = 0.5, force: bool = False) -> dict:
    """``Var[O|x]`` for every row of ``cfg_path``'s input pool, or a prior run's cached result.

    Returns ``{"var": [...], "var_samp": mean, "n": N, "artifact": name, "is_expectation": bool}``.
    ``is_expectation`` records whether ``var`` is the exact single-shot variance or the delta-method
    asymptotic one, so a caller never has to re-derive which case it is in from the observable name.

    Loads the config's already-generated dataset (raises if it has not been generated yet -- same
    convention as :func:`learner.cache.cached_fit`, this does not call :mod:`pipeline.generate`
    itself) and scores every row's exact distribution in one batched
    :meth:`~observable.base.Observable.effective_variance` call, rather than looping row by row.
    """
    from config import load_config
    from observable import resolve_observable
    from pipeline.artifact import exact_path
    from pipeline.distribution import load_dist
    from pipeline.score import context_for

    cfg = load_config(cfg_path)
    from model import build_model
    model = build_model(cfg)
    path = exact_path(cfg, model, out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run `python -m pipeline.generate --config "
                         f"{cfg_path}` first")

    dist = load_dist(path)
    artifact_name = str(dist.meta["hash"])
    source = "exact"

    cache_path = _cache_path(scores_root, artifact_name, source, observable)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    ctx = context_for(dist.meta, dist.keys, dist.probs_at_zero.numpy(), graph_density=graph_density)
    obs = resolve_observable(observable, ctx)
    if not obs.is_differentiable:
        raise ValueError(f"{observable!r} is not differentiable in p (is_differentiable=False); "
                         f"Var[O|x] via the influence function is undefined for it")

    var = obs.effective_variance(dist.probs).detach().double().tolist()

    result = {
        "var": var,
        "var_samp": statistics.mean(var),
        "n": len(var),
        "artifact": artifact_name,
        "observable": observable,
        "is_expectation": bool(obs.is_expectation),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Var[O|x] over a config's input pool, cached.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for obs in args.observables:
        res = cached_shot_variance(args.config, obs, out_root=args.out_root,
                                   scores_root=args.scores_root, force=args.force)
        kind = "exact" if res["is_expectation"] else "asymptotic (delta-method)"
        print(f"{obs:32s} Var_samp={res['var_samp']:.6g}  n={res['n']:<8d} [{kind}]")


if __name__ == "__main__":
    main()
