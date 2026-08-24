"""The global shot-requirement bound (Makarovskiy et al., Proposition 1): resolving an arbitrary,
independently-drawn pair of points -- as opposed to :mod:`metrics.local_ratio`'s own construction,
restricted to a point's nearest neighbours in x-space.

    from metrics.global_ratio import cached_global_ratio, eps_global_of_N
    result = cached_global_ratio("configs/photonic.yaml", "parity")
    eps_global_of_N(result, N=10000, t=1.0, delta=0.1)

**The bound.**  For epsilon > 0, delta in (0, 1], t > 0, and theta, theta' drawn independently from
the parameter distribution, with probability at least 1 - 2/t, taking

    N >= (t / (delta * epsilon^2)) * Var_samp(O) / Var_circ(O)

samples at each setting yields loss estimates within epsilon * sqrt(Var_circ(O)) of the true value,
simultaneously with probability at least 1 - 2*delta.  Solving for epsilon at fixed N (the same
"replace the free variable with N" move as :func:`metrics.local_ratio.eps_local_of_N`):

    eps_global(N) = sqrt[ t * Var_samp(O) / (delta * N * Var_circ(O)) ]

**Averaged over x, not theta -- the same substitution as :mod:`metrics.circuit_variance`.**
``Var_samp``/``Var_circ`` here are this project's x-averaged construction (built on
:mod:`metrics.shot_variance`/:mod:`metrics.circuit_variance`), not Makarovskiy et al.'s own
theta-averaged (Haar-random-circuit-draw) construction -- analogous, not proven equivalent; state
this substitution wherever ``eps_global`` is reported, same caveat as :mod:`metrics.circuit_variance`.

**Why this is the comparison point for the local bound.**  The global bound answers "how many shots
to resolve an arbitrary pair anywhere in the distribution" -- necessarily conservative, since it has
to cover the worst case over the whole space.  The local bound
(:mod:`metrics.local_ratio`) answers the narrower "how many shots to resolve x from its actual
nearest neighbours" -- plotting ``eps_global(N)`` alongside ``eps_local(N)``/``min_delta``
(:mod:`eval.local_ratio_plot`) shows how much slack the local, neighbour-restricted question buys
over the global, arbitrary-pair one at the same N.

Exact-only, same convention as every other module in this package -- no ``shots=`` kwarg.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__ in (None, ""):                    # allow `python metrics/global_ratio.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"


def _cache_path(scores_root: str | Path, artifact_name: str, observable: str) -> Path:
    return Path(scores_root) / artifact_name / "exact" / "global_ratio" / f"{observable}.json"


def cached_global_ratio(cfg_path: str | Path, observable: str, *,
                        out_root: str | Path = "datasets", scores_root: str | Path = "scores",
                        graph_density: float = 0.5, force: bool = False) -> dict:
    """``{"var_samp": ..., "var_circ": ..., "artifact": ..., "observable": ...}`` -- the two
    ingredients Proposition 1's bound needs, from the already-cached
    :mod:`metrics.shot_variance`/:mod:`metrics.circuit_variance` results (no new computation of its
    own -- this module only assembles the ratio and its N-dependence).
    """
    from .circuit_variance import cached_circuit_variance
    from .shot_variance import cached_shot_variance

    sv = cached_shot_variance(cfg_path, observable, out_root=out_root, scores_root=scores_root,
                              graph_density=graph_density, force=force)
    cv = cached_circuit_variance(cfg_path, observable, out_root=out_root, scores_root=scores_root,
                                 graph_density=graph_density, force=force)

    cache_path = _cache_path(scores_root, cv["artifact"], observable)
    result = {
        "var_samp": sv["var_samp"],
        "var_circ": cv["var_circ"],
        "artifact": cv["artifact"],
        "observable": observable,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def eps_global_of_N(result: dict, *, N: int, t: float = 1.0, delta: float = 0.1) -> float:
    """``eps_global(N) = sqrt[t * Var_samp / (delta * N * Var_circ)]`` -- the resolvable relative
    error, in units of ``sqrt(Var_circ)``, at shot budget ``N``.  ``t=1`` is the loosest
    (least-conservative) choice consistent with Proposition 1's own hypothesis ``t > 0`` -- raise
    it for the probability guarantee on bounding the per-point variance by ``t * Var_samp`` (see
    the module docstring's Markov-inequality step) to hold with higher confidence.
    """
    var_samp, var_circ = result["var_samp"], result["var_circ"]
    return (float(t) * var_samp / (float(delta) * int(N) * max(var_circ, 1e-300))) ** 0.5


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Global (Proposition 1) shot-requirement bound, "
                                             "cached.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for obs in args.observables:
        res = cached_global_ratio(args.config, obs, out_root=args.out_root,
                                  scores_root=args.scores_root, force=args.force)
        eps = eps_global_of_N(res, N=args.N, t=args.t, delta=args.delta)
        print(f"{obs:32s} Var_samp={res['var_samp']:.6g}  Var_circ={res['var_circ']:.6g}  "
             f"eps_global(N={args.N})={eps:.6g}")


if __name__ == "__main__":
    main()
