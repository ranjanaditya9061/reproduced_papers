"""The local shot-requirement bound: resolving ``x`` from its nearest neighbours, and the
resolvable-gap ``epsilon(N)`` this project's own construction gives (as opposed to Makarovskiy et
al.'s global, arbitrary-pair Proposition 1 -- see :mod:`metrics.global_ratio`).

    from metrics.local_ratio import cached_local_ratio
    result = cached_local_ratio("configs/photonic.yaml", "parity", k=10)
    result["min_delta"]     # (N,) list, the bottleneck neighbour gap per point
    result["eps_local"]     # (N,) list, epsilon_local(N) at a chosen N, for the SAME points

**The bound, and why Delta plays epsilon's role.**  Following Makarovskiy et al.'s own two-point
Chebyshev derivation (Appendix A.2) but restricted to a neighbour set ``g`` in x-space rather than an
arbitrary independently-drawn point:

    N >= max_g (Var[O|x] + Var[O|g]) / (delta * Delta(x,g)^2),   Delta(x,g) := <O>_x - <O>_g

Unlike the global bound, there is no separate free relative-error knob here: the *point* of this
metric is "can x be told apart from its neighbours at all", so the tolerance is naturally the gap
itself (``epsilon_relative = 1`` in Makarovskiy et al.'s Appendix A.2 notation) -- Delta(x,g) is not
approximated by anything, it is looked up directly from the already-cached exact ``<O>`` (this
project's ``Var_circ`` machinery, :mod:`metrics.circuit_variance`, computes it for the whole pool at
once). The bottleneck is the **worst** (smallest) gap among x's k nearest neighbours -- the ``max``
over ``g`` in the shots bound is maximised exactly when Delta is smallest, so ``min_g Delta(x,g)`` and
``max_g [.../Delta(x,g)^2]`` pick out the same neighbour.

**Solving the SAME bound for a resolvable Delta at fixed N** (mirroring
:func:`metrics.global_ratio.epsilon_of_N`'s move, "replace Delta with epsilon"):

    eps_local(N) = sqrt[ (Var[O|x] + Var[O|min-gap neighbour]) / (delta * N) ]

Plot ``eps_local(N)`` against the actual ``min_delta`` at the same points: wherever
``eps_local(N) < min_delta``, that point's hardest neighbour is resolvable at budget ``N``;
wherever ``eps_local(N) > min_delta``, it is not.

**Neighbours are found in x-space** (:func:`scipy.spatial.cKDTree`, k nearest by Euclidean distance
in the input pool), the same convention as
:func:`eval_legacy.resolution_ceiling.estimate_epsilon_x` -- reused here for the KD-tree call, not
its z-gate/shot-noise machinery, since this module works on the exact ``Var[O|x]``/``<O>_x`` (see
:mod:`metrics.shot_variance`/:mod:`metrics.circuit_variance`), not a shots-measured sigma.

**Exact-only.**  No shots anywhere in the neighbour-finding or the Var[O|x]/Delta lookup -- ``N`` is
a *hypothetical* shot budget being asked "would this many shots resolve this pair", not a set of
shots actually drawn. See :mod:`metrics.shot_sampler` for the module that draws real shots when
that's what's wanted instead.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__ in (None, ""):                    # allow `python metrics/local_ratio.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"


def _cache_path(scores_root: str | Path, artifact_name: str, observable: str, k: int) -> Path:
    return (Path(scores_root) / artifact_name / "exact" / "local_ratio"
           / f"{observable}__k{k}.json")


def _nearest_neighbour_gaps(X, soft, k: int):
    """``(min_delta, min_delta_idx)``, both ``(N,)`` -- for each point, the smallest ``|<O>_x -
    <O>_g|`` among its ``k`` nearest x-neighbours, and which neighbour achieved it."""
    import numpy as np
    from scipy.spatial import cKDTree

    N = len(X)
    tree = cKDTree(X)
    k_eff = min(k + 1, N)
    _, idxs = tree.query(X, k=k_eff)              # column 0 is self, at distance 0
    if k_eff == 1:
        return np.zeros(N), np.arange(N)
    idxs = idxs[:, 1:]

    y = np.asarray(soft, dtype=np.float64)
    gaps = np.abs(y[:, None] - y[idxs])            # (N, k)
    min_j = gaps.argmin(axis=1)
    min_delta = gaps[np.arange(N), min_j]
    min_delta_idx = idxs[np.arange(N), min_j]
    return min_delta, min_delta_idx


def cached_local_ratio(cfg_path: str | Path, observable: str, *, k: int = 10,
                       out_root: str | Path = "datasets", scores_root: str | Path = "scores",
                       graph_density: float = 0.5, force: bool = False) -> dict:
    """Per-point ``min_delta`` (bottleneck neighbour gap) and the ingredients for
    ``eps_local(N)`` at any ``N``/``delta`` a caller chooses later -- this caches ``Var[O|x]``,
    the neighbour's own ``Var[O|g]``, and ``min_delta``, not a single fixed-``N`` number, so the
    same cache serves any downstream ``N`` sweep without recomputation.

    Returns ``{"var_x": [...], "var_neighbour": [...], "min_delta": [...], "n": N, ...}``, all
    ``(N,)`` lists aligned to the config's input pool row order.
    """
    from config import load_config
    from pipeline.score import load_dataset

    from .shot_variance import cached_shot_variance

    cfg = load_config(cfg_path)
    if cfg.generation.shots:
        raise ValueError(f"{cfg_path!r} has generation.shots={cfg.generation.shots}; "
                         "the local ratio is exact-only -- point this at an exact (shots=0) config")

    cache_path = None  # resolved after we know artifact_name

    X, soft, artifact_name = load_dataset(cfg, observable, out_root=out_root,
                                          scores_root=scores_root, graph_density=graph_density)
    cache_path = _cache_path(scores_root, artifact_name, observable, k)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    sv = cached_shot_variance(cfg_path, observable, out_root=out_root, scores_root=scores_root,
                              graph_density=graph_density, force=force)
    var_x = sv["var"]

    min_delta, min_delta_idx = _nearest_neighbour_gaps(X.numpy(), soft.numpy(), k)
    var_neighbour = [var_x[int(j)] for j in min_delta_idx]

    result = {
        "var_x": var_x,
        "var_neighbour": var_neighbour,
        "min_delta": min_delta.tolist(),
        "min_delta_idx": min_delta_idx.tolist(),
        "k": int(k),
        "n": len(var_x),
        "artifact": artifact_name,
        "observable": observable,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def eps_local_of_N(result: dict, *, N: int, delta: float = 0.1):
    """``eps_local(N) = sqrt[(Var[O|x] + Var[O|neighbour]) / (delta * N)]``, ``(N,)`` list --
    the resolvable gap at shot budget ``N``, from :func:`cached_local_ratio`'s per-point cache.

    Compare against ``result["min_delta"]`` directly: ``eps_local(N)[i] < min_delta[i]`` means
    point ``i``'s hardest neighbour is resolvable at this ``N``.
    """
    var_x = result["var_x"]
    var_g = result["var_neighbour"]
    return [((vx + vg) / (float(delta) * int(N))) ** 0.5 for vx, vg in zip(var_x, var_g)]


def main(argv=None) -> None:
    import argparse
    import statistics

    ap = argparse.ArgumentParser(description="Local neighbour-resolution bound: min_delta and "
                                             "eps_local(N), cached.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for obs in args.observables:
        res = cached_local_ratio(args.config, obs, k=args.k, out_root=args.out_root,
                                 scores_root=args.scores_root, force=args.force)
        eps = eps_local_of_N(res, N=args.N, delta=args.delta)
        resolvable = sum(e < d for e, d in zip(eps, res["min_delta"]))
        print(f"{obs:32s} median(min_delta)={statistics.median(res['min_delta']):.4g}  "
             f"median(eps_local@N={args.N})={statistics.median(eps):.4g}  "
             f"resolvable={resolvable}/{res['n']}")


if __name__ == "__main__":
    main()
