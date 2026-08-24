"""``Var_circ(O)``, cached: the variance of the observable's true value across the input pool.

    from metrics.circuit_variance import cached_circuit_variance
    result = cached_circuit_variance("configs/photonic.yaml", "parity")
    result["var_circ"]     # Var_x[<O>_x], one scalar

**What this computes.**  ``Var_circ(O) := Var_x[<O>_x]``, ``<O>_x = T(p_x)`` -- the spread of the
observable's *true* value as the input ``x`` varies across the pool.  This is the ``x``-averaged
construction :mod:`METRICS.md` and this project's own draft adopt in place of Makarovskiy et al.'s
``theta``-averaged ``Var_circ = Var_theta[l(theta)]`` (averaged over a Haar-random circuit draw) --
the same law-of-total-variance object, averaged over the encoded input rather than the circuit's own
parameters.  Not proven equivalent to the source paper's construction, only analogous; state this
substitution explicitly wherever ``Var_circ`` is reported in the paper.

**No shots, no new simulation.**  ``<O>_x = T(p_x)`` is exactly the ``soft`` score vector
:mod:`pipeline.score` already computes and caches for every ``(config, observable)`` pair scored so
far -- the same tensor :mod:`learner.cache` trains a learner on.  This module does not recompute
``T(p_x)``; it loads that existing cache via :func:`pipeline.score.load_dataset` and takes one
variance over it.  Exact-only, like :mod:`metrics.shot_variance` -- do not add a ``shots=`` kwarg
here; a shots-based ``Var_circ`` would need shot-noisy labels, which is a different quantity (the
plug-in bias/variance question :mod:`metrics.shot_variance`'s own docstring already flags).

**Pairing with ``Var_samp``.**  ``Var_circ`` on its own is not the trainability/utility criterion --
see :mod:`metrics.shot_variance` for ``Var_samp(O) = E_x[Var[O|x]]``, the other half of the ratio
``Var_samp/Var_circ`` this project's Circuit Variance section builds on.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python metrics/circuit_variance.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"


def _cache_path(scores_root: str | Path, artifact_name: str, observable: str) -> Path:
    return Path(scores_root) / artifact_name / "exact" / "circuit_variance" / f"{observable}.json"


def cached_circuit_variance(cfg_path: str | Path, observable: str, *,
                            out_root: str | Path = "datasets", scores_root: str | Path = "scores",
                            graph_density: float = 0.5, force: bool = False) -> dict:
    """``Var_circ(O) = Var_x[<O>_x]`` over ``cfg_path``'s input pool, or a prior run's cache.

    Returns ``{"var_circ": scalar, "mean": scalar, "n": N, "artifact": name, "observable": name}``.
    Requires ``cfg.generation.shots == 0`` (the exact branch) -- raises via
    :func:`pipeline.score.load_dataset`'s own ``SystemExit`` if the exact artifact does not exist or
    the config is shots-only, matching :mod:`metrics.shot_variance`'s exact-only convention.
    """
    from config import load_config
    from pipeline.score import load_dataset

    cfg = load_config(cfg_path)
    if cfg.generation.shots:
        raise ValueError(f"{cfg_path!r} has generation.shots={cfg.generation.shots}; "
                         "Var_circ is exact-only -- point this at an exact (shots=0) config")

    X, soft, artifact_name = load_dataset(cfg, observable, out_root=out_root,
                                          scores_root=scores_root, graph_density=graph_density)

    cache_path = _cache_path(scores_root, artifact_name, observable)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    soft_d = soft.detach().double()
    result = {
        "var_circ": float(soft_d.var(unbiased=False)),
        "mean": float(soft_d.mean()),
        "n": int(soft_d.shape[0]),
        "artifact": artifact_name,
        "observable": observable,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Var_circ(O) = Var_x[<O>_x] over a config's input "
                                             "pool, cached.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for obs in args.observables:
        res = cached_circuit_variance(args.config, obs, out_root=args.out_root,
                                      scores_root=args.scores_root, force=args.force)
        print(f"{obs:32s} Var_circ={res['var_circ']:.6g}  mean={res['mean']:+.5g}  n={res['n']}")


if __name__ == "__main__":
    main()
