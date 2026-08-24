"""R^2-at-10-seeds companion to :mod:`eval.gradient_vs_hardness`: is low R^2 always a collapsed
gradient, or can an observable be hard to *learn* while its gradient stays healthy?

    python eval/r2_vs_gradient.py

Same "average per learner over seeds, then max across learners" protocol as
:func:`eval.best_of_grid.sweep_best_of_grid`, so results are directly comparable to every other
R^2 number reported in this repo -- but loads each ``(config, observable)``'s dataset **once**
via :func:`~learner.cache.preload_dataset` and reuses it across every learner/seed's
:func:`~learner.cache.cached_fit` call, rather than calling :func:`learner.auto.run_config` in a
loop (which reloads ``dist.npz`` from disk on every single call). Measured at ``m=12`` (12,376
outcomes x 10,000 rows, ~495 MB): ~12-20s per reload, almost entirely I/O -- with 4 sizes x 6
observables x 3 learners x 10 seeds = 720 fits, that reload cost alone would dominate wall time by
roughly two orders of magnitude versus loading once per (config, observable) cell (24 loads) and
fitting in memory. The fit itself is still cached per (learner, seed) via
:func:`~learner.cache.cached_fit`'s own on-disk cache, so a second run of this module is cheap even
across process restarts, not just within one.

Runs against the SAME configs (:mod:`configs.size.size_photonic_fock`, ``m in {6,8,10,12}``) and
the SAME observable list as :mod:`eval.gradient_vs_hardness`, so the two JSONs can be joined on
``(m, observable)`` for the final comparison plot.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learner.auto import DEFAULT_SWEEP_LEARNERS
from learner.cache import cached_fit, preload_dataset
from learner import embedding, kernel, nn  # noqa: F401 -- registration side effects

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "size" / "size_photonic_fock"
CONFIGS = {6: "m06k03.yaml", 8: "m08k04.yaml", 10: "m10k05.yaml", 12: "m12k06.yaml"}

#: Matches eval.gradient_vs_hardness.OBSERVABLES exactly -- the join key for the final plot.
OBSERVABLES = ["parity", "n_first", "ent", "osc", "sq_parity", "xent_parity"]


def best_r2(cfg_path: Path, observable: str, *, n_seeds: int, out_root: str,
           scores_root: str) -> dict:
    """Max-of-per-learner-mean R^2 for one (config, observable): loads (X, soft) ONCE via
    :func:`~learner.cache.preload_dataset`, then loops learners/seeds through
    :func:`~learner.cache.cached_fit` (which reuses that preload rather than reloading) -- one cell
    of best_of_grid's protocol, standalone."""
    preloaded = preload_dataset(cfg_path, observable, out_root=out_root, scores_root=scores_root)

    per_learner_means = {}
    for lname, kwargs in DEFAULT_SWEEP_LEARNERS:
        scores = []
        for seed in range(int(n_seeds)):
            k = dict(kwargs)
            k["seed"] = seed
            try:
                res = cached_fit(cfg_path, observable, lname, out_root=out_root,
                                 scores_root=scores_root, split_seed=seed, _preloaded=preloaded,
                                 **k)
                scores.append(res["r2"])
            except (Exception, SystemExit) as exc:         # noqa: BLE001
                print(f"  [r2] {observable}/{lname}/seed={seed} failed: {exc}")
        if scores:
            per_learner_means[lname] = statistics.mean(scores)
    if not per_learner_means:
        return {"r2": None, "per_learner": {}}
    best_name = max(per_learner_means, key=per_learner_means.get)
    return {"r2": per_learner_means[best_name], "best_learner": best_name,
            "per_learner": per_learner_means}


def run(*, n_seeds: int = 10, out_root: str = "datasets", scores_root: str = "scores") -> dict:
    out = {"observables": OBSERVABLES, "n_seeds": n_seeds, "sizes": []}
    for m, fname in CONFIGS.items():
        cfg_path = CONFIGS_DIR / fname
        row = {"m": m, "per_obs": {}}
        for obs in OBSERVABLES:
            try:
                r = best_r2(cfg_path, obs, n_seeds=n_seeds, out_root=out_root,
                           scores_root=scores_root)
            except (Exception, SystemExit) as exc:            # noqa: BLE001 -- one bad (m, obs)
                # cell (most commonly a missing dataset -- load_dataset raises SystemExit, not
                # Exception, which a bare `except Exception` here would silently miss) must not
                # abort the rest of the sweep
                print(f"  [r2] m={m}/{obs} failed: {exc}")
                r = {"r2": None, "per_learner": {}}
            row["per_obs"][obs] = r
            r2_txt = f"{r['r2']:.4f}" if r["r2"] is not None else "FAILED"
            print(f"m={m:2d} {obs:<14} best_r2={r2_txt} "
                  f"({r.get('best_learner', '-')})", flush=True)
        out["sizes"].append(row)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "r2_vs_gradient.json"))
    args = ap.parse_args(argv)

    res = run(n_seeds=args.n_seeds, out_root=args.out_root, scores_root=args.scores_root)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
