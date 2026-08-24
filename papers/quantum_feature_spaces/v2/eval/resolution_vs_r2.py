"""Local and global resolution vs. R^2, across the SAME architecture grid
:mod:`eval.best_of_grid` swept -- a single config's per-point resolution plot
(:mod:`eval.local_ratio_plot`) tells you nothing on its own; this is the summary that connects it
to learnability.

    python eval/resolution_vs_r2.py --eval-dir configs/eval/photonic_encoding --observable parity \
        --N 10000
    python eval/resolution_vs_r2.py --configs configs/eval/*/*.yaml --observable parity --N 10000

**Local**: ``mean(min_delta - eps_local(N))`` per config -- the average resolvability margin across
the pool's nearest-neighbour pairs (:mod:`metrics.local_ratio`), plotted against that config's
measured R^2 at ``shots=N`` (from :mod:`eval.best_of_grid_shots`'s own cached results, not
recomputed here).

**Global**: ``eps_global(N)`` per config (Makarovskiy et al.'s Proposition 1,
:mod:`metrics.global_ratio`) -- already a single scalar per config, no per-point average needed --
plotted directly against the same measured R^2 at ``shots=N``.

**Plotting only.**  No simulation, no observable scoring, no neighbour search happens in this file;
every number comes from the cached metric modules and from :func:`learner.cache.cached_fit` via
:func:`learner.auto.run_config`.
"""

from __future__ import annotations

import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/resolution_vs_r2.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _r2_at_shots(cfg_path, observable: str, shots: int, *, n_seeds: int, out_root: str,
                 scores_root: str) -> float:
    """Max over :data:`learner.auto.DEFAULT_SWEEP_LEARNERS`, each averaged over ``n_seeds``
    reseeded splits, at the given shot budget -- same convention as
    :func:`eval.best_of_grid.sweep_best_of_grid`, evaluated on an in-memory
    ``generation.shots``-overridden config (:func:`eval.r2_vs_shots._shots_variant_config`, no
    sibling YAML written to disk)."""
    from eval.r2_vs_shots import _shots_variant_config
    from learner.auto import DEFAULT_SWEEP_LEARNERS, run_config
    from pipeline.generate import generate_shots

    cfg_n = _shots_variant_config(cfg_path, shots)
    generate_shots(cfg_n, root=out_root)

    means = []
    for lname, kwargs in DEFAULT_SWEEP_LEARNERS:
        scores = []
        for seed in range(int(n_seeds)):
            res = run_config(cfg_n, observable, lname, out_root=out_root,
                             scores_root=scores_root, split_seed=seed, seed=seed, **kwargs)
            scores.append(res["r2"])
        if scores:
            means.append(statistics.mean(scores))
    return max(means) if means else float("nan")


def collect(cfg_paths: list[str | Path], observable: str, *, N: int = 10000, k: int = 10,
           delta: float = 0.1, t: float = 1.0, n_seeds: int = 10, out_root: str = "datasets",
           scores_root: str = "scores") -> dict:
    """One row per config: mean local margin, global eps, and R^2 at ``shots=N``."""
    from metrics.global_ratio import cached_global_ratio, eps_global_of_N
    from metrics.local_ratio import cached_local_ratio, eps_local_of_N

    labels, local_margin, global_eps, r2 = [], [], [], []
    for cfg_path in cfg_paths:
        local = cached_local_ratio(cfg_path, observable, k=k, out_root=out_root,
                                   scores_root=scores_root)
        eps_l = eps_local_of_N(local, N=N, delta=delta)
        margin = [md - e for md, e in zip(local["min_delta"], eps_l)]

        glob = cached_global_ratio(cfg_path, observable, out_root=out_root,
                                   scores_root=scores_root)
        eps_g = eps_global_of_N(glob, N=N, t=t, delta=delta)

        r2_val = _r2_at_shots(cfg_path, observable, N, n_seeds=n_seeds, out_root=out_root,
                              scores_root=scores_root)

        labels.append(Path(cfg_path).stem)
        local_margin.append(statistics.mean(margin))
        global_eps.append(eps_g)
        r2.append(r2_val)
        print(f"{labels[-1]:24s} mean_local_margin={local_margin[-1]:.4g}  "
             f"eps_global(N={N})={eps_g:.4g}  R^2={r2_val:.4g}")

    return {"configs": labels, "mean_local_margin": local_margin, "eps_global": global_eps,
           "r2": r2, "observable": observable, "N": N, "k": k, "n_seeds": n_seeds}


def plot_resolution_vs_r2(result: dict, *, save_path: str | Path | None = None,
                          show: bool = False):
    """Two panels: mean local resolvability margin vs. R^2, and eps_global(N) vs. R^2 -- each one
    point per config, labelled.  Kept as separate axes: the local margin is signed (positive =
    resolvable) while eps_global is a magnitude (Proposition 1's own convention), so they are not
    on a directly comparable scale.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1.scatter(result["mean_local_margin"], result["r2"], s=30, alpha=0.7, color="tab:purple")
    for x, y, label in zip(result["mean_local_margin"], result["r2"], result["configs"]):
        ax1.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax1.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("mean(min_delta - eps_local(N)) over pool")
    ax1.set_ylabel(f"R^2 at shots={result['N']}")
    ax1.set_title("Local resolvability margin vs. R^2")
    ax1.set_ylim(-0.05, 1.02)
    ax1.grid(alpha=0.3)

    ax2.scatter(result["eps_global"], result["r2"], s=30, alpha=0.7, color="tab:red")
    for x, y, label in zip(result["eps_global"], result["r2"], result["configs"]):
        ax2.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel(f"eps_global(N={result['N']})  [smaller = more resolvable]")
    ax2.set_ylabel(f"R^2 at shots={result['N']}")
    ax2.set_title("Global resolvability bound vs. R^2")
    ax2.set_ylim(-0.05, 1.02)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(f"observable={result['observable']}  N={result['N']}  k={result['k']}  "
                f"n_seeds={result['n_seeds']}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Local and global resolution vs. R^2, across a "
                                             "configs/eval/ group, or an explicit flat list of "
                                             "configs.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval-dir", help="e.g. configs/eval/photonic_encoding")
    group.add_argument("--configs", nargs="+", help="explicit config paths/globs, e.g. "
                                                     "configs/eval/*/*.yaml to combine every "
                                                     "subfolder")
    ap.add_argument("--observable", required=True)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    if args.configs:
        cfg_paths = [Path(p) for p in args.configs]
    else:
        from eval.best_of_grid import _variants_in

        variants = _variants_in(Path(args.eval_dir))
        if not variants:
            raise SystemExit(f"no *.yaml configs found in {args.eval_dir}")
        cfg_paths = [path for _, path in variants]

    result = collect(cfg_paths, args.observable, N=args.N, k=args.k, delta=args.delta, t=args.t,
                     n_seeds=args.n_seeds, out_root=args.out_root, scores_root=args.scores_root)

    out_json = args.out_json or f"resolution_vs_r2__{args.observable}.json"
    out_png = args.out_png or f"resolution_vs_r2__{args.observable}.png"
    Path(out_json).write_text(json.dumps(result, indent=2))
    plot_resolution_vs_r2(result, save_path=out_png)
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
