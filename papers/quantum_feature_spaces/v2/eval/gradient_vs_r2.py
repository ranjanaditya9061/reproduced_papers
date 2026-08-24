"""Gradient vs. R^2, and normalized gradient vs. R^2 -- two scatter plots, one point per config,
mean over the sampled pool points with an error bar spanning [min, max].

    python eval/gradient_vs_r2.py --configs configs/eval/photonic_encoding/*.yaml \
        --observable parity

**Mean, not a bare minimum -- and why the spread matters too.**  A single sampled point's gradient
being small does not by itself mean the observable is hard: even a healthy, non-plateaued
observable has points near a local extremum of ``<O>_x`` where the instantaneous slope is genuinely
close to zero (the way ``d(sin x)/dx = 0`` at ``x = pi/2`` without ``sin`` being flat anywhere else).
Taking the bare min over ``n_x`` sampled points is an order statistic that is likely to land small
even in a healthy landscape, so it conflates "this one point happens to sit near a local flat spot"
with "the whole landscape is flat" -- the standard barren-plateau criterion (Makarovskiy et al.'s
gradient second moment, ``E_x[G^2]``) is closer to a MEAN over the landscape, not a minimum. The
min/max range is kept as the error bar precisely so a reader can see how much a single point's
reading would have misled on its own, rather than dropping that information the way a bare min
does.

**Plotting only.**  Every number here comes from :func:`metrics.gradient.cached_gradient`
(cache-hit-or-compute) and :func:`learner.cache.cached_fit` (via :func:`learner.auto.run_config`,
same cache-hit-or-compute discipline) -- no simulation, no observable scoring, no gradient
computation happens in this file itself.
"""

from __future__ import annotations

import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/gradient_vs_r2.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _best_r2(cfg_path, observable: str, *, n_seeds: int, out_root: str, scores_root: str) -> float:
    """Max over :data:`learner.auto.DEFAULT_SWEEP_LEARNERS`, each averaged over ``n_seeds``
    reseeded splits -- the same convention as :func:`eval.best_of_grid.sweep_best_of_grid`."""
    from learner.auto import DEFAULT_SWEEP_LEARNERS, run_config

    means = []
    for lname, kwargs in DEFAULT_SWEEP_LEARNERS:
        scores = []
        for seed in range(int(n_seeds)):
            res = run_config(cfg_path, observable, lname, out_root=out_root,
                             scores_root=scores_root, split_seed=seed, seed=seed, **kwargs)
            scores.append(res["r2"])
        if scores:
            means.append(statistics.mean(scores))
    return max(means) if means else float("nan")


#: Sentinel distinct from ``None`` (which now means "all rows" to :func:`metrics.gradient.
#: cached_gradient`) so this module's own default stays "use that function's default" (100 points)
#: without silently falling back to the full, expensive pool.
_USE_GRADIENT_DEFAULT = object()


def collect(cfg_paths: list[str | Path], observable: str, *, n_x=_USE_GRADIENT_DEFAULT,
           n_seeds: int = 10, out_root: str = "datasets", scores_root: str = "scores") -> dict:
    """One row per config: ``mean||g||`` (with its min/max range), the same for the normalized
    gradient, and ``R^2`` -- each config is one point in the two scatter plots, with the
    min-to-max spread across the ``n_x`` sampled points shown as an error bar.  A bare minimum
    alone conflates "this observable is flat everywhere" (a genuine plateau) with "one sampled
    point happened to land near a local extremum of an otherwise healthy landscape" (not a
    plateau) -- the mean is the closer read to the standard barren-plateau criterion
    (``E_x[||g||]`` small, cf. Makarovskiy et al.'s gradient-second-moment definition), and the
    min/max range shows how much a single point's reading could mislead on its own.  ``n_x``
    defaults to :func:`metrics.gradient.cached_gradient`'s own default (100 points, not the full
    pool); pass ``n_x=None`` explicitly for every row.
    """
    import statistics

    from metrics.gradient import cached_gradient

    gradient_kwargs = {} if n_x is _USE_GRADIENT_DEFAULT else {"n_x": n_x}
    labels, g_mean, g_min, g_max, gn_mean, gn_min, gn_max, r2 = [], [], [], [], [], [], [], []
    for cfg_path in cfg_paths:
        grad = cached_gradient(cfg_path, observable, out_root=out_root,
                               scores_root=scores_root, **gradient_kwargs)
        r2_val = _best_r2(cfg_path, observable, n_seeds=n_seeds, out_root=out_root,
                          scores_root=scores_root)
        labels.append(Path(cfg_path).stem)
        g_mean.append(statistics.mean(grad["g_norm"]))
        g_min.append(min(grad["g_norm"]))
        g_max.append(max(grad["g_norm"]))
        gn_mean.append(statistics.mean(grad["g_norm_normalized"]))
        gn_min.append(min(grad["g_norm_normalized"]))
        gn_max.append(max(grad["g_norm_normalized"]))
        r2.append(r2_val)
        print(f"{labels[-1]:24s} ||g||={g_mean[-1]:.4g} [{g_min[-1]:.4g}, {g_max[-1]:.4g}]  "
             f"||g||_norm={gn_mean[-1]:.4g} [{gn_min[-1]:.4g}, {gn_max[-1]:.4g}]  "
             f"R^2={r2_val:.4g}")

    return {"configs": labels, "mean_g_norm": g_mean, "min_g_norm": g_min, "max_g_norm": g_max,
           "mean_g_norm_normalized": gn_mean, "min_g_norm_normalized": gn_min,
           "max_g_norm_normalized": gn_max, "r2": r2, "observable": observable, "n_x": n_x,
           "n_seeds": n_seeds}


def _asymmetric_xerr(mean, lo, hi):
    """``(2, N)`` array for ``ax.errorbar(xerr=...)``: distance from mean down to ``lo`` and up to
    ``hi`` -- clamped at 0 so float round-off (mean landing a hair outside [lo, hi]) never gives
    ``errorbar`` a negative width."""
    import numpy as np

    mean, lo, hi = np.asarray(mean), np.asarray(lo), np.asarray(hi)
    return np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])


def plot_gradient_vs_r2(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """Two panels, side by side: mean||g|| vs. R^2 (unbounded, own scale) and mean normalized
    ||g|| vs. R^2 (scale-invariant, comparable across configs), each point's horizontal error bar
    spanning [min, max] over the sampled pool points -- kept as separate axes rather than overlaid,
    same reasoning as :func:`eval_legacy.gradient_vs_hardness.plot_gradient`'s own split: the raw
    and normalized gradients are not on comparable scales and plotting them together invites
    reading a scale difference as a hardness difference.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    xerr1 = _asymmetric_xerr(result["mean_g_norm"], result["min_g_norm"], result["max_g_norm"])
    ax1.errorbar(result["mean_g_norm"], result["r2"], xerr=xerr1, fmt="o", markersize=6,
                alpha=0.7, color="tab:blue", ecolor="tab:blue", elinewidth=1, capsize=3)
    for x, y, label in zip(result["mean_g_norm"], result["r2"], result["configs"]):
        ax1.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("||dT/dx|| over pool (mean, error bar = min-max)")
    ax1.set_ylabel("R^2 (max over learners)")
    ax1.set_title("Raw gradient vs. R^2")
    ax1.set_ylim(-0.05, 1.02)
    ax1.grid(alpha=0.3, which="both")

    xerr2 = _asymmetric_xerr(result["mean_g_norm_normalized"], result["min_g_norm_normalized"],
                             result["max_g_norm_normalized"])
    ax2.errorbar(result["mean_g_norm_normalized"], result["r2"], xerr=xerr2, fmt="o", markersize=6,
                alpha=0.7, color="tab:orange", ecolor="tab:orange", elinewidth=1, capsize=3)
    for x, y, label in zip(result["mean_g_norm_normalized"], result["r2"], result["configs"]):
        ax2.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel("||dT/dx|| / sqrt(Var_circ) over pool (mean, error bar = min-max)")
    ax2.set_ylabel("R^2 (max over learners)")
    ax2.set_title("Normalized gradient vs. R^2")
    ax2.set_ylim(-0.05, 1.02)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(f"observable={result['observable']}  n_x={result['n_x'] or 'all'}  "
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

    ap = argparse.ArgumentParser(description="Min gradient / min normalized gradient vs. R^2, "
                                             "one point per config.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--observable", required=True)
    ap.add_argument("--n-x", type=int, default=100, help="subsample this many rows per config "
                                                         "(default 100; pass 0 for all rows)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    n_x = None if args.n_x == 0 else args.n_x
    result = collect(args.configs, args.observable, n_x=n_x, n_seeds=args.n_seeds,
                     out_root=args.out_root, scores_root=args.scores_root)

    out_json = args.out_json or f"gradient_vs_r2__{args.observable}.json"
    out_png = args.out_png or f"gradient_vs_r2__{args.observable}.png"
    Path(out_json).write_text(json.dumps(result, indent=2))
    plot_gradient_vs_r2(result, save_path=out_png)
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
