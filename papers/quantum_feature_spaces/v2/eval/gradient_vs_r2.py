"""Gradient vs. R^2, normalized gradient vs. R^2, and Var_circ vs. R^2 -- three scatter plots, one
point per config, mean over the sampled pool points with an error bar spanning [min, max] (Var_circ
is a single pool-level scalar, so it has no such spread).

    python eval/gradient_vs_r2.py --configs configs/eval/photonic_encoding/*.yaml \
        --observable parity

**Why all three, not just the normalized one.**  ``||g|| = ||g_norm|| * sqrt(Var_circ)`` by
construction, so a raw-``||g||``-vs-R^2 correlation is ambiguous between two different stories: (a)
it is riding on ``Var_circ`` alone (bigger-scale observables/configs just happen to also be harder,
an artifact of each observable's arbitrary output units -- rescaling any observable ``O -> c*O``
scales ``||g||`` and ``Var_circ`` but leaves ``||g_norm||`` and R^2 exactly fixed, so this
possibility cannot be ruled out from ``||g||`` alone), or (b) it reflects genuine shape -- the label
function oscillates fast relative to its own spread, which is what ``||g_norm||`` isolates.
Plotting ``Var_circ`` alongside the other two, and printing all three Pearson correlations against
R^2, is the actual test: if ``Var_circ`` alone correlates about as strongly as raw ``||g||`` does,
story (a) cannot be ruled out; if ``||g_norm||`` correlates comparably to raw ``||g||`` while
``Var_circ`` alone does not, that isolates shape (story (b)) as the real driver.

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


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient, no numpy/scipy dependency for a two-column, N<1000 case."""
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return cov / (va ** 0.5 * vb ** 0.5)


def collect(cfg_paths: list[str | Path], observable: str, *, n_x=_USE_GRADIENT_DEFAULT,
           n_seeds: int = 10, out_root: str = "datasets", scores_root: str = "scores") -> dict:
    """One row per config: ``mean||g||`` (with its min/max range), the same for the normalized
    gradient, ``Var_circ`` (a single scalar per config -- see :func:`~metrics.circuit_variance.
    cached_circuit_variance`, already computed as a side effect of :func:`~metrics.gradient.
    cached_gradient`'s own normalization, so this costs nothing extra), and ``R^2`` -- each config
    is one point in the three scatter plots, with the min-to-max spread across the ``n_x`` sampled
    points shown as an error bar on the two gradient panels.  A bare minimum alone conflates "this
    observable is flat everywhere" (a genuine plateau) with "one sampled point happened to land near
    a local extremum of an otherwise healthy landscape" (not a plateau) -- the mean is the closer
    read to the standard barren-plateau criterion (``E_x[||g||]`` small, cf. Makarovskiy et al.'s
    gradient-second-moment definition), and the min/max range shows how much a single point's
    reading could mislead on its own.  ``n_x`` defaults to :func:`metrics.gradient.cached_gradient`'s
    own default (100 points, not the full pool); pass ``n_x=None`` explicitly for every row.
    """
    import statistics

    from metrics.gradient import cached_gradient

    gradient_kwargs = {} if n_x is _USE_GRADIENT_DEFAULT else {"n_x": n_x}
    labels, g_mean, g_min, g_max = [], [], [], []
    gn_mean, gn_min, gn_max, var_circ, r2 = [], [], [], [], []
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
        var_circ.append(grad["sqrt_var_circ"] ** 2)
        r2.append(r2_val)
        print(f"{labels[-1]:24s} ||g||={g_mean[-1]:.4g} [{g_min[-1]:.4g}, {g_max[-1]:.4g}]  "
             f"||g||_norm={gn_mean[-1]:.4g} [{gn_min[-1]:.4g}, {gn_max[-1]:.4g}]  "
             f"Var_circ={var_circ[-1]:.4g}  R^2={r2_val:.4g}")

    corr_g = _pearson(g_mean, r2)
    corr_gn = _pearson(gn_mean, r2)
    corr_var = _pearson(var_circ, r2)
    print()
    print(f"Pearson r vs R^2:  ||g|| (mean)={corr_g:+.3f}   ||g_norm|| (mean)={corr_gn:+.3f}   "
         f"Var_circ={corr_var:+.3f}")
    print("(if Var_circ's |r| is close to ||g||'s, the raw ||g|| correlation may just be riding "
         "on Var_circ, i.e. an artifact of the observable's own scale -- see the module docstring)")

    return {"configs": labels, "mean_g_norm": g_mean, "min_g_norm": g_min, "max_g_norm": g_max,
           "mean_g_norm_normalized": gn_mean, "min_g_norm_normalized": gn_min,
           "max_g_norm_normalized": gn_max, "var_circ": var_circ, "r2": r2,
           "corr_g_mean_r2": corr_g, "corr_g_norm_mean_r2": corr_gn, "corr_var_circ_r2": corr_var,
           "observable": observable, "n_x": n_x, "n_seeds": n_seeds}


def collect_observables(cfg_path: str | Path, observables: list[str], *, n_x=_USE_GRADIENT_DEFAULT,
                        n_seeds: int = 10, out_root: str = "datasets",
                        scores_root: str = "scores") -> dict:
    """The transpose of :func:`collect`: ONE circuit config fixed, one row per OBSERVABLE instead
    of one row per config -- the section-2 axis (:mod:`eval.eval_obs`'s fixed-circuit,
    varying-observable picture) applied to the gradient-vs-R^2 screen instead of to the plain
    best-of-learners R^2 bar chart.  Same three-panel plot, same Pearson-correlation printout,
    same underlying :func:`metrics.gradient.cached_gradient`/:func:`learner.auto.run_config`
    cache-hit-or-compute calls -- only which axis is held fixed changes, so this reuses every
    other function in this module unmodified.

    Returns the same shape as :func:`collect`, with ``"configs"`` renamed ``"observables"`` (still
    aligned 1:1 with every other list) and ``"config"`` (singular, the one fixed circuit) in place
    of ``"observable"``.
    """
    import statistics

    from metrics.gradient import cached_gradient

    gradient_kwargs = {} if n_x is _USE_GRADIENT_DEFAULT else {"n_x": n_x}
    labels, g_mean, g_min, g_max = [], [], [], []
    gn_mean, gn_min, gn_max, var_circ, r2 = [], [], [], [], []
    for obs in observables:
        grad = cached_gradient(cfg_path, obs, out_root=out_root, scores_root=scores_root,
                               **gradient_kwargs)
        r2_val = _best_r2(cfg_path, obs, n_seeds=n_seeds, out_root=out_root,
                          scores_root=scores_root)
        labels.append(obs)
        g_mean.append(statistics.mean(grad["g_norm"]))
        g_min.append(min(grad["g_norm"]))
        g_max.append(max(grad["g_norm"]))
        gn_mean.append(statistics.mean(grad["g_norm_normalized"]))
        gn_min.append(min(grad["g_norm_normalized"]))
        gn_max.append(max(grad["g_norm_normalized"]))
        var_circ.append(grad["sqrt_var_circ"] ** 2)
        r2.append(r2_val)
        print(f"{labels[-1]:32s} ||g||={g_mean[-1]:.4g} [{g_min[-1]:.4g}, {g_max[-1]:.4g}]  "
             f"||g||_norm={gn_mean[-1]:.4g} [{gn_min[-1]:.4g}, {gn_max[-1]:.4g}]  "
             f"Var_circ={var_circ[-1]:.4g}  R^2={r2_val:.4g}")

    corr_g = _pearson(g_mean, r2)
    corr_gn = _pearson(gn_mean, r2)
    corr_var = _pearson(var_circ, r2)
    print()
    print(f"Pearson r vs R^2:  ||g|| (mean)={corr_g:+.3f}   ||g_norm|| (mean)={corr_gn:+.3f}   "
         f"Var_circ={corr_var:+.3f}")

    return {"configs": labels, "mean_g_norm": g_mean, "min_g_norm": g_min, "max_g_norm": g_max,
           "mean_g_norm_normalized": gn_mean, "min_g_norm_normalized": gn_min,
           "max_g_norm_normalized": gn_max, "var_circ": var_circ, "r2": r2,
           "corr_g_mean_r2": corr_g, "corr_g_norm_mean_r2": corr_gn, "corr_var_circ_r2": corr_var,
           "observable": f"config={Path(cfg_path).stem}", "n_x": n_x, "n_seeds": n_seeds}


def _asymmetric_xerr(mean, lo, hi):
    """``(2, N)`` array for ``ax.errorbar(xerr=...)``: distance from mean down to ``lo`` and up to
    ``hi`` -- clamped at 0 so float round-off (mean landing a hair outside [lo, hi]) never gives
    ``errorbar`` a negative width."""
    import numpy as np

    mean, lo, hi = np.asarray(mean), np.asarray(lo), np.asarray(hi)
    return np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])


def plot_gradient_vs_r2(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """Three panels, side by side: mean||g|| vs. R^2 (unbounded, own scale), mean normalized ||g||
    vs. R^2 (scale-invariant, comparable across configs), and Var_circ vs. R^2 (the scale
    ingredient alone, no gradient) -- each gradient point's horizontal error bar spans [min, max]
    over the sampled pool points; ``Var_circ`` is a single pool-level scalar per config, so it has
    none.  Kept as three separate axes rather than overlaid, same reasoning as
    :func:`eval_legacy.gradient_vs_hardness.plot_gradient`'s own split: none of the three are on a
    directly comparable scale, and plotting them together invites reading a scale difference as a
    hardness difference.  Comparing panel 1 against panel 3 is the actual test for whether raw
    ||g||'s correlation with R^2 is riding on Var_circ alone -- see the module docstring.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))

    xerr1 = _asymmetric_xerr(result["mean_g_norm"], result["min_g_norm"], result["max_g_norm"])
    ax1.errorbar(result["mean_g_norm"], result["r2"], xerr=xerr1, fmt="o", markersize=6,
                alpha=0.7, color="tab:blue", ecolor="tab:blue", elinewidth=1, capsize=3)
    for x, y, label in zip(result["mean_g_norm"], result["r2"], result["configs"]):
        ax1.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("||dT/dx|| over pool (mean, error bar = min-max)")
    ax1.set_ylabel("R^2 (max over learners)")
    ax1.set_title(f"Raw gradient vs. R^2  (r={result.get('corr_g_mean_r2', float('nan')):+.2f})")
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
    ax2.set_title(f"Normalized gradient vs. R^2  "
                 f"(r={result.get('corr_g_norm_mean_r2', float('nan')):+.2f})")
    ax2.set_ylim(-0.05, 1.02)
    ax2.grid(alpha=0.3, which="both")

    if "var_circ" in result:
        ax3.scatter(result["var_circ"], result["r2"], s=30, alpha=0.7, color="tab:green")
        for x, y, label in zip(result["var_circ"], result["r2"], result["configs"]):
            ax3.annotate(label, (x, y), fontsize=7, alpha=0.7)
        ax3.set_xscale("log")
        ax3.set_xlabel("Var_circ = Var_x[<O>_x]  (scale ingredient alone)")
        ax3.set_ylabel("R^2 (max over learners)")
        ax3.set_title(f"Var_circ vs. R^2  (r={result.get('corr_var_circ_r2', float('nan')):+.2f})")
        ax3.set_ylim(-0.05, 1.02)
        ax3.grid(alpha=0.3, which="both")

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

    ap = argparse.ArgumentParser(description="Min gradient / min normalized gradient vs. R^2. "
                                             "--configs (many configs, one observable) and "
                                             "--config (one config, many observables) are "
                                             "mutually exclusive.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", nargs="+", help="many configs, --observable fixed -- one "
                                                     "point per config")
    group.add_argument("--config", help="one config, --observables varying -- one point per "
                                        "observable (the eval.eval_obs axis)")
    ap.add_argument("--observable", help="required with --configs")
    ap.add_argument("--observables", nargs="+", help="required with --config; defaults to "
                                                      "eval.eval_obs.DEFAULT_FAMILIES flattened "
                                                      "if omitted")
    ap.add_argument("--n-x", type=int, default=100, help="subsample this many rows per config "
                                                         "(default 100; pass 0 for all rows)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    n_x = None if args.n_x == 0 else args.n_x

    if args.configs:
        if not args.observable:
            ap.error("--observable is required with --configs")
        result = collect(args.configs, args.observable, n_x=n_x, n_seeds=args.n_seeds,
                         out_root=args.out_root, scores_root=args.scores_root)
        tag = args.observable
    else:
        observables = args.observables
        if not observables:
            from eval.eval_obs import _flatten_families, DEFAULT_FAMILIES
            observables = _flatten_families(DEFAULT_FAMILIES)
        result = collect_observables(args.config, observables, n_x=n_x, n_seeds=args.n_seeds,
                                     out_root=args.out_root, scores_root=args.scores_root)
        tag = Path(args.config).stem

    out_json = args.out_json or f"gradient_vs_r2__{tag}.json"
    out_png = args.out_png or f"gradient_vs_r2__{tag}.png"
    Path(out_json).write_text(json.dumps(result, indent=2))
    plot_gradient_vs_r2(result, save_path=out_png)
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
