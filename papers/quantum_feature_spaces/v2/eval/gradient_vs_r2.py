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


def _rank(a: list[float]) -> list[float]:
    """Average (fractional) ranks, 1-indexed, ties broken by the mean rank of the tied block --
    the standard convention :func:`_spearman` needs (matches ``scipy.stats.rankdata`` default)."""
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for idx in order[i:j + 1]:
            ranks[idx] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation -- Pearson correlation on ranks, no numpy/scipy dependency.
    Unlike Pearson, this only assumes a monotonic (not linear) relationship and is insensitive to
    one point's absolute scale, so it survives a multi-order-of-magnitude outlier (e.g.
    ``connected_paritynumloops_pair``'s ``Var_circ`` sitting ~280x every other observable's) without
    that point dominating the statistic the way it would a least-squares fit or even Pearson's own
    second-moment sums."""
    if len(a) < 2:
        return float("nan")
    return _pearson(_rank(a), _rank(b))


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

    corr_g, scorr_g = _pearson(g_mean, r2), _spearman(g_mean, r2)
    corr_gn, scorr_gn = _pearson(gn_mean, r2), _spearman(gn_mean, r2)
    corr_var, scorr_var = _pearson(var_circ, r2), _spearman(var_circ, r2)
    print()
    print(f"Pearson r vs R^2:   ||g|| (mean)={corr_g:+.3f}   ||g_norm|| (mean)={corr_gn:+.3f}   "
         f"Var_circ={corr_var:+.3f}")
    print(f"Spearman r vs R^2:  ||g|| (mean)={scorr_g:+.3f}   ||g_norm|| (mean)={scorr_gn:+.3f}   "
         f"Var_circ={scorr_var:+.3f}")
    print("(if Var_circ's |r| is close to ||g||'s, the raw ||g|| correlation may just be riding "
         "on Var_circ, i.e. an artifact of the observable's own scale -- see the module docstring)")

    return {"configs": labels, "mean_g_norm": g_mean, "min_g_norm": g_min, "max_g_norm": g_max,
           "mean_g_norm_normalized": gn_mean, "min_g_norm_normalized": gn_min,
           "max_g_norm_normalized": gn_max, "var_circ": var_circ, "r2": r2,
           "corr_g_mean_r2": corr_g, "corr_g_norm_mean_r2": corr_gn, "corr_var_circ_r2": corr_var,
           "spearman_g_mean_r2": scorr_g, "spearman_g_norm_mean_r2": scorr_gn,
           "spearman_var_circ_r2": scorr_var,
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

    corr_g, scorr_g = _pearson(g_mean, r2), _spearman(g_mean, r2)
    corr_gn, scorr_gn = _pearson(gn_mean, r2), _spearman(gn_mean, r2)
    corr_var, scorr_var = _pearson(var_circ, r2), _spearman(var_circ, r2)
    print()
    print(f"Pearson r vs R^2:   ||g|| (mean)={corr_g:+.3f}   ||g_norm|| (mean)={corr_gn:+.3f}   "
         f"Var_circ={corr_var:+.3f}")
    print(f"Spearman r vs R^2:  ||g|| (mean)={scorr_g:+.3f}   ||g_norm|| (mean)={scorr_gn:+.3f}   "
         f"Var_circ={scorr_var:+.3f}")

    return {"configs": labels, "mean_g_norm": g_mean, "min_g_norm": g_min, "max_g_norm": g_max,
           "mean_g_norm_normalized": gn_mean, "min_g_norm_normalized": gn_min,
           "max_g_norm_normalized": gn_max, "var_circ": var_circ, "r2": r2,
           "corr_g_mean_r2": corr_g, "corr_g_norm_mean_r2": corr_gn, "corr_var_circ_r2": corr_var,
           "spearman_g_mean_r2": scorr_g, "spearman_g_norm_mean_r2": scorr_gn,
           "spearman_var_circ_r2": scorr_var,
           "observable": f"config={Path(cfg_path).stem}", "n_x": n_x, "n_seeds": n_seeds}


def _asymmetric_xerr(mean, lo, hi):
    """``(2, N)`` array for ``ax.errorbar(xerr=...)``: distance from mean down to ``lo`` and up to
    ``hi`` -- clamped at 0 so float round-off (mean landing a hair outside [lo, hi]) never gives
    ``errorbar`` a negative width."""
    import numpy as np

    mean, lo, hi = np.asarray(mean), np.asarray(lo), np.asarray(hi)
    return np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])


#: Display label for a circuit config stem that isn't already readable title-cased -- mirrors
#: :mod:`eval.eval_obs`'s ``OBSERVABLE_LABELS`` convention, but for the config-stem side of the
#: title instead of the observable side.
CIRCUIT_LABELS: dict[str, str] = {
    "phase": "Photonic Phase Encoding",
}


def _axis_title(result: dict) -> str:
    """``"Circuit: ..., Observable: ..."`` -- whichever axis was held fixed gets named, the swept
    axis reads "Variable". :func:`collect` (circuit axis) stores the fixed observable name in
    ``result["observable"]``; :func:`collect_observables` (observable axis) stores
    ``f"config={stem}"`` there instead -- that prefix is how the two cases are told apart, since
    both funnel into the same ``"observable"`` dict key. The config stem is looked up in
    :data:`CIRCUIT_LABELS` for a human-readable name, falling back to the raw stem if absent.
    """
    obs_field = result.get("observable", "")
    if isinstance(obs_field, str) and obs_field.startswith("config="):
        circuit = obs_field[len("config="):]
        circuit = CIRCUIT_LABELS.get(circuit, circuit)
        return f"Circuit: {circuit}, Observable: Variable"
    return f"Circuit: Variable, Observable: {obs_field}"


def plot_gradient_vs_r2(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """Two panels, side by side, BOTH with R^2 on the x-axis and log-scaled y: left panel overlays
    Gradient (``tab:blue``) and Circuit Variance (``tab:green``) as two series on the SAME x-axis
    (R^2) but each with its OWN y-axis (Gradient on the left, Circuit Variance on a ``twinx()``
    right axis) -- different units/scales, so a shared axis squashed one series against the other;
    right panel is Normalized Gradient (``tab:orange``) alone.

    **Why every y-axis here is log-scaled.** Raw ``||g||`` and ``Var_circ`` can span multiple
    orders of magnitude across observables purely from each observable's own arbitrary output scale
    (e.g. ``connected_paritynumloops_pair``'s ``Var_circ`` measured ~280x every other observable's
    in one run) -- log-y keeps that spread readable instead of collapsing every other point to the
    axis floor. ``||g_norm||`` removes most of that scale dependence by construction
    (``||g|| = ||g_norm|| * sqrt(Var_circ)``, see the module docstring) but can still range over an
    order of magnitude or more across observables, so its panel is log-scaled too for the same
    reason and for visual consistency with the left panel.

    **No fit line.** A least-squares fit line (linear or log-space) is a stronger claim than this
    data supports -- these are three-to-fourteen points per config/observable sweep, and a fit
    invites reading a slope as if it were a physical rate. Only the scatter and the correlation
    coefficients are shown.

    **Pearson AND Spearman, not just Pearson.** Pearson only detects a *linear* relationship and is
    sensitive to any single point's absolute scale; Spearman (rank correlation) only assumes a
    monotonic relationship and is insensitive to how far out an outlier sits, only its rank --
    exactly the property that matters here given the Circuit-Variance outlier described above.
    Both are computed once in :func:`collect`/:func:`collect_observables` (``corr_*_r2``/
    ``spearman_*_r2`` keys) and simply displayed here, not recomputed.

    Title is normal (default) font size -- see :func:`_axis_title` for the "Circuit: ...,
    Observable: ..." template, which states which axis was held fixed and which one swept.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    title = _axis_title(result)

    def _scatter(ax, r2_vals, y_vals, yerr, color):
        ax.errorbar(r2_vals, y_vals, yerr=yerr, fmt="o", markersize=6,
                   alpha=0.7, color=color, ecolor=color, elinewidth=1, capsize=3)

    def _corr_text(corr_key, spearman_key):
        pearson_r = result.get(corr_key, float("nan"))
        spearman_r = result.get(spearman_key, float("nan"))
        return f"Pearson = {pearson_r:+.3f}\nSpearman = {spearman_r:+.3f}"

    #: Coefficient-of-Determination axis label -- the house style already used by
    #: eval.best_of_grid/eval.eval_obs/eval.size_r2_multi, single line here (no "\n").
    R2_LABEL = "Coefficient of Determination ($R^2$)"

    #: Single font size for EVERY piece of text in the figure -- title, axis labels, tick labels,
    #: and the boxed correlation text -- so nothing reads as more or less important by size alone;
    #: matplotlib's defaults would otherwise make the suptitle larger than the axis labels and the
    #: boxed text smaller than both.
    FONT_SIZE = 10

    yerr1 = _asymmetric_xerr(result["mean_g_norm"], result["min_g_norm"], result["max_g_norm"])
    _scatter(ax1, result["r2"], result["mean_g_norm"], yerr1, "tab:blue")
    ax1.set_yscale("log")
    ax1.set_xlabel(R2_LABEL, fontsize=FONT_SIZE)
    ax1.set_ylabel("Gradient", color="tab:blue", fontsize=FONT_SIZE)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=FONT_SIZE)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE)
    ax1.grid(alpha=0.3, which="both")

    #: Var_circ shares the x-axis (R^2) but gets its OWN y-axis on the right -- different units
    #: and a different natural scale from the gradient, so squashing them onto one shared axis
    #: (the original combined-panel attempt) made both series hard to read.
    ax1_var = ax1.twinx()
    _scatter(ax1_var, result["r2"], result["var_circ"], None, "tab:green")
    ax1_var.set_yscale("log")
    ax1_var.set_ylabel("Circuit Variance", color="tab:green", fontsize=FONT_SIZE)
    ax1_var.tick_params(axis="y", labelcolor="tab:green", labelsize=FONT_SIZE)

    #: Two black-bordered boxes ABOVE the axes (y slightly > 1 in axes-fraction coords, so they sit
    #: outside the plot area rather than over any data point) -- Gradient at top-left, Circuit
    #: Variance at top-right, matching the left-Gradient/right-Var_circ axis convention already
    #: used for the twin y-axes. Each box is identified by its text color, not a written-out name
    #: or a matplotlib legend. A small y-offset above 1.0 gives breathing room between the boxes
    #: and the plot border.
    ax1.text(0.0, 1.04, _corr_text("corr_g_mean_r2", "spearman_g_mean_r2"), transform=ax1.transAxes,
            fontsize=FONT_SIZE, va="bottom", ha="left", color="tab:blue",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
            zorder=10, clip_on=False)
    ax1.text(1.0, 1.04, _corr_text("corr_var_circ_r2", "spearman_var_circ_r2"), transform=ax1.transAxes,
            fontsize=FONT_SIZE, va="bottom", ha="right", color="tab:green",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
            zorder=10, clip_on=False)

    yerr2 = _asymmetric_xerr(result["mean_g_norm_normalized"], result["min_g_norm_normalized"],
                             result["max_g_norm_normalized"])
    _scatter(ax2, result["r2"], result["mean_g_norm_normalized"], yerr2, "tab:orange")
    ax2.set_yscale("log")
    ax2.set_xlabel(R2_LABEL, fontsize=FONT_SIZE)
    ax2.set_ylabel("Normalized Gradient", fontsize=FONT_SIZE)
    ax2.tick_params(axis="both", labelsize=FONT_SIZE)
    ax2.grid(alpha=0.3, which="both")
    ax2.text(0.0, 1.04, _corr_text("corr_g_norm_mean_r2", "spearman_g_norm_mean_r2"),
            transform=ax2.transAxes, fontsize=FONT_SIZE, va="bottom", ha="left", color="tab:orange",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
            clip_on=False)

    fig.suptitle(title, fontsize=FONT_SIZE)
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
    ap.add_argument("--exclude-observables", nargs="*", default=["osc"],
                    help="drop these observables (exact match) from --observables before "
                         "plotting -- default drops 'osc', whose Var_circ/gradient sit orders of "
                         "magnitude above every other observable and dominate the panels; pass "
                         "--exclude-observables with no values to keep everything")
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
        excluded = set(args.exclude_observables or [])
        observables = [obs for obs in observables if obs not in excluded]
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
