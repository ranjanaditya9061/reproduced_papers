"""Plot the ``m = 2k`` size sweep written by ``eval/sweep_size.py``.

Three figures:

1. ``bunched_mass`` -- mean bunched mass against ``m``, Boson Sampling against the determinant based
   sampler.  The one plot that says whether ``bunching_s = k/m`` is doing its job, since matching this
   quantity is what the rule is calibrated for.
2. ``mass_split`` -- the same masses decomposed into unbunched + bunched, one panel per arm.
   Stacked, because the two are a partition of 1 (``unbunched = 1 - bunched`` exactly), so a
   parts-of-a-whole form is the honest one; plotting them as two free lines would imply two
   independent measurements.
3. ``fisher_eigs`` -- five panels in a row, one per eigenvalue of the input Fisher matrix from
   largest to smallest, each comparing the two arms against ``m``.  All five panels share one
   ``y`` limit, so the panels are directly comparable and the decay across the spectrum is visible
   in the plot rather than only in the tick labels.

Series colours are categorical slots 1-2 of the reference palette; everything else is matplotlib's
default chrome.  Identity is carried by legend *and* marker shape, never colour alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_STYLE = {
    "photonic": {"label": "Boson Sampling", "color": "#2a78d6", "marker": "o"},
    "fermion": {"label": "Determinant Based Sampler", "color": "#eb6834", "marker": "s"},
}
NEUTRAL = "#d8d8d2"                    # the shared unbunched sector in figure 2

LINE = dict(linewidth=2.0, solid_capstyle="round", solid_joinstyle="round")
MARK = dict(markersize=8, markeredgecolor="black", markeredgewidth=1.0)

XLABEL = "Circuit size  m   (k = m/2)"

#: Every legend: boxed, fully opaque, black border -- so gridlines and fills never show through.
LEGEND = dict(frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")

#: x offsets so the two arms' min-max whiskers do not overlap each other
DODGE = (-0.13, 0.13)


def _style(ax, *, xlabel=None, ylabel=None, title=None):
    ax.grid(True)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")


def _series(res, arm, key):
    return [row["arms"][arm][key] for row in res["sizes"]]


def _plot_arm(ax, ms, lo, mid, hi, st, dodge=0.0):
    """Line through the mean with a capped min-max whisker at each point: ``|---o---|``.

    The whisker is the full spread over the input pool, not the SEM -- it answers "how much does this
    vary with x", which is the wider and far more honest interval.  ``dodge`` shifts the whiskers a
    little along x so the two arms' ranges do not draw on top of each other; the line itself stays on
    the true x so the trend is not distorted.
    """
    xs = [m + dodge for m in ms]
    ax.errorbar(xs, mid,
                yerr=[[a - b for a, b in zip(mid, lo)], [b - a for a, b in zip(mid, hi)]],
                fmt="none", ecolor=st["color"], elinewidth=1.4, capsize=5, capthick=1.4,
                alpha=0.75, zorder=1)
    ax.plot(ms, mid, color=st["color"], marker=st["marker"], label=st["label"], zorder=2,
            **LINE, **MARK)


def fig_bunched(res, out: Path):
    ms = [row["m"] for row in res["sizes"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for j, (arm, st) in enumerate(ARM_STYLE.items()):
        _plot_arm(ax, ms, _series(res, arm, "bunched_min"), _series(res, arm, "bunched_mean"),
                  _series(res, arm, "bunched_max"), st, dodge=DODGE[j])

    # No per-series end labels: the two curves coincide (which is the result), so two numbers at the
    # right edge either overlap or drift far enough from their markers to be ambiguous.  The quantity
    # worth labelling is the gap, once.  Exact values are in the table view the CLI prints.
    b_, f_ = _series(res, "photonic", "bunched_mean"), _series(res, "fermion", "bunched_mean")
    worst = max(range(len(ms)), key=lambda i: abs(f_[i] - b_[i]) / b_[i])
    ax.annotate(f"Worst gap {100 * (f_[worst] - b_[worst]) / b_[worst]:+.1f}%  at m = {ms[worst]}",
                (ms[worst], min(b_[worst], f_[worst])), textcoords="offset points",
                xytext=(12, -26), fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.7", linewidth=1.0))

    _style(ax, xlabel=XLABEL, ylabel="Mean bunched mass",
           title="Bunched mass tracks Boson Sampling across the sweep")
    ax.set_xticks(ms)
    ax.set_ylim(0, 1)
    ax.margins(x=0.10)
    ax.legend(loc="lower right", **LEGEND)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_mass_split(res, out: Path):
    """Unbunched + bunched as a partition of 1, one panel per arm."""
    ms = [row["m"] for row in res["sizes"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for j, (ax, (arm, st)) in enumerate(zip(axes, ARM_STYLE.items())):
        b = _series(res, arm, "bunched_mean")
        u = [1.0 - v for v in b]
        # The unbunched share is the SAME quantity in both panels, so it gets one neutral tone; the
        # arm's hue marks only the bunched share.  A tinted blue here would read as Boson Sampling
        # inside the determinant panel and miscode identity.
        ax.stackplot(ms, u, b, colors=[NEUTRAL, st["color"]],
                     labels=["Unbunched (collision-free)", "Bunched"], edgecolor="white",
                     linewidth=2.0)
        for i, (x, v) in enumerate(zip(ms, u)):
            # first/last labels would run into the y-axis ticks and the panel edge
            ha = "left" if i == 0 else ("right" if i == len(ms) - 1 else "center")
            off = 5 if i == 0 else (-5 if i == len(ms) - 1 else 0)
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(off, 7),
                        ha=ha, fontsize=8)
        _style(ax, xlabel=XLABEL,
               ylabel="Share of probability mass" if j == 0 else None, title=st["label"])
        ax.set_xticks(ms)
        ax.set_ylim(0, 1)
        ax.set_xlim(min(ms), max(ms))
        # One legend per panel, in that panel's own hue: a single shared legend would show one arm's
        # colour for "Bunched" and contradict the other panel.
        ax.legend(loc="center right", **LEGEND)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_fisher(res, out: Path):
    ms = [row["m"] for row in res["sizes"]]
    n_f = len(res["sizes"][0]["arms"]["photonic"]["eigs_mean"])

    # One y limit for all five panels, from the tallest whisker top anywhere in the figure.
    top = max(row["arms"][a]["eigs_max"][i]
              for row in res["sizes"] for a in ARM_STYLE for i in range(n_f))

    fig, axes = plt.subplots(1, n_f, figsize=(3.0 * n_f, 3.6), sharey=True)
    for i, ax in enumerate(axes):
        for j, (arm, st) in enumerate(ARM_STYLE.items()):
            col = lambda key: [row["arms"][arm][key][i] for row in res["sizes"]]  # noqa: B023
            _plot_arm(ax, ms, col("eigs_min"), col("eigs_mean"), col("eigs_max"), st,
                      dodge=DODGE[j])
        _style(ax, xlabel="m", ylabel="Eigenvalue of F" if i == 0 else None,
               title=f"$\\lambda_{{{i + 1}}}$")
        ax.set_xticks(ms)
        ax.set_ylim(0, top * 1.06)
        ax.margins(x=0.12)

    # Top-right, one entry per row.  Anchored inside the last panel rather than on the figure: a
    # figure-level legend at upper right clips on the right edge and collides with the panel title,
    # whereas the shared y limit is set by a lambda_1 whisker so this panel's upper region is empty.
    axes[-1].legend(ncol=1, loc="upper right", fontsize=9, **LEGEND)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(here / "sweep_size.json"))
    ap.add_argument("--out-dir", default=str(here / "figures"))
    args = ap.parse_args(argv)

    res = json.loads(Path(args.data).read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_bunched(res, out / "bunched_mass.png")
    fig_mass_split(res, out / "mass_split.png")
    fig_fisher(res, out / "fisher_eigs.png")
    print(f"wrote {out / 'bunched_mass.png'}\n      {out / 'mass_split.png'}\n"
          f"      {out / 'fisher_eigs.png'}")

    # a table view alongside the figures: identity and values never colour-only
    print(f"\nn_features = {res['n_features']}, n_x = {res['n_x']}, FD eps = {res['eps']}")
    print(f"{'m':>3} {'k':>2} {'outcomes':>9}" + "".join(f" {a[:4] + '_bunch':>12}"
                                                         for a in res["arms"]))
    for row in res["sizes"]:
        line = f"{row['m']:3d} {row['k']:2d} {row['arms']['photonic']['n_outcomes']:9d}"
        for a in res["arms"]:
            line += f" {row['arms'][a]['bunched_mean']:12.4f}"
        print(line)


if __name__ == "__main__":
    main()
