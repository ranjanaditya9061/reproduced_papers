"""Plot the Hellinger-distance sweep written by ``eval/sweep_delta.py``.

One figure per circuit size ``m``, ``hellinger_dist_m{m}.png``: one subplot per arm actually
present in the data (``res["arms"]``, so 2 or 3 depending on the sweep), each showing the **PDF**
of ``H(p(x0), p(x))`` over the ``n_x`` points drawn in the delta-ball, sharing one x-range so the
shapes are directly comparable. A KDE is used when SciPy is available (smoother, and the natural
choice for a density read), falling back to a normalised-histogram density otherwise so no new
hard dependency is introduced.

A second figure, ``hellinger_dist_overview.png``, stacks one such row per swept ``m`` so the whole
sweep is visible at a glance.

Series colours and labels come from ``ARM_STYLE`` (shared in spirit with ``eval/plot_size.py``, so
the analyses read as one system), extended with a ``qubit`` entry alongside ``photonic``/``fermion``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_STYLE = {
    "photonic": {"label": "Boson Sampling", "color": "#2a78d6", "marker": "o"},
    "fermion": {"label": "Determinant Based Sampler", "color": "#eb6834", "marker": "s"},
    "qubit": {"label": "Qubit IQP", "color": "#3fa34d", "marker": "^"},
}

#: Every legend: boxed, fully opaque, black border -- so gridlines and fills never show through.
LEGEND = dict(frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")

N_GRID = 256


def _style(ax, *, xlabel=None, ylabel=None, title=None):
    ax.grid(True)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")


def _density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """PDF of ``values`` evaluated on ``grid``: a Gaussian KDE, or a histogram density fallback."""
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(values)(grid)
    except ImportError:
        counts, edges = np.histogram(values, bins=max(10, len(values) // 5), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, counts, left=0.0, right=0.0)


def _plot_arms(axes, res_row, arms, *, xmax: float):
    """Draw every arm's Hellinger-distance PDF for one sweep row onto ``axes`` (one per arm)."""
    grid = np.linspace(0.0, max(xmax, 1e-12), N_GRID)
    for ax, arm in zip(axes, arms):
        st = ARM_STYLE[arm]
        values = np.asarray(res_row["arms"][arm]["hellinger"], dtype=float)
        if np.ptp(values) < 1e-12:                    # delta=0 / degenerate: a point mass at 0
            ax.axvline(float(values.mean()) if len(values) else 0.0, color=st["color"],
                       linewidth=2.0)
        else:
            density = _density(values, grid)
            ax.fill_between(grid, density, color=st["color"], alpha=0.35)
            ax.plot(grid, density, color=st["color"], linewidth=2.0)
        ax.set_xlim(0.0, grid[-1])
        _style(ax, xlabel="Hellinger distance  H(p(x0), p(x))",
               ylabel="Density", title=f"{st['label']}  (m={res_row['m']})")


def fig_per_m(res, out_dir: Path):
    arms = res["arms"]
    xmax = max(max(row["arms"][a]["hellinger"], default=0.0)
              for row in res["sizes"] for a in arms)
    for row in res["sizes"]:
        fig, axes = plt.subplots(1, len(arms), figsize=(4.8 * len(arms), 4.0), sharey=True)
        if len(arms) == 1:
            axes = [axes]
        _plot_arms(axes, row, arms, xmax=xmax)
        fig.suptitle(f"Hellinger distance from p(x0), delta={res['delta']:g}, n_x={res['n_x']}")
        fig.tight_layout()
        fig.savefig(out_dir / f"hellinger_dist_m{row['m']}.png", dpi=200)
        plt.close(fig)


def fig_overview(res, out: Path):
    arms = res["arms"]
    ms = [row["m"] for row in res["sizes"]]
    xmax = max(max(row["arms"][a]["hellinger"], default=0.0)
              for row in res["sizes"] for a in arms)
    fig, axes = plt.subplots(len(ms), len(arms),
                             figsize=(4.8 * len(arms), 3.4 * len(ms)), sharex=True, sharey=True)
    if len(ms) == 1:
        axes = axes.reshape(1, len(arms))
    if len(arms) == 1:
        axes = axes.reshape(len(ms), 1)
    for r, row in enumerate(res["sizes"]):
        _plot_arms(axes[r], row, arms, xmax=xmax)
    fig.suptitle(f"Hellinger distance from p(x0) across the m = 2k sweep  "
                 f"(delta={res['delta']:g}, n_x={res['n_x']})")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", type=float, default=0.01,
                     help="ball radius used by sweep_delta.py; selects results/sweep_delta_<delta>.json")
    ap.add_argument("--data", default=None,
                     help="explicit path to a sweep_delta output; overrides --delta")
    ap.add_argument("--out-dir", default=str(here / "figures"))
    args = ap.parse_args(argv)

    if args.data is not None:
        data_path = Path(args.data)
    elif args.delta is not None:
        data_path = here / f"results/sweep_delta_{args.delta:.0e}.json"
    else:
        data_path = here / "sweep_delta.json"

    res = json.loads(data_path.read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_per_m(res, out)
    fig_overview(res, out / f"results/hellinger_dist_overview{args.delta:.0e}.png")
    print(f"wrote {out / 'hellinger_dist_m*.png'}\n      {out / 'hellinger_dist_overview.png'}")

    print(f"\nn_features = {res['n_features']}, n_x = {res['n_x']}, delta = {res['delta']}")
    print(f"{'m':>3} {'k':>2}" + "".join(f" {a[:4] + '_hel':>18}" for a in res["arms"]))
    for row in res["sizes"]:
        line = f"{row['m']:3d} {row['k']:2d}"
        for a in res["arms"]:
            r = row["arms"][a]
            line += f" {r['hellinger_mean']:8.4f}+-{r['hellinger_sem']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
