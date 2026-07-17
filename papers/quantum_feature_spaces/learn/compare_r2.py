"""Overlay saved :mod:`learn.scaling` experiments as test-R^2 vs #features learning curves.

Complementary view to :mod:`learn.compare_scaling`: instead of "min #features to reach a
threshold", this plots the **whole curve** -- test R^2 (y) against the explicit feature budget
``n_feat`` (x) -- reconstructed from each run's saved ``(order x degree)`` grid.  One subplot per
swept complexity value (e.g. ``m``), one line per saved experiment, averaged over the experiment's
average group.

    # qubit vs photonic learning curves, a subplot per m
    python -m learn.compare_r2 \\
        scaling/scaling_1_qubit_m.json scaling/scaling_6_photonic_m.json \\
        --labels qubit photonic --save img/compare_r2_qubit_vs_photonic.png
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.compare_r2` and `python learn/compare_r2.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import numpy as np

from .scaling import load_results


def r2_curves(payload, basis, *, envelope=False):
    """``{x_value: {n_feat, mean, std}}`` test-R^2 vs feature budget for one experiment.

    Reads every run's saved grid curve for ``basis``.  Within a run the best (max) test R^2 at
    each ``n_feat`` is kept (best model achievable at that budget); those are then averaged over
    the average group at each shared ``n_feat``.  With ``envelope=True`` the per-x mean curve is
    made monotone (cumulative max over increasing ``n_feat``) -- the classic learning-curve shape.
    """
    from collections import defaultdict

    per_x = defaultdict(lambda: defaultdict(list))       # x -> n_feat -> [best r2 per repeat]
    for run in payload["runs"]:
        x = run["x"]
        if basis not in run["per_basis"]:
            continue
        best_at = {}                                     # n_feat -> max test_r2 within this run
        for pt in run["per_basis"][basis]["curve"]:
            r2 = pt["test_r2"]
            if r2 is None:
                continue
            nf = pt["n_feat"]
            if nf not in best_at or r2 > best_at[nf]:
                best_at[nf] = r2
        for nf, r2 in best_at.items():
            per_x[x][nf].append(r2)

    out = {}
    for x, d in per_x.items():
        nfeats = sorted(d)
        means = [float(np.mean(d[nf])) for nf in nfeats]
        stds = [float(np.std(d[nf])) for nf in nfeats]
        if envelope:
            means = list(np.maximum.accumulate(means))
        out[x] = {"n_feat": nfeats, "mean": means, "std": stds}
    return out


def plot_compare_r2(payloads, labels, save_path, *, basis="fourier", envelope=False, show=False):
    """Row of subplots (one per swept complexity value) of test R^2 vs #features.

    Each subplot fixes one x value (e.g. one ``m``) and overlays each experiment's
    :func:`r2_curves` learning curve, with one line per experiment.  Payloads lacking ``basis``
    are skipped; x values are the union across experiments (subplots only draw the experiments
    that ran at that value)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = [(r2_curves(p, basis, envelope=envelope), l, p) for p, l in zip(payloads, labels)
              if basis in p["meta"]["bases"]]
    if not curves:
        raise ValueError(f"none of the payloads contain basis {basis!r}")
    xl = curves[0][2]["meta"]["x_label"]
    xvals = sorted({x for c, _, _ in curves for x in c},
                   key=lambda v: (0, v) if isinstance(v, (int, float)) else (1, str(v)))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, len(xvals), figsize=(5.5 * len(xvals), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, x in zip(axes, xvals):
        for pi, (c, label, _) in enumerate(curves):
            if x not in c:
                continue
            d = c[x]
            color = colors[pi % len(colors)]
            ax.plot(d["n_feat"], d["mean"], marker="o", ms=3, label=label, color=color)
            lo = [m - s for m, s in zip(d["mean"], d["std"])]
            hi = [m + s for m, s in zip(d["mean"], d["std"])]
            ax.fill_between(d["n_feat"], lo, hi, color=color, alpha=0.15)
        ax.set_xscale("log")                             # n_feat spans orders of magnitude
        ax.set_xlabel("# explicit features")
        ax.set_title(f"{xl} = {x}")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel(f"test R²{' (envelope)' if envelope else ''}  (mean ± std, basis={basis})")
    axes[-1].legend(title="experiment")
    fig.suptitle(f"Test R² vs feature budget, per {xl}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[compare_r2] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.compare_r2",
        description="Overlay saved learn.scaling experiments as test-R^2 vs #features curves, "
                    "one subplot per swept complexity value.")
    ap.add_argument("files", nargs="+", help="two or more saved learn.scaling results JSONs")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend labels, one per file (default: the file stems)")
    ap.add_argument("--basis", default="fourier",
                    help="which basis to compare across experiments (default: fourier)")
    ap.add_argument("--envelope", action="store_true",
                    help="make each curve monotone (cumulative-max R² over #features)")
    ap.add_argument("--save", default=None, help="path to write the comparison PNG")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    if len(args.files) < 2:
        ap.error("give at least two saved results JSONs to compare")
    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        ap.error(f"--labels count ({len(labels)}) must match files count ({len(args.files)})")
    payloads = [load_results(f) for f in args.files]

    save = args.save
    if save is None and not args.show:
        stems = "_vs_".join(Path(f).stem.replace("scaling_", "") for f in args.files)
        save = str(Path("img") / f"compare_r2_{stems}.png")
    plot_compare_r2(payloads, labels, save, basis=args.basis, envelope=args.envelope,
                    show=args.show)


if __name__ == "__main__":
    main()
