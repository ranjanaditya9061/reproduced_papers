"""Overlay two (or more) saved :mod:`learn.scaling` experiments at several thresholds.

Reuses the payloads written by ``learn.scaling`` (each run stores its full ``(order x degree)``
grid curve, so min-``n_feat`` can be recomputed at *any* threshold without recomputing).  Draws a
row of subplots -- one per ``--thresholds`` value -- of the mean min # features to reach test
R^2 >= threshold vs the swept problem size, with one line per experiment.

    # compare the qubit vs photonic feature-budget scaling at three thresholds
    python -m learn.compare_scaling \\
        scaling/scaling_1_qubit_m.json scaling/scaling_6_photonic_m.json \\
        --labels qubit photonic --thresholds 0.5 0.7 0.9 \\
        --save img/compare_qubit_vs_photonic.png
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.compare_scaling` and `python learn/compare_scaling.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

from .scaling import aggregate_at, load_results


def plot_compare(payloads, labels, thresholds, save_path, *, basis="fourier", show=False):
    """Row of subplots (one per threshold) overlaying each experiment's min-``n_feat`` scaling.

    ``payloads``/``labels`` are the saved :func:`learn.scaling.run_scaling` results and their
    legend names.  For each threshold, min-``n_feat`` is recomputed from the saved grid curves via
    :func:`learn.scaling.aggregate_at`, averaged over the experiment's average group, and plotted
    vs the swept x-field.  Payloads lacking ``basis`` are skipped."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [(p, l) for p, l in zip(payloads, labels) if basis in p["meta"]["bases"]]
    if not usable:
        raise ValueError(f"none of the payloads contain basis {basis!r}")
    thresholds = list(thresholds)
    xl = usable[0][0]["meta"]["x_label"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.5 * len(thresholds), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, thr in zip(axes, thresholds):
        for pi, (payload, label) in enumerate(usable):
            agg = aggregate_at(payload, thr)
            sweep = payload["meta"]["sweep"]
            idxs = list(range(len(sweep)))
            xvals = [agg[basis][i]["x"] for i in idxs]
            numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xvals)
            xpos = xvals if numeric else idxs
            means = [agg[basis][i]["mean"] for i in idxs]
            stds = [agg[basis][i]["std"] for i in idxs]
            color = colors[pi % len(colors)]
            ax.plot(xpos, means, marker="o", label=label, color=color)
            lo = [max(m - s, 1) for m, s in zip(means, stds)]    # clamp for the log axis
            ax.fill_between(xpos, lo, [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)
            if not numeric:
                ax.set_xticks(xpos)
                ax.set_xticklabels([str(x) for x in xvals])
            for i in idxs:                                       # flag partial-coverage points
                a = agg[basis][i]
                if 0 < a["n_reached"] < a["n_total"]:
                    ax.annotate(f"{a['n_reached']}/{a['n_total']}", (xpos[i], a["mean"]),
                                fontsize=7, color="grey", xytext=(3, 3), textcoords="offset points")
        # ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_title(f"test R² ≥ {thr}")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel(f"min # features  (mean ± std over averages, basis={basis})")
    axes[-1].legend(title="experiment")
    fig.suptitle(f"Feature budget to learn vs {xl}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[compare_scaling] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.compare_scaling",
        description="Overlay saved learn.scaling experiments as a row of subplots, one per "
                    "threshold, of min # features to learn vs problem size.")
    ap.add_argument("files", nargs="+", help="two or more saved learn.scaling results JSONs")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend labels, one per file (default: the file stems)")
    ap.add_argument("--basis", default="fourier",
                    help="which basis to compare across experiments (default: fourier)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                    help="one subplot per threshold")
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
        save = str(Path("img") / f"compare_{stems}.png")
    plot_compare(payloads, labels, args.thresholds, save, basis=args.basis, show=args.show)


if __name__ == "__main__":
    main()