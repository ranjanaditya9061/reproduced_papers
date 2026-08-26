"""Observable-family bar chart, one fixed circuit, one cell = best learner, averaged over seeds.

    python eval/eval_obs.py --config configs/photonic.yaml

The transpose of :mod:`eval.best_of_grid`: that script fixes a small observable set (parity/
majority/n_first) and varies the *circuit* across columns; this one fixes a single circuit config
and varies the *observable*, grouped by family (boson-phase counting, polynomial, graph, nonlinear)
so the walkthrough from simple score-vector observables to nonlinear functionals reads as one
picture instead of a bare unordered list.

Reuses :func:`eval.best_of_grid.sweep_best_of_grid`'s statistic unchanged (max over
``learner.auto.DEFAULT_SWEEP_LEARNERS``, each itself averaged over ``--n-seeds`` reseeded
train/test splits) so numbers here are directly comparable to :mod:`eval.best_of_grid`'s -- same
rigor, same seed-averaging discipline, just the fixed/varying axes swapped. See that module's
docstring for why max-of-means (not mean-of-maxes) and why the seed axis is really a split-seed
axis for ridge/svr.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/eval_obs.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"

#: ``{family_label: [observable_key, ...]}`` -- the walkthrough order for the presentation's
#: section 2 (boson-phase counting -> polynomial -> graph -> nonlinear). Curated, not derived from
#: the registry automatically, since the registry has far more variants (``_lo{N}``, angle
#: variants, per-reading graph families) than are worth showing in one picture -- add entries here
#: deliberately rather than enumerating every registered observable.
DEFAULT_FAMILIES: dict[str, list[str]] = {
    "Boson-phase (counting)": ["parity", "majority", "n_first", "bunching"],
    "Polynomial": ["prod_parity__lo3", "prod_parity_consecutive", "prod_parity_diag_prime_sqrt"],
    "Graph": ["connected_maxcc_pair", "connected_paritymaxcc_pair",
             "connected_numloops_pair", "connected_paritynumloops_pair"],
    "Nonlinear": ["sq_parity", "pairprod", "ent", "osc"],
}

#: Display label for an observable key that isn't already readable title-cased -- mirrors
#: :mod:`eval.best_of_grid`'s ``OBSERVABLE_LABELS`` convention (``n_first`` really scores mode 0's
#: parity, not a raw count; kept here rather than imported to avoid a cross-module label coupling
#: for what is otherwise a self-contained script).
OBSERVABLE_LABELS: dict[str, str] = {
    "n_first": "Mode 1\nParity",
    "prod_parity__lo3": "Degree 3\nProduct",
    "prod_parity_consecutive": "Consecutive\nProduct",
    "prod_parity_diag_prime_sqrt": "Quadratic\nMonbroussou et al.",
    "connected_maxcc_pair": "Max Size\nConnected",
    "connected_paritymaxcc_pair": "Parity x\nMax Size\nConnected",
    "connected_numloops_pair": "Num Loops",
    "connected_paritynumloops_pair": "Parity x\nNum Loops",
    "sq_parity": "Quadratic\nParity",
    "pairprod": "Quadratic\nProduct",
    "ent": "Entropy-\nBased",
    "osc": "Oscillatory",
}


def _flatten_families(families: dict[str, list[str]]) -> list[str]:
    """Families dict -> one flat observable list, family order preserved, no de-duplication
    (an observable repeated across families would repeat as a row -- not expected in practice but
    not silently collapsed either)."""
    return [obs for obs_list in families.values() for obs in obs_list]


def _safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def sweep_eval_obs(cfg_path: str | Path, families: dict[str, list[str]] = DEFAULT_FAMILIES, *,
                   learners=None, n_seeds: int = 10, out_root: str = "datasets",
                   scores_root: str = "scores", n_train: int | None = None,
                   graph_density: float = 0.5, n_jobs: int = 1) -> dict:
    """One circuit config, every observable in ``families`` -- same per-cell statistic as
    :func:`eval.best_of_grid.sweep_best_of_grid`, called with a single-variant list so its grid
    degenerates to one column; reshaped here into a flat per-observable result plus the family
    grouping for the bar-chart renderer.

    Returns ``{"families": {label: [obs, ...]}, "observables": [...], "r2": [...],
    "r2_std": [...], "detail": [...], "n_seeds": n_seeds, "learners": [...],
    "config": str(cfg_path)}`` -- ``r2``/``r2_std``/``detail`` are flat lists aligned with
    ``observables``, not nested per-variant, since there is exactly one variant here. ``r2_std`` is
    the stdev of the per-seed best-of-learners scores (see :func:`sweep_best_of_grid`'s
    best-of-three-per-seed reduction), for the bar chart's error bars.
    """
    from eval.best_of_grid import sweep_best_of_grid

    observables = _flatten_families(families)
    variant = [(Path(cfg_path).stem, Path(cfg_path))]
    grid = sweep_best_of_grid(variant, observables, learners=learners, n_seeds=n_seeds,
                              out_root=out_root, scores_root=scores_root, n_train=n_train,
                              graph_density=graph_density, n_jobs=n_jobs)

    r2 = [row[0] for row in grid["r2"]]
    r2_std = [row[0] for row in grid["r2_std"]]
    detail = [row[0] for row in grid["detail"]]
    #: parity's own R^2, carried separately so every per-family chart (including one that doesn't
    #: contain "parity" itself, e.g. the Graph family) can still draw it as a reference line --
    #: None if "parity" wasn't in the swept observable list at all.
    parity_r2 = r2[observables.index("parity")] if "parity" in observables else None
    return {"families": {label: list(obs_list) for label, obs_list in families.items()},
            "observables": observables, "r2": r2, "r2_std": r2_std, "detail": detail,
            "parity_r2": parity_r2, "n_seeds": n_seeds, "learners": grid["learners"],
            "config": str(cfg_path)}


def plot_eval_obs(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """One bar per observable, grouped and coloured by family, R^2 on the y-axis -- the
    walkthrough picture for section 2 (boson-phase -> polynomial -> graph -> nonlinear), same
    circuit throughout so only the observable changes between bars."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    observables, r2 = result["observables"], result["r2"]
    r2_arr = np.array([np.nan if v is None else v for v in r2], dtype=float)
    r2_std = np.array([0.0 if s is None else s for s in result.get("r2_std", [0.0] * len(r2))],
                      dtype=float)

    labels = [OBSERVABLE_LABELS.get(o, o.replace("_", " ").title()) for o in observables]
    #: bar height is clamped to >=0 (axis starts at 0, per house style); a negative R^2 draws as a
    #: zero-height bar with its true value labelled in red above the baseline instead of dipping
    #: below the axis.
    heights = np.clip(np.nan_to_num(r2_arr, nan=0.0), 0.0, None)
    light_colour = "#a8d5e2"

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(observables) + 1.5), 3))
    x = np.arange(len(observables))
    err = np.where(np.isfinite(r2_arr), r2_std, 0.0)
    #: error bars at +/- one stdev of the per-seed best-of-learners scores (see
    #: eval.best_of_grid.sweep_best_of_grid's best-of-three-per-seed reduction).
    ax.bar(x, heights, yerr=err, color=light_colour, edgecolor="black", linewidth=0.5, zorder=3,
          capsize=3, error_kw={"linewidth": 0.8, "zorder": 4})
    #: value-label y-position: a short bar (value in [0.02, 0.15]) would collide with the fixed
    #: baseline below, so its label sits above the bar's full visual extent -- value PLUS its own
    #: error-bar whisker, so the label never overlaps the whisker -- plus 0.03; every other bar --
    #: including a negative value or a failed ("n/a") cell -- gets the fixed y=0.03 baseline, so
    #: most labels still read as one consistent row.
    for i, v in enumerate(r2_arr):
        if not np.isfinite(v):
            ax.text(i, 0.03, "n/a", ha="center", va="bottom", rotation=90)
        elif v < 0:
            ax.text(i, 0.03, f"{v:.2f}", ha="center", va="bottom", color="red")
        elif 0.02 <= v <= 0.15:
            ax.text(i, v + err[i] + 0.03, f"{v:.2f}", ha="center", va="bottom")
        else:
            ax.text(i, 0.03, f"{v:.2f}", ha="center", va="bottom")

    #: parity's own R^2 as a reference line -- shown on every chart, including a per-family one
    #: that doesn't contain "parity" itself (e.g. Graph), since parity_r2 is carried through
    #: _result_for_family regardless of which family was sliced out. Labelled with a direct text
    #: annotation rather than a legend box, matching the house no-legend style.
    parity_r2 = result.get("parity_r2")
    if parity_r2 is not None and np.isfinite(parity_r2):
        ax.axhline(parity_r2, color="red", linestyle=":", linewidth=1.2, zorder=2)
        ax.text(len(observables) - 0.5, parity_r2, f"Parity ({parity_r2:.2f})", color="red",
               fontsize=8, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient of\nDetermination ($R^2$)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _result_for_family(result: dict, label: str) -> dict:
    """Slice one family out of a full :func:`sweep_eval_obs` result -- no re-fitting, since every
    observable's R^2 was already computed in the one combined sweep; this just filters the flat
    ``observables``/``r2``/``detail`` lists down to one family's rows so
    :func:`plot_eval_obs` renders a single-family chart at full bar width instead of a busy
    multi-family one."""
    obs_list = result["families"][label]
    idx = {obs: i for i, obs in enumerate(result["observables"])}
    keep = [idx[o] for o in obs_list]
    return {"families": {label: obs_list},
            "observables": [result["observables"][i] for i in keep],
            "r2": [result["r2"][i] for i in keep],
            "r2_std": [result["r2_std"][i] for i in keep],
            "detail": [result["detail"][i] for i in keep],
            "n_seeds": result["n_seeds"], "learners": result["learners"],
            "config": result["config"], "parity_r2": result.get("parity_r2")}


def run(*, cfg_path: Path, families: dict[str, list[str]] = DEFAULT_FAMILIES, n_seeds: int = 10,
       out_root: str = "datasets", scores_root: str = "scores", n_train: int | None = None,
       n_jobs: int = 1, out_dir: Path | None = None, per_family: bool = True) -> dict:
    """One combined chart (all families) plus, when ``per_family`` (default), one additional
    filtered chart per family -- e.g. for a talk that gives each family its own slide (section 2's
    boson-phase / polynomial / graph / nonlinear walkthrough), so each slide gets a focused image
    at full detail instead of reusing the crowded combined one four times."""
    result = sweep_eval_obs(cfg_path, families, n_seeds=n_seeds, out_root=out_root,
                            scores_root=scores_root, n_train=n_train, n_jobs=n_jobs)
    out_dir = out_dir or cfg_path.parent
    tag = _safe_tag(cfg_path.stem)
    out_json = out_dir / f"eval_obs__{tag}.json"
    out_png = out_dir / f"eval_obs__{tag}.png"
    out_json.write_text(json.dumps(result, indent=2))
    plot_eval_obs(result, save_path=out_png)
    print(f"    wrote {out_png}", flush=True)

    if per_family:
        for label in families:
            family_result = _result_for_family(result, label)
            family_png = out_dir / f"eval_obs__{tag}__{_safe_tag(label)}.png"
            plot_eval_obs(family_result, save_path=family_png)
            print(f"    wrote {family_png}", flush=True)

    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Observable-family bar chart for one fixed circuit "
                                             "config -- the transpose of eval.best_of_grid.")
    ap.add_argument("--config", required=True, help="single circuit config, e.g. "
                    "configs/eval/photonic_encoding/phase.yaml")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-dir", default=None, help="defaults to the config's own directory")
    ap.add_argument("--no-per-family", action="store_true",
                    help="skip the extra per-family charts, write only the combined one")
    args = ap.parse_args(argv)

    run(cfg_path=Path(args.config), n_seeds=args.n_seeds, out_root=args.root,
       scores_root=args.scores_root, n_train=args.n_train, n_jobs=args.n_jobs,
       out_dir=Path(args.out_dir) if args.out_dir else None,
       per_family=not args.no_per_family)


if __name__ == "__main__":
    main()
