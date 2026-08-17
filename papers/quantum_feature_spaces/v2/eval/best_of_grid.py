"""Model x observable heatmap, one cell = best learner, averaged over seeds.

    python eval/best_of_grid.py --eval-dir configs/eval/photonic_encoding
    python eval/best_of_grid.py --eval-dir configs/eval/classical_vs_quantum --n-seeds 10

For one ``configs/eval/<group>/`` folder (e.g. ``havlicek/``, ``phase/`` -- each ``*.yaml`` inside is
one model variant), builds a grid with **observables as rows and variants as columns**: the layout
:mod:`configs.run_eval_heatmaps` does not produce (that script's grids are variant x observable or
variant x learner, one folder = one axis of variants only; this is the cross-model-family reading
where each folder's files are already different ``model.kind``s, not encodings of a single model).

**Cell value is ``max`` over :data:`learner.auto.DEFAULT_SWEEP_LEARNERS`, each itself averaged over
``--n-seeds`` reseeded fits** -- not the other way around.  Averaging first (per learner, across
seeds) then maxing (across learners) is deliberate: it reports "the best-performing learner's typical
score," not "the best score any learner got on its luckiest seed," which would be an optimistic
outlier pick rather than a stable reading.

**The seed axis varies the train/test split, not (only) the learner's own internal randomness.**
``ridge`` (at the default ``fourier_poly`` basis, no random features) and ``svr`` (an exact QP
solve given fixed data) are both fully deterministic given ``(X, y)`` -- reseeding only their own
``seed`` kwarg changes nothing about their score, since neither reads it anywhere on that path.
Only ``mlp`` (via ``torch.manual_seed``, weight init + batch order) varies with the learner ``seed``
alone.  So every seed iteration here also draws a fresh ``split_seed`` (via
:func:`learner.auto.run_config`'s ``split_seed`` override) -- this is what actually gives ridge and
svr ``n_seeds`` genuinely different train/test partitions to be evaluated on, not ``n_seeds``
identical repeats of the same fit.  The learner's own ``seed`` kwarg is reseeded in lockstep purely
so ``mlp``'s within-split variance is also captured, not because it does anything for the other two.

**Nothing here is thrown away.**  The saved JSON keeps every ``(learner, seed)`` cell's raw R^2 --
:func:`sweep_best_of_grid`'s ``"detail"`` field -- so which learner actually won a cell, and how
much the seeds spread, is always recoverable later even though the printed table/heatmap only shows
the final max-of-means number per :func:`STATUS.md`/the paired-protocol discipline of never trusting
a single learner's score without knowing whether others agree.

Requires every variant's dataset to already be generated -- this only fits learners on saved
artifacts, exactly like :mod:`configs.run_eval_heatmaps`; it does not call
:mod:`pipeline.generate` itself.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/best_of_grid.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"
DEFAULT_OBSERVABLES = ["parity", "majority", "n_first"]

#: Per-folder ``{file_stem: display_label}`` -- the yaml stem on disk (e.g. ``bs_phase``) stays the
#: config's identity; this only controls what the heatmap's axis prints, so the two can diverge
#: (compact/unambiguous filenames vs. readable plot labels) without a rename touching any config
#: loader.  A stem missing from its folder's mapping falls back to a title-cased, underscore-split
#: version of the stem itself (see :func:`_display_label`), so a new variant file never breaks the
#: plot -- it just prints a slightly blunter label until an entry is added here.
VARIANT_LABELS: dict[str, dict[str, str]] = {
    "photonic_encoding": {
        "phase": "Phase",
        "bs": "Beamsplitter",
        "bs_phase": "Phase +\nBeamsplitter",
        "havlicek": "Havlicek",
    },
    "qubit_encoding": {
        "phase": "Phase",
        "iqp": "IQP (Havlicek)",
    },
    "complex_qubit_encoding": {
        "photonic_phase": "Photonic,\nphase",
        "photonic_havlicek": "Photonic,\nHavlicek",
        "qubit_phase": "Qubit,\nphase",
        "qubit_havlicek": "Qubit,\nHavlicek (IQP)",
    },
    "classical_vs_quantum": {
        "quadratic_fock": "Quadratic Fock\n(classical)",
        "fermion_phase": "Fermion,\nphase",
        "fermion_havlicek": "Fermion,\nHavlicek",
    },
    "spin_magic": {
        "ghz_enccirc": "GHZ",
        "linear_encboth": "Linear cluster,\n(spin-and-circuit)",
        "linear_enccirc": "Linear cluster",
        "linear_encspin": "Linear cluster,\n(spin-encoded)",
        "linear_u3_enccirc": "Arbitrary cluster, \n(random U3)",
    },
    "spin": {
        "encboth_l1": "Encoded both,\n1 layer",
        "enccirc_l1": "Circuit-encoded,\n1 layer",
        "enccirc_l3": "Circuit-encoded,\n3 layers",
        "encspin_l1": "Spin-encoded,\n1 layer",
    },
}

#: Per-folder x-axis title -- what the varying axis actually represents, shown instead of the
#: generic "model" label so the heatmap states its own comparison without needing the caption.
AXIS_TITLES: dict[str, str] = {
    "photonic_encoding": "photonic encoding",
    "qubit_encoding": "qubit encoding",
    "complex_qubit_encoding": "Complex encoding (Havlicek)",
    "classical_vs_quantum": "Classical equivalent model",
    "spin_magic": "Single-Qubit Emitter Variant",
    "spin": "multi-qubit emitter (spin) variant",
}


def _display_label(folder_name: str, stem: str) -> str:
    """``VARIANT_LABELS[folder_name][stem]`` if mapped, else a title-cased fallback from the stem
    itself (``bs_phase`` -> ``Bs Phase``) so an unmapped variant still prints something readable."""
    mapped = VARIANT_LABELS.get(folder_name, {})
    if stem in mapped:
        return mapped[stem]
    return stem.replace("_", " ").title()


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _variants_in(subfolder: Path) -> list[tuple[str, Path]]:
    """``[(stem, path), ...]`` for every ``*.yaml`` directly inside ``subfolder``, sorted by stem."""
    return [(p.stem, p) for p in sorted(subfolder.glob("*.yaml"))]


def sweep_best_of_grid(variants: list[tuple[str, Path]], observables: list[str], *,
                       learners=None, n_seeds: int = 10, out_root: str = "datasets",
                       scores_root: str = "scores", n_train: int | None = None,
                       graph_density: float = 0.5) -> dict:
    """For every ``(observable, variant)`` cell: fit each learner in ``learners`` at ``n_seeds``
    reseeded draws, average each learner's R^2 over seeds, then take the max across learners.

    Returns ``{"observables": [...], "variants": [...], "r2": [[...]], "detail": [[...]]}`` --
    ``r2[i][j]`` is the max-of-per-learner-means for ``(observables[i], variants[j])``; ``detail``
    holds the same shape but each cell is ``{learner_name: [r2_seed0, r2_seed1, ...]}`` so no raw
    number is lost even though only the summary feeds the heatmap.  A cell that raises for every
    learner/seed is recorded as ``None`` in ``r2`` (empty dict in ``detail``) and printed, not fatal
    to the rest of the grid -- same per-cell failure discipline as every other sweep in
    :mod:`learner.auto`.
    """
    import math

    from learner.auto import DEFAULT_SWEEP_LEARNERS, run_config

    learners = learners or DEFAULT_SWEEP_LEARNERS
    variant_names = [name for name, _ in variants]

    r2_grid, detail_grid = [], []
    for obs in observables:
        r2_row, detail_row = [], []
        for vname, vpath in variants:
            cell_detail: dict[str, list[float]] = {}
            learner_means = []
            for lname, base_kwargs in learners:
                seed_scores = []
                for seed in range(int(n_seeds)):
                    kwargs = dict(base_kwargs)
                    kwargs["seed"] = seed
                    try:
                        res = run_config(vpath, obs, lname, out_root=out_root,
                                         scores_root=scores_root, n_train=n_train,
                                         graph_density=graph_density, split_seed=seed, **kwargs)
                        seed_scores.append(res["r2"])
                    except Exception as exc:                # noqa: BLE001 -- one bad seed must not
                        print(f"[best_of_grid] {obs}/{vname}/{lname}/seed={seed} failed: {exc}")
                        seed_scores.append(None)
                cell_detail[lname] = seed_scores
                valid = [s for s in seed_scores if s is not None]
                if valid:
                    learner_means.append(statistics.mean(valid))
            r2_row.append(max(learner_means) if learner_means else None)
            detail_row.append(cell_detail)
        r2_grid.append(r2_row)
        detail_grid.append(detail_row)

    return {"observables": list(observables), "variants": variant_names, "r2": r2_grid,
            "detail": detail_grid, "n_seeds": n_seeds,
            "learners": [name for name, _ in learners]}


def plot_best_of_grid(result: dict, *, title: str = "", x_label: str = "model",
                      variant_labels: list[str] | None = None,
                      save_path: str | Path | None = None, show: bool = False):
    """Observables on y, variants on x, cell = max-of-per-learner-mean-R^2 -- same visual
    conventions as :func:`learner.auto.plot_heatmap` (RdYlGn, grey for missing cells).

    ``variant_labels``, if given, are the display strings shown on the x-axis instead of
    ``result["variants"]`` (the raw yaml stems) -- same length/order, see :data:`VARIANT_LABELS`
    and :func:`_display_label`.  ``x_label`` replaces the generic ``"model"`` axis title with
    whatever the varying axis actually represents for this folder (e.g. ``"photonic encoding"``).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    obs, variants = result["observables"], result["variants"]
    labels = variant_labels if variant_labels is not None else variants
    r2 = np.array([[np.nan if v is None else v for v in row] for row in result["r2"]], dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(variants) + 2), max(3, 0.7 * len(obs) + 1.5)))
    masked = np.ma.masked_invalid(r2)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticks(range(len(obs)))
    ax.set_yticklabels(obs)
    ax.set_xlabel(x_label)
    ax.set_ylabel("observable")
    ax.set_title(title or f"best-of-{len(result['learners'])} R^2 "
                          f"(mean over {result['n_seeds']} seeds)")

    for i in range(len(obs)):
        for j in range(len(variants)):
            v = r2[i][j]
            text = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            colour = "black" if not np.isfinite(v) or -0.4 < v < 0.75 else "white"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=9)

    fig.colorbar(im, ax=ax, label="R^2")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def run(*, eval_dir: Path, observables: list[str] = DEFAULT_OBSERVABLES, n_seeds: int = 10,
       out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None) -> dict:
    """One ``configs/eval/<group>/`` folder -> one saved grid (PNG + JSON) inside that folder."""
    variants = _variants_in(eval_dir)
    if not variants:
        raise SystemExit(f"no *.yaml configs found in {eval_dir}")
    print(f"=== {eval_dir.name}: {len(variants)} variants -> {[n for n, _ in variants]}",
          flush=True)

    result = sweep_best_of_grid(variants, observables, n_seeds=n_seeds, out_root=out_root,
                                scores_root=scores_root, n_train=n_train)
    tag = "-".join(_safe_tag(o) for o in observables)
    out_json = eval_dir / f"best_of_grid__{tag}.json"
    out_png = eval_dir / f"best_of_grid__{tag}.png"
    out_json.write_text(json.dumps(result, indent=2))
    labels = [_display_label(eval_dir.name, stem) for stem in result["variants"]]
    x_label = AXIS_TITLES.get(eval_dir.name, "model")
    plot_best_of_grid(result, title=f"{eval_dir.name}: best-of-3 R^2 (mean over {n_seeds} seeds)",
                      x_label=x_label, variant_labels=labels, save_path=out_png)
    print(f"    wrote {out_png}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Observable x model heatmap, one cell = best learner "
                                              "(mean over reseeded fits) for one configs/eval/ group.")
    ap.add_argument("--eval-dir", required=True, help="e.g. configs/eval/havlicek")
    ap.add_argument("--observables", nargs="+", default=DEFAULT_OBSERVABLES)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    run(eval_dir=Path(args.eval_dir), observables=args.observables, n_seeds=args.n_seeds,
       out_root=args.root, scores_root=args.scores_root, n_train=args.n_train)


if __name__ == "__main__":
    main()
