"""Model x observable heatmap, one cell = best learner, averaged over seeds.

    python eval/best_of_grid.py --eval-dir configs/eval/photonic_encoding
    python eval/best_of_grid.py --eval-dir configs/eval/classical_vs_quantum --n-seeds 10
    python eval/best_of_grid.py --configs configs/eval/*/*.yaml   # combine every subfolder

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

#: ``{observable_key: display_label}`` for the y-axis -- ``n_first`` is a misleading name kept for
#: historical/on-disk-compatibility reasons (see :mod:`observable.scorers.counting`): it actually
#: scores the *parity* of mode 0's photon count (``n_0 mod 2``), not a raw photon number, so its
#: display label says "Parity" too rather than repeating the internal name's inaccuracy.
OBSERVABLE_LABELS: dict[str, str] = {
    "parity": "Parity",
    "majority": "Majority",
    "n_first": "Mode 1\nParity",
}

#: Per-folder ``{file_stem: display_label}`` -- the yaml stem on disk (e.g. ``bs_phase``) stays the
#: config's identity; this only controls what the heatmap's axis prints, so the two can diverge
#: (compact/unambiguous filenames vs. readable plot labels) without a rename touching any config
#: loader.  A stem missing from its folder's mapping falls back to a title-cased, underscore-split
#: version of the stem itself (see :func:`_display_label`), so a new variant file never breaks the
#: plot -- it just prints a slightly blunter label until an entry is added here.  This dict's key
#: order also drives the x-axis column order (see :func:`_variants_in`) -- reordering keys here
#: reorders the plot, no other change needed.
VARIANT_LABELS: dict[str, dict[str, str]] = {
    "photonic_encoding": {
        "phase": "Phase",
        "bs": "Beamsplitter",
        "bs_phase": "Phase +\nBeamsplitter",
        "havlicek": "Havlicek",
    },
    "photonic_reencoding": {f"depth{_l}": str(_l) for _l in range(1, 11)},
    "qubit_encoding": {
        "phase": "Phase",
        "iqp": "IQP (Havlicek)",
    },
    "complex_qubit_encoding": {
        "photonic_phase": "Photonic\nPhase",
        "photonic_havlicek": "Photonic\nHavlicek",
        "qubit_phase": "Qubit\nPhase",
        "qubit_havlicek": "Qubit\nHavlicek (IQP)",
    },
    "classical_vs_quantum": {
        "quadratic_fock": "Quadratic",
        "fermion_phase": "Fermion\nPhase",
        "fermion_bs": "Fermion,\nBeamsplitter",
        "fermion_bs_phase": "Fermion\nPhase +\nBeamsplitter",
        "fermion_havlicek": "Fermion\nHavlicek",
    },
    "spin_magic": {
        "ghz_enccirc": "GHZ",
        "linear_enccirc": "Linear",
        "linear_u3_enccirc": "Arbitrary \nRandom U3",
        "linear_encspin": "Linear\nSpin-Encoded",
        "linear_encboth": "Linear\nSpin-and-\nCircuit",
    },
    "spin": {
        "enccirc_l1": "Circuit-\nEncoded\n1 Layer",
        "encspin_l1": "Spin-\nEncoded\n1 Layer",
        "encboth_l1": "Spin-and\n-Circuit\n1 Layer",
        "enccirc_l3": "Circuit\nEncoded\n3 Layers",
    },
}

#: Per-folder x-axis title -- what the varying axis actually represents, shown instead of the
#: generic "model" label so the heatmap states its own comparison without needing the caption.
AXIS_TITLES: dict[str, str] = {
    "photonic_encoding": "Photonic Encoding",
    "photonic_reencoding": "# of Re-encodings",
    "qubit_encoding": "Qubit Encoding",
    "complex_qubit_encoding": "Havlicek Encoding",
    "classical_vs_quantum": "Classical Equivalent Model",
    "spin_magic": "Single-Qubit Emitted Cluster",
    "spin": "Multi-Qubit Emitted Variant",
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
    """``[(stem, path), ...]`` for every ``*.yaml`` directly inside ``subfolder``, ordered per
    :data:`VARIANT_LABELS`'s key order for this folder (that dict is curated in the order the
    columns should read left-to-right) with any stem missing from the mapping appended
    alphabetically after -- so a new variant file still shows up, just at the end until an entry
    is added to :data:`VARIANT_LABELS`."""
    stem_to_path = {p.stem: p for p in subfolder.glob("*.yaml")}
    order = list(VARIANT_LABELS.get(subfolder.name, {}))
    ordered_stems = [s for s in order if s in stem_to_path]
    ordered_stems += sorted(s for s in stem_to_path if s not in order)
    return [(s, stem_to_path[s]) for s in ordered_stems]


def _grid_seed_worker(task):
    """One ``(observable, variant, learner, seed)`` fit -- module-level so it's picklable for
    :func:`~v2.circuit.spin.parallel_row_map`'s process pool.  Returns ``(r2, error)`` rather than
    raising/printing here: printing happens in the parent, once results are back in task order, so
    output from a parallel run reads the same as the old serial loop instead of interleaving across
    worker processes."""
    (vpath, obs, lname, out_root, scores_root, n_train, graph_density, seed,
     base_kwargs) = task
    from learner.auto import run_config

    kwargs = dict(base_kwargs)
    kwargs["seed"] = seed
    try:
        res = run_config(vpath, obs, lname, out_root=out_root, scores_root=scores_root,
                         n_train=n_train, graph_density=graph_density, split_seed=seed, **kwargs)
        return res["r2"], None
    except (Exception, SystemExit) as exc:                  # noqa: BLE001 -- one bad seed must not
        return None, str(exc)                                # abort the rest of the sweep
                                                              # (run_config raises SystemExit, not
                                                              # Exception, on a missing dataset)


def sweep_best_of_grid(variants: list[tuple[str, Path]], observables: list[str], *,
                       learners=None, n_seeds: int = 10, out_root: str = "datasets",
                       scores_root: str = "scores", n_train: int | None = None,
                       graph_density: float = 0.5, n_jobs: int = 1) -> dict:
    """For every ``(observable, variant)`` cell: fit each learner in ``learners`` at ``n_seeds``
    reseeded draws, average each learner's R^2 over seeds, then take the max across learners.

    Returns ``{"observables": [...], "variants": [...], "r2": [[...]], "detail": [[...]]}`` --
    ``r2[i][j]`` is the max-of-per-learner-means for ``(observables[i], variants[j])``; ``detail``
    holds the same shape but each cell is ``{learner_name: [r2_seed0, r2_seed1, ...]}`` so no raw
    number is lost even though only the summary feeds the heatmap.  A cell that raises for every
    learner/seed is recorded as ``None`` in ``r2`` (empty dict in ``detail``) and printed, not fatal
    to the rest of the grid -- same per-cell failure discipline as every other sweep in
    :mod:`learner.auto`.

    Every ``(observable, variant, learner, seed)`` fit is independent, so they are flattened into
    one task list and handed to :func:`~v2.circuit.spin.parallel_row_map`, the same ordered
    process-pool map :mod:`v2.circuit.prep`'s row workers use, rather than looping in serial.
    ``n_jobs=1`` (the default) stays serial; ``-1`` uses all-but-one CPU.
    """
    from circuit.spin import parallel_row_map
    from learner.auto import DEFAULT_SWEEP_LEARNERS

    learners = learners or DEFAULT_SWEEP_LEARNERS
    variant_names = [name for name, _ in variants]

    cells = [(obs, vname, vpath, lname, base_kwargs)
            for obs in observables
            for vname, vpath in variants
            for lname, base_kwargs in learners]
    tasks = [(vpath, obs, lname, out_root, scores_root, n_train, graph_density, seed, base_kwargs)
            for obs, vname, vpath, lname, base_kwargs in cells
            for seed in range(int(n_seeds))]
    flat = parallel_row_map(_grid_seed_worker, tasks, n_jobs)

    n = int(n_seeds)
    cell_seed_scores: dict[tuple[str, str, str], list] = {}
    for i, (obs, vname, vpath, lname, base_kwargs) in enumerate(cells):
        seed_results = flat[i * n:(i + 1) * n]
        seed_scores = []
        for seed, (r2, err) in enumerate(seed_results):
            if err is not None:
                print(f"[best_of_grid] {obs}/{vname}/{lname}/seed={seed} failed: {err}")
            else:
                print(vname, lname, seed, r2)
            seed_scores.append(r2)
        cell_seed_scores[(obs, vname, lname)] = seed_scores

    r2_grid, detail_grid = [], []
    for obs in observables:
        r2_row, detail_row = [], []
        for vname, vpath in variants:
            cell_detail: dict[str, list[float]] = {}
            learner_means = []
            for lname, base_kwargs in learners:
                seed_scores = cell_seed_scores[(obs, vname, lname)]
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


def plot_best_of_grid(result: dict, *, x_label: str = "model",
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
    ax.set_yticklabels([OBSERVABLE_LABELS.get(o, o.replace("_", " ").title()) for o in obs])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Observable")

    for i in range(len(obs)):
        for j in range(len(variants)):
            v = r2[i][j]
            text = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            colour = "black" if not np.isfinite(v) or -0.4 < v < 0.75 else "white"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=9)

    fig.colorbar(im, ax=ax, label="Coefficient of Determination (R^2)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_best_of_grid_bar(result: dict, *, x_label: str = "model",
                          variant_labels: list[str] | None = None,
                          save_path: str | Path | None = None, show: bool = False):
    """Single-observable variant of :func:`plot_best_of_grid`: one bar per variant instead of a
    1-row heatmap.  Only sensible when ``result["observables"]`` has exactly one entry (e.g. a
    ``--observables parity`` run) -- raises otherwise, since a bar plot has nowhere to put a second
    row and silently picking one would hide the dropped data rather than surfacing it.

    House bar-plot style (matches :func:`eval.eval_obs.plot_eval_obs`): no title/legend, axis
    starts at 0 (a negative R^2 draws as a zero-height bar with its value labelled in red at the
    baseline instead of dipping below the axis), one light flat colour, small figure size, light
    background grid.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    obs, variants = result["observables"], result["variants"]
    if len(obs) != 1:
        raise ValueError(f"plot_best_of_grid_bar needs exactly one observable, got {obs!r}")

    labels = variant_labels if variant_labels is not None else variants
    r2 = np.array([np.nan if v is None else v for v in result["r2"][0]], dtype=float)
    heights = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, None)
    light_colour = "#a8d5e2"

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(variants) + 1.5), 3))
    x = np.arange(len(variants))
    ax.bar(x, heights, color=light_colour, edgecolor="black", linewidth=0.5, zorder=3)
    for i, v in enumerate(r2):
        if not np.isfinite(v):
            ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
        elif v < 0:
            ax.text(i, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="red")
        else:
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("R^2", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def run(*, eval_dir: Path, observables: list[str] = DEFAULT_OBSERVABLES, n_seeds: int = 10,
       out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None, n_jobs: int = 1) -> dict:
    """One ``configs/eval/<group>/`` folder -> one saved grid (PNG + JSON) inside that folder."""
    variants = _variants_in(eval_dir)
    if not variants:
        raise SystemExit(f"no *.yaml configs found in {eval_dir}")
    print(f"=== {eval_dir.name}: {len(variants)} variants -> {[n for n, _ in variants]}",
          flush=True)

    result = sweep_best_of_grid(variants, observables, n_seeds=n_seeds, out_root=out_root,
                                scores_root=scores_root, n_train=n_train, n_jobs=n_jobs)
    tag = "-".join(_safe_tag(o) for o in observables)
    out_json = eval_dir / f"best_of_grid__{tag}.json"
    out_png = eval_dir / f"best_of_grid__{tag}.png"
    out_json.write_text(json.dumps(result, indent=2))
    labels = [_display_label(eval_dir.name, stem) for stem in result["variants"]]
    x_label = AXIS_TITLES.get(eval_dir.name, "model")
    plot_fn = plot_best_of_grid_bar if len(observables) == 1 else plot_best_of_grid
    plot_fn(result, x_label=x_label, variant_labels=labels, save_path=out_png)
    print(f"    wrote {out_png}", flush=True)
    return result


def run_configs(*, config_paths: list[str | Path], observables: list[str] = DEFAULT_OBSERVABLES,
                n_seeds: int = 10, out_root: str = "datasets", scores_root: str = "scores",
                n_train: int | None = None, n_jobs: int = 1,
                out_json: str | Path | None = None, out_png: str | Path | None = None) -> dict:
    """Same grid as :func:`run`, but over an explicit flat list of config paths instead of one
    ``--eval-dir`` folder -- for combining variants across several ``configs/eval/<group>/``
    folders in one grid (e.g. ``configs/eval/*/*.yaml``), matching :mod:`eval.gradient_vs_r2`'s
    own ``--configs`` convention rather than introducing a second, folder-scoped flag pattern.

    Columns are labelled by each config's bare filename stem -- no :data:`VARIANT_LABELS`/
    :data:`AXIS_TITLES` lookup, since those are curated per single-folder group and a combined
    run mixes variants from different families that were never meant to share one labelling
    scheme; deliberately unopinionated rather than guessing a grouped label.
    """
    variants = [(Path(p).stem, Path(p)) for p in config_paths]
    if not variants:
        raise SystemExit("no config paths given")
    print(f"=== {len(variants)} configs -> {[n for n, _ in variants]}", flush=True)

    result = sweep_best_of_grid(variants, observables, n_seeds=n_seeds, out_root=out_root,
                                scores_root=scores_root, n_train=n_train, n_jobs=n_jobs)
    tag = "-".join(_safe_tag(o) for o in observables)
    out_json = Path(out_json) if out_json else Path(f"best_of_grid__{tag}.json")
    out_png = Path(out_png) if out_png else Path(f"best_of_grid__{tag}.png")
    out_json.write_text(json.dumps(result, indent=2))
    plot_best_of_grid(result, x_label="model", variant_labels=result["variants"],
                      save_path=out_png)
    print(f"    wrote {out_png}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Observable x model heatmap, one cell = best learner "
                                              "(mean over reseeded fits) for one configs/eval/ "
                                              "group, or an explicit flat list of configs.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval-dir", help="e.g. configs/eval/havlicek -- one folder's variants")
    group.add_argument("--configs", nargs="+", help="explicit config paths/globs, e.g. "
                                                     "configs/eval/*/*.yaml to combine every "
                                                     "subfolder into one grid")
    ap.add_argument("--observables", nargs="+", default=DEFAULT_OBSERVABLES)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel fits across (observable, "
                                                            "variant, learner, seed) cells; "
                                                            "1 = serial, -1 = all-but-one CPU "
                                                            "(default)")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-json", default=None, help="--configs mode only")
    ap.add_argument("--out-png", default=None, help="--configs mode only")
    args = ap.parse_args(argv)

    if args.configs:
        run_configs(config_paths=args.configs, observables=args.observables,
                   n_seeds=args.n_seeds, out_root=args.root, scores_root=args.scores_root,
                   n_train=args.n_train, n_jobs=args.n_jobs, out_json=args.out_json,
                   out_png=args.out_png)
    else:
        run(eval_dir=Path(args.eval_dir), observables=args.observables, n_seeds=args.n_seeds,
           out_root=args.root, scores_root=args.scores_root, n_train=args.n_train,
           n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
