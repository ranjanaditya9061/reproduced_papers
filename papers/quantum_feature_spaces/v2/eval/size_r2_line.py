"""Held-out R^2 vs. circuit size, for one learner and one observable, as a line plot.

    python eval/size_r2_line.py --size-dir configs/size --learner ridge --observable parity

For each subfolder of ``configs/size/`` (one ``m{MM}k{KK}.yaml`` per size tier -- see
``configs/run_size_sweep.py``'s ``GROWTH_STEMS`` for which model families these mirror), this fits
one fixed ``--learner`` on one fixed ``--observable`` at every size tier
(:func:`learner.auto.run_config`, the same fit/score path :mod:`learner.auto`'s own heatmaps use --
no new fitting machinery here), and plots held-out R^2 against ``m``. One line per subfolder, all on
the same axes, so different model families' size-scaling can be compared directly.

**Each ``(family, m)`` point is a mean over ``--n-seeds`` reseeded fits**, not a single fit -- same
seeding discipline as :mod:`eval.best_of_grid`: every seed iteration overrides both the learner's own
``seed`` kwarg (so ``mlp``'s weight-init/batch-order variance is captured) and ``split_seed`` (so
``ridge``/``svr``, deterministic given a split, still get genuinely different train/test partitions
across seeds instead of ``n_seeds`` identical repeats). See that module's docstring for the full
reasoning. The raw per-seed scores are kept in the saved JSON's ``"detail"`` field even though only
the mean feeds the line.

This is the direct trainability read on the hardness question :mod:`eval.efficiency_line` approaches
via Fisher information: does a fixed learner's ability to predict the observable from ``x`` degrade
as the circuit grows, and does that degradation differ between families (e.g. a permanent-based
sampler vs. its determinant-based twin, or a feature map with weaker Fisher signal vs. one with
stronger)?  See that module's docstring for why small Fisher trace is a near-guarantee of poor
learnability but large trace is not a guarantee of good learnability -- R^2 here is what actually
settles it for a given learner, rather than bounding what any learner could achieve.

Requires every size tier's dataset to already be generated (:mod:`configs.run_size_sweep`, whose
``GROWTH_STEMS`` produces exactly the ``(m, k)`` tiers these configs mirror) -- this only reads saved
artifacts, it does not generate them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/size_r2_line.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE_DIR = Path(__file__).resolve().parents[1] / "configs" / "size"
DEFAULT_LEARNER = "ridge"
DEFAULT_OBSERVABLE = "parity"
DEFAULT_N_SEEDS = 10

_SIZE_STEM_RE = re.compile(r"^m(\d+)k(\d+)$")


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _cell_stds(row: dict) -> list[float]:
    """``row["r2_std"]`` if present; otherwise recomputed from ``row["detail"]`` (older saved
    JSONs, from before ``r2_std`` was added, still carry the raw per-seed scores needed to derive
    it -- only a JSON with neither field falls back to all-zero bars)."""
    if row.get("r2_std") is not None:
        return row["r2_std"]
    import statistics
    stds = []
    for seed_scores in row.get("detail") or [[] for _ in row["r2"]]:
        valid = [s for s in seed_scores if s is not None]
        stds.append(statistics.stdev(valid) if len(valid) > 1 else 0.0)
    return stds


def _sizes_in(subfolder: Path) -> list[tuple[int, Path]]:
    """``[(m, path), ...]`` for every ``m{MM}k{KK}.yaml`` directly inside ``subfolder``, sorted by
    ``m`` ascending -- the config's own ``problem.m``/``problem.k`` are the source of truth
    (:func:`config.load_config` reads them), the filename is only used for sorting.
    """
    out = []
    for p in sorted(subfolder.glob("*.yaml")):
        m = _SIZE_STEM_RE.match(p.stem)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def _seed_fit_worker(task):
    """One ``(cell, seed)`` fit -- module-level so it's picklable for
    :func:`~v2.circuit.spin.parallel_row_map`'s process pool.  Returns ``(r2, error)`` rather than
    raising/printing here: printing happens in the parent, once results are back in task order,
    so output from a parallel run reads the same as the old serial loop instead of interleaving
    across worker processes."""
    (cfg_path, observable, learner_name, out_root, scores_root, n_train, seed, force,
     learner_kwargs) = task
    from learner.auto import run_config
    print(task)

    kwargs = dict(learner_kwargs)
    kwargs["seed"] = seed
    try:
        res = run_config(cfg_path, observable, learner_name, out_root=out_root,
                         scores_root=scores_root, n_train=n_train, split_seed=seed, force=force,
                         **kwargs)
        return res["r2"], None
    except (Exception, SystemExit) as exc:                  # noqa: BLE001 -- one bad seed must not
        return None, str(exc)                                # abort the rest of the sweep
                                                              # (run_config raises SystemExit, not
                                                              # Exception, on a missing dataset)


def sweep_size_r2(families: list[tuple[str, list[tuple[int, Path]]]], observable: str,
                  learner_name: str, *, out_root: str = "datasets", scores_root: str = "scores",
                  n_train: int | None = None, n_seeds: int = DEFAULT_N_SEEDS, n_jobs: int = 1,
                  force: bool = False, **learner_kwargs) -> dict:
    """Held-out R^2 for every ``(family, size)`` pair, at one fixed ``observable``/``learner_name``,
    each pair averaged over ``n_seeds`` reseeded fits (see module docstring).

    ``families`` is ``[(name, [(m, cfg_path), ...]), ...]`` -- see :func:`_sizes_in`.  One bad seed
    (missing artifact, a fit that raises) is recorded as ``None`` within that cell's seed list and
    printed, not fatal to the rest -- same per-cell failure handling as the other eval drivers.  A
    cell where every seed failed reports ``r2=None`` for that cell.

    Every ``(family, m, seed)`` fit is independent -- nothing here shares state across seeds or
    cells -- so they are flattened into one task list and handed to
    :func:`~v2.circuit.spin.parallel_row_map`, the same ordered process-pool map
    :mod:`v2.circuit.prep`'s row workers use, rather than looping in serial.  ``n_jobs=1`` (the
    default) stays serial; ``-1`` uses all-but-one CPU.  Learner fits release the GIL for their
    actual compute (numpy/torch), but with hundreds of small fits the process-per-worker model
    still wins over threads by avoiding GIL contention on the Python-side bookkeeping between fits.
    """
    from circuit.spin import parallel_row_map

    cells = [(name, m, cfg_path) for name, sizes in families for m, cfg_path in sizes]
    tasks = [(cfg_path, observable, learner_name, out_root, scores_root, n_train, seed, force,
             learner_kwargs)
            for _, _, cfg_path in cells for seed in range(int(n_seeds))]
    flat = parallel_row_map(_seed_fit_worker, tasks, n_jobs)

    import statistics

    rows_by_family: dict[str, dict] = {}
    n = int(n_seeds)
    for i, (name, m, _) in enumerate(cells):
        seed_results = flat[i * n:(i + 1) * n]
        seed_scores = []
        for seed, (r2, err) in enumerate(seed_results):
            if err is not None:
                print(f"[size_r2_line] {name}/m={m}/seed={seed} failed: {err}")
            seed_scores.append(r2)
        valid = [s for s in seed_scores if s is not None]
        row = rows_by_family.setdefault(name, {"family": name, "m": [], "r2": [], "r2_std": [],
                                               "detail": []})
        row["m"].append(m)
        row["r2"].append(statistics.mean(valid) if valid else None)
        row["r2_std"].append(statistics.stdev(valid) if len(valid) > 1 else 0.0)
        row["detail"].append(seed_scores)

    rows = [rows_by_family[name] for name, _ in families]
    return {"families": [r["family"] for r in rows], "rows": rows, "observable": observable,
            "learner": learner_name, "n_seeds": n_seeds}


def plot_size_r2_line(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """One line per family from :func:`sweep_size_r2`'s output: ``m`` on x, held-out R^2 on y,
    error bars at +/- one seed stdev (:func:`_cell_stds`, 0 for a cell fit at only one seed).

    A missing cell (``None``) leaves a gap in that family's line rather than plotting as 0 -- a
    fit that could not even be attempted is not the same as a fit that scored 0.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    observable, learner = result["observable"], result["learner"]
    n_seeds = result.get("n_seeds", 1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in result["rows"]:
        ms = row["m"]
        r2s = [np.nan if v is None else v for v in row["r2"]]
        stds = [np.nan if v is None else s for v, s in zip(row["r2"], _cell_stds(row))]
        ax.errorbar(ms, r2s, yerr=stds, marker="o", linewidth=2, capsize=4, label=row["family"])

    ax.axhline(0.0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("m")
    ax.set_ylabel("held-out R^2")
    ax.set_title(f"R^2 vs. size: {observable} x {learner} (mean over {n_seeds} seeds)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def run(*, size_dir: Path = SIZE_DIR, observable: str = DEFAULT_OBSERVABLE,
       learner: str = DEFAULT_LEARNER, out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None, n_seeds: int = DEFAULT_N_SEEDS, n_jobs: int = 1,
       force: bool = False) -> dict:
    """Walk every subfolder of ``size_dir``, build one combined R^2-vs-size plot, and save it
    (PNG + JSON) directly into ``size_dir``.

    Unlike :func:`eval.efficiency_line.run` / :func:`eval.violin.run` (one plot per subfolder), this
    produces ONE plot with one line per subfolder, since the point is comparing families against
    each other on the same size axis rather than describing one family in isolation.

    Output filenames are tagged with ``observable`` and ``learner`` (filesystem-safe fragments) so
    two runs at different observables/learners do not overwrite each other's plot/JSON.

    **Always re-walks and re-derives the sweep** -- no separate whole-sweep cache of its own.  Every
    ``(family, m, seed)`` fit goes through :func:`~learner.auto.run_config`, which is itself cached
    per-cell in :func:`~learner.cache.cached_fit` (one on-disk cache, shared by every learner-fitting
    call site in this repo, rather than this module keeping its own second, coarser cache blind to
    ``n_train``/``graph_density``/hyperparameter overrides on top of it). ``force`` is forwarded to
    every ``run_config`` call so it still means "ignore the cache and refit", same as elsewhere --
    the assembled JSON is written every run purely as a durable output artifact, not read back as a
    cache.
    """
    tag = f"{_safe_tag(observable)}__{_safe_tag(learner)}"
    out_json = size_dir / f"size_r2__{tag}.json"
    out_png = size_dir / f"size_r2__{tag}.png"

    subfolders = sorted(p for p in size_dir.iterdir() if p.is_dir())
    if not subfolders:
        raise SystemExit(f"no subfolders found in {size_dir}")

    families = []
    for sub in subfolders:
        sizes = _sizes_in(sub)
        if not sizes:
            print(f"=== {sub.name}: no m{{MM}}k{{KK}}.yaml configs, skipping", flush=True)
            continue
        print(f"=== {sub.name}: sizes -> {[m for m, _ in sizes]}", flush=True)
        families.append((sub.name, sizes))

    result = sweep_size_r2(families, observable, learner, out_root=out_root,
                           scores_root=scores_root, n_train=n_train, n_seeds=n_seeds,
                           n_jobs=n_jobs, force=force)
    out_json.write_text(json.dumps(result, indent=2))

    _print_result(result)
    plot_size_r2_line(result, save_path=out_png)
    print(f"wrote {out_png}", flush=True)
    return result


def _print_result(result: dict) -> None:
    """One line per ``(family, m)`` cell: ``r2 = mean +/- stdev`` over ``result["n_seeds"]``
    seeds, ``n/a`` for a cell where every seed failed."""
    n_seeds = result.get("n_seeds", 1)
    print(f"--- {result['observable']} x {result['learner']} (mean over {n_seeds} seeds) ---")
    for row in result["rows"]:
        for m, r2, std in zip(row["m"], row["r2"], _cell_stds(row)):
            cell = "n/a" if r2 is None else f"{r2:.4f} +/- {std:.4f}"
            print(f"{row['family']:>20}  m={m:<4}  r2 = {cell}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plot held-out R^2 vs. circuit size, one line per "
                                              "configs/size/ subfolder.")
    ap.add_argument("--size-dir", default=str(SIZE_DIR))
    ap.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    ap.add_argument("--learner", default=DEFAULT_LEARNER)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel fits across (family, m, seed) "
                                                            "cells; 1 = serial, -1 = all-but-one CPU "
                                                            "(default)")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true", help="ignore any cached per-cell fit "
                                                          "(learner.cache) and refit every cell")
    args = ap.parse_args(argv)

    run(size_dir=Path(args.size_dir), observable=args.observable, learner=args.learner,
       out_root=args.root, scores_root=args.scores_root, n_train=args.n_train,
       n_seeds=args.n_seeds, n_jobs=args.n_jobs, force=args.force)


if __name__ == "__main__":
    main()
