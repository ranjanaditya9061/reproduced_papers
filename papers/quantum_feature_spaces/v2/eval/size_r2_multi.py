"""Held-out R^2 vs. circuit size, photonic vs. fermion, for several observables at once -- one
line plot per observable, both arms on the same axes.

    python eval/size_r2_multi.py \\
        --photonic-dir configs/size/size_photonic_fock --fermion-dir configs/size/size_fermion \\
        --observables parity prod_parity_diag_prime_sqrt connected_paritymaxcc_pair

The presentation's size-scaling section: three observables (a plain counting baseline, the
Monbroussou et al. quadratic number-phase observable, and the mode-pair graph reading), each shown
as its own R^2-vs-m line plot with two lines -- photonic ("bosonic") and fermion -- so a size trend
can be read off directly and compared against the paired-arm floor-check discipline
(:mod:`learner.compare`'s verdict table; see ``STATUS.md`` section 9.6 for why a shared collapse on
both arms licenses no conclusion about photonic specifically).

**Shots cutover past m=14.**  Every size tier at or below ``SHOTS_CUTOVER_M`` (14) is read exact
(``generation.shots=0``, whatever is already on disk); every tier strictly above it gets
``generation.shots = SHOTS_ABOVE_CUTOVER`` (10000) applied **in memory**, the same
:func:`config.ExperimentConfig`-mutation pattern :mod:`eval.r2_vs_shots` uses (no sibling YAML
written) -- both arms switch at the same ``m``, so the exact-vs-shots regime change never
confounds the photonic-vs-fermion comparison at a given size. Rationale: past ``m=14`` the exact
Fock-basis outcome count is large enough that generating/scoring the full exact distribution
becomes the bottleneck (`STATUS.md` notes the Fisher Jacobian alone is ~2GB at ``m=14``'s
77,520-outcome basis for a *different* metric; scoring/learning at that support size and beyond is
the same kind of cost), so a fixed finite shot budget is used instead, uniformly on both arms.

Reuses :func:`eval.size_r2_line._sizes_in`'s ``m{MM}k{KK}.yaml`` folder convention and
:func:`learner.auto.run_config` for fitting -- no new fitting machinery, only the cross-observable
loop and the shots cutover are new here.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/size_r2_multi.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_OBSERVABLES = ["parity", "prod_parity_diag_prime_sqrt", "connected_paritymaxcc_pair"]
DEFAULT_N_SEEDS = 10
SHOTS_CUTOVER_M = 14
SHOTS_ABOVE_CUTOVER = 10_000

_SIZE_STEM_RE = re.compile(r"^m(\d+)k(\d+)$")

#: Display label for an observable key -- kept local rather than importing from best_of_grid/
#: eval_obs to avoid coupling three otherwise-independent scripts to one shared label dict.
OBSERVABLE_LABELS: dict[str, str] = {
    "prod_parity_diag_prime_sqrt": "Monbroussou\n(p = O(sqrt k))",
    "connected_paritymaxcc_pair": "Parity x MaxCC\n(pair graph)",
}


def _safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _sizes_in(family_dir: Path) -> list[tuple[int, Path]]:
    """``[(m, path), ...]`` for every ``m{MM}k{KK}.yaml`` directly inside ``family_dir``, sorted by
    ``m`` ascending -- same convention as :func:`eval.size_r2_line._sizes_in`."""
    out = []
    for p in sorted(family_dir.glob("*.yaml")):
        mo = _SIZE_STEM_RE.match(p.stem)
        if not mo:
            continue
        out.append((int(mo.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def _cfg_for_m(cfg_path: Path, m: int, *, shots_cutover_m: int, shots_above: int):
    """Load ``cfg_path`` and, if ``m > shots_cutover_m``, return an in-memory copy with
    ``generation.shots = shots_above`` set -- otherwise return the config unmodified (exact,
    whatever ``generation.shots`` already is on disk, normally 0). Mirrors
    :func:`eval.r2_vs_shots._shots_variant_config`: no sibling YAML is written, and
    :func:`learner.auto.run_config`/:func:`learner.cache.cached_fit` accept an
    :class:`~config.ExperimentConfig` object directly, so this composes with the existing cache
    exactly as the shots sweep does.
    """
    from config import load_config

    cfg = load_config(cfg_path)
    if m > shots_cutover_m:
        cfg = copy.deepcopy(cfg)
        cfg.generation.shots = int(shots_above)
    return cfg


def _seed_fit_worker(task):
    """One ``(cell, seed)`` fit -- module-level so it's picklable for
    :func:`~circuit.spin.parallel_row_map`'s process pool. Generates the shots branch first when
    the cell is past the cutover (a no-op / cache hit if already generated)."""
    (cfg_path, m, observable, learner_name, out_root, scores_root, n_train, graph_density, seed,
     shots_cutover_m, shots_above, base_kwargs) = task
    from learner.auto import run_config
    from pipeline.generate import generate_shots

    cfg = _cfg_for_m(cfg_path, m, shots_cutover_m=shots_cutover_m, shots_above=shots_above)
    if cfg.generation.shots > 0:
        generate_shots(cfg, root=out_root)

    kwargs = dict(base_kwargs)
    kwargs["seed"] = seed
    try:
        res = run_config(cfg, observable, learner_name, out_root=out_root, scores_root=scores_root,
                         n_train=n_train, graph_density=graph_density, split_seed=seed, **kwargs)
        return res["r2"], None
    except (Exception, SystemExit) as exc:                  # noqa: BLE001 -- one bad seed must not
        return None, str(exc)                                # abort the rest of the sweep


def sweep_size_r2_multi(families: list[tuple[str, list[tuple[int, Path]]]],
                        observables: list[str], *, learners=None, n_seeds: int = DEFAULT_N_SEEDS,
                        out_root: str = "datasets", scores_root: str = "scores",
                        n_train: int | None = None, graph_density: float = 0.5,
                        shots_cutover_m: int = SHOTS_CUTOVER_M,
                        shots_above: int = SHOTS_ABOVE_CUTOVER, n_jobs: int = 1) -> dict:
    """R^2 for every ``(observable, family, m)`` cell -- best-of-``learners`` **at each seed**,
    then mean/stdev over ``n_seeds`` reseeded splits.  Same reduction order as
    :func:`eval.best_of_grid.sweep_best_of_grid` (see that function's docstring for why this
    differs from mean-per-learner-then-max, and why the difference matters whenever different
    learners win at different seeds).

    Returns ``{"observables": [...], "families": [...], "results": {obs: {"rows": [...]}},
    "shots_cutover_m": ..., "shots_above": ..., "n_seeds": ...}`` -- one ``rows`` list per
    observable, each row ``{"family": name, "m": [...], "r2": [...], "r2_std": [...]}``, the same
    shape :func:`eval.size_r2_line.plot_size_r2_line` already knows how to plot (this module's own
    :func:`plot_size_r2_multi` reuses that shape per observable).
    """
    from circuit.spin import parallel_row_map
    from learner.auto import DEFAULT_SWEEP_LEARNERS

    learners = learners or DEFAULT_SWEEP_LEARNERS
    family_names = [name for name, _ in families]

    cells = [(obs, name, m, cfg_path, lname, base_kwargs)
            for obs in observables
            for name, sizes in families
            for m, cfg_path in sizes
            for lname, base_kwargs in learners]
    tasks = [(cfg_path, m, obs, lname, out_root, scores_root, n_train, graph_density, seed,
             shots_cutover_m, shots_above, base_kwargs)
            for obs, name, m, cfg_path, lname, base_kwargs in cells
            for seed in range(int(n_seeds))]
    flat = parallel_row_map(_seed_fit_worker, tasks, n_jobs)

    n = int(n_seeds)
    cell_seed_scores: dict[tuple[str, str, int, str], list] = {}
    for i, (obs, name, m, cfg_path, lname, base_kwargs) in enumerate(cells):
        seed_results = flat[i * n:(i + 1) * n]
        seed_scores = []
        for seed, (r2, err) in enumerate(seed_results):
            if err is not None:
                print(f"[size_r2_multi] {obs}/{name}/m={m}/{lname}/seed={seed} failed: {err}")
            seed_scores.append(r2)
        cell_seed_scores[(obs, name, m, lname)] = seed_scores

    results: dict[str, dict] = {}
    for obs in observables:
        rows = []
        for name, sizes in families:
            ms, r2s, stds = [], [], []
            for m, _ in sizes:
                # best-of-learners AT EACH SEED, then mean/stdev over seeds.
                per_seed_best = []
                for seed in range(n):
                    seed_vals = [cell_seed_scores[(obs, name, m, lname)][seed]
                                for lname, _ in learners]
                    valid = [v for v in seed_vals if v is not None]
                    if valid:
                        per_seed_best.append(max(valid))
                ms.append(m)
                r2s.append(statistics.mean(per_seed_best) if per_seed_best else None)
                stds.append(statistics.stdev(per_seed_best) if len(per_seed_best) > 1 else 0.0)
            rows.append({"family": name, "m": ms, "r2": r2s, "r2_std": stds})
        results[obs] = {"rows": rows}

    return {"observables": list(observables), "families": family_names, "results": results,
            "shots_cutover_m": shots_cutover_m, "shots_above": shots_above, "n_seeds": n_seeds,
            "learners": [name for name, _ in learners]}


def plot_size_r2_multi(result: dict, *, save_dir: Path, tag_prefix: str = "size_r2") -> list[Path]:
    """One PNG per observable -- ``m`` on x, held-out R^2 on y, one line per family (photonic vs.
    fermion), error bars at +/- one stdev of the per-seed best-of-learners scores
    (``row["r2_std"]``, see :func:`sweep_size_r2_multi`'s best-of-three-per-seed reduction). House
    line-plot style: no title, y-axis fixed to ``[0, 1]``, light dashed grid, a vertical marker at
    the shots cutover so the exact/shots regime change is visible on the plot rather than only in
    the caption.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_paths = []
    cutover_m = result["shots_cutover_m"]
    for obs in result["observables"]:
        rows = result["results"][obs]["rows"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for row in rows:
            ms = row["m"]
            r2s = [np.nan if v is None else max(0.0, v) for v in row["r2"]]
            stds = [0.0 if s is None else s for s in row.get("r2_std", [0.0] * len(ms))]
            err = [s if np.isfinite(v) else 0.0 for v, s in zip(r2s, stds)]
            ax.errorbar(ms, r2s, yerr=err, marker="o", linewidth=1.5, markersize=4, capsize=3,
                       label=row["family"])

        if any(m > cutover_m for row in rows for m in row["m"]):
            ax.axvline(cutover_m, color="grey", linestyle=":", linewidth=1)

        ax.set_xlabel("m")
        ax.set_ylabel("Coefficient of\nDetermination ($R^2$)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.legend(loc="best", frameon=False)

        fig.tight_layout()
        out_path = save_dir / f"{tag_prefix}__{_safe_tag(obs)}.png"
        fig.savefig(out_path, dpi=150)
        out_paths.append(out_path)
        print(f"    wrote {out_path}", flush=True)
    return out_paths


def run(*, photonic_dir: Path, fermion_dir: Path, observables: list[str] = DEFAULT_OBSERVABLES,
       n_seeds: int = DEFAULT_N_SEEDS, out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None, graph_density: float = 0.5,
       shots_cutover_m: int = SHOTS_CUTOVER_M, shots_above: int = SHOTS_ABOVE_CUTOVER,
       n_jobs: int = 1, save_dir: Path | None = None) -> dict:
    photonic_sizes = _sizes_in(photonic_dir)
    fermion_sizes = _sizes_in(fermion_dir)
    if not photonic_sizes:
        raise SystemExit(f"no m{{MM}}k{{KK}}.yaml configs found in {photonic_dir}")
    if not fermion_sizes:
        raise SystemExit(f"no m{{MM}}k{{KK}}.yaml configs found in {fermion_dir}")
    print(f"=== photonic sizes -> {[m for m, _ in photonic_sizes]}", flush=True)
    print(f"=== fermion sizes  -> {[m for m, _ in fermion_sizes]}", flush=True)

    families = [("photonic", photonic_sizes), ("fermion", fermion_sizes)]
    result = sweep_size_r2_multi(families, observables, n_seeds=n_seeds, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train,
                                 graph_density=graph_density, shots_cutover_m=shots_cutover_m,
                                 shots_above=shots_above, n_jobs=n_jobs)

    save_dir = save_dir or photonic_dir.parent
    (save_dir / "size_r2_multi.json").write_text(json.dumps(result, indent=2))
    plot_size_r2_multi(result, save_dir=save_dir)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Held-out R^2 vs. size, photonic vs. fermion, for "
                                             "several observables -- one line plot per observable.")
    ap.add_argument("--photonic-dir", required=True, help="e.g. configs/size/size_photonic_fock")
    ap.add_argument("--fermion-dir", required=True, help="e.g. configs/size/size_fermion")
    ap.add_argument("--observables", nargs="+", default=DEFAULT_OBSERVABLES)
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--graph-density", type=float, default=0.5)
    ap.add_argument("--shots-cutover-m", type=int, default=SHOTS_CUTOVER_M)
    ap.add_argument("--shots-above", type=int, default=SHOTS_ABOVE_CUTOVER)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--save-dir", default=None, help="defaults to --photonic-dir's parent")
    args = ap.parse_args(argv)

    run(photonic_dir=Path(args.photonic_dir), fermion_dir=Path(args.fermion_dir),
       observables=args.observables, n_seeds=args.n_seeds, out_root=args.root,
       scores_root=args.scores_root, n_train=args.n_train, graph_density=args.graph_density,
       shots_cutover_m=args.shots_cutover_m, shots_above=args.shots_above, n_jobs=args.n_jobs,
       save_dir=Path(args.save_dir) if args.save_dir else None)


if __name__ == "__main__":
    main()
