"""``eval/best_of_grid.py``, at a fixed shot budget instead of the exact distribution.

    python eval/best_of_grid_shots.py --eval-dir configs/eval/photonic_encoding --shots 10000

Same grid (observables x variants, one cell = max over
:data:`learner.auto.DEFAULT_SWEEP_LEARNERS`, each averaged over ``--n-seeds`` reseeded splits) and
the same output convention as :mod:`eval.best_of_grid` -- this module only adds the shots layer on
top, reusing every other piece of it directly rather than re-implementing the grid/plot logic.

**Why a separate module, not a ``--shots`` flag on ``best_of_grid.py``.**  Every variant's config
needs ``generation.shots`` set to the swept budget before its shots are generated and its learners
fit -- :func:`eval.r2_vs_shots._shots_variant_config` builds that as an in-memory
:class:`~config.ExperimentConfig` copy (no sibling YAML written to disk), one per variant in the
folder, then hands those config objects to :func:`eval.best_of_grid.sweep_best_of_grid` in place of
the usual variant paths.  :func:`learner.auto.run_config`/:func:`learner.cache.cached_fit` accept
either (:func:`~learner.cache._resolve_config`), and the on-disk fit/score cache is keyed off the
resulting dataset identity, not off a path, so this is cache-equivalent to writing the file --
:func:`~circuit.spin.parallel_row_map`'s process-pool workers pickle the config objects across
process boundaries the same way they already pickle every other task argument.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/best_of_grid_shots.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(*, eval_dir: Path, shots: int, observables=None, n_seeds: int = 10,
       out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None, n_jobs: int = 1) -> dict:
    """Same as :func:`eval.best_of_grid.run`, but every variant's config is first given an
    in-memory ``generation.shots = shots`` override (:func:`eval.r2_vs_shots.
    _shots_variant_config`) and its shots generated (:func:`pipeline.generate.generate_shots`)
    before the grid is swept."""
    from eval.best_of_grid import (DEFAULT_OBSERVABLES, _display_label, _safe_tag, _variants_in,
                                   plot_best_of_grid, sweep_best_of_grid)
    from eval.r2_vs_shots import _shots_variant_config
    from pipeline.generate import generate_shots

    from eval.best_of_grid import AXIS_TITLES

    observables = observables or DEFAULT_OBSERVABLES
    base_variants = _variants_in(eval_dir)
    if not base_variants:
        raise SystemExit(f"no *.yaml configs found in {eval_dir}")

    shots_variants = []
    for stem, path in base_variants:
        cfg_n = _shots_variant_config(path, shots)
        generate_shots(cfg_n, root=out_root)
        shots_variants.append((stem, cfg_n))

    print(f"=== {eval_dir.name} @ shots={shots}: {len(shots_variants)} variants -> "
         f"{[n for n, _ in shots_variants]}", flush=True)

    result = sweep_best_of_grid(shots_variants, observables, n_seeds=n_seeds, out_root=out_root,
                                scores_root=scores_root, n_train=n_train, n_jobs=n_jobs)
    tag = "-".join(_safe_tag(o) for o in observables)
    out_json = eval_dir / f"best_of_grid_shots{shots}__{tag}.json"
    out_png = eval_dir / f"best_of_grid_shots{shots}__{tag}.pdf"
    out_json.write_text(json.dumps(result, indent=2))
    labels = [_display_label(eval_dir.name, stem) for stem, _ in base_variants]
    x_label = AXIS_TITLES.get(eval_dir.name, "model")
    plot_best_of_grid(result, x_label=x_label, variant_labels=labels, save_path=out_png)
    print(f"    wrote {out_png}", flush=True)
    return result


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Observable x model heatmap at a fixed shot budget, "
                                             "one cell = best learner (mean over reseeded fits).")
    ap.add_argument("--eval-dir", required=True, help="e.g. configs/eval/photonic_encoding")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--observables", nargs="+", default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    run(eval_dir=Path(args.eval_dir), shots=args.shots, observables=args.observables,
       n_seeds=args.n_seeds, out_root=args.root, scores_root=args.scores_root,
       n_train=args.n_train, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
