"""Drive both heatmap grids over every subfolder of ``configs/eval/``.

    python configs/run_eval_heatmaps.py
    python configs/run_eval_heatmaps.py --eval-dir configs/eval --mode both
    python configs/run_eval_heatmaps.py --learner svr --observables parity majority

Each subfolder of ``configs/eval/`` (e.g. ``photonic_encoding/``, ``spin_magic/``) holds one
``*.yaml`` per circuit variant to compare -- the encodings/structures/ablations that belong on the
same axis (see each folder's own configs).  For every subfolder this script builds:

* ``variant_observable.png`` / ``.json`` -- rows are the subfolder's variants, columns are
  ``--observables``, at one fixed ``--learner``
  (:func:`learner.auto.sweep_variant_observable_grid`).
* ``variant_learner.png`` / ``.json`` -- rows are the subfolder's variants, columns are
  ``--learners`` (default :data:`learner.auto.DEFAULT_SWEEP_LEARNERS`), at one fixed
  ``--observable`` (:func:`learner.auto.sweep_variant_learner_grid`).

``--mode`` picks one or both (default ``both``).  Both write into the subfolder itself, next to the
configs they were built from, so a subfolder's directory is self-contained: its inputs (the
per-variant YAMLs) and its outputs (the grids) sit together.

Requires every variant's dataset to already be generated (:mod:`configs.run_size_sweep` or
``python -m pipeline.generate``) -- this script only fits learners on saved artifacts, it does not
generate them.  A variant whose artifact is missing fails that cell (prints and continues; see
:func:`learner.auto.sweep_variant_observable_grid`), it does not stop the subfolder or the run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python configs/run_eval_heatmaps.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learner.auto import (
    DEFAULT_SWEEP_LEARNERS,
    plot_variant_learner_grid,
    plot_variant_observable_grid,
    sweep_variant_learner_grid,
    sweep_variant_observable_grid,
)

EVAL_DIR = Path(__file__).parent / "eval"
DEFAULT_OBSERVABLES = ["parity", "majority", "ent", "osc"]


def _variants_in(subfolder: Path) -> list[tuple[str, Path]]:
    """``[(stem, path), ...]`` for every ``*.yaml`` directly inside ``subfolder``, sorted by stem
    so the grid's row order is stable across runs."""
    return [(p.stem, p) for p in sorted(subfolder.glob("*.yaml"))]


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run(*, eval_dir: Path = EVAL_DIR, mode: str = "both",
       observables: list[str] = DEFAULT_OBSERVABLES, learner: str = "ridge",
       learners: tuple[tuple[str, dict], ...] = DEFAULT_SWEEP_LEARNERS, observable: str = "parity",
       out_root: str = "datasets", scores_root: str = "scores",
       n_train: int | None = None) -> list[tuple[Path, Exception]]:
    """For every subfolder of ``eval_dir``, build the grid(s) named by ``mode`` and save them
    (PNG + JSON) into that subfolder.

    Returns the ``(subfolder, exception)`` pairs that failed outright (e.g. an empty subfolder, or
    both grids raising) -- per-cell failures inside a grid are handled by the sweep functions
    themselves and do not appear here.
    """
    subfolders = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not subfolders:
        raise SystemExit(f"no subfolders found in {eval_dir}")

    failures = []
    for sub in subfolders:
        variants = _variants_in(sub)
        if not variants:
            print(f"=== {sub.name}: no *.yaml configs, skipping", flush=True)
            continue
        print(f"=== {sub.name}: {len(variants)} variants -> {[n for n, _ in variants]}", flush=True)

        if mode in ("observable", "both"):
            try:
                result = sweep_variant_observable_grid(
                    variants, observables, learner, out_root=out_root, scores_root=scores_root,
                    n_train=n_train)
                tag = f"{'-'.join(_safe_tag(o) for o in observables)}__{_safe_tag(learner)}"
                (sub / f"variant_observable__{tag}.json").write_text(json.dumps(result, indent=2))
                plot_variant_observable_grid(result, save_path=sub / f"variant_observable__{tag}.png")
                print(f"    wrote {sub / f'variant_observable__{tag}.png'}", flush=True)
            except Exception as exc:                       # noqa: BLE001 -- one bad subfolder must
                print(f"    variant_observable grid FAILED: {exc}", flush=True)  # not stop the rest
                failures.append((sub, exc))

        if mode in ("learner", "both"):
            try:
                result = sweep_variant_learner_grid(
                    variants, observable, learners=learners, out_root=out_root,
                    scores_root=scores_root, n_train=n_train)
                tag = f"{'-'.join(_safe_tag(n) for n, _ in learners)}__{_safe_tag(observable)}"
                (sub / f"variant_learner__{tag}.json").write_text(json.dumps(result, indent=2))
                plot_variant_learner_grid(result, save_path=sub / f"variant_learner__{tag}.png")
                print(f"    wrote {sub / f'variant_learner__{tag}.png'}", flush=True)
            except Exception as exc:                       # noqa: BLE001 -- see above
                print(f"    variant_learner grid FAILED: {exc}", flush=True)
                failures.append((sub, exc))

    return failures


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build variant-comparison heatmaps for every "
                                              "configs/eval/ subfolder.")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--mode", choices=["observable", "learner", "both"], default="both",
                    help="observable = variant x observable grid (one learner); "
                    "learner = variant x learner grid (one observable); both = both (default)")
    ap.add_argument("--observables", nargs="+", default=DEFAULT_OBSERVABLES,
                    help="columns for the variant x observable grid")
    ap.add_argument("--learner", default="ridge",
                    help="the one learner used for the variant x observable grid")
    ap.add_argument("--observable", default="parity",
                    help="the one observable used for the variant x learner grid")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    failures = run(eval_dir=Path(args.eval_dir), mode=args.mode, observables=args.observables,
                   learner=args.learner, observable=args.observable, out_root=args.root,
                   scores_root=args.scores_root, n_train=args.n_train)

    if failures:
        print(f"\n{len(failures)} subfolder grid(s) failed outright:", flush=True)
        for sub, exc in failures:
            print(f"  {sub.name}: {exc}", flush=True)
        raise SystemExit(1)
    print("all eval heatmaps generated successfully", flush=True)


if __name__ == "__main__":
    main()
