"""Per-variant observable-label violin plots -- no learner involved, just the raw ``(N,)`` labels.

    python eval/violin.py --eval-dir configs/eval
    python eval/violin.py --eval-dir configs/eval --observable majority

For each subfolder of ``configs/eval/`` (one ``*.yaml`` per circuit variant -- see
``configs/run_eval_heatmaps.py``'s docstring for what the folders mean), this loads the saved
artifact for every variant and scores it against one fixed observable
(:func:`pipeline.score.load_soft`), then draws one violin per variant showing how that observable's
values are distributed across the dataset. This is upstream of learnability: before asking "can a
learner recover this observable from x", it shows what the observable's own label distribution
looks like per variant -- e.g. a label collapsed onto a few values is a different failure mode from
a learner failing to fit a genuinely spread-out one.

Requires every variant's dataset to already be generated (:mod:`configs.run_size_sweep` or
``python -m pipeline.generate``) -- this only reads saved artifacts, it does not generate them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/violin.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"
DEFAULT_OBSERVABLE = "parity"


def _variants_in(subfolder: Path) -> list[tuple[str, Path]]:
    """``[(stem, path), ...]`` for every ``*.yaml`` directly inside ``subfolder``, sorted by stem
    so the violin order is stable across runs -- mirrors
    :func:`configs.run_eval_heatmaps._variants_in`."""
    return [(p.stem, p) for p in sorted(subfolder.glob("*.yaml"))]


def variant_labels(cfg_path: str | Path, observable: str, *, out_root: str = "datasets",
                   scores_root: str = "scores", graph_density: float = 0.5):
    """``(N,)`` raw observable labels for one config -- the same load path :func:`learner.auto.run_config`
    uses, minus the fit/split/learner step: every row's label, not just the held-out ones, since a
    violin describes the whole dataset's spread rather than a train/test split.
    """
    from config import load_config
    from model import build_model
    from pipeline.artifact import exact_path
    from pipeline.distribution import load_dist
    from pipeline.score import load_soft

    cfg = load_config(cfg_path)
    path = exact_path(cfg, build_model(cfg), out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run pipeline.generate --config {cfg_path}")

    load_dist(path)                                  # validates the artifact is readable
    return load_soft(path, observable, scores_root=scores_root, graph_density=graph_density)


def sweep_variant_observable_dist(variants: list[tuple[str, "str | Path"]], observable: str, *,
                                  out_root: str = "datasets", scores_root: str = "scores") -> dict:
    """``(N,)`` labels for every variant, at one fixed ``observable``.

    One bad variant (e.g. a missing artifact) is recorded as an empty list and printed, not fatal
    to the rest -- same per-cell failure handling as :mod:`learner.auto`'s grid sweeps.
    """
    names, labels = [], []
    for name, cfg_path in variants:
        try:
            y = variant_labels(cfg_path, observable, out_root=out_root, scores_root=scores_root)
            labels.append([float(v) for v in y.double()])
        except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad variant must not
                                                        # (pipeline.score/artifact raise SystemExit,
                                                        # not Exception, on a missing dataset -- a
                                                        # bare `except Exception` here silently lets
                                                        # that kill the whole process instead of
                                                        # skipping just this variant, per the module's
                                                        # own documented per-cell resilience)
            print(f"[violin] {name} failed: {exc}")     # abort the rest of the plot
            labels.append([])
        names.append(name)

    return {"variants": names, "observable": observable, "labels": labels}


def plot_variant_observable_violin(result: dict, *, save_path: str | Path | None = None,
                                   show: bool = False):
    """One violin per variant, from :func:`sweep_variant_observable_dist`'s output.

    A variant with no labels (a failed load) still gets its own x position, left empty, so the
    variant order and count always match the subfolder's configs regardless of which ones failed.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names, observable = result["variants"], result["observable"]
    labels = result["labels"]
    positions = list(range(1, len(names) + 1))
    present = [(pos, np.asarray(y, dtype=float)) for pos, y in zip(positions, labels) if y]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names) + 2), 5))
    if present:
        pos, data = zip(*present)
        parts = ax.violinplot(data, positions=list(pos), showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor("#4C72B0")
            body.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("variant")
    ax.set_ylabel(f"{observable} label")
    ax.set_title(f"Label distribution: {observable} x variant")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment for one CLI input -- keeps ``[A-Za-z0-9_.-]``, replaces
    everything else with ``_``, so ``--observable`` names always produce a valid path component
    across the platforms this repo runs on (notably Windows, where ``:``/``<``/``>`` etc. are
    illegal in filenames)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run(*, eval_dir: Path = EVAL_DIR, observable: str = DEFAULT_OBSERVABLE,
       out_root: str = "datasets", scores_root: str = "scores") -> list[tuple[Path, Exception]]:
    """For every subfolder of ``eval_dir``, build one violin plot and save it (PNG + JSON) into
    that subfolder.  Mirrors :func:`configs.run_eval_heatmaps.run`'s walk and failure handling.

    Output filenames are tagged with ``observable`` (:func:`_safe_tag`) so two runs at different
    observables do not overwrite each other's plot/JSON in the same subfolder.
    """
    subfolders = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not subfolders:
        raise SystemExit(f"no subfolders found in {eval_dir}")

    tag = _safe_tag(observable)
    failures = []
    for sub in subfolders:
        variants = _variants_in(sub)
        if not variants:
            print(f"=== {sub.name}: no *.yaml configs, skipping", flush=True)
            continue
        print(f"=== {sub.name}: {len(variants)} variants -> {[n for n, _ in variants]}", flush=True)
        try:
            result = sweep_variant_observable_dist(variants, observable, out_root=out_root,
                                                    scores_root=scores_root)
            (sub / f"variant_violin__{tag}.json").write_text(json.dumps(result, indent=2))
            plot_variant_observable_violin(result, save_path=sub / f"variant_violin__{tag}.png")
            print(f"    wrote {sub / f'variant_violin__{tag}.png'}", flush=True)
        except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad subfolder must not
            print(f"    violin plot FAILED: {exc}", flush=True)  # stop the rest of the run
            failures.append((sub, exc))

    return failures


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build a variant x observable-label violin plot for "
                                              "every configs/eval/ subfolder.")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    failures = run(eval_dir=Path(args.eval_dir), observable=args.observable, out_root=args.root,
                   scores_root=args.scores_root)

    if failures:
        print(f"\n{len(failures)} subfolder plot(s) failed outright:", flush=True)
        for sub, exc in failures:
            print(f"  {sub.name}: {exc}", flush=True)
        raise SystemExit(1)
    print("all eval violin plots generated successfully", flush=True)


if __name__ == "__main__":
    main()
