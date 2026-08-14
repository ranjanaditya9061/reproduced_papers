"""Held-out R^2 vs. circuit size, for one learner and one observable, as a line plot.

    python eval/size_r2_line.py --size-dir configs/size --learner ridge --observable parity

For each subfolder of ``configs/size/`` (one ``m{MM}k{KK}.yaml`` per size tier -- see
``configs/run_size_sweep.py``'s ``GROWTH_STEMS`` for which model families these mirror), this fits
one fixed ``--learner`` on one fixed ``--observable`` at every size tier
(:func:`learner.auto.run_config`, the same fit/score path :mod:`learner.auto`'s own heatmaps use --
no new fitting machinery here), and plots held-out R^2 against ``m``. One line per subfolder, all on
the same axes, so different model families' size-scaling can be compared directly.

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

_SIZE_STEM_RE = re.compile(r"^m(\d+)k(\d+)$")


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


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


def sweep_size_r2(families: list[tuple[str, list[tuple[int, Path]]]], observable: str,
                  learner_name: str, *, out_root: str = "datasets", scores_root: str = "scores",
                  n_train: int | None = None, **learner_kwargs) -> dict:
    """Held-out R^2 for every ``(family, size)`` pair, at one fixed ``observable``/``learner_name``.

    ``families`` is ``[(name, [(m, cfg_path), ...]), ...]`` -- see :func:`_sizes_in`.  One bad cell
    (missing artifact, a fit that raises) is recorded as ``None`` and printed, not fatal to the rest
    -- same per-cell failure handling as the other eval drivers.
    """
    from learner.auto import run_config

    rows = []
    for name, sizes in families:
        ms, r2s = [], []
        for m, cfg_path in sizes:
            try:
                res = run_config(cfg_path, observable, learner_name, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train, **learner_kwargs)
                r2s.append(res["r2"])
            except Exception as exc:                       # noqa: BLE001 -- one bad cell must not
                print(f"[size_r2_line] {name}/m={m} failed: {exc}")  # abort the rest of the sweep
                r2s.append(None)
            ms.append(m)
        rows.append({"family": name, "m": ms, "r2": r2s})

    return {"families": [r["family"] for r in rows], "rows": rows, "observable": observable,
            "learner": learner_name}


def plot_size_r2_line(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """One line per family from :func:`sweep_size_r2`'s output: ``m`` on x, held-out R^2 on y.

    A missing cell (``None``) leaves a gap in that family's line rather than plotting as 0 -- a
    fit that could not even be attempted is not the same as a fit that scored 0.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    observable, learner = result["observable"], result["learner"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in result["rows"]:
        ms = row["m"]
        r2s = [np.nan if v is None else v for v in row["r2"]]
        ax.plot(ms, r2s, marker="o", linewidth=2, label=row["family"])

    ax.axhline(0.0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("m")
    ax.set_ylabel("held-out R^2")
    ax.set_title(f"R^2 vs. size: {observable} x {learner}")
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
       n_train: int | None = None) -> dict:
    """Walk every subfolder of ``size_dir``, build one combined R^2-vs-size plot, and save it
    (PNG + JSON) directly into ``size_dir``.

    Unlike :func:`eval.efficiency_line.run` / :func:`eval.violin.run` (one plot per subfolder), this
    produces ONE plot with one line per subfolder, since the point is comparing families against
    each other on the same size axis rather than describing one family in isolation.

    Output filenames are tagged with ``observable`` and ``learner`` (filesystem-safe fragments) so
    two runs at different observables/learners do not overwrite each other's plot/JSON.
    """
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
                           scores_root=scores_root, n_train=n_train)
    tag = f"{_safe_tag(observable)}__{_safe_tag(learner)}"
    (size_dir / f"size_r2__{tag}.json").write_text(json.dumps(result, indent=2))
    plot_size_r2_line(result, save_path=size_dir / f"size_r2__{tag}.png")
    print(f"wrote {size_dir / f'size_r2__{tag}.png'}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plot held-out R^2 vs. circuit size, one line per "
                                              "configs/size/ subfolder.")
    ap.add_argument("--size-dir", default=str(SIZE_DIR))
    ap.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    ap.add_argument("--learner", default=DEFAULT_LEARNER)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    run(size_dir=Path(args.size_dir), observable=args.observable, learner=args.learner,
       out_root=args.root, scores_root=args.scores_root, n_train=args.n_train)


if __name__ == "__main__":
    main()
