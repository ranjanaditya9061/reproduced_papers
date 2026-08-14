"""Held-out R^2 vs. circuit size for ONE model family, all three learners overlaid.

    python eval/size_learner_line.py --family configs/size/size_photonic_fock --observable parity

Where :mod:`eval.size_r2_line` fixes one learner and compares model families against each other
across size, this fixes one model family and compares :data:`learner.auto.DEFAULT_SWEEP_LEARNERS`
(``ridge``/``svr``/``mlp``) against each other across the SAME size axis -- the learner-agreement
check this session has repeatedly needed: a single-learner R^2 trend (e.g. the ``svr`` crash at
``photonic_fock``'s ``m=14``) can be a fit artifact rather than a real size effect, and the only way
to tell is to see whether the other two learners agree at every size, not just at one.

One ``m{MM}k{KK}.yaml`` per size tier inside ``--family`` (the same layout
:func:`eval.size_r2_line._sizes_in` reads -- see ``configs/run_size_sweep.py``'s ``GROWTH_STEMS``
for which families have these generated). Fits every learner in ``--learners`` on every size tier via
:func:`learner.auto.run_config` (no new fitting machinery), one line per learner, ``m`` on the x-axis.

Requires every size tier's dataset to already be generated -- this only reads saved artifacts, it
does not generate them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/size_learner_line.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_OBSERVABLE = "parity"
_SIZE_STEM_RE = re.compile(r"^m(\d+)k(\d+)$")


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _sizes_in(family_dir: Path) -> list[tuple[int, Path]]:
    """``[(m, path), ...]`` for every ``m{MM}k{KK}.yaml`` directly inside ``family_dir``, sorted by
    ``m`` ascending -- identical convention to :func:`eval.size_r2_line._sizes_in`."""
    out = []
    for p in sorted(family_dir.glob("*.yaml")):
        m = _SIZE_STEM_RE.match(p.stem)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def sweep_size_learners(sizes: list[tuple[int, Path]], observable: str,
                        learners: tuple[tuple[str, dict], ...], *, out_root: str = "datasets",
                        scores_root: str = "scores", n_train: int | None = None) -> dict:
    """Held-out R^2 for every ``(learner, size)`` pair on one family, at one fixed ``observable``.

    One bad cell (missing artifact, a fit that raises) is recorded as ``None`` and printed, not
    fatal to the rest -- same per-cell failure handling as every other eval driver in this package.
    """
    from learner.auto import run_config

    learner_names = [name for name, _ in learners]
    rows = []
    for learner_name, kwargs in learners:
        ms, r2s = [], []
        for m, cfg_path in sizes:
            try:
                res = run_config(cfg_path, observable, learner_name, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train, **kwargs)
                r2s.append(res["r2"])
            except Exception as exc:                       # noqa: BLE001 -- one bad cell must not
                print(f"[size_learner_line] {learner_name}/m={m} failed: {exc}")  # not stop the rest
                r2s.append(None)
            ms.append(m)
        rows.append({"learner": learner_name, "m": ms, "r2": r2s})

    return {"learners": learner_names, "rows": rows, "observable": observable}


def plot_size_learner_line(result: dict, *, family_name: str = "", save_path: str | Path | None = None,
                           show: bool = False):
    """One line per learner from :func:`sweep_size_learners`'s output: ``m`` on x, held-out R^2 on y.

    A missing cell (``None``) leaves a gap in that learner's line rather than plotting as 0 -- a fit
    that could not even be attempted is not the same as a fit that scored 0.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    observable = result["observable"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in result["rows"]:
        ms = row["m"]
        r2s = [np.nan if v is None else v for v in row["r2"]]
        ax.plot(ms, r2s, marker="o", linewidth=2, label=row["learner"])

    ax.axhline(0.0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("m")
    ax.set_ylabel("held-out R^2")
    title = f"R^2 vs. size: {observable}"
    if family_name:
        title += f" ({family_name})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def run(*, family: Path, observable: str = DEFAULT_OBSERVABLE, out_root: str = "datasets",
       scores_root: str = "scores", n_train: int | None = None) -> dict:
    """Sweep and plot one family's ``ridge``/``svr``/``mlp`` R^2 vs. size, saving PNG + JSON into
    ``family`` itself.

    Output filenames are tagged with ``observable`` (:func:`_safe_tag`) so two runs at different
    observables do not overwrite each other's plot/JSON.
    """
    from learner.auto import DEFAULT_SWEEP_LEARNERS

    sizes = _sizes_in(family)
    if not sizes:
        raise SystemExit(f"no m{{MM}}k{{KK}}.yaml configs found in {family}")
    print(f"=== {family.name}: sizes -> {[m for m, _ in sizes]}, "
          f"learners -> {[n for n, _ in DEFAULT_SWEEP_LEARNERS]}", flush=True)

    result = sweep_size_learners(sizes, observable, DEFAULT_SWEEP_LEARNERS, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train)
    tag = _safe_tag(observable)
    (family / f"size_learner_r2__{tag}.json").write_text(json.dumps(result, indent=2))
    plot_size_learner_line(result, family_name=family.name,
                           save_path=family / f"size_learner_r2__{tag}.png")
    print(f"wrote {family / f'size_learner_r2__{tag}.png'}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plot held-out R^2 vs. size for one configs/size/ "
                                              "family, one line per learner (ridge/svr/mlp).")
    ap.add_argument("--family", required=True,
                    help="path to one configs/size/size_<name>/ folder (e.g. "
                    "configs/size/size_photonic_fock)")
    ap.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    args = ap.parse_args(argv)

    run(family=Path(args.family), observable=args.observable, out_root=args.root,
       scores_root=args.scores_root, n_train=args.n_train)


if __name__ == "__main__":
    main()
