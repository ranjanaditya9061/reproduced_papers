"""Learnability (R^2) vs. shot budget N, at fixed model/size/observable/encoding -- alongside the
local and global resolvability margins at the SAME N values, so the plot shows whether R^2 actually
starts climbing where the resolvability threshold crosses.

    python eval/r2_vs_shots.py --config configs/photonic.yaml --observable parity \
        --shots 100 1000 10000 100000 --n-seeds 10

Default target: boson sampling, m=6 k=3, phase encoding -- pass ``--config`` pointing at that
config (e.g. ``configs/eval/photonic_encoding/phase.yaml``).

**Resolvability margins are cheap to add across the whole ``N`` sweep.**  Unlike R^2 (needs shots
actually drawn and a learner fit per ``N``), :func:`metrics.local_ratio.eps_local_of_N`/
:func:`metrics.global_ratio.eps_global_of_N` are closed-form functions of ``N``, evaluated from the
already-cached exact ``Var[O|x]``/``Var_circ`` -- no new shots, no refitting, so every ``N`` in the
sweep gets both margins essentially for free.  ``mean_local_margin(N) = mean(min_delta -
eps_local(N))`` over the pool; ``global_margin(N) = sqrt(Var_circ) - eps_global(N)`` (Proposition
1's own natural unit).  Both are POSITIVE where resolvable, NEGATIVE where shot noise swamps the
signal -- plotted on a second panel, sharing the log-``N`` x-axis with R^2, so a reader can see
whether the zero-crossing lines up with where R^2 actually starts climbing.

**Real shots only.**  This sweep goes through the model's own native sampler
(:meth:`model.base.DistributionModel.shot_counts` -- Clifford-and-Clifford for photonic ``fock``,
resolved directly, no fallback) via :mod:`pipeline.generate`/:mod:`pipeline.score`, exactly the
same path :mod:`learner.cache` uses for any shots-based config.  It does **not** use
:mod:`metrics.shot_sampler`'s multinomial-from-``p`` fallback -- that module is explicitly scoped to
models without a native sampler (``model.supports_shots == False``), and boson sampling has one.

**One config, N overridden in memory per sweep point.**  ``generation.shots`` is a config field, but
writing one YAML file per ``N`` in a sweep is unnecessary busywork -- :class:`config.ExperimentConfig`
is a plain (non-frozen) dataclass, so this loads the base config once and sets ``cfg.generation.shots
= N`` per iteration before calling :func:`pipeline.generate.generate_shots`/
:func:`pipeline.score.load_dataset` directly.  Each ``N`` still lands in its own, correctly-tagged
cache directory (:func:`pipeline.shots.shot_source_tag` folds the shot count into the path), so nothing
here bypasses or collides with the normal on-disk cache -- re-running this script is a cache hit for
every ``N`` already generated.

Fits :data:`learner.auto.DEFAULT_SWEEP_LEARNERS` (ridge/svr/mlp) at each ``N``, each averaged over
``--n-seeds`` reseeded train/test splits, and reports the max across learners -- the same
max-of-per-learner-means convention as :func:`eval.best_of_grid.sweep_best_of_grid`, for the same
reason (typical best-learner performance, not a lucky-seed outlier).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/r2_vs_shots.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _shots_variant_path(cfg_path: str | Path, n_shots: int) -> Path:
    """A real, permanent sibling config with ``generation.shots = n_shots`` set -- not a temp file.

    :func:`learner.auto.run_config`/:func:`learner.cache.cached_fit` only accept a **path**
    (they call :func:`config.load_config` unconditionally, so an in-memory
    :class:`config.ExperimentConfig` cannot be passed through them), so each ``N`` in the sweep
    needs its own config file on disk to reuse the normal caching path -- written once, alongside
    the base config, at ``<base>__shots<N>.yaml``, so a re-run of this script is a cache hit both
    at the config-file level and at :func:`~pipeline.generate.generate_shots`'s own shots-store
    level, exactly like any other config in this repo (matching the ``configs/size_sweep_shots/``
    naming convention already used elsewhere for this exact purpose).
    """
    import yaml

    base = Path(cfg_path)
    variant_path = base.with_name(f"{base.stem}__shots{int(n_shots)}{base.suffix}")
    if not variant_path.exists():
        raw = yaml.safe_load(base.read_text()) or {}
        raw.setdefault("generation", {})["shots"] = int(n_shots)
        variant_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return variant_path


def _r2_at_shots(cfg_path: str | Path, observable: str, n_shots: int, *, learners, n_seeds: int,
                 out_root: str, scores_root: str, n_train: int | None) -> dict:
    """``{learner_name: [r2_seed0, ...]}`` for one ``(cfg_path, observable, n_shots)`` cell,
    generating the shots branch first if it is not already on disk at this budget."""
    from config import load_config
    from learner.auto import run_config
    from pipeline.generate import generate_shots

    variant_path = _shots_variant_path(cfg_path, n_shots)
    cfg_n = load_config(variant_path)
    generate_shots(cfg_n, root=out_root)

    per_learner: dict[str, list[float]] = {}
    for lname, kwargs in learners:
        scores = []
        for seed in range(int(n_seeds)):
            res = run_config(variant_path, observable, lname, out_root=out_root,
                             scores_root=scores_root, n_train=n_train, split_seed=seed, seed=seed,
                             **kwargs)
            scores.append(res["r2"])
        per_learner[lname] = scores
    return per_learner


def _resolvability_at_shots(cfg_path: str | Path, observable: str, shots: list[int], *,
                            k: int, delta: float, t: float, out_root: str,
                            scores_root: str) -> dict:
    """``mean_local_margin(N)``/``global_margin(N)`` for every ``N`` in ``shots``, from the
    already-cached exact resolution metrics -- no shots drawn, no learner fit."""
    from metrics.global_ratio import cached_global_ratio, eps_global_of_N
    from metrics.local_ratio import cached_local_ratio, eps_local_of_N

    local = cached_local_ratio(cfg_path, observable, k=k, out_root=out_root,
                               scores_root=scores_root)
    glob = cached_global_ratio(cfg_path, observable, out_root=out_root, scores_root=scores_root)
    sqrt_var_circ = max(glob["var_circ"], 1e-300) ** 0.5

    mean_local_margin, global_margin = [], []
    for n_shots in shots:
        eps_l = eps_local_of_N(local, N=n_shots, delta=delta)
        margin = [md - e for md, e in zip(local["min_delta"], eps_l)]
        mean_local_margin.append(statistics.mean(margin))

        eps_g = eps_global_of_N(glob, N=n_shots, t=t, delta=delta)
        global_margin.append(sqrt_var_circ - eps_g * sqrt_var_circ)

    return {"mean_local_margin": mean_local_margin, "global_margin": global_margin}


def sweep_r2_vs_shots(cfg_path: str | Path, observable: str, *, shots: list[int],
                      learners=None, n_seeds: int = 10, k: int = 10, delta: float = 0.1,
                      t: float = 1.0, out_root: str = "datasets", scores_root: str = "scores",
                      n_train: int | None = None) -> dict:
    """For every ``N`` in ``shots``: generate (or reuse) that many shots, fit every learner at
    ``n_seeds`` reseeded splits, average per learner, take the max across learners -- plus the
    local/global resolvability margins at the same ``N`` values (see
    :func:`_resolvability_at_shots`), computed once from the cached exact metrics rather than
    inside the loop.

    Returns ``{"shots": [...], "r2": [...], "detail": [...], "learners": [...],
    "mean_local_margin": [...], "global_margin": [...]}`` -- ``r2[i]`` is the
    max-of-per-learner-means at ``shots[i]``; ``detail[i]`` is ``{learner: [r2_seed0, ...]}``, same
    nothing-thrown-away discipline as :func:`eval.best_of_grid.sweep_best_of_grid`.
    """
    from learner.auto import DEFAULT_SWEEP_LEARNERS

    learners = learners or DEFAULT_SWEEP_LEARNERS

    r2_row, detail_row = [], []
    for n_shots in shots:
        per_learner = _r2_at_shots(cfg_path, observable, n_shots, learners=learners,
                                   n_seeds=n_seeds, out_root=out_root, scores_root=scores_root,
                                   n_train=n_train)
        means = [statistics.mean(v) for v in per_learner.values() if v]
        r2_row.append(max(means) if means else None)
        detail_row.append(per_learner)
        print(f"N={n_shots:<10d} r2={r2_row[-1]}")

    resolvability = _resolvability_at_shots(cfg_path, observable, shots, k=k, delta=delta, t=t,
                                            out_root=out_root, scores_root=scores_root)

    return {"shots": list(shots), "r2": r2_row, "detail": detail_row,
           "learners": [name for name, _ in learners], "observable": observable,
           "n_seeds": int(n_seeds), **resolvability}


def plot_r2_vs_shots(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """R^2 (max-of-per-learner-means) vs. shot budget N, log-x, on top; local and global
    resolvability margins vs. the SAME N values on a second panel below, sharing the x-axis -- so a
    reader can see whether R^2 starts climbing near the same N where the margins cross zero
    (resolvable).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = result["shots"]
    r2 = [v if v is not None else float("nan") for v in result["r2"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.semilogx(shots, r2, "o-", color="tab:blue")
    ax1.set_ylabel("R^2 (max over learners, mean over seeds)")
    ax1.set_title(f"{result['observable']}: learnability vs. shot budget\n"
                 f"n_seeds={result['n_seeds']}, learners={result['learners']}")
    ax1.set_ylim(-0.05, 1.02)
    ax1.grid(alpha=0.3, which="both")

    if "mean_local_margin" in result:
        ax2.semilogx(shots, result["mean_local_margin"], "o-", color="tab:purple",
                    label="mean local margin")
    if "global_margin" in result:
        ax2.semilogx(shots, result["global_margin"], "o-", color="tab:red",
                    label="global margin")
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("shots (N)")
    ax2.set_ylabel("resolvability margin\n(positive = resolvable)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="R^2 vs. shot budget N, at fixed model/observable.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", required=True)
    ap.add_argument("--shots", nargs="+", type=int,
                    default=[100, 1000, 10000, 100000])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--k", type=int, default=10, help="neighbours for the local resolvability "
                                                      "margin")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--t", type=float, default=1.0, help="Proposition 1's t parameter for the "
                                                         "global margin")
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    result = sweep_r2_vs_shots(args.config, args.observable, shots=args.shots,
                               n_seeds=args.n_seeds, k=args.k, delta=args.delta, t=args.t,
                               out_root=args.out_root, scores_root=args.scores_root,
                               n_train=args.n_train)

    out_json = args.out_json or f"r2_vs_shots__{args.observable}.json"
    out_png = args.out_png or f"r2_vs_shots__{args.observable}.png"
    Path(out_json).write_text(json.dumps(result, indent=2))
    plot_r2_vs_shots(result, save_path=out_png)
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
