"""Per-variant observable efficiency (``eta``), averaged over sampled points, as a line plot.

    python eval/efficiency_line.py --eval-dir configs/eval --observable parity

For each subfolder of ``configs/eval/`` (see ``configs/run_eval_heatmaps.py``'s docstring for what
the folders mean), this loads the saved artifact for every variant, samples ``n_x`` input points,
and at each one computes :func:`metrics.observable.eta` -- the exact joint efficiency
``g^T F^+ g / V_eff``, where ``g = d<O>/dx`` (the observable's gradient, via its closed-form
influence function times the distribution's finite-differenced Jacobian -- see
:func:`metrics.observable.influence_terms`) and ``F`` is the input Fisher matrix of the
distribution (:func:`metrics.observable.fisher_at`).  ``eta`` is exact and bounded in ``[0, 1]``
because it accounts for correlations between input directions through ``F``'s pseudo-inverse,
unlike a plain per-direction ``sum_i g_i^2 / F_ii`` which would ignore ``F``'s off-diagonal
structure -- see that module's docstring for why ``eta = 1`` iff the observable's influence
function *is* the score.

Reports the mean over the ``n_x`` points (not a single-point value) so the line reflects the
observable's efficiency across the input space rather than whatever one row happened to be. One
line per subfolder plot, x-axis = variant, y-axis = ``eta_mean`` at the fixed ``--observable``.

Requires every variant's dataset to already be generated (:mod:`configs.run_size_sweep` or
``python -m pipeline.generate``) -- this only reads saved artifacts, it does not generate them. A
variant whose observable is non-differentiable there (e.g. ``max_prob``) is skipped with a printed
reason, not silently omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/efficiency_line.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"
DEFAULT_OBSERVABLE = "parity"
DEFAULT_N_X = 48


def _variants_in(subfolder: Path) -> list[tuple[str, Path]]:
    """``[(stem, path), ...]`` for every ``*.yaml`` directly inside ``subfolder``, sorted by stem --
    mirrors :func:`configs.run_eval_heatmaps._variants_in` / :func:`eval.violin._variants_in`."""
    return [(p.stem, p) for p in sorted(subfolder.glob("*.yaml"))]


def variant_eta(cfg_path: str | Path, observable: str, *, n_x: int = DEFAULT_N_X,
                out_root: str = "datasets", graph_density: float = 0.5) -> float:
    """Mean :func:`metrics.observable.eta` for one config, over ``n_x`` sampled input points.

    Raises if ``observable`` is non-differentiable at this config (mirrors
    :func:`metrics.observable.analyse`'s exclusion, but as a single named observable rather than a
    filtered list, since this is a per-variant scalar rather than a multi-observable table).
    """
    from config import load_config
    from metrics.observable import eta, fisher_at, influence_terms
    from model import build_model, sample_X
    from observable import resolve_observable
    from pipeline.artifact import exact_path
    from pipeline.distribution import load_dist
    from pipeline.score import context_for

    cfg = load_config(cfg_path)
    model = build_model(cfg)
    path = exact_path(cfg, model, out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run pipeline.generate --config {cfg_path}")

    dist = load_dist(path, size=1)
    ctx = context_for(dist.meta, dist.keys, graph_density=graph_density)
    obs = resolve_observable(observable, ctx)
    if not obs.is_differentiable:
        raise ValueError(f"observable {observable!r} is not differentiable "
                         "(is_differentiable=False) -- eta is undefined for it")

    X = sample_X(n_x, cfg.problem.n_features, cfg.seeds.sample_seed)
    etas = []
    for x in X:
        p, dp, F = fisher_at(model, x)
        g, V = influence_terms(obs, p, dp)
        etas.append(eta(g, V, F))

    import torch
    return float(torch.tensor(etas).mean())


def sweep_variant_eta(variants: list[tuple[str, "str | Path"]], observable: str, *,
                      n_x: int = DEFAULT_N_X, out_root: str = "datasets") -> dict:
    """``eta_mean`` for every variant, at one fixed ``observable``.

    One bad variant (missing artifact, non-differentiable observable) is recorded as ``None`` and
    printed, not fatal to the rest -- same per-cell failure handling as the other eval drivers.
    """
    names, values = [], []
    for name, cfg_path in variants:
        try:
            values.append(variant_eta(cfg_path, observable, n_x=n_x, out_root=out_root))
        except Exception as exc:                       # noqa: BLE001 -- one bad variant must not
            print(f"[efficiency_line] {name} failed: {exc}")  # abort the rest of the line
            values.append(None)
        names.append(name)

    return {"variants": names, "observable": observable, "n_x": n_x, "eta_mean": values}


def plot_variant_eta_line(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """``eta_mean`` line plot from :func:`sweep_variant_eta`'s output: variant on x, ``eta_mean`` on y.

    A variant with no value (a failed load, or a non-differentiable observable there) leaves a gap
    in the line rather than plotting as 0 -- ``eta = 0`` is a real, meaningfully different value
    (a variant whose distribution genuinely carries no information about the observable).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names, observable, n_x = result["variants"], result["observable"], result["n_x"]
    values = [np.nan if v is None else v for v in result["eta_mean"]]
    positions = list(range(1, len(names) + 1))

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names) + 2), 5))
    ax.plot(positions, values, marker="o", linewidth=2, color="#4C72B0")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("variant")
    ax.set_ylabel("eta_mean = mean(g^T F^+ g / V_eff)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Observable efficiency: {observable} x variant  (n_x={n_x})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def run(*, eval_dir: Path = EVAL_DIR, observable: str = DEFAULT_OBSERVABLE,
       n_x: int = DEFAULT_N_X, out_root: str = "datasets") -> list[tuple[Path, Exception]]:
    """For every subfolder of ``eval_dir``, build one efficiency line plot and save it (PNG + JSON)
    into that subfolder.  Mirrors :func:`eval.violin.run`'s walk and failure handling.
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
        try:
            result = sweep_variant_eta(variants, observable, n_x=n_x, out_root=out_root)
            (sub / "variant_efficiency.json").write_text(json.dumps(result, indent=2))
            plot_variant_eta_line(result, save_path=sub / "variant_efficiency.png")
            print(f"    wrote {sub / 'variant_efficiency.png'}", flush=True)
        except Exception as exc:                       # noqa: BLE001 -- one bad subfolder must not
            print(f"    efficiency line FAILED: {exc}", flush=True)  # stop the rest of the run
            failures.append((sub, exc))

    return failures


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build a variant x eta_mean line plot for every "
                                              "configs/eval/ subfolder.")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    ap.add_argument("--n-x", type=int, default=DEFAULT_N_X)
    ap.add_argument("--root", default="datasets")
    args = ap.parse_args(argv)

    failures = run(eval_dir=Path(args.eval_dir), observable=args.observable, n_x=args.n_x,
                   out_root=args.root)

    if failures:
        print(f"\n{len(failures)} subfolder plot(s) failed outright:", flush=True)
        for sub, exc in failures:
            print(f"  {sub.name}: {exc}", flush=True)
        raise SystemExit(1)
    print("all eval efficiency line plots generated successfully", flush=True)


if __name__ == "__main__":
    main()
