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
observable's efficiency across the input space rather than whatever one row happened to be. Two
lines per subfolder plot, x-axis = variant: ``eta_mean`` (left axis, ``[0, 1]``) at the fixed
``--observable``, and ``trace_mean`` (right axis, unbounded) -- the mean of ``tr(F)``, the
distribution's own Fisher information with no observable involved. Plotting both together separates
two different questions: how much total input-information the *distribution* carries
(``trace_mean``, observable-independent) versus how much of that information the *readout* actually
captures (``eta_mean``, observable-dependent) -- a variant can have a large ``trace_mean`` and small
``eta_mean`` (an information-rich distribution this observable fails to read out) or the reverse.

**``trace_mean`` is reported on the unprojected ``F``, unlike ``eta_mean``.**
:func:`metrics.observable.fisher_at` always projects ``F`` onto the ``(n_f-1)``-dim complement of
``1/sqrt(n_f)`` (:func:`metrics.distribution.project_physical`) before returning it -- a projection
that only removes a genuinely null direction when ``n_features == m``.  Every config under
``configs/eval/`` has ``n_features=5, m=6`` (``n_features < m``), where
:func:`metrics.distribution.phase_eigenvalue`'s own docstring calls that same direction "genuinely
informative" rather than null.  So ``trace_mean``/the debug eigenvalues here come from
:func:`_unprojected_fisher` -- the full ``n_features``-dimensional ``F``, with no direction dropped
-- while ``eta_mean`` still goes through the projected ``F^+`` (every other quantity in
:mod:`metrics.observable` is only defined on that subspace, so ``eta`` cannot be un-projected the
same way without reimplementing that module's machinery).

Requires every variant's dataset to already be generated (:mod:`configs.run_size_sweep` or
``python -m pipeline.generate``) -- this only reads saved artifacts, it does not generate them. A
variant whose observable is non-differentiable there (e.g. ``max_prob``) is skipped with a printed
reason, not silently omitted.
"""

from __future__ import annotations

import argparse
import json
import re
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


def _unprojected_fisher(p, dp):
    """``F`` built straight from ``(p, dp)`` -- no :func:`metrics.distribution.project_physical`
    mean-subtraction.

    :func:`metrics.observable.fisher_at` always projects ``F`` onto the ``(n_f-1)``-dim complement
    of ``1/sqrt(n_f)`` before returning it, unconditionally -- regardless of whether
    ``n_features == m``.  That projection is only guaranteed to remove a genuinely null (exactly
    unobservable, to round-off) direction when ``n_features == m``; when ``n_features < m`` (every
    config under ``configs/eval/`` uses ``n_features=5, m=6``),
    :func:`metrics.distribution.phase_eigenvalue`'s own docstring calls the projected-out direction
    "genuinely informative" rather than null.  ``eta``/``F^+`` still need the projected ``F`` (every
    other quantity in :mod:`metrics.observable` is defined on that subspace), so this is a separate,
    diagnostic-only ``F`` used just for the reported ``trace``/``eigenvalues`` here -- it is not fed
    back into ``eta``.
    """
    from metrics.distribution import SUPPORT_TOL

    keep = p > SUPPORT_TOL
    J = (dp[keep] / p[keep].sqrt().unsqueeze(1)).double()
    return J.T @ J


def variant_eta(cfg_path: str | Path, observable: str, *, n_x: int = DEFAULT_N_X,
                out_root: str = "datasets", graph_density: float = 0.5,
                debug: bool = False) -> tuple[float, float, dict | None]:
    """``(eta_mean, trace_mean, debug_points)`` for one config, over ``n_x`` sampled input points.

    ``trace_mean`` is the mean of ``tr(F)`` on the **unprojected** Fisher matrix
    (:func:`_unprojected_fisher`) -- the distribution's own Fisher information, with no observable
    involved, over the full ``n_features``-dimensional input space rather than the ``(n_f-1)``-dim
    subspace :func:`metrics.observable.fisher_at` reports on.  ``eta`` itself still uses the
    projected ``F`` from ``fisher_at`` (see :func:`_unprojected_fisher`'s docstring for why).

    ``debug_points`` is ``None`` unless ``debug=True``, in which case it is a per-point breakdown
    (``eta``, ``trace``, ``eigenvalues`` of the unprojected ``F``, ``V_eff``, and ``||g||``) for all
    ``n_x`` points -- printed here and also returned so it lands in the caller's JSON, for
    diagnosing whether a suspicious ``trace_mean`` is a real effect or a sign the Jacobian/Fisher
    computation silently degenerated for this variant.

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
    etas, traces = [], []
    points = [] if debug else None
    for i, x in enumerate(X):
        p, dp, F_proj = fisher_at(model, x)
        g, V = influence_terms(obs, p, dp)
        e = eta(g, V, F_proj)
        F = _unprojected_fisher(p, dp)                  # trace/eigenvalues reported on THIS one
        tr = float(F.diagonal().sum())
        etas.append(e)
        traces.append(tr)
        if debug:
            import torch as _torch
            eigvals = _torch.linalg.eigvalsh(F.double()).flip(0)   # descending, matches trace order
            point = {"i": i, "eta": float(e), "trace": tr, "V_eff": float(V),
                    "g_norm": float(g.double().norm()),
                    "eigenvalues": [float(v) for v in eigvals]}
            points.append(point)
            print(f"    [{i:2d}] eta={point['eta']:.6g}  trace={point['trace']:.6g}  "
                  f"V_eff={point['V_eff']:.6g}  |g|={point['g_norm']:.6g}  "
                  f"eigs={['%.4g' % v for v in point['eigenvalues']]}", flush=True)

    import torch
    eta_mean, trace_mean = float(torch.tensor(etas).mean()), float(torch.tensor(traces).mean())
    debug_out = None
    if debug:
        def _spread(name, vals):
            t = torch.tensor(vals, dtype=torch.float64)
            s = {"mean": float(t.mean()), "std": float(t.std()) if len(vals) > 1 else 0.0,
                 "min": float(t.min()), "max": float(t.max())}
            print(f"    {name:<8} over {n_x} points: mean={s['mean']:.6g}  std={s['std']:.6g}  "
                  f"min={s['min']:.6g}  max={s['max']:.6g}", flush=True)
            return s

        summary = {
            "eta": _spread("eta", etas),
            "trace": _spread("trace", traces),
            "V_eff": _spread("V_eff", [pt["V_eff"] for pt in points]),
            "g_norm": _spread("g_norm", [pt["g_norm"] for pt in points]),
        }
        debug_out = {"variant_points": points, "summary": summary}
    return eta_mean, trace_mean, debug_out


def sweep_variant_eta(variants: list[tuple[str, "str | Path"]], observable: str, *,
                      n_x: int = DEFAULT_N_X, out_root: str = "datasets",
                      debug: bool = False) -> dict:
    """``eta_mean``/``trace_mean`` for every variant, at one fixed ``observable``.

    One bad variant (missing artifact, non-differentiable observable) is recorded as ``None`` for
    both and printed, not fatal to the rest -- same per-cell failure handling as the other eval
    drivers.  With ``debug=True``, also collects each variant's per-point breakdown
    (:func:`variant_eta`'s ``debug_points``) under ``result["debug"][name]``.
    """
    names, eta_values, trace_values, debug_by_name = [], [], [], {}
    for name, cfg_path in variants:
        try:
            if debug:
                print(f"  -- {name} --", flush=True)
            eta_mean, trace_mean, dbg = variant_eta(cfg_path, observable, n_x=n_x,
                                                     out_root=out_root, debug=debug)
            eta_values.append(eta_mean)
            trace_values.append(trace_mean)
            if debug:
                debug_by_name[name] = dbg
        except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad variant must not
                                                        # (variant_eta raises SystemExit, not
                                                        # Exception, on a missing artifact -- see
                                                        # eval.violin's identical fix)
            print(f"[efficiency_line] {name} failed: {exc}")  # abort the rest of the line
            eta_values.append(None)
            trace_values.append(None)
        names.append(name)

    result = {"variants": names, "observable": observable, "n_x": n_x, "eta_mean": eta_values,
              "trace_mean": trace_values}
    if debug:
        result["debug"] = debug_by_name
    return result


def plot_variant_eta_line(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """``eta_mean`` and ``trace_mean`` line plot from :func:`sweep_variant_eta`'s output: variant on
    x, ``eta_mean`` on the left y-axis, ``tr(F)`` on a twin right y-axis (different scale and no
    ``[0, 1]`` bound, so it cannot share the left axis).

    A variant with no value (a failed load, or a non-differentiable observable there) leaves a gap
    in the corresponding line rather than plotting as 0 -- ``eta = 0`` is a real, meaningfully
    different value (a variant whose distribution genuinely carries no information about the
    observable), and the same goes for a genuinely near-zero ``tr(F)``.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names, observable, n_x = result["variants"], result["observable"], result["n_x"]
    eta_values = [np.nan if v is None else v for v in result["eta_mean"]]
    trace_values = [np.nan if v is None else v for v in result["trace_mean"]]
    positions = list(range(1, len(names) + 1))

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names) + 2), 5))
    eta_color, trace_color = "#4C72B0", "#DD8452"
    l1, = ax.plot(positions, eta_values, marker="o", linewidth=2, color=eta_color, label="eta_mean")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("variant")
    ax.set_ylabel("eta_mean = mean(g^T F^+ g / V_eff)", color=eta_color)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelcolor=eta_color)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    l2, = ax2.plot(positions, trace_values, marker="s", linewidth=2, linestyle="--",
                   color=trace_color, label="trace_mean")
    ax2.set_ylabel("trace_mean = mean(tr(F))", color=trace_color)
    ax2.tick_params(axis="y", labelcolor=trace_color)

    ax.set_title(f"Observable efficiency & distribution Fisher info: {observable} x variant  "
                f"(n_x={n_x})")
    ax.legend(handles=[l1, l2], loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _safe_tag(s: str) -> str:
    """Filesystem-safe filename fragment -- see :func:`eval.violin._safe_tag`, same convention."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run(*, eval_dir: Path = EVAL_DIR, observable: str = DEFAULT_OBSERVABLE,
       n_x: int = DEFAULT_N_X, out_root: str = "datasets",
       debug: bool = False) -> list[tuple[Path, Exception]]:
    """For every subfolder of ``eval_dir``, build one efficiency line plot and save it (PNG + JSON)
    into that subfolder.  Mirrors :func:`eval.violin.run`'s walk and failure handling.

    ``debug=True`` prints and saves the per-point breakdown from :func:`variant_eta` for every
    variant -- see that function's docstring for what it reports and why.

    Output filenames are tagged with ``observable`` and ``n_x`` (:func:`_safe_tag`) so two runs at
    different observables or sample counts do not overwrite each other's plot/JSON.
    """
    subfolders = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not subfolders:
        raise SystemExit(f"no subfolders found in {eval_dir}")

    tag = f"{_safe_tag(observable)}__nx{int(n_x)}"
    failures = []
    for sub in subfolders:
        variants = _variants_in(sub)
        if not variants:
            print(f"=== {sub.name}: no *.yaml configs, skipping", flush=True)
            continue
        print(f"=== {sub.name}: {len(variants)} variants -> {[n for n, _ in variants]}", flush=True)
        try:
            result = sweep_variant_eta(variants, observable, n_x=n_x, out_root=out_root, debug=debug)
            (sub / f"variant_efficiency__{tag}.json").write_text(json.dumps(result, indent=2))
            plot_variant_eta_line(result, save_path=sub / f"variant_efficiency__{tag}.png")
            print(f"    wrote {sub / f'variant_efficiency__{tag}.png'}", flush=True)
        except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad subfolder must not
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
    ap.add_argument("--debug", action="store_true",
                    help="print + save per-point eta/trace/eigenvalues/V_eff/g_norm for every "
                    "point of every variant, plus each quantity's mean/std/min/max -- use this to "
                    "check whether a suspicious trace_mean (e.g. clustered at some round number "
                    "across every point) is real or a sign of a degenerate Jacobian")
    args = ap.parse_args(argv)

    failures = run(eval_dir=Path(args.eval_dir), observable=args.observable, n_x=args.n_x,
                   out_root=args.root, debug=args.debug)

    if failures:
        print(f"\n{len(failures)} subfolder plot(s) failed outright:", flush=True)
        for sub, exc in failures:
            print(f"  {sub.name}: {exc}", flush=True)
        raise SystemExit(1)
    print("all eval efficiency line plots generated successfully", flush=True)


if __name__ == "__main__":
    main()
