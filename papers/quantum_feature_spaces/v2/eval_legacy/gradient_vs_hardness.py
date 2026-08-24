"""Does "hard to learn" only ever mean "barren plateau" (the observable's gradient collapses)?

    python eval/gradient_vs_hardness.py

**Why look at the gradient directly, not the Fisher matrix.** A barren plateau is, by definition,
a collapsing *gradient*: ``d<O>/dx -> 0`` as the system grows. That is exactly
``g = influence_terms(obs, p, dp)[0]`` (:mod:`metrics.observable`'s own docstring: "which is also
d<T>/dx_i"), computed straight from one autograd Jacobian -- no Fisher matrix needed. ``eta``
(``g^T F^+ g / V_eff``) answers a different, harder question ("how much of the *distribution's
total* input-information does this readout capture"), and STATUS.md is explicit that magnitude,
not rank, is what a plateau test needs (:mod:`metrics.distribution`'s ``fisher_spectrum``
docstring) -- ``g`` is the most direct magnitude there is.

**The test.** For each observable and each ``m`` in the sweep, compute the mean gradient norm
``E_x[||g||]`` directly. If "hard to learn" only ever meant "plateau", ||g|| should collapse
exactly where the observable becomes hard to learn (low eta / low R^2) and stay healthy where it
is easy -- one clean threshold, no exceptions. The known counterexample already flagged in
STATUS.md (Analysis B, section 3.3): ``osc`` has ``V_eff`` four orders of magnitude above
``parity`` at the *same* x -- a large influence function, not a collapsing one -- so if ``osc`` is
still hard to learn while its ||g|| stays large (or even grows), that is a case of "hard to learn"
NOT explained by gradient collapse: the observable is throwing away information some other way
(e.g. by weighting outcomes so unevenly that its own label variance swamps the shot budget, or its
influence oscillates so ``G_O`` -- the gradient-FREE screen -- disagrees with ``||g||`` in sign of
implication, per :func:`metrics.observable.g_ratio`'s own one-directional Poincare caveat).

**Jacobian source: finite differences (:mod:`metrics.fisher`), not autograd, at every size --
and this is about memory, not shots.** The autograd path (``jacrev`` through merlin,
:func:`metrics.distribution.probs_and_jacobian`) OOM-kills at ``m=14`` even for a SINGLE input
point on a 46GB machine: ``torch.func.jacrev`` batches its backward pass over every one of the
``n_outcomes`` output dimensions at once (77520 at m=14), so the graph it retains scales with
``n_outcomes``, not with the ``(n_outcomes, n_features)`` Jacobian it eventually returns. Central
differences (:func:`metrics.fisher.probs_and_jacobian`, ``2*n_features + 1`` plain forward calls
to ``model.probs(x, grad=False)``, no graph at all) sidestep this entirely -- measured on this
machine: m=14 7.6s/point, m=16 20.9s/point, m=18 119s/point (~38-42 us/outcome, consistent across
sizes), m=20 extrapolates to ~13 min/point. **Crucially this stays on the EXACT distribution the
whole way** (``shots=0`` in :func:`metrics.fisher.probs_fn`) -- it is not the shots branch, and
none of :mod:`metrics.fisher`'s own documented shot-noise finding applies (see next paragraph). The
only cost is wall-clock, which is why ``m in (18, 20)`` want a smaller ``--n-x`` than the default
(see ``configs`` used by the actual sweep run, not this module's default).

**Why ``eta`` -- unlike a SHOT-based Fisher matrix -- is still exact here.**
:mod:`metrics.fisher`'s own module docstring measures a *different* failure: a Fisher matrix
plugged in from ``model.shot_counts``'s empirical distribution is biased **2.2x** even at 1e5
shots, because ``F_ii`` divides by ``p_n`` and shot noise on rare outcomes doesn't cancel that
division (``E[1/p_hat] > 1/p``). That finding is about ``shots > 0`` in
:func:`metrics.fisher.probs_fn`; this module always passes ``shots=0``, i.e. every finite
difference is taken on ``model.probs()`` itself -- the exact distribution, differenced instead of
autograd-differentiated. There is no plug-in shot noise anywhere in this file's numbers, at any
``m``, so ``eta`` is exact (to float32 + truncation error, the same ``~1.6e-4`` :mod:`metrics.fd`
measures for FD-vs-autograd agreement on the exact distribution) all the way through the sweep --
it is not capped by the shot-noise argument at all. What DOES cap it is whatever ``m`` the wall
clock allows.

Writes one JSON, plus TWO separate plots (:func:`plot_gradient`, :func:`plot_efficiency`) -- kept
apart deliberately, see those functions' docstrings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import ExperimentConfig, ModelConfig, ProblemConfig
from metrics.distribution import SUPPORT_TOL, project_physical
from metrics.fisher import EPS as FD_EPS
from metrics.fisher import probs_and_jacobian as fd_probs_and_jacobian
from metrics.fisher import probs_fn as fd_probs_fn
from metrics.observable import eta, influence_terms
from model import build_model, sample_X
from observable import ObservableContext, resolve_observable

#: Deliberately includes osc -- the in-repo case flagged (STATUS.md 3.3, 3.5) as having an unusually
#: large influence function (V_eff four orders above parity) and an oscillating score vector that
#: the gradient-free screen G_O is known to misjudge -- exactly the profile a plateau-only story
#: would not predict correctly.
OBSERVABLES = ["parity", "n_first", "ent", "osc", "sq_parity", "xent_parity"]

#: Full range: m=6..12 cost seconds total, m=14/16 low minutes, m=18/20 tens of minutes to hours
#: per --n-x=40 point (see module docstring) -- callers sweeping the tail should pass a smaller
#: --n-x rather than trimming this list, so every m in one run shares the same observable set.
M_VALUES = [6, 8, 10, 12, 14, 16, 18, 20]


def config_for(m: int, k: int, n_features: int) -> ExperimentConfig:
    return ExperimentConfig(problem=ProblemConfig(n_features=n_features, m=m, k=k),
                            model=ModelConfig(kind="photonic"))


def _fisher_from_pd(p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    """``F`` on the physical subspace from an already-computed ``(p, dp)`` pair -- the same
    construction as :func:`metrics.observable.fisher_at`, minus that function's hardcoded autograd
    call, so it works identically whether ``dp`` came from autograd or (as here) finite differences.
    """
    keep = p > SUPPORT_TOL
    J = project_physical(dp[keep] / p[keep].sqrt().unsqueeze(1)).double()
    return J.T @ J


def run(*, m_values=M_VALUES, n_features: int = 5, n_x: int = 40, seed: int = 42,
       fd_eps: float = FD_EPS, checkpoint_path: str | Path | None = None) -> dict:
    """Every ``(p, dp)`` in this sweep comes from :func:`metrics.fisher.probs_and_jacobian` at
    ``shots=0`` -- finite differences on the exact distribution, never autograd (OOMs past m=12)
    and never the shots branch (biased Fisher, per :mod:`metrics.fisher`'s own measurement) -- see
    the module docstring for both.

    Each size's row records the full config identity it was computed from (``n_features``, ``m``,
    ``k``, ``model_kind/prep/encoding``, ``model_seed``, ``sample_seed``) so a later scatter/join
    plot can cite exactly which config produced which point, rather than a bare ``(m, observable)``
    pair.

    ``checkpoint_path``: write ``out`` to this path after EVERY size, not just at the end, and skip
    any ``m`` already present in it on entry -- a single large-``m`` point can cost minutes to
    tens of minutes (see the module docstring's measured FD cost), so writing only once at the end
    means an interruption (killed process, machine restart) loses everything computed so far, not
    just the in-flight point. Written via a temp-file-then-rename so a kill mid-write never leaves
    a truncated/corrupt JSON on disk -- the checkpoint is always either the previous complete state
    or the new one, never a half-written one.
    """
    X = sample_X(n_x, n_features, seed)
    out = {"n_features": n_features, "n_x": n_x, "seed": seed, "fd_eps": fd_eps,
           "observables": OBSERVABLES, "sizes": []}
    done_m = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        out = json.loads(Path(checkpoint_path).read_text())
        done_m = {row["m"] for row in out["sizes"]}
        if done_m:
            print(f"resuming from checkpoint {checkpoint_path}: m={sorted(done_m)} already done",
                 flush=True)

    def _checkpoint():
        if checkpoint_path is None:
            return
        tmp = Path(checkpoint_path).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(checkpoint_path)

    for m in m_values:
        if m in done_m:
            continue
        k = m // 2
        cfg = config_for(m, k, n_features)
        model = build_model(cfg)
        fn = fd_probs_fn(model, shots=0)
        keys = model.outcome_keys()
        ctx = ObservableContext(m=m, k=k, keys=keys, seed=model.seed,
                                input_state=model.input_state(),
                                reference_probs=model.probs_at_zero().numpy())
        obs = {n: resolve_observable(n, ctx) for n in OBSERVABLES}

        row = {"m": m, "k": k, "n_outcomes": len(keys),
              "config": {"n_features": n_features, "m": m, "k": k,
                        "model_kind": cfg.model.kind, "model_prep": cfg.model.prep,
                        "model_encoding": cfg.model.encoding, "model_seed": model.seed,
                        "sample_seed": seed, "jacobian": "finite_difference", "fd_eps": fd_eps,
                        "shots": 0},
              "per_obs": {n: {} for n in OBSERVABLES}}
        g_norms = {n: [] for n in OBSERVABLES}
        etas = {n: [] for n in OBSERVABLES}
        for x in X:
            p, dp = fd_probs_and_jacobian(fn, x, eps=fd_eps)
            F = _fisher_from_pd(p, dp)
            for n in OBSERVABLES:
                g, V = influence_terms(obs[n], p, dp)
                g_norms[n].append(float(g.norm()))
                etas[n].append(eta(g, V, F))

        for n in OBSERVABLES:
            gn = torch.tensor(g_norms[n])
            et = torch.tensor(etas[n])
            row["per_obs"][n] = {
                "g_norm_mean": float(gn.mean()), "g_norm_min": float(gn.min()),
                "g_norm_max": float(gn.max()), "g_norm_std": float(gn.std()) if n_x > 1 else 0.0,
                "eta_mean": float(et.mean()), "eta_min": float(et.min()),
            }
        out["sizes"].append(row)
        print(f"m={m:2d} k={k}  " + "  ".join(
            f"{n}: ||g||={row['per_obs'][n]['g_norm_mean']:.4g} eta={row['per_obs'][n]['eta_mean']:.4f}"
            for n in OBSERVABLES), flush=True)
    return out


def plot_gradient(result: dict, save_path: str | Path) -> None:
    """``||g|| = ||d<O>/dx||`` vs. size, one line per observable -- the raw plateau signal, on its
    own axis and its own figure.  Deliberately NOT sharing a figure (or an axis) with
    :func:`plot_efficiency`: ``||g||`` is unbounded and not comparable across observables with
    different native scales (``V_eff`` spans four orders of magnitude across this observable list,
    STATUS.md 3.3), so it answers only "is the raw signal collapsing", never "is this observable
    relatively hard" -- that second, cross-observable-comparable question is what
    :func:`plot_efficiency`'s bounded ``eta`` is for. Plotting them on one axis pair invites reading
    a scale difference as a hardness difference, which is exactly the conflation this module's
    docstring warns against.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [row["m"] for row in result["sizes"]]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for n in result["observables"]:
        gn = [row["per_obs"][n]["g_norm_mean"] for row in result["sizes"]]
        ax.semilogy(ms, gn, "o-", label=n)

    ax.set_xlabel("m")
    ax.set_ylabel("mean ||g|| = ||d<O>/dx||  (log scale)")
    ax.set_title("Gradient magnitude vs. size\n(collapse here = genuine barren plateau)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def plot_efficiency(result: dict, save_path: str | Path) -> None:
    """``eta`` (information efficiency, exact and bounded in ``[0, 1]``) vs. size, one line per
    observable -- the cross-observable-comparable "hard to learn" proxy, kept off
    :func:`plot_gradient`'s figure for the reason documented there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [row["m"] for row in result["sizes"]]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for n in result["observables"]:
        et = [row["per_obs"][n]["eta_mean"] for row in result["sizes"]]
        ax.plot(ms, et, "o-", label=n)

    ax.set_xlabel("m")
    ax.set_ylabel("mean eta (information efficiency)")
    ax.set_title("Observable efficiency vs. size\n(low here = 'hard to learn')")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def print_manifest(result: dict) -> None:
    """One line per plotted (m, observable) cell, naming exactly the config it came from --
    printed so a scatter/join plot built from this JSON later is traceable back to a source
    without re-opening the file. Mirrors ``row["config"]`` already saved in the JSON.
    """
    print("\n--- manifest: which config produced each point -------------------------------")
    for row in result["sizes"]:
        c = row["config"]
        print(f"  m={c['m']:<3} k={c['k']:<3} n_f={c['n_features']}  "
              f"model={c['model_kind']}/{c['model_prep']}/{c['model_encoding']}  "
              f"model_seed={c['model_seed']} sample_seed={c['sample_seed']}  "
              f"jacobian={c['jacobian']}(eps={c['fd_eps']:g})  n_outcomes={row['n_outcomes']}  "
              f"observables={result['observables']}")
    print("---------------------------------------------------------------------------------\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", nargs="+", type=int, default=M_VALUES)
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-x", type=int, default=40,
                    help="sampled input points averaged per size; the default is fine through "
                    "m=16 but costs tens of minutes to hours per size at m=18/20 (see module "
                    "docstring's measured us/outcome) -- lower this for the large-m tail rather "
                    "than dropping sizes, so every m in one run shares the same observable set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fd-eps", type=float, default=FD_EPS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                        "gradient_vs_hardness.json"))
    args = ap.parse_args(argv)

    res = run(m_values=args.m, n_features=args.n_features, n_x=args.n_x, seed=args.seed,
             fd_eps=args.fd_eps)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    print_manifest(res)

    gradient_path = Path(args.out).with_name(Path(args.out).stem + "_gnorm.png")
    plot_gradient(res, gradient_path)
    print(f"wrote {gradient_path}")

    efficiency_path = Path(args.out).with_name(Path(args.out).stem + "_eta.png")
    plot_efficiency(res, efficiency_path)
    print(f"wrote {efficiency_path}")


if __name__ == "__main__":
    main()
