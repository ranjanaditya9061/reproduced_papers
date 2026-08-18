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

Writes one JSON + one plot: ||g|| (log scale) and eta, both vs m, one line per observable.
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
from metrics.distribution import finite_difference_jacobian, probs_and_jacobian
from metrics.observable import eta, fisher_at, influence_terms
from model import build_model, sample_X
from observable import ObservableContext, resolve_observable

#: Deliberately includes osc -- the in-repo case flagged (STATUS.md 3.3, 3.5) as having an unusually
#: large influence function (V_eff four orders above parity) and an oscillating score vector that
#: the gradient-free screen G_O is known to misjudge -- exactly the profile a plateau-only story
#: would not predict correctly.
OBSERVABLES = ["parity", "n_first", "ent", "osc", "sq_parity", "xent_parity"]

M_VALUES = [6, 8, 10, 12]


def config_for(m: int, k: int, n_features: int) -> ExperimentConfig:
    return ExperimentConfig(problem=ProblemConfig(n_features=n_features, m=m, k=k),
                            model=ModelConfig(kind="photonic"))


def run(*, m_values=M_VALUES, n_features: int = 5, n_x: int = 40, seed: int = 42,
       cross_check_fd: bool = True, fd_eps: float = 1e-4) -> dict:
    """``cross_check_fd``: also compute ``||g||`` via :func:`finite_difference_jacobian` on the
    FIRST input point only (expensive -- one ``2*n_features`` forward-eval pass per observable is
    cheap, but doing it for all of ``X`` would double the whole sweep's cost for a check that only
    needs to answer "does autograd agree with an independent numerical method", not build its own
    full statistic). Reported per size as ``g_norm_fd`` beside the autograd ``g_norm_mean`` so a
    large gap flags an autograd/merlin problem rather than a real effect.
    """
    X = sample_X(n_x, n_features, seed)
    out = {"n_features": n_features, "n_x": n_x, "observables": OBSERVABLES, "sizes": []}

    for m in m_values:
        k = m // 2
        model = build_model(config_for(m, k, n_features))
        keys = model.outcome_keys()
        ctx = ObservableContext(m=m, k=k, keys=keys, seed=model.seed,
                                input_state=model.input_state(),
                                reference_probs=model.probs_at_zero().numpy())
        obs = {n: resolve_observable(n, ctx) for n in OBSERVABLES}

        row = {"m": m, "k": k, "n_outcomes": len(keys), "per_obs": {n: {} for n in OBSERVABLES}}
        g_norms = {n: [] for n in OBSERVABLES}
        etas = {n: [] for n in OBSERVABLES}
        for x in X:
            p, dp = probs_and_jacobian(model, x)
            _, _, F = fisher_at(model, x)
            for n in OBSERVABLES:
                g, V = influence_terms(obs[n], p, dp)
                g_norms[n].append(float(g.norm()))
                etas[n].append(eta(g, V, F))

        g_norms_fd = {}
        if cross_check_fd:
            p0, _ = probs_and_jacobian(model, X[0])
            dp_fd = finite_difference_jacobian(model, X[0], eps=fd_eps)
            for n in OBSERVABLES:
                g_fd, _ = influence_terms(obs[n], p0, dp_fd)
                g_norms_fd[n] = float(g_fd.norm())

        for n in OBSERVABLES:
            gn = torch.tensor(g_norms[n])
            et = torch.tensor(etas[n])
            row["per_obs"][n] = {
                "g_norm_mean": float(gn.mean()), "g_norm_min": float(gn.min()),
                "g_norm_max": float(gn.max()),
                "eta_mean": float(et.mean()), "eta_min": float(et.min()),
                "g_norm_fd_x0": g_norms_fd.get(n),
                "g_norm_autograd_x0": g_norms[n][0],
            }
        out["sizes"].append(row)
        print(f"m={m:2d} k={k}  " + "  ".join(
            f"{n}: ||g||={row['per_obs'][n]['g_norm_mean']:.4g} eta={row['per_obs'][n]['eta_mean']:.4f}"
            + (f" (fd={row['per_obs'][n]['g_norm_fd_x0']:.4g} vs ag={row['per_obs'][n]['g_norm_autograd_x0']:.4g})"
               if cross_check_fd else "")
            for n in OBSERVABLES), flush=True)
    return out


def plot(result: dict, save_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [row["m"] for row in result["sizes"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for n in result["observables"]:
        gn = [row["per_obs"][n]["g_norm_mean"] for row in result["sizes"]]
        et = [row["per_obs"][n]["eta_mean"] for row in result["sizes"]]
        ax1.semilogy(ms, gn, "o-", label=n)
        ax2.plot(ms, et, "o-", label=n)

    ax1.set_xlabel("m")
    ax1.set_ylabel("mean ||g|| = ||d<O>/dx||  (log scale)")
    ax1.set_title("Gradient magnitude vs. size\n(collapse here = genuine barren plateau)")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("m")
    ax2.set_ylabel("mean eta (information efficiency)")
    ax2.set_title("Observable efficiency vs. size\n(low here = 'hard to learn')")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle("Is 'hard to learn' only ever a barren plateau?")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", nargs="+", type=int, default=M_VALUES)
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-x", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                        "gradient_vs_hardness.json"))
    args = ap.parse_args(argv)

    res = run(m_values=args.m, n_features=args.n_features, n_x=args.n_x, seed=args.seed)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    plot_path = Path(args.out).with_suffix(".png")
    plot(res, plot_path)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
