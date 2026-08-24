"""Size sweep at ``m = 2k``: bunched mass and input-Fisher spectrum, boson against determinant.

Joins existing pieces only -- :func:`metrics.fisher.probs_fn` and :func:`metrics.fisher.fisher` for
both arms, so the two are measured by the **same** central-difference code at the same ``eps``.  That
uniformity is the point: the boson arm goes through merlin and has no usable autograd path at this
size (``jacrev`` vmaps ``n_outcomes`` cotangents and the determinant arm runs out of memory at
``m=12``), so using FD for one arm and autograd for the other would confound the comparison with the
differentiation method.  ``metrics.fisher`` already does FD for both; nothing here re-implements it.

``m = 2k`` fixes the filling fraction at ``k/m = 1/2``, so the determinant arm's ``bunching_s``
default is ``0.5`` at every size and the sweep varies circuit size alone.  ``n_features = 5`` keeps
the Fisher matrix ``5 x 5`` across the sweep -- required for the spectra to be stackable -- and
sits strictly below ``m``, so there is no exact global-phase null direction and all five eigenvalues
carry information (``phase_direction`` is reported so that stays checkable rather than assumed).

The basis grows as ``C(m+k-1, k)``: 56, 330, 2002, 12376 at ``m = 6, 8, 10, 12``.  ``m = 14`` is
77520 and ``m = 16`` is 490314, where the Fisher Jacobian alone is ~2 GB, so the sweep stops at 12.

Writes one JSON, consumed by ``eval/plot_size.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):                        # allow `python eval/sweep_size.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig, ModelConfig, ProblemConfig
from metrics.fisher import EPS, fisher, probs_fn
from model import build_model, sample_X

ARMS = ("photonic", "fermion")


def config_for(kind: str, m: int, k: int, n_features: int) -> ExperimentConfig:
    return ExperimentConfig(problem=ProblemConfig(n_features=n_features, m=m, k=k),
                            model=ModelConfig(kind=kind))


def measure(kind: str, m: int, k: int, X: torch.Tensor, *, eps: float) -> dict:
    """Bunched mass and the mean Fisher spectrum for one arm at one size."""
    model = build_model(config_for(kind, m, k, X.shape[1]))
    fn = probs_fn(model)                                     # exact (shots=0), FD-differentiated
    keys = model.outcome_keys()
    bunched = torch.tensor([max(int(c) for c in key) > 1 for key in keys])

    eigs, mass, supp, phase = [], [], [], []
    for x in X:
        p = fn(x)
        mass.append(float(p[bunched].sum() / p.sum()))
        _, detail = fisher(fn, x, eps=eps)           # the matrix itself is not needed, only eigs
        eigs.append(detail["eigenvalues"])
        supp.append(int(detail["n_support"]))
        phase.append(float(detail["phase_direction"]))

    E = torch.stack(eigs).double()                           # (n_x, n_f), descending per row
    M = torch.tensor(mass, dtype=torch.float64)
    return {
        "n_outcomes": len(keys),
        "n_support_mean": float(sum(supp) / len(supp)),
        "phase_direction_mean": float(sum(phase) / len(phase)),
        "bunched_mean": float(M.mean()),
        "bunched_sem": float(M.std(unbiased=True) / len(M) ** 0.5),
        "bunched_min": float(M.min()),
        "bunched_max": float(M.max()),
        "eigs_mean": E.mean(dim=0).tolist(),
        "eigs_sem": (E.std(dim=0, unbiased=True) / E.shape[0] ** 0.5).tolist(),
        # Full spread over the input pool, for the min-max whiskers the plots draw.  Reported
        # alongside the SEM because they answer different questions: the SEM is how well the mean is
        # pinned down, the min-max is how much the quantity actually varies with x.
        "eigs_min": E.min(dim=0).values.tolist(),
        "eigs_max": E.max(dim=0).values.tolist(),
        "bunching_s": float(getattr(model, "bunching_s", float("nan"))),
    }


def run(*, m_values, n_features: int, n_x: int, seed: int, eps: float) -> dict:
    out = {"n_features": n_features, "n_x": n_x, "eps": eps, "sample_seed": seed,
           "arms": list(ARMS), "sizes": []}
    X = sample_X(n_x, n_features, seed)
    for m in m_values:
        if m % 2:
            raise ValueError(f"m = 2k requires even m (got {m})")
        k = m // 2
        row = {"m": m, "k": k, "arms": {}}
        for kind in ARMS:
            t0 = time.time()
            row["arms"][kind] = measure(kind, m, k, X, eps=eps)
            row["arms"][kind]["seconds"] = round(time.time() - t0, 1)
            r = row["arms"][kind]
            print(f"  m={m:2d} k={k} {kind:9s} outcomes={r['n_outcomes']:6d} "
                  f"support={r['n_support_mean']:8.1f} bunched={r['bunched_mean']:.4f} "
                  f"lam={['%.3f' % v for v in r['eigs_mean']]} ({r['seconds']}s)", flush=True)
        out["sizes"].append(row)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16],
                    help="circuit sizes; even, since k = m/2")
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-x", type=int, default=100, help="input points averaged over")
    ap.add_argument("--seed", type=int, default=42, help="sample_X seed")
    ap.add_argument("--eps", type=float, default=EPS, help="central-difference step")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "sweep_size.json"))
    args = ap.parse_args(argv)

    print(f"m = 2k sweep, n_features={args.n_features}, n_x={args.n_x}, eps={args.eps} "
          f"(FD for both arms)")
    res = run(m_values=args.m, n_features=args.n_features, n_x=args.n_x, seed=args.seed,
              eps=args.eps)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
