"""Hellinger-distance sweep: how ``p(x)`` moves away from ``p(x0)`` inside a delta-ball, across
photonic, fermion, and qubit.

Same ``m = 2k`` sweep as ``eval/sweep_size.py``, extended to a third arm, and reuses its
distribution accessor (:func:`metrics.fisher.probs_fn`, exact -- ``shots=0`` -- so no Fisher
matrix is ever built) -- only two things change relative to that script: the point set (``n_x``
points in a ball of radius ``delta`` around one fixed ``x0 = X[0]``, instead of ``n_x`` i.i.d.
draws over the full input cube) and the comparison quantity (the Hellinger distance
``H(p(x0), p(x_i))``, instead of the eigenvalues of the input Fisher matrix).

``qubit`` reuses the same ``(m, k)`` pair as the Fock arms: ``m`` is its qubit count (decoupled
from ``n_features`` -- see :mod:`model.qubit`) and ``k`` its variational depth, so the sweep grows
the qubit circuit exactly as it grows the photonic/fermion one, at the same fixed ``n_features``.

Hellinger is used rather than KL or Wasserstein: it is a proper, bounded (``[0, 1]``) metric with
no ground-metric choice needed over the Fock outcome basis (unlike Wasserstein, which would need
one), and its local quadratic expansion around ``p`` **is** the Fisher information metric --
``4 H(p, p+dp)^2 -> dp^T F dp`` as ``dp -> 0`` -- so it is the natural bridge back to the
AIRM-on-Fisher analysis this script started from, without needing to build ``F`` itself.

Writes one JSON, consumed by ``eval/plot_delta.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):                        # allow `python eval/sweep_delta.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig, ModelConfig, ProblemConfig
from metrics.fisher import probs_fn
from model import build_model, sample_X

ARMS = ("photonic", "fermion", "qubit")


def config_for(kind: str, m: int, k: int, n_features: int) -> ExperimentConfig:
    return ExperimentConfig(problem=ProblemConfig(n_features=n_features, m=m, k=k),
                            model=ModelConfig(kind=kind))


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """``H(p, q) = sqrt(0.5 * sum_n (sqrt(p_n) - sqrt(q_n))^2)`` over the shared outcome basis.

    Bounded in ``[0, 1]``, symmetric, and a proper metric (triangle inequality holds), unlike KL.
    Both models share one outcome basis at fixed ``(m, k)`` (:mod:`v2.model.fermion` reuses the
    photonic Fock keys), so no support-alignment step is needed -- and no ground metric between
    outcomes is needed either, unlike Wasserstein.  Needs no Jacobian and hence no finite
    difference at all: this is a direct distance between two exact probability vectors.
    """
    p, q = p.double().clamp(min=0.0), q.double().clamp(min=0.0)
    return float((0.5 * (p.sqrt() - q.sqrt()).pow(2).sum()).sqrt())


def sample_ball(x0: torch.Tensor, delta: float, n: int, seed: int) -> torch.Tensor:
    """``n`` points drawn uniformly in the L2 ball of radius ``delta`` around ``x0``: ``(n, n_f)``.

    Direction uniform on the sphere (a normalised Gaussian), radius uniform on ``[0, delta]`` --
    the standard rejection-free construction for a uniform ball draw.  Seeded exactly like
    :func:`model.sampler.sample_X`, with its own seed argument so the ball draw is independent of
    (but reproducible alongside) ``sample_seed``.  ``delta = 0`` degenerates to ``n`` exact copies
    of ``x0``, which is the point: it is the base case that pins ``hellinger_distance`` at 0.
    """
    n_f = x0.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    directions = torch.randn(n, n_f, generator=gen, dtype=x0.dtype)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-12)
    radii = float(delta) * torch.rand(n, 1, generator=gen, dtype=x0.dtype).pow(1.0 / n_f)
    return x0.unsqueeze(0) + directions * radii


def measure(kind: str, m: int, k: int, x0: torch.Tensor, X_ball: torch.Tensor) -> dict:
    """``H(p(x0), p(x))`` for every ``x`` in ``X_ball``, one arm at one size."""
    model = build_model(config_for(kind, m, k, x0.shape[0]))
    fn = probs_fn(model)                                       # exact (shots=0)
    keys = model.outcome_keys()

    p0 = fn(x0)
    dists = [hellinger_distance(p0, fn(x)) for x in X_ball]

    D = torch.tensor(dists, dtype=torch.float64)
    return {
        "n_outcomes": len(keys),
        "hellinger": [float(v) for v in D],
        "hellinger_mean": float(D.mean()),
        "hellinger_sem": float(D.std(unbiased=True) / len(D) ** 0.5) if len(D) > 1 else 0.0,
        "hellinger_median": float(D.median()),
        "hellinger_min": float(D.min()),
        "hellinger_max": float(D.max()),
    }


def run(*, m_values, n_features: int, n_x: int, delta: float, seed: int) -> dict:
    out = {"n_features": n_features, "n_x": n_x, "delta": delta, "sample_seed": seed,
           "arms": list(ARMS), "sizes": []}
    x0 = sample_X(1, n_features, seed)[0]
    X_ball = sample_ball(x0, delta, n_x, seed)
    for m in m_values:
        if m % 2:
            raise ValueError(f"m = 2k requires even m (got {m})")
        k = m // 2
        row = {"m": m, "k": k, "arms": {}}
        for kind in ARMS:
            t0 = time.time()
            row["arms"][kind] = measure(kind, m, k, x0, X_ball)
            row["arms"][kind]["seconds"] = round(time.time() - t0, 1)
            r = row["arms"][kind]
            print(f"  m={m:2d} k={k} {kind:9s} outcomes={r['n_outcomes']:6d} "
                  f"hellinger={r['hellinger_mean']:.4f}+-{r['hellinger_sem']:.4f} "
                  f"[{r['hellinger_min']:.4f}, {r['hellinger_max']:.4f}] ({r['seconds']}s)",
                  flush=True)
            print("ratio", r['hellinger_sem']/r['hellinger_mean'] )
        out["sizes"].append(row)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", nargs="+", type=int, default=[6, 8, 10, 12],
                    help="circuit sizes; even, since k = m/2")
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-x", type=int, default=100, help="points sampled inside the delta-ball")
    ap.add_argument("--delta", type=float, default=0.000001, help="ball radius around x0 (radians)")
    ap.add_argument("--seed", type=int, default=42, help="sample_X / ball seed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args(argv)

    print(f"m = 2k Hellinger sweep, n_features={args.n_features}, n_x={args.n_x}, "
          f"delta={args.delta} (exact probs for all arms, no Fisher matrix)")
    res = run(m_values=args.m, n_features=args.n_features, n_x=args.n_x, delta=args.delta,
              seed=args.seed)
    path_out = Path(args.out) / f"results/sweep_delta_{args.delta:.0e}.json"
    Path(path_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {path_out}")


if __name__ == "__main__":
    main()
