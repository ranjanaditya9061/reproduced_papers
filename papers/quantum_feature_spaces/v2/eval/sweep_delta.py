"""Hellinger-distance sweep: how ``p(x)`` moves away from ``p(x0)`` inside a delta-ball, across
every Fock-basis-comparable model kind (see ``ARMS``): photonic, fermion, qubit, mlp_fock,
quadratic_fock, and the photonic spin / spin_magic preps.

Same ``m = 2k`` sweep as ``eval/sweep_size.py``, at exact (``shots=0``) probabilities so no Fisher
matrix is ever built -- but ``probs_batch`` calls :meth:`model.base.DistributionModel.probs`
directly on the **whole** delta-ball in one chunk, rather than row-by-row like
:func:`metrics.fisher.probs_fn`: ``spin``/``spin_magic`` discover their outcome basis from the
backend per call and union it only within that call's batch (:func:`circuit.prep._align_rows`), so
a per-row call would align ``p(x0)`` and ``p(x)`` to two different bases. The point set also
differs from ``sweep_size.py``'s: ``n_x`` points in a ball of radius ``delta`` around one fixed
``x0 = X[0]``, instead of ``n_x`` i.i.d. draws over the full input cube -- and the comparison
quantity is the Hellinger distance ``H(p(x0), p(x_i))``, instead of the eigenvalues of the input
Fisher matrix.

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
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):                        # allow `python eval/sweep_delta.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from model import build_model, sample_X

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "sample"

#: One YAML per arm.  Each config's own ``model`` section (kind, prep, encoding, +knobs) and
#: ``problem.n_features`` are used as-is; only ``problem.m``/``problem.k`` are overridden per
#: sweep point (``k = m/2``), so an arm like ``quadratic_fock`` keeps its own ``n_features``
#: while still being grown across the same ``m`` values as everyone else.
#:
#: ``mlp``/``analytical`` are deliberately excluded: both emit a fixed 2-outcome distribution
#: (:mod:`model.classical`), so their support does not grow with ``m`` the way every Fock-basis
#: arm's does, and a Hellinger distance over a 2-point basis is not comparable to one over
#: ``C(m+k-1, k)`` outcomes.  ``spin``/``spin_magic`` are the actually comparable "classical-ish"
#: photonic preps: same ``m``-mode Fock outcome space (``spin_magic`` after the readout=0
#: post-selection in :func:`readout_zero_probs`), different state preparation.
#:
#: Their reported ``n_outcomes`` is only the basis perceval actually reaches for this batch
#: (:func:`circuit.prep._align_rows`), which is usually the full ``C(m+k-1, k)`` once the
#: Haar-random interferometer mixes every mode, but is not guaranteed to be -- an unreachable
#: outcome is never enumerated at all (not even as an explicit zero), so treat ``n_outcomes``
#: here as a lower bound rather than assuming parity with the other arms' fixed count.
ARMS = {
    "photonic": "photonic_fock_phase.yaml",
    "fermion": "fermion.yaml",
    "qubit": "qubit.yaml",
    "mlp_fock": "mlp_fock.yaml",
    "quadratic_fock": "quadratic_fock.yaml",
    "spin": "photonic_spin.yaml",
    # "spin_magic_ghz": "photonic_spin_magic_ghz.yaml",
    # "spin_magic_linear": "photonic_spin_magic_linear.yaml",
    # "spin_magic_linear_u3": "photonic_spin_magic_linear_u3.yaml",
    # "spin_magic_data_on_spin": "photonic_spin_magic_data_on_spin.yaml",
}


def config_for(base, m: int, k: int):
    cfg = dataclasses.replace(base, problem=dataclasses.replace(base.problem, m=m, k=k))
    cfg.validate()
    return cfg


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


def probs_batch(model, X: torch.Tensor) -> torch.Tensor:
    """``(N, n_outcomes)`` exact probs for the whole batch in **one** chunk.

    Unlike :func:`metrics.fisher.probs_fn` (one row per call), this is required for a prep that
    discovers its outcome basis from the backend per call (``spin``/``spin_magic``,
    :meth:`circuit.prep.StatePrep.outcome_keys` returning ``None``): each such call independently
    unions its rows onto one basis (:func:`circuit.prep._align_rows`) and overwrites
    ``model._keys`` with it, so two *separate* single-row calls -- one for ``x0``, one for an
    ``X_ball`` point -- are generally aligned to two different bases and not comparable.  Forcing
    everything through one ``model.probs`` call with ``forward_batch`` raised past ``N`` makes the
    whole delta-ball share one basis, the same way :func:`model.base.DistributionModel.probs`'s
    own row-chunking would if a chunk boundary fell inside it.
    """
    model.forward_batch = max(model.forward_batch, X.shape[0])
    return model.probs(X, grad=False).double()


def readout_zero_probs(model, P: torch.Tensor):
    """Condition ``P`` (``(N, n_outcomes)``, aligned to ``model.outcome_keys()``) on the readout
    modes reading dual-rail ``0``, renormalize, and return ``(P_conditioned, data_keys)``.

    ``spin_magic`` reports two readout modes (:meth:`model.photonic.PhotonicModel.readout_modes`,
    forwarding :meth:`circuit.prep.SpinMagicPrep.readout_modes` -- the ``(m, m+1)`` pair a readout
    photon is emitted into).  Dual-rail ``0`` is one photon in the first mode, none in the second
    (:func:`circuit.fock.binary_keys`'s ``(1, 0)`` convention).  Nothing post-selects this inside
    the prep -- see :mod:`circuit.prep`'s module docstring -- so it is applied here, offline, and
    is exactly what makes the outcome basis returned comparable to every other (data-mode-only)
    arm's.  A no-op (returns ``P``, ``keys`` unchanged) when ``readout_modes()`` is empty, which is
    every arm except ``spin_magic*``: :meth:`circuit.prep.SpinPrep.probs` already traces its spin
    out inside perceval, so ``spin``'s distribution is already the bare photonic one.
    """
    readout = model.readout_modes()
    keys = model.outcome_keys()
    if not readout:
        return P, keys
    r0, r1 = readout
    mask = torch.tensor([int(key[r0]) == 1 and int(key[r1]) == 0 for key in keys])
    data_keys = [tuple(v for j, v in enumerate(key) if j not in readout)
                 for key, keep in zip(keys, mask.tolist()) if keep]

    Pc = P[:, mask]
    totals = Pc.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("readout=0 post-selection has zero mass at some x")
    return Pc / totals, data_keys


def measure(cfg, delta: float, n_x: int, seed: int) -> dict:
    """``H(p(x0), p(x))`` for every ``x`` in a delta-ball around ``x0``, one arm at one size.

    ``x0``/``X_ball`` are sampled here, at this arm's own ``cfg.problem.n_features``, rather than
    shared across arms -- each arm's YAML pins its own ``n_features``
    (:class:`config.ProblemConfig`), and that must stay fixed within an arm's comparison but need
    not match another arm's.
    """
    model = build_model(cfg)
    x0 = sample_X(1, cfg.problem.n_features, seed)[0]
    X_ball = sample_ball(x0, delta, n_x, seed)
    X_all = torch.cat([x0.unsqueeze(0), X_ball], dim=0)

    P = probs_batch(model, X_all)                               # exact (shots=0), one basis
    P, keys = readout_zero_probs(model, P)
    p0, P_ball = P[0], P[1:]
    dists = [hellinger_distance(p0, p) for p in P_ball]

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


def run(*, m_values, n_x: int, delta: float, seed: int) -> dict:
    bases = {kind: load_config(CONFIGS_DIR / path) for kind, path in ARMS.items()}
    out = {"n_x": n_x, "delta": delta, "sample_seed": seed, "arms": list(ARMS),
           "n_features": {kind: cfg.problem.n_features for kind, cfg in bases.items()},
           "sizes": []}
    for m in m_values:
        if m % 2:
            raise ValueError(f"m = 2k requires even m (got {m})")
        k = m // 2
        row = {"m": m, "k": k, "arms": {}}
        for kind, base in bases.items():
            t0 = time.time()
            cfg = config_for(base, m, k)
            row["arms"][kind] = measure(cfg, delta, n_x, seed)
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
    ap.add_argument("--n-x", type=int, default=100, help="points sampled inside the delta-ball")
    ap.add_argument("--delta", type=float, default=0.001, help="ball radius around x0 (radians)")
    ap.add_argument("--seed", type=int, default=42, help="sample_X / ball seed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args(argv)

    print(f"m = 2k Hellinger sweep, n_x={args.n_x}, delta={args.delta} "
          f"(exact probs for all arms, no Fisher matrix; n_features per arm's own config)")
    res = run(m_values=args.m, n_x=args.n_x, delta=args.delta, seed=args.seed)
    path_out = Path(args.out) / f"results/sweep_delta_{args.delta:.0e}.json"
    Path(path_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {path_out}")


if __name__ == "__main__":
    main()
