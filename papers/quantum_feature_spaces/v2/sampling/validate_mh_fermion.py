"""Validate :meth:`model.fermion.FermionModel.shot_counts` (MH-backed) at ``m=6, k=3``.

Same protocol as :mod:`sampling.validate_mh_boson`, on the determinant arm instead of the boson
one: ``m=6,k=3`` gives ``C(m+k-1,k)=56`` outcomes, small enough that :meth:`FermionModel.probs`
can enumerate the full basis and give an exact ground truth to check the MH-backed
:meth:`FermionModel.shot_counts` against -- the one regime where this is possible at all, since
past this size the whole point of using MH is that the full-basis normalisation
:func:`~model.fermion.determinant_probs` needs is no longer affordable.

No independent physical sampler exists for the determinant readout (unlike the boson arm's
``CliffordClifford2017``) -- ``supports_shots`` on this model *is* MH, so this script's only
comparison is MH's own TV-distance-to-exact as a function of sample size, checking for the same
plateau-vs-shrink signature that caught the sign bug on the boson arm.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from circuit.fock import fock_keys
from model.fermion import FermionModel

M, K, N_FEATURES = 6, 3, 5
SEED = 0
N_MAX = 100_000
SIZES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum())


def outcomes_to_probs(outcomes: list[tuple], keys: list[tuple]) -> torch.Tensor:
    counts: dict = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    total = len(outcomes)
    return torch.tensor([counts.get(key, 0) / total for key in keys], dtype=torch.float64)


def main() -> None:
    torch.manual_seed(SEED)
    model = FermionModel(m=M, k=K, n_features=N_FEATURES, seed=42)
    keys = fock_keys(M, K)
    x = torch.rand(1, N_FEATURES) * 2 * torch.pi

    p_exact = model.probs(x)[0].double()
    p_exact = p_exact / p_exact.sum()

    t0 = time.time()
    mh_outcomes = model.shot_counts(x, shots=N_MAX, shot_seed=SEED)[0]
    print(f"m={M}, k={K}, n_outcomes={len(keys)}, bunching_s={model.bunching_s}")
    print(f"MH: {N_MAX} steps in {time.time() - t0:.1f}s")
    print()

    print(f"{'n':>10}{'TV MH':>12}")
    rows = []
    for n in SIZES:
        p_mh_n = outcomes_to_probs(mh_outcomes[:n], keys)
        tv_n = tv_distance(p_exact, p_mh_n)
        rows.append((n, tv_n))
        print(f"{n:>10}{tv_n:>12.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ns = np.array([r[0] for r in rows], dtype=float)
    tv_arr = np.array([r[1] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.loglog(ns, tv_arr, "s-", label="MH (fermion determinant)")
    ref = tv_arr[0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref, "k--", alpha=0.4, label=r"$n^{-1/2}$ reference")
    ax.set_xlabel("number of samples n")
    ax.set_ylabel("TV distance to exact distribution")
    ax.set_title(f"Fermion MH: TV distance vs. sample size, m={M}, k={K}")
    ax.legend()
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "mh_validation_fermion_tv_vs_n.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
