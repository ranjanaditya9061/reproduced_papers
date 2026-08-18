"""Validate :mod:`sampling.mh` against ground truth on boson sampling at ``m=6, k=3``.

``m=6,k=3`` gives ``C(m+k-1,k) = 56`` outcomes -- small enough to enumerate exactly, which is
the whole point: this is the one regime where MH's answer can be checked against something
that isn't itself a sampler. Three distributions are compared against the same fixed input `x`:

1. **Exact** -- ``PhotonicModel._probs`` / ``boson_probs_reference``, the ground-truth
   ``|Perm(U[S,T])|^2 / prod_j n_j!`` over the full Fock basis.
2. **Physical shots** -- ``PhotonicModel.shot_counts``, via ``CliffordClifford2017``: a real
   boson sampler, not a resample of the stored exact vector. Independent baseline.
3. **MH** -- :func:`sampling.mh.mh_chain`, targeting the *unnormalised* weight
   ``|Perm(U[S,T])|^2 / prod_j n_j!`` computed the same way as (1)/(2), via a "move one photon
   from an occupied mode to a uniformly random mode" proposal. This proposal is NOT symmetric
   (moving a photon out of a crowded mode is more likely, forward, than the reverse move back
   into it), so the Hastings correction is tracked explicitly rather than assumed zero -- see
   :func:`make_propose`, and :data:`sampling.mh.ProposeFn` for the sign convention this module's
   first draft got backwards (caught by this exact TV-vs-n sweep: a flipped sign produces a
   healthy-looking chain whose TV distance to the true distribution plateaus around 0.33 instead
   of continuing to shrink with more samples).

Rather than a single sample-size snapshot (noisy, easy to over/under-interpret one draw of), this
sweeps **TV distance to exact vs. sample size** for both samplers on one shared log-log plot, at
sizes ``10, 100, 1000, ...``, by taking growing prefixes of one long draw/chain -- this shows
directly whether an estimator's error keeps shrinking like a normal sampler (slope ~ -1/2 on
log-log axes) or is stuck at a floor, which is the signature of a real bias rather than finite-
sample noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import torch

from circuit.fock import fock_keys
from circuit.photonic_circuit import default_input_state, sandwich_unitaries, sandwich_unitary_at
from model.fermion import boson_probs_reference, occupied_modes
from model.photonic import PhotonicModel
from sampling.mh import acceptance_rate, effective_sample_size, mh_chain

M, K, N_FEATURES = 6, 3, 5
SEED = 0
N_MAX = 100_000                                   # largest prefix checked
MH_BURN_IN = 2_000
SIZES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum())


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((0.5 * (p.sqrt() - q.sqrt()).pow(2).sum()).sqrt())


def outcomes_to_probs(outcomes: list[tuple], keys: list[tuple]) -> torch.Tensor:
    counts: dict = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    total = len(outcomes)
    return torch.tensor([counts.get(key, 0) / total for key in keys], dtype=torch.float64)


def make_propose(m: int, k: int, rng: random.Random):
    """Pick a photon at random (by occupied-mode list), move it to a uniformly random mode
    (including back to where it started -- a null move).

    **Not symmetric**, despite appearances: ``q(s -> s') = n_i(s)/k * 1/m`` where ``n_i(s)`` is
    the occupancy of the source mode *in the current state*. The reverse move ``s' -> s`` draws
    from ``n_j(s')/k * 1/m`` where ``j`` is the destination mode -- and ``n_j(s') = n_j(s) + 1``,
    not ``n_i(s)``, whenever the two modes started at different occupancies. Per
    :data:`sampling.mh.ProposeFn`'s convention (backward-over-forward), the correction returned
    here for the step ``s -> s'`` is ``log q(s' -> s) - log q(s -> s') = log(n_j(s) + 1) -
    log(n_i(s))``, zero only when source and destination had equal occupancy before the move.
    Verified against ground truth in this module's ``main()`` -- getting this backwards produces
    a chain that runs cleanly (healthy acceptance rate, no errors) but is biased: it
    systematically over-visits unbunched outcomes relative to bunched ones by a constant factor
    per Fock "shape" class, and its TV distance to the true distribution plateaus instead of
    shrinking with more samples.
    """

    def propose(state: tuple) -> tuple[tuple, float]:
        occ = list(occupied_modes(state))          # length k, one entry per photon
        src_mode = rng.choice(occ)
        dst_mode = rng.randrange(m)
        new = list(state)
        new[src_mode] -= 1
        new[dst_mode] += 1
        n_i = state[src_mode]                       # occupancy of source mode *before* the move
        n_j_after = new[dst_mode]                   # occupancy of dest mode *after* the move
        # backward-over-forward, per sampling.mh.ProposeFn: log q(s'->s) - log q(s->s')
        log_q_correction = 0.0 if src_mode == dst_mode else (
            torch.log(torch.tensor(float(n_j_after))) - torch.log(torch.tensor(float(n_i)))
        ).item()
        return tuple(new), log_q_correction

    return propose


def main() -> None:
    torch.manual_seed(SEED)
    model = PhotonicModel(m=M, k=K, n_features=N_FEATURES, seed=42)
    keys = fock_keys(M, K)
    key_index = {key: i for i, key in enumerate(keys)}
    x = torch.rand(1, N_FEATURES) * 2 * torch.pi

    # 1. Exact ground truth.
    p_exact = model.probs(x)[0].double()
    p_exact = p_exact / p_exact.sum()

    # 2. Physical shots: one long draw, real boson sampler, independent of the exact-probs path.
    shots_outcomes = None
    if model.supports_shots:
        draws = model.shot_counts(x, shots=N_MAX, shot_seed=SEED)[0]
        shots_outcomes = [tuple(int(v) for v in o) for o in draws]

    # 3. MH: one long chain targeting the same unnormalised |Perm|^2/prod(n_j!) weight.
    W1, W2 = sandwich_unitaries(M, model.seed)
    U = sandwich_unitary_at(W1, W2, x, N_FEATURES, model.encoding)   # (1, m, m)
    s_modes = occupied_modes(default_input_state(M, K))

    def log_weight(state: tuple) -> float:
        w = boson_probs_reference(U, s_modes, [state])[0, 0]
        return float(torch.log(w.clamp(min=1e-300)))

    x0 = tuple(default_input_state(M, K))
    rng = random.Random(SEED)
    mh_states = mh_chain(x0, propose=make_propose(M, K, rng), log_weight=log_weight,
                         n_steps=N_MAX, burn_in=MH_BURN_IN, seed=SEED)

    accept_rate = acceptance_rate(mh_states)
    idx_chain = [float(key_index[s]) for s in mh_states]
    ess = effective_sample_size(idx_chain)
    print(f"m={M}, k={K}, n_outcomes={len(keys)}")
    print(f"MH: {N_MAX} steps ({MH_BURN_IN} burn-in), acceptance rate={accept_rate:.3f}, "
          f"ESS(outcome index)={ess:.1f} (of {N_MAX} raw samples)")
    print()

    # TV distance to exact at growing prefixes, for both samplers.
    header = f"{'n':>10}{'TV shots':>12}{'TV MH':>12}"
    print(header)
    rows = []
    for n in SIZES:
        p_shots_n = outcomes_to_probs(shots_outcomes[:n], keys) if shots_outcomes else None
        p_mh_n = outcomes_to_probs(mh_states[:n], keys)
        tv_shots_n = tv_distance(p_exact, p_shots_n) if p_shots_n is not None else float("nan")
        tv_mh_n = tv_distance(p_exact, p_mh_n)
        rows.append((n, tv_shots_n, tv_mh_n))
        print(f"{n:>10}{tv_shots_n:>12.4f}{tv_mh_n:>12.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ns = np.array([r[0] for r in rows], dtype=float)
    tv_shots_arr = np.array([r[1] for r in rows], dtype=float)
    tv_mh_arr = np.array([r[2] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    if shots_outcomes:
        ax.loglog(ns, tv_shots_arr, "o-", label="shots (physical sampler)")
    ax.loglog(ns, tv_mh_arr, "s-", label="MH")
    # reference n^-1/2 slope, anchored at the first MH point, to show what "just noise" looks like
    ref = tv_mh_arr[0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref, "k--", alpha=0.4, label=r"$n^{-1/2}$ reference")
    ax.set_xlabel("number of samples n")
    ax.set_ylabel("TV distance to exact distribution")
    ax.set_title(f"TV distance vs. sample size, m={M}, k={K}")
    ax.legend()
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "mh_validation_tv_vs_n.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
