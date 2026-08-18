"""A model-agnostic Metropolis-Hastings core: state in, state out, no knowledge of Fock keys,
determinants, or circuits.

    chain = mh_chain(x0, propose=my_proposal, log_weight=my_log_weight, n_steps=10_000, seed=0)

**Why MH at all, here.**  Both :mod:`model.fermion` (the bunched, ``s = k/m``-corrected determinant
readout) and :mod:`model.quadratic_fock` normalise by summing ``|det(...)|^2`` /
``|a(x, n)|^2`` over the **entire** ``C(m+k-1, k)``-outcome basis
(``model.fermion.determinant_probs``'s ``p / p.sum(dim=1, ...)``) -- unlike a real boson sampler,
where unitarity makes ``sum_n p_n = 1`` automatic and a single outcome can be evaluated (and hence
sampled) without ever touching the rest of the basis.  Past the point where enumerating that basis
is affordable (:func:`pipeline.distribution.check_size`'s wall), there is no way to draw an
exactly-normalised sample from either model directly.

MH sidesteps this exactly because its acceptance ratio only ever needs a *ratio* of two single-outcome
weights, ``alpha = P~(x_new) / P~(x_current)`` -- the shared normaliser cancels, so the expensive
whole-basis sum is never computed.  This is a real, structural difference from ``photonic``/``qubit``
(whose ``shot_counts`` are true physical/algorithmic samplers, not MH): those need no normaliser at
all, and the docstrings of :mod:`model.fermion`/:mod:`model.quadratic_fock` are explicit that this is
not the case for them.  A model with a genuinely exact sampler (the collision-free free-fermion
sector's determinantal point process, named in :meth:`model.fermion.FermionModel.shot_counts`'s
``NotImplementedError``) should use that instead -- it is exact and needs no burn-in, unlike MH's
correlated, asymptotically-correct draws.  MH is for the sector that HAS no such exact sampler: the
full bunched distribution and ``quadratic_fock``'s bilinear amplitude, both empirical/unphysical
constructions with no known efficient exact sampler.

**Correlated samples, not i.i.d.**  A single MH chain's draws are autocorrelated by construction
(each step depends on the last); :func:`mh_chain` returns every accepted-or-repeated state along the
chain (the standard MH convention: a rejected proposal repeats the current state, it is not
dropped), not a thinned or deduplicated i.i.d. sample.  Callers that need approximately independent
draws should thin by the chain's autocorrelation time; this module does not estimate that time
itself -- see :func:`effective_sample_size` for a cheap diagnostic, not a guarantee.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import torch

State = TypeVar("State")

#: ``(state) -> (proposed_state, log_q_bwd_minus_fwd)``.  The second element is
#: ``log q(proposed -> state) - log q(state -> proposed)``, the Hastings correction for an
#: asymmetric proposal -- ``0.0`` for any proposal that is its own reverse.  **Get this sign
#: backwards and the chain still runs, still has a healthy-looking acceptance rate, and still
#: *looks* converged on a coarse check (e.g. matches on total mass or on a single summary
#: statistic) -- but its stationary distribution is not the target.**  Verified concretely in
#: :mod:`sampling.validate_mh_boson`: with the sign flipped, TV distance to the true distribution
#: plateaus around 0.33 instead of continuing to shrink with more samples, and the visitation
#: ratio between two states of unequal proposal-symmetry (e.g. a bunched vs. unbunched Fock
#: outcome, whose forward/backward proposal probabilities differ) is off by a constant factor
#: instead of matching ``p(s)/p(s')``. Concretely: ``mh_step``'s accept ratio is
#: ``alpha(s -> s') = min(1, [w(s') q(s' -> s)] / [w(s) q(s -> s')])``, so in log form the
#: correction term added to ``log w(s') - log w(s)`` must be ``log q(s' -> s) - log q(s -> s')``
#: -- backward over forward, not forward over backward.
ProposeFn = Callable[[State], tuple[State, float]]

#: ``(state) -> log P~(state)``, the log of the UNNORMALISED target weight.  Working in log-space
#: throughout (never ``P~`` itself) is what keeps this numerically safe: ``|det|^2``/``|a|^2`` can
#: be many orders of magnitude apart across the basis, and the acceptance ratio only ever needs
#: ``log_weight(new) - log_weight(current)``, so nothing is lost by never exponentiating until the
#: final ``min(1, exp(...))`` accept/reject draw.
LogWeightFn = Callable[[State], float]


def mh_step(state: State, *, propose: ProposeFn, log_weight: LogWeightFn,
           log_w_current: float | None = None, gen: torch.Generator) -> tuple[State, float, bool]:
    """One Metropolis-Hastings step: propose, accept or reject, return the resulting state.

    ``log_w_current`` lets a caller avoid recomputing ``log_weight(state)`` every step (it is the
    accepted state's weight from the previous call) -- ``None`` computes it fresh, which
    :func:`mh_chain` only does once, at ``x0``.

    Returns ``(new_state, log_w_new, accepted)``: ``new_state is state`` (same object, not a copy)
    on rejection, so a caller comparing by identity can tell accept from reject without a separate
    flag if it does not need ``accepted`` itself.
    """
    if log_w_current is None:
        log_w_current = float(log_weight(state))

    proposed, log_q_correction = propose(state)
    log_w_proposed = float(log_weight(proposed))

    log_alpha = (log_w_proposed - log_w_current) + log_q_correction
    accept = log_alpha >= 0.0 or float(torch.rand((), generator=gen)) < float(torch.exp(
        torch.tensor(min(log_alpha, 0.0))))

    if accept:
        return proposed, log_w_proposed, True
    return state, log_w_current, False


def mh_chain(x0: State, *, propose: ProposeFn, log_weight: LogWeightFn, n_steps: int,
            burn_in: int = 0, seed: int = 0) -> list[State]:
    """``n_steps`` states from a single MH chain started at ``x0``, ``burn_in`` steps discarded
    from the front.

    Every element of the returned list is a real chain state (rejected steps repeat the previous
    one, per :func:`mh_step`) -- ``n_steps`` states means ``n_steps`` return values, not
    ``n_steps`` acceptances.  Deterministic in ``seed`` (a fresh ``torch.Generator``, not the
    global RNG), so two calls with the same ``x0``/``seed`` reproduce the same chain exactly.
    """
    gen = torch.Generator().manual_seed(int(seed))
    state = x0
    log_w = float(log_weight(state))

    for _ in range(int(burn_in)):
        state, log_w, _ = mh_step(state, propose=propose, log_weight=log_weight,
                                  log_w_current=log_w, gen=gen)

    out = []
    for _ in range(int(n_steps)):
        state, log_w, _ = mh_step(state, propose=propose, log_weight=log_weight,
                                  log_w_current=log_w, gen=gen)
        out.append(state)
    return out


def acceptance_rate(chain: list[State]) -> float:
    """Fraction of consecutive pairs that differ -- a cheap proxy for the accept rate (an exact
    count needs :func:`mh_step`'s own ``accepted`` flag, not available post-hoc when states repeat
    validly; this undercounts by the chance a genuine accept proposes an identical state, which is
    zero for every proposal in this module since no proposal can return its own input).

    Textbook guidance targets roughly 20-50% for a well-tuned random-walk proposal: much lower and
    the chain is barely moving (proposal steps too large, or the target too peaked), much higher and
    consecutive samples are so similar the chain is not exploring efficiently either.
    """
    if len(chain) < 2:
        return 0.0
    changed = sum(1 for a, b in zip(chain[:-1], chain[1:]) if a != b)
    return changed / (len(chain) - 1)


def effective_sample_size(chain: list[float], *, max_lag: int | None = None) -> float:
    """A rough ESS for a scalar summary of the chain (e.g. one coordinate, or ``log_weight``
    itself): ``n / (1 + 2 sum_{lag>=1} rho_lag)``, the standard batch-means-free estimator, summing
    the autocorrelation until it first goes non-positive (Geyer's initial positive sequence rule --
    stops the sum before accumulated noise in the tail inflates it).

    A diagnostic, not a guarantee: this is for deciding a thinning interval or sanity-checking that
    the chain is mixing at all, not a substitute for checking multiple independent chains agree.
    """
    n = len(chain)
    if n < 2:
        return float(n)
    x = torch.tensor(chain, dtype=torch.float64)
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0.0:
        return float(n)                                       # constant chain: nothing to decorrelate

    lag_cap = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    rho_sum = 0.0
    lag = 1
    while lag <= lag_cap:
        cov = float((x[:-lag] * x[lag:]).mean())
        rho = cov / var
        if rho <= 0.0:
            break
        rho_sum += rho
        lag += 1
    return n / (1.0 + 2.0 * rho_sum)
