"""Score vectors built from marked outcomes or a fixed reference distribution.

Two families whose ``v(n)`` needs one piece of *circuit* context beyond the outcome basis:

* ``single_output`` -- the signed contrast between the injected input state and its mode-reversal,
  peak-normalised.  Needs ``input_state``.
* ``xent`` / ``xent_<base>`` -- ``sum_n <base>(n) p(n) log q(n)`` with ``q`` the circuit's output
  at ``x = 0``.  Needs ``reference_probs``.

**Both are expectation values.**  ``xent`` in particular: because ``q`` does not depend on ``x``,
``sum_n p_x(n) log q(n)`` is the ordinary expectation ``<log q>`` of a fixed bounded diagonal
observable, so it sits in the classically *easy* regime (single-shot estimable) and is a probe
rather than a hardness witness.  It is the second half of ``KL(p || q) = ent(x) - xent(x)``, and
all the hardness of that combination lives in the ``ent`` term (:mod:`v2.observable.prob_fn`).
``single_output`` divides by the row peak, which is still linear per row.

**Both are offline-re-scorable in v2, unlike the legacy pipeline.**  There, ``input_state`` and
``q`` were not persisted, so ``single_output`` could not be re-scored at all and ``xent`` needed a
matched-seed teacher rebuilt by hand.  Both are functions of the circuit alone, so
:mod:`v2.pipeline.artifact` stores them in the artifact (``input_state`` in ``meta.json``,
``probs_at_zero`` in the ``.npz``) and the context is populated automatically.

Carried from ``model/photonic_observables/{single_output,cross_entropy}.py``.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from ..base import (Expectation, Observable, ObservableContext, ObservableFamily, as_vec,
                    base_score_vec, register)
from .counting import require_even_m

#: Bases usable under ``xent_<base>``; bare ``xent`` means weight ``1`` everywhere.
XENT_BASES = ("parity", "majority", "bunching", "n_first")

#: Floor for ``q`` before taking its log, bounding ``log q >= log(XENT_EPS) ~ -27.6``.  Unlike
#: the entropy clamp -- where ``p = 0`` kills its own term exactly, so clamping *is* the true
#: ``p log p -> 0`` limit -- this one is a genuine regularisation: if ``q(n) ~ 0`` while
#: ``p_x(n) > 0`` the exact score is ``-inf``, and the clamp replaces it with a finite bound.
XENT_EPS = 1e-12

_XENT_RE = re.compile(r"^xent(?:_(.+))?$")


# --- single_output ------------------------------------------------------------------------- #


def single_output_score(key, input_state) -> int:
    """``+1`` on the input state, ``-1`` on its reverse, ``0`` on every other outcome."""
    kl = [int(key[i]) for i in range(len(input_state))]
    if kl == list(input_state):
        return 1
    if kl == list(reversed(input_state)):
        return -1
    return 0


class SingleOutputObservable(Expectation):
    """``(probs @ v) / max_n p(n)`` -- the marked-outcome contrast, peak-normalised.

    Dividing by the row peak keeps the value comparable across inputs: a nearly flat distribution
    would otherwise give a vanishing contrast for every ``x``.

    The normalisation makes this a *ratio* of two functionals rather than a plain mean, so the
    inherited ``psi = v`` is wrong for it -- but it is still differentiable away from ties in the
    max, so it stays in scope for analysis B with its own influence function::

        T(p) = (p.v) / p_{n*},    n* = argmax_n p_n
        psi_n = v_n / p_{n*}  -  (p.v) / p_{n*}^2 . 1[n = n*]

    :attr:`~v2.observable.base.Observable.is_expectation` is therefore left False: ``V_eff`` here is
    the asymptotic variance, not an exact single-shot one.  (Ties in the argmax are measure-zero for
    a continuous circuit but are checked, since a degenerate distribution would make ``psi``
    ambiguous.)
    """

    #: The score is a ratio, so V_eff is asymptotic rather than an exact single-shot variance.
    is_expectation = False

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs @ self.score_vec) / probs.max(dim=1).values.clamp(min=1e-10)

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi`` of the peak-normalised ratio -- see the class docstring for the derivation."""
        peak, arg = probs.max(dim=1)
        peak = peak.clamp(min=1e-10)
        num = probs @ self.score_vec                                    # (N,)
        psi = self.score_vec.unsqueeze(0) / peak.unsqueeze(1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, arg.unsqueeze(1), 1.0)
        return psi - onehot * (num / (peak * peak)).unsqueeze(1)


class SingleOutputFamily(ObservableFamily):
    describe = "single_output"

    def matches(self, name: str) -> bool:
        return name == "single_output"

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.input_state is None:
            raise ValueError(
                "observable 'single_output' needs the circuit's input_state. v2 stores it in the "
                "artifact's meta.json, so this means the artifact predates that field or the "
                "context was built by hand; pass input_state=<sequence> explicitly."
            )
        return SingleOutputObservable(
            [single_output_score(key, ctx.input_state) for key in ctx.keys])

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        return {"observable": "single_output", "normalisation": "peak"}


# --- xent --------------------------------------------------------------------------------- #


def parse_xent_observable(observable: str):
    """Split into ``(is_xent, base)``; ``base`` is ``None`` for bare ``xent`` (weight ``1``)."""
    mo = _XENT_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_xent_observable(observable: str) -> bool:
    """True for ``xent`` or a well-formed ``xent_<base>`` (``base`` in :data:`XENT_BASES`)."""
    is_xent, base = parse_xent_observable(observable)
    return is_xent and (base is None or base in XENT_BASES)


def reference_log_probs(reference_probs, *, eps: float = XENT_EPS) -> np.ndarray:
    """``log q``, floored at ``eps`` so the vector is finite and bounded below."""
    return np.log(np.maximum(np.asarray(reference_probs, dtype=np.float64), float(eps)))


def xent_score_vec(reference_probs, base: str | None, keys, *, m: int, k: int,
                   eps: float = XENT_EPS) -> np.ndarray:
    """Score vector ``<base>(n) . log q(n)``; ``base=None`` leaves the weight at ``1``."""
    log_q = reference_log_probs(reference_probs, eps=eps)
    if base is None:
        return log_q
    if base == "majority":
        require_even_m(m, "xent_majority")
    weight = base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                            allowed=XENT_BASES, label="xent")
    return weight * log_q


class XEntFamily(ObservableFamily):
    describe = f"xent / xent_<base> (base in {XENT_BASES})"

    def matches(self, name: str) -> bool:
        return is_xent_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_xent_observable(name)[1]
        q = ctx.resolve_reference_probs(f"observable {name!r}")
        obs = Expectation(xent_score_vec(q, base, ctx.keys, m=ctx.m, k=ctx.k))
        # q is folded into score_vec and not needed to score, but keep it reachable for
        # inspection and so a caller can hand it to an offline re-score.
        obs.register_buffer("reference_probs", as_vec(q))
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        # q is fully determined by the circuit (m, k, n_features, seed), which the artifact
        # identity already fixes, so no extra field is needed beyond the base and the clamp.
        base = parse_xent_observable(name)[1]
        return {"observable": "xent", "base": base, "eps": XENT_EPS}


register(SingleOutputFamily())
register(XEntFamily())
