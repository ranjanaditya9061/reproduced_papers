"""Non-polynomial observables: ``sum_n v_n phi(p_n)`` for a scalar nonlinearity ``phi``.

The one shape that is **not** the expectation of any fixed diagonal observable.  ``sq_<base>`` and
``pairprod`` are degree-2 in ``p`` and so need two copies of the state; these are not polynomials
in ``p`` at all, because their ``phi`` has no Taylor expansion at ``p = 0``.  No bounded number of
single-shot samples estimates them to additive error, which is the strongest
not-classically-easy claim of any family here -- and the reason the utility metrics reject them
(there is no ``Var(O)`` to divide by; see
:meth:`v2.observable.base.Observable.variance`).

Three families:

* ``ent`` / ``ent_<base>`` -- ``phi(p) = p log p``.  Bare ``ent`` is ``sum_n p log p = -H(p)``, the
  negative Shannon entropy: a whole-distribution concentration measure where ``max_prob`` reads
  only the peak.  Entropy estimation over ``n_out`` outcomes needs ``~n_out / log n_out`` samples.
* ``osc`` / ``osc_<base>`` -- ``phi(p) = p sin(1/(p + eps))``, by a wide margin the hardest to
  estimate from samples (see :data:`OSC_EPS`).
* ``max_prob`` -- ``max_n p(n)``, a concentration probe with no score vector at all.

**Sign, and how to label.**  Every ``p log p`` term is ``<= 0``, so the weights decide entirely:
``ent`` and ``ent_n_first`` are non-positive (``n_first`` scores ``n_0 mod 2 in {0, 1}``, a
non-negative weight, so it cannot flip a term), meaning ``sign(O)`` yields ONE class -- threshold
those instead, e.g. at the median.  ``ent_parity`` / ``ent_majority`` carry mixed ``+-1`` weights
and come out genuinely mixed-sign (measured at ``m=6, k=3``: ~57% / ~35% positive).
``ent_bunching`` is mixed in principle but ~99% positive in practice (most mass sits on
collision-free outcomes where the weight is ``+1``), so threshold it too.

Carried from ``model/photonic_observables/{entropy,oscillatory,max_prob}.py``.
"""

from __future__ import annotations

import re

import torch

from .base import (Observable, ObservableContext, ObservableFamily, ProbFunction, base_score_vec,
                   register)
from .scorers.counting import require_even_m

#: Bases usable under ``ent_<base>`` / ``osc_<base>`` -- the plain counting scorers.  Bare
#: ``ent`` / ``osc`` mean weight ``1`` everywhere.
POINTWISE_BASES = ("parity", "majority", "bunching", "n_first")

ENT_BASES = POINTWISE_BASES
OSC_BASES = POINTWISE_BASES

#: Oscillation scale for ``osc``.  **Not a monotone dial** -- both ends degrade, for different
#: reasons -- and it genuinely changes the observable, so it is part of the score-cache identity.
#: Fixed as a module constant rather than exposed as a config knob.
OSC_EPS = 1e-3

#: Floor inside ``log`` for ``ent``.  Unlike the ``xent`` clamp this is the *true* limit, not a
#: regularisation: ``p log p -> 0`` as ``p -> 0``, and a zero-probability outcome contributes
#: exactly ``0 * log(eps) = 0``.
ENT_EPS = 1e-12

_ENT_RE = re.compile(r"^ent(?:_(.+))?$")
_OSC_RE = re.compile(r"^osc(?:_(.+))?$")


class EntropyWeighted(ProbFunction):
    """``phi(p) = p log p``, so ``O(x) = sum_n v_n . p(n) log p(n)`` -- a weighted neg-entropy."""

    def __init__(self, score_vec=None, *, eps: float = ENT_EPS):
        super().__init__(score_vec)
        self.eps = float(eps)

    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        return probs * torch.log(probs.clamp(min=self.eps))

    def transform_grad(self, probs: torch.Tensor) -> torch.Tensor:
        """``phi'(p) = log p + 1``, so bare ``ent`` has ``V_eff = Var_p(log p)``.

        Finite and exactly computable here (the ``+1`` shifts every entry equally and drops out of
        the variance, consistent with ``psi`` being defined only up to a constant).
        """
        return torch.log(probs.clamp(min=self.eps)) + 1.0


class Oscillatory(ProbFunction):
    """``phi(p) = p sin(1/(p + eps))``, so ``O(x) = sum_n v_n . p sin(1/(p + eps))``.

    The reciprocal inside the sine makes ``phi`` oscillate faster as ``p -> 0``: the argument runs
    from ``1/eps`` down toward 0 across the range a distribution spans, so the sine passes through
    many periods.  The ``p`` prefactor damps the amplitude, giving ``|phi(p)| <= p`` hence
    ``|O| <= 1`` with no normalisation, and ``phi(0) = 0`` exactly.

    ``phi'(p) = sin(1/(p+eps)) - p cos(1/(p+eps))/(p+eps)^2`` reaches ``~1/eps`` near ``p ~ eps``.
    Measured at ``m=6, k=3, eps=1e-3``, Pearson ``r`` between the score from exact ``p`` and from
    an ``S``-shot empirical ``p``:

    ======  ======  ======  ========
    shots   osc     ent     parity
    ======  ======  ======  ========
    100     -0.07    0.82     0.79
    1000     0.33    0.98     0.97
    10000    0.64    0.998    0.997
    100000   0.89    0.9998   0.9997
    ======  ======  ======  ========

    **So generate ``osc`` datasets with ``generation.shots = 0``.**  Measured: shot-scored ``osc`` labels correlate only **0.06** with the exact ones at 100 shots.  A shot-sampled ``osc`` dataset is not a
    noisy version of this observable; at 100 shots it is nearly unrelated to it.

    Smoothness in ``x`` is a separate question from estimability, and the prefactor saves it: the
    fast oscillation lives at ``p ~ eps`` where the term is negligible, while the score is
    dominated by large-``p`` outcomes where the argument is small and slowly varying.  Measured
    ``r`` between ``O(X)`` and ``O(X + 0.1)``: ``0.72``, against ``0.997`` for ``ent``.  Rougher,
    not noise.
    """

    def __init__(self, score_vec=None, *, eps: float = OSC_EPS):
        super().__init__(score_vec)
        self.eps = float(eps)

    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        return probs * torch.sin(1.0 / (probs + self.eps))

    def transform_grad(self, probs: torch.Tensor) -> torch.Tensor:
        """``phi'(p) = sin(1/(p+eps)) - p cos(1/(p+eps)) / (p+eps)^2``.

        Reaches magnitude ``~1/eps`` around ``p ~ eps``, which is exactly why ``V_eff`` is huge for
        ``osc`` and its efficiency comes out low -- the quantitative version of "hardest family here
        to estimate from samples".
        """
        arg = 1.0 / (probs + self.eps)
        return torch.sin(arg) - probs * torch.cos(arg) / (probs + self.eps) ** 2


class MaxProb(Observable):
    """``max_n p(n)`` per row, clamped to ``>= 1e-10``.

    A concentration probe with no per-outcome score vector, so it subclasses
    :class:`~v2.observable.base.Observable` directly rather than any of the three shapes.

    **Excluded from analysis B for non-differentiability, not for lacking a variance.**  Its
    influence function is ``psi_n = 1[n = argmax]``, well-defined only away from ties, so
    :attr:`is_differentiable` is False and :meth:`influence` raises with that reason -- unlike the
    quadratic and pointwise shapes, which are differentiable and fully in scope.

    **Three separate facts, previously conflated.**  (1) The exclusion above is about smoothness and
    holds in every regime.  (2) It is nonetheless ``partial_basis_safe``: a zero can never be the
    maximum, so ``max`` over the *observed* outcomes equals ``max`` over the full basis, and it scores
    fine on a shot draw or an un-enumerable basis.  (3) Its shot plug-in is separately **biased
    upward** -- the sample max of a noisy vector.  Measured against an exact value of ``0.0940``:

    ======  ==========  ========
    shots   bias        corr
    ======  ==========  ========
    100     ``+0.0177``   0.758
    1000    ``+0.0013``   0.972
    10000   ``-0.0003``   0.996
    ======  ==========  ========

    So it is usable from shots at a large budget and misleading at a small one, which is a statement
    about the estimator, not about availability.
    """

    is_differentiable = False

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return probs.max(dim=1).values.clamp(min=1e-10)

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "max_prob is not differentiable in p: psi_n = 1[n = argmax] is undefined at a tie, and "
            "its plug-in estimator is biased upward. This is a smoothness failure, not a missing "
            "variance -- Quadratic and ProbFunction observables ARE in scope via their influence "
            "functions."
        )


def _parse(regex, observable: str):
    mo = regex.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def parse_ent_observable(observable: str):
    """Split into ``(is_ent, base)``; ``base`` is ``None`` for bare ``ent`` (weight ``1``)."""
    return _parse(_ENT_RE, observable)


def is_ent_observable(observable: str) -> bool:
    is_ent, base = parse_ent_observable(observable)
    return is_ent and (base is None or base in ENT_BASES)


def parse_osc_observable(observable: str):
    """Split into ``(is_osc, base)``; ``base`` is ``None`` for bare ``osc`` (weight ``1``)."""
    return _parse(_OSC_RE, observable)


def is_osc_observable(observable: str) -> bool:
    is_osc, base = parse_osc_observable(observable)
    return is_osc and (base is None or base in OSC_BASES)


def pointwise_base_vec(base: str | None, keys, *, m: int, k: int, label: str, allowed):
    """Per-outcome weight vector, or ``None`` for the bare (unweighted) spelling."""
    if base is None:
        return None
    if base == "majority":
        require_even_m(m, f"{label}_majority")
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=allowed, label=label)


class EntFamily(ObservableFamily):
    describe = f"ent / ent_<base> (base in {ENT_BASES})"

    def matches(self, name: str) -> bool:
        return is_ent_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_ent_observable(name)[1]
        vec = pointwise_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k, label="ent", allowed=ENT_BASES)
        return EntropyWeighted(vec, eps=ENT_EPS)

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        return {"observable": "ent", "base": parse_ent_observable(name)[1], "eps": ENT_EPS}


class OscFamily(ObservableFamily):
    describe = f"osc / osc_<base> (base in {OSC_BASES})"

    def matches(self, name: str) -> bool:
        return is_osc_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_osc_observable(name)[1]
        vec = pointwise_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k, label="osc", allowed=OSC_BASES)
        return Oscillatory(vec, eps=OSC_EPS)

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        # eps genuinely changes the observable, so it belongs to the identity.
        return {"observable": "osc", "base": parse_osc_observable(name)[1], "eps": OSC_EPS}


class MaxProbFamily(ObservableFamily):
    describe = "max_prob"

    def matches(self, name: str) -> bool:
        return name == "max_prob"

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        return MaxProb()

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        return {"observable": "max_prob"}


register(EntFamily())
register(OscFamily())
register(MaxProbFamily())
