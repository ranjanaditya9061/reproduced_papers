"""``osc`` / ``osc_<base>``: ``O(x) = Σ_n <base>(n) · p(n) sin(1/(p(n) + eps))``.

The reciprocal inside the sine makes the integrand oscillate faster and faster as ``p -> 0``: the
argument ``1/(p + eps)`` runs from ``1/eps`` at ``p = 0`` down towards ``0`` as ``p`` grows, so
across the probability range a Fock distribution actually spans, the sine passes through many
full periods.  The ``p`` prefactor damps the amplitude, so ``|φ(p)| <= p``, hence ``|O| <= 1`` with
no normalisation, and ``φ(0) = 0`` exactly -- ``eps > 0`` keeps the argument finite and the
prefactor kills the term regardless.

Like :mod:`.entropy` this is a :class:`~.base.PointwiseObservable`: ``Σ_n v_n φ(p(n))`` for a
scalar nonlinearity ``φ``, differing only in the choice of ``φ``.  ``φ'(p) = sin(1/(p+eps)) -
p cos(1/(p+eps))/(p+eps)^2`` reaches magnitude ~``1/eps`` around ``p ~ eps``, so it is far more
sensitive to a perturbation of ``p`` than ``p log p`` is.

**This is by a wide margin the hardest family here to estimate from samples**, which is the point
of it.  Measured at ``m=6, k=3, eps=1e-3`` -- Pearson ``r`` between the score computed from the
exact ``p`` and from an ``S``-shot empirical ``p`` (i.e. what ``nsample=S`` would give), over 300
inputs:

======  ======  ======  ========
shots   osc     ent     parity
======  ======  ======  ========
100     -0.07    0.82     0.79
1000     0.33    0.98     0.97
10000    0.64    0.998    0.997
100000   0.89    0.9998   0.9997
======  ======  ======  ========

``ent`` is barely harder than plain ``parity`` at these budgets; ``osc`` is uncorrelated with its
own exact value at 100 shots and still short of it at 100k.  **Practical consequence: generate
``osc`` datasets with ``nsample = 0`` (exact probabilities).**  A shot-sampled ``osc`` dataset is
not a noisy version of this observable, it is nearly unrelated to it.

Smoothness in ``x`` is a *separate* question from estimability, and here the ``p`` prefactor
saves it: the fast oscillation lives at ``p ~ eps``, where the prefactor makes the term
negligible, while the score is dominated by the large-``p`` outcomes where ``1/(p+eps)`` is small
and slowly varying.  So the score stays a genuine (if rough) function of the input rather than a
hash of ``p`` -- measured ``r`` between ``O(X)`` and ``O(X + 0.1)``: ``0.72`` at ``eps=1e-3``,
against ``0.997`` for ``ent`` and ``0.998`` for ``parity``.  Rougher, not noise.

:data:`OSC_EPS` is the dial, and it is not monotone -- both ends degrade, for different reasons.
"""

from __future__ import annotations

import re

from .base import (Observable, ObservableContext, ObservableFamily, OscillatoryObservable,
                   base_score_vec, register)
from .majority import require_even_m
from .sq import SQ_BASES

#: Per-Fock-state bases usable under an ``osc_<base>`` observable -- the same elementary scorers
#: ``sq_<base>`` / ``ent_<base>`` accept.  Bare ``osc`` means weight ``1`` everywhere.
OSC_BASES = SQ_BASES

#: Oscillation scale for ``φ(p) = p sin(1/(p + eps))``.  The sine's argument is bounded by
#: ``1/eps``, so this sets how many periods ``φ`` sweeps over the probabilities a Fock
#: distribution actually occupies -- and it is NOT a monotone hardness dial.  Measured at
#: ``m=6, k=3``: periods swept, smoothness ``r(O(X), O(X+0.1))``, and the output spread ``std(O)``
#: that any learner needs there to be signal in at all:
#:
#: =======  =======  ==========  ========
#: eps      periods  r(x, x+.1)  std(O)
#: =======  =======  ==========  ========
#: 1e-3       158      0.72       0.141
#: 1e-2        15      0.81       0.152
#: 1e-1       1.1      0.996      0.176
#: 5e-1       0.1      0.996      0.005
#: 1e+0       0.0      0.997      0.005
#: =======  =======  ==========  ========
#:
#: Past ~``1e-1`` the observable degenerates: ``1/(p+eps) -> 1/eps`` for every occupied ``p``, so
#: ``φ(p) -> p sin(1/eps)`` and the score collapses to a near-constant multiple of ``Σ p = 1``
#: (spread drops ~30x).  So ``eps`` too LARGE kills the signal, ``eps`` too SMALL kills sample
#: estimability (see the module docstring); ``1e-3`` to ``1e-1`` is the usable band.
#:
#: ``eps`` is part of the dataset identity but is fixed here as a module constant (like
#: :data:`~.graphs.MAX_VERTEX_DEGREE`) rather than encoded in the observable string -- so changing
#: it silently redefines any dataset already generated under ``osc``.  Bump it deliberately, and
#: regenerate.
OSC_EPS = 1e-3

_OSC_RE = re.compile(r"^osc(?:_(.+))?$")


def parse_osc_observable(observable: str):
    """Split into ``(is_osc, base)``; ``base`` is ``None`` for bare ``osc`` (weight ``1``).

    Plain observables return ``(False, observable)``, matching
    :func:`~.sq.parse_sq_observable` / :func:`~.entropy.parse_ent_observable`.
    """
    mo = _OSC_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_osc_observable(observable: str) -> bool:
    """True for ``osc`` or a well-formed ``osc_<base>`` (``base`` in :data:`OSC_BASES`)."""
    is_osc, base = parse_osc_observable(observable)
    return is_osc and (base is None or base in OSC_BASES)


def osc_base_vec(base: str, keys, *, m: int, k: int):
    """Per-Fock-state ``<base>`` weight list for an ``osc_<base>`` observable.

    Reuses the plain observables' scorers via :data:`~.base.BASE_SCORERS`, so ``osc_parity`` and
    ``ent_parity`` weight by the identical vector.
    """
    if base == "majority":
        require_even_m(m, "osc_majority")
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=OSC_BASES, label="osc")


class OscFamily(ObservableFamily):
    describe = f"osc / osc_<base> (base in {OSC_BASES})"

    def matches(self, name: str) -> bool:
        return is_osc_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_osc_observable(name)[1]
        vec = None if base is None else osc_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k)
        return OscillatoryObservable(vec, eps=OSC_EPS)


register(OscFamily())
