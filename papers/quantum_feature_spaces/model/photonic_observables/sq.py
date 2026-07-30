"""``sq_<base>``: ``O(x) = Σ_n <base>(n) · p(n)^2`` -- the diagonal degree-2 observable.

Every *linear* observable ``O(x) = Σ_n p_x(n)·P(n) = <P>`` is a bounded diagonal expectation,
additive-error estimable from single-shot samples in ~1/eps^2 shots (just average ``P`` over the
samples -- no permanents).  That is the classically "easy" regime and a weak basis for a hardness
/ quantum-advantage teacher.  ``sq_<base>`` is degree-2 in ``p``: a collision / 2-copy quantity
(a signed purity term) that is NOT single-shot additive estimable -- you need two independent
samples to hit the same outcome.

The score is signed (mixed ±1 through ``<base>``), so ``sign(O)`` yields two classes with no
centering or balancing.  Deterministic in ``(observable, m, k)`` -> reproducible / hashable.

This *is* ``p^T K p`` with ``K = diag(<base>)`` -- :class:`~.base.SquaredObservable` subclasses
:class:`~.base.QuadraticObservable` and exists only to store that diagonal alone, which drops the
cost from O(n_fock^2) to O(n_fock) and is why ``sq_<base>`` has no Fock-dimension cap where
``pairprod`` does.  See :mod:`.pairprod` for the dense (non-separable) kernel and
:mod:`.entropy` for the non-polynomial sibling.
"""

from __future__ import annotations

import re

from .base import (Observable, ObservableContext, ObservableFamily, SquaredObservable,
                   base_score_vec, register)
from .majority import require_even_m

#: Per-Fock-state bases usable under a ``sq_<base>`` observable (the plain ±1/float scorers).
SQ_BASES = ("parity", "majority", "bunching", "n_first")

_SQ_RE = re.compile(r"^sq_(.+)$")


def parse_sq_observable(observable: str):
    """Split ``sq_<base>`` into ``(is_sq, base)`` (plain observables -> ``(False, observable)``)."""
    mo = _SQ_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_sq_observable(observable: str) -> bool:
    """True for a ``sq_<base>`` observable (score ``Σ_n <base>(n) p(n)^2``; base in :data:`SQ_BASES`)."""
    is_sq, base = parse_sq_observable(observable)
    return is_sq and base in SQ_BASES


def sq_base_vec(base: str, keys, *, m: int, k: int):
    """Per-Fock-state ``<base>`` score list for a ``sq_<base>`` observable.

    Reuses the plain observables' scorers via :data:`~.base.BASE_SCORERS`, so the two paths can
    never drift.  Kept on the ``(base, keys, *, m, k)`` signature because
    :mod:`data.photonic_quantum` scores the batched generator path with it directly.
    """
    if base == "majority":
        require_even_m(m, "sq_majority")
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=SQ_BASES, label="sq")


class SqFamily(ObservableFamily):
    describe = f"sq_<base> (base in {SQ_BASES})"

    def matches(self, name: str) -> bool:
        return is_sq_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_sq_observable(name)[1]
        return SquaredObservable(sq_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k))


register(SqFamily())