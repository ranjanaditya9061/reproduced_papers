"""``ent`` / ``ent_<base>``: ``O(x) = Σ_n <base>(n) · p(n) log p(n)`` -- surprisal-weighted scores.

The non-polynomial sibling of :mod:`.sq` and :mod:`.pairprod`.  Those are degree-2 in ``p`` and
so need two copies of the state to estimate; this one is not a polynomial in ``p`` at all, since
``p log p`` has no Taylor expansion at ``p = 0``.  The value is therefore not the expectation of
any fixed diagonal observable, and no bounded number of single-shot samples estimates it to
additive error -- entropy estimation over ``n_fock`` outcomes needs ~``n_fock / log n_fock``
samples.  That is the strongest not-classically-easy claim of any family here.

Two spellings:

* ``ent``          -- weight ``1`` on every outcome, i.e. ``Σ_n p log p = -H(p)``, the plain
  negative Shannon entropy of the output distribution.  A whole-distribution concentration
  measure, where ``max_prob`` reads only the peak.
* ``ent_<base>``   -- weight each outcome's surprisal contribution by its ``<base>`` score, so
  the result reports *where* the distribution concentrates rather than only how much.

Sign, and how to label -- every ``p log p`` term is ``<= 0``, so the weights decide entirely:

* ``ent`` and ``ent_n_first`` are non-positive, so ``sign(O)`` gives ONE class.  (``n_first``
  scores ``n_0 mod 2 ∈ {0, 1}``, a *non-negative* weight -- see :mod:`.n_first` -- so it cannot
  flip any term.)  Label these by a threshold, e.g. the median.
* ``ent_parity`` / ``ent_majority`` carry mixed ±1 weights and come out genuinely mixed-sign
  (measured at ``m=6, k=3``: ~57% / ~35% positive), so they label by sign like the rest.
* ``ent_bunching`` is mixed-sign in principle but heavily biased in practice (~99% positive at
  ``m=6, k=3``, since most mass sits on collision-free outcomes where the weight is ``+1``), so
  prefer a threshold there too.

Deterministic in ``(observable, m, k)`` -> reproducible / hashable, and fully offline re-scorable
(it needs only the persisted counts + probs).
"""

from __future__ import annotations

import re

from .base import (EntropyWeightedObservable, Observable, ObservableContext, ObservableFamily,
                   base_score_vec, register)
from .majority import require_even_m
from .sq import SQ_BASES

#: Per-Fock-state bases usable under an ``ent_<base>`` observable -- the same elementary
#: scorers ``sq_<base>`` accepts.  ``ent`` with no base means weight ``1`` everywhere.
ENT_BASES = SQ_BASES

_ENT_RE = re.compile(r"^ent(?:_(.+))?$")


def parse_ent_observable(observable: str):
    """Split into ``(is_ent, base)``; ``base`` is ``None`` for bare ``ent`` (weight ``1``).

    Plain observables return ``(False, observable)``, matching
    :func:`~.sq.parse_sq_observable`'s shape.
    """
    mo = _ENT_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_ent_observable(observable: str) -> bool:
    """True for ``ent`` or a well-formed ``ent_<base>`` (``base`` in :data:`ENT_BASES`)."""
    is_ent, base = parse_ent_observable(observable)
    return is_ent and (base is None or base in ENT_BASES)


def ent_base_vec(base: str, keys, *, m: int, k: int):
    """Per-Fock-state ``<base>`` weight list for an ``ent_<base>`` observable.

    Reuses the plain observables' scorers via :data:`~.base.BASE_SCORERS`, so ``ent_parity`` and
    ``sq_parity`` weight by the identical vector.
    """
    if base == "majority":
        require_even_m(m, "ent_majority")
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=ENT_BASES, label="ent")


class EntFamily(ObservableFamily):
    describe = f"ent / ent_<base> (base in {ENT_BASES})"

    def matches(self, name: str) -> bool:
        return is_ent_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_ent_observable(name)[1]
        vec = None if base is None else ent_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k)
        return EntropyWeightedObservable(vec)


register(EntFamily())
