"""``prod_parity``: ``(-1)^P(n)`` with ``P(n)`` a signed sum of square-free monomials in the counts.

``P(n) = Σ_monomials c·Π_{i in monomial} n_i`` over the per-mode photon counts (``c`` an integer
coefficient), and the score is ``(-1)^P(n) ∈ {+1, -1}`` -- a product-generalisation of ``parity``
(which is just ``(-1)^{Σ n_i}``, the sum of all size-1 monomials).  Each mode appears at most once
per monomial (square-free; "no higher powers"), but a count itself may exceed 1.

The monomial set is chosen by ``__``-suffixes on the observable string.  A segment is a named
preset or an explicit (possibly subtracted) monomial, and the segments are summed (this is how
``custom = presetA + presetB`` is expressed):

* preset ``full`` / ``top``  -- the single highest monomial, all m modes  (``Π_i n_i``)
* preset ``lo<t>``           -- "leave-t-out": every monomial with t fewer modes than the highest,
  i.e. all ``(m-t)``-subsets.  ``lo1`` = every monomial with exactly one mode dropped.
* explicit ``M<i>-<j>-...``  -- the monomial ``+n_i·n_j·...`` (dash-joined mode indices)
* explicit ``N<i>-<j>-...``  -- the *subtracted* monomial ``-n_i·n_j·...`` (same syntax as ``M``)

Because the score is ``(-1)^P(n)`` it depends only on ``P(n) mod 2``: coefficients are summed per
monomial (``M`` adds +1, ``N`` adds -1) and only the monomials with an *odd* total survive -- the
sign itself never affects the score, and a monomial that accumulates an even coefficient cancels
(added twice ``M__M``, or added then subtracted ``M__N``).

Examples: ``prod_parity`` (== ``prod_parity__full``) = ``n_0·n_1·…·n_{m-1}``;
``prod_parity__lo1`` = ``Σ_j Π_{i≠j} n_i``; ``prod_parity__full__lo1`` = the sum of both;
``prod_parity__M0-1__M2-3`` = ``n_0·n_1 + n_2·n_3``; ``prod_parity__M0-1__N2-3-4`` =
``n_0·n_1 - n_2·n_3·n_4``.

Two siblings derive their monomials from the problem geometry ``(m[, k])`` instead of from the
string: :data:`PROD_PARITY_CONSECUTIVE` (sliding windows of every order 2..min(k,m)) and
:data:`PROD_PARITY_SECOND` (the order-2 slice).  All three feed the same ``(-1)^P(n)`` scorer, so
:func:`prod_family_monomials` is the single entry point.
"""

from __future__ import annotations

import re
from itertools import combinations

from .base import LinearObservable, Observable, ObservableContext, ObservableFamily, register

#: Named presets for :func:`parse_prod_parity` (``lo<t>`` is a family, any t < m).
PROD_PARITY_PRESETS = ("full", "top", "lo<t>")

#: ``prod_parity`` variants whose monomial set is derived from the problem geometry ``(m[, k])``
#: rather than the observable string.
PROD_PARITY_CONSECUTIVE = "prod_parity_consecutive"
PROD_PARITY_SECOND = "prod_parity_second"

_PROD_PARITY_RE = re.compile(r"^prod_parity(?:__.+)?$")
_LO_RE = re.compile(r"^lo(\d+)$")


def is_prod_parity_observable(observable: str) -> bool:
    """True for a ``prod_parity`` observable (bare, or with ``__`` monomial segments)."""
    return bool(_PROD_PARITY_RE.match(observable))


def is_prod_parity_consecutive(observable: str) -> bool:
    """True for the ``prod_parity_consecutive`` observable (monomials come from ``(m, k)``)."""
    return observable == PROD_PARITY_CONSECUTIVE


def is_prod_parity_second(observable: str) -> bool:
    """True for the ``prod_parity_second`` observable (the order-2 slice; from ``m``)."""
    return observable == PROD_PARITY_SECOND


def is_prod_family(observable: str) -> bool:
    """True for any prod-parity-family observable (``prod_parity[...]``, ``_consecutive``, ``_second``)."""
    return (is_prod_parity_observable(observable) or is_prod_parity_consecutive(observable)
            or is_prod_parity_second(observable))


def parse_prod_segment(seg: str, m: int):
    """One ``__`` segment -> list of ``(coeff, monomial)`` pairs.

    ``coeff`` is ``+1`` for an added term (presets and ``M<...>``) or ``-1`` for a
    subtracted one (``N<...>``); ``monomial`` is a frozenset of mode indices.
    """
    if seg in ("full", "top"):
        return [(1, frozenset(range(m)))]
    mo = _LO_RE.match(seg)
    if mo is not None:
        size = m - int(mo.group(1))
        if size < 1:
            raise ValueError(f"prod_parity segment {seg!r}: leaves fewer than 1 mode (m={m})")
        return [(1, frozenset(c)) for c in combinations(range(m), size)]
    if seg[:1] in ("M", "N"):
        coeff = 1 if seg[0] == "M" else -1
        try:
            idx = [int(v) for v in seg[1:].split("-") if v != ""]
        except ValueError as exc:
            raise ValueError(f"bad prod_parity monomial {seg!r}: expected "
                             "M<i>-<j>-... (added) or N<i>-<j>-... (subtracted), "
                             "dash-joined mode indices") from exc
        if not idx:
            raise ValueError(f"empty prod_parity monomial {seg!r}")
        if any(i < 0 or i >= m for i in idx):
            raise ValueError(f"prod_parity monomial {seg!r} has a mode index outside [0, {m})")
        return [(coeff, frozenset(idx))]
    raise ValueError(f"bad prod_parity segment {seg!r}; expected a preset "
                     f"({PROD_PARITY_PRESETS}) or an explicit monomial "
                     "M<i>-<j>-... (added) / N<i>-<j>-... (subtracted)")


def parse_prod_parity(observable: str, m: int):
    """Canonical monomial list for a ``prod_parity`` observable (needs ``m`` to expand presets).

    Returns a sorted list of sorted mode-index tuples: the monomials whose summed
    coefficient over the ``__`` segments is *odd* (``M`` adds +1, ``N`` subtracts 1).
    Because the score is ``(-1)^P(n)`` only ``P(n) mod 2`` matters, so a monomial with an
    even total coefficient cancels -- added twice (``M__M``) or added then subtracted
    (``M__N``) -- and the sign never survives.  Bare ``prod_parity`` defaults to ``full``
    (the single highest monomial).  Deterministic in ``(observable, m)`` so equal specs --
    e.g. ``prod_parity__M0-1__M2-3`` and ``prod_parity__M2-3__M0-1`` -- canonicalise (and
    thus hash) identically.
    """
    if not is_prod_parity_observable(observable):
        raise ValueError(f"{observable!r} is not a prod_parity observable")
    segments = observable.split("__")[1:] or ["full"]
    coeffs: dict = {}
    for seg in segments:
        for coeff, mono in parse_prod_segment(seg, m):
            coeffs[mono] = coeffs.get(mono, 0) + coeff
    monos = {mono for mono, c in coeffs.items() if c % 2 != 0}
    if not monos:
        raise ValueError(f"prod_parity observable {observable!r} has no surviving monomials "
                         "(every term cancelled to an even coefficient mod 2)")
    return sorted(tuple(sorted(mono)) for mono in monos)


def second_monomials(m: int):
    """Second-order (pair) monomials over ``m`` modes: ``P(n) = Σ_{i <= j} n_i·n_j``.

    Independent of ``k``.  Returns a sorted list of sorted mode-index tuples, matching
    :func:`parse_prod_parity`.

    NOTE: the name and the original docstring say *consecutive* pairs
    (``n0n1 + n1n2 + …``), but the implementation enumerates **every** pair ``i <= j``,
    including the diagonal self-terms ``(i, i)`` -- i.e. ``n_i^2``, which makes the set
    non-square-free.  That is what every dataset under ``prod_parity_second`` (and under the
    ``*_second_*`` angle variants) was generated with, so it is preserved verbatim.
    """
    m = int(m)
    if m < 2:
        raise ValueError(f"prod_parity_second needs m >= 2 (m={m}): there is no pair product")
    return [(i, j) for i in range(m) for j in range(m) if i <= j]


def consecutive_monomials(m: int, k: int):
    """Consecutive sliding-window monomials of every order ``2..min(k, m)`` over ``m`` modes.

    ``P(n) = Σ_{w=2}^{min(k,m)} Σ_{i=0}^{m-w} Π_{j=i}^{i+w-1} n_j`` -- e.g. ``m=6, k=3`` gives
    ``n0n1 + n1n2 + … + n4n5``  (all consecutive pairs)  ``+  n0n1n2 + n1n2n3 + … + n3n4n5``
    (all consecutive triples).  Orders run from 2 (the first genuine *product*) up to the
    photon count ``k`` (a monomial of order > k is 0 on every k-photon outcome), and are
    capped at ``m`` (a window cannot exceed the mode count).  Returns a sorted list of sorted
    mode-index tuples, matching :func:`parse_prod_parity`.
    """
    m, k = int(m), int(k)
    top = min(k, m)
    if top < 2:
        raise ValueError(f"prod_parity_consecutive needs min(k, m) >= 2 (m={m}, k={k}): "
                         "there is no consecutive product below order 2")
    monos = {tuple(range(i, i + w)) for w in range(2, top + 1) for i in range(m - w + 1)}
    return sorted(monos)


def prod_family_monomials(observable: str, m: int, k: int):
    """Monomials for any prod-parity-family observable.

    Dispatches to :func:`consecutive_monomials` (orders 2..k, needs ``k``) for
    ``prod_parity_consecutive``, to :func:`second_monomials` (order-2 pairs, ``k`` unused) for
    ``prod_parity_second``, else to :func:`parse_prod_parity`.  All feed the same ``(-1)^P(n)``
    scorer :func:`prod_parity_score`.
    """
    if is_prod_parity_consecutive(observable):
        return consecutive_monomials(m, k)
    if is_prod_parity_second(observable):
        return second_monomials(m)
    return parse_prod_parity(observable, m)


def monomial_product(key, mono) -> int:
    """``Π_{i in mono} n_i`` for one Fock outcome (short-circuits on a zero count)."""
    prod = 1
    for i in mono:
        prod *= int(key[i])
        if prod == 0:
            break                                       # a zero count kills the monomial
    return prod


def prod_parity_score(key, monomials) -> int:
    """``(-1)^{P(n)}`` for one Fock outcome (``P`` = Σ over monomials of the count product)."""
    total = sum(monomial_product(key, mono) for mono in monomials)
    return 1 if total % 2 == 0 else -1


class ProdParityFamily(ObservableFamily):
    describe = ("prod_parity[__<preset|M<i>-<j>-...|N<i>-<j>-...>...] / "
                f"{PROD_PARITY_CONSECUTIVE} / {PROD_PARITY_SECOND}")

    def matches(self, name: str) -> bool:
        return is_prod_family(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        monomials = prod_family_monomials(name, ctx.m, ctx.k)
        return LinearObservable([prod_parity_score(key, monomials) for key in ctx.keys])

    def hash_spec(self, name: str, ctx: ObservableContext) -> dict:
        # Canonicalise to the monomial set: equivalent spellings (segment order, duplicates,
        # preset/explicit mixes -- and prod_parity_consecutive vs the explicit prod_parity that
        # expands to the same monomials) map to one dataset.
        return {"observable": "prod_parity",
                "monomials": [list(mono) for mono in
                              prod_family_monomials(name, ctx.m, ctx.k)]}


register(ProdParityFamily())