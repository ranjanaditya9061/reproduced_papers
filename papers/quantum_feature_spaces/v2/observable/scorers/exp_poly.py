"""``exp(poly)`` score vectors: ``(-1)^P(n)`` and ``cos(P(n))`` for a count polynomial ``P``.

``P(n) = sum_mono c . prod_{i in mono} n_i`` over the per-mode photon counts.  Two readings:

* ``prod_parity[...]`` -- integer coefficients, score ``(-1)^P(n) in {+1, -1}``.  A product
  generalisation of ``parity`` (which is ``(-1)^{sum n_i}``, all size-1 monomials).
* ``prod_parity_{consecutive,second}_{pi,random}`` -- a per-monomial *angle*, score
  ``Re(exp(i P(n))) = cos(P(n)) in [-1, 1]``.  Contains the above as the ``theta = pi`` case,
  since ``cos(pi N) = (-1)^N`` for integer ``N``.

**These are expectation values.**  However elaborate the polynomial, the result is ``probs @ v``
with ``v(n) = (-1)^{P(n)}`` or ``cos(P(n))`` -- a fixed, bounded, diagonal observable.  The
``exp(poly)`` structure is a way of *constructing* ``v``, not a different functional shape.

**Relevant to the metrics**: these are the score vectors most likely to alternate sign rapidly
across the outcome basis, hence the family whose *mean* is most likely to oscillate in ``x``.  A
low global variance ratio ``G_O`` therefore does not imply this family is uninformative -- see the
one-directionality caveat in :mod:`v2.metrics.observable`, which is why ``G_O`` may prioritise work
but must never exclude a cell.

**Two identities and a degeneracy, measured at ``m=6, k=3`` -- know these before reading a result:**

* ``prod_parity_consecutive_pi`` is *numerically identical* to ``prod_parity_consecutive``, which
  is the ``cos(pi N) = (-1)^N`` claim above holding in practice.  A useful free cross-check: if
  they ever diverge, the angle path has a bug.
* ``prod_parity_second`` coincides with ``bunching``.  Not a coincidence, and not restricted to
  one seed: ``sum_{i<=j} n_i n_j = (k^2 + sum_i n_i^2) / 2``, which at ``k = 3`` is even exactly
  on the collision-free outcomes.  So at ``k = 3`` this observable *is* the collision-free
  indicator, and reporting both as independent readouts double-counts one measurement.
* **Bare ``prod_parity`` is degenerate whenever ``k < m``.**  It defaults to the ``full`` monomial
  ``n_0 . n_1 ... n_{m-1}``, which requires *every* mode occupied; with ``k`` photons in ``m > k``
  modes the product is identically 0, so the score is the constant ``+1`` with zero variance and
  carries no information at all.  Use an explicit segment (``prod_parity__lo3``,
  ``prod_parity__M0-1__M2-3``) or the geometry-derived ``prod_parity_consecutive``, which is
  capped at order ``min(k, m)`` precisely to avoid this.

Carried from ``model/photonic_observables/{prod_parity,prod_angle}.py``, including the two
docstring corrections recorded there (``second_monomials`` enumerates every pair ``i <= j``
*including* the diagonal ``n_i^2`` self-terms, and the ``_random`` draw is ``[0, 2pi]``): both are
what every existing dataset was generated with, so they are preserved verbatim.
"""

from __future__ import annotations

import math
import re
from itertools import combinations

import numpy as np

from ..base import Expectation, Observable, ObservableContext, ObservableFamily, register

#: Named presets for :func:`parse_prod_parity` (``lo<t>`` is a family, any ``t < m``).
PROD_PARITY_PRESETS = ("full", "top", "lo<t>")

#: Variants whose monomial set comes from the problem geometry rather than the observable string.
PROD_PARITY_CONSECUTIVE = "prod_parity_consecutive"
PROD_PARITY_SECOND = "prod_parity_second"

_PROD_PARITY_RE = re.compile(r"^prod_parity(?:__.+)?$")
_LO_RE = re.compile(r"^lo(\d+)$")
_PROD_ANGLE_RE = re.compile(r"^prod_parity_(consecutive|second)_(pi|random)$")


def is_prod_parity_observable(observable: str) -> bool:
    """True for a ``prod_parity`` observable (bare, or with ``__`` monomial segments)."""
    return bool(_PROD_PARITY_RE.match(observable))


def is_prod_parity_consecutive(observable: str) -> bool:
    return observable == PROD_PARITY_CONSECUTIVE


def is_prod_parity_second(observable: str) -> bool:
    return observable == PROD_PARITY_SECOND


def is_prod_family(observable: str) -> bool:
    """True for any integer-coefficient prod-parity observable."""
    return (is_prod_parity_observable(observable) or is_prod_parity_consecutive(observable)
            or is_prod_parity_second(observable))


def is_prod_parity_angle(observable: str) -> bool:
    """True for an angle/phase variant (score ``cos(P(n))``)."""
    return bool(_PROD_ANGLE_RE.match(observable))


def parse_prod_segment(seg: str, m: int):
    """One ``__`` segment -> list of ``(coeff, monomial)`` pairs.

    ``coeff`` is ``+1`` for an added term (presets and ``M<...>``) or ``-1`` for a subtracted one
    (``N<...>``); ``monomial`` is a frozenset of mode indices.
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
            raise ValueError(f"bad prod_parity monomial {seg!r}: expected M<i>-<j>-... (added) "
                             "or N<i>-<j>-... (subtracted), dash-joined mode indices") from exc
        if not idx:
            raise ValueError(f"empty prod_parity monomial {seg!r}")
        if any(i < 0 or i >= m for i in idx):
            raise ValueError(f"prod_parity monomial {seg!r} has a mode index outside [0, {m})")
        return [(coeff, frozenset(idx))]
    raise ValueError(f"bad prod_parity segment {seg!r}; expected a preset "
                     f"({PROD_PARITY_PRESETS}) or an explicit monomial M<i>-... / N<i>-...")


def parse_prod_parity(observable: str, m: int):
    """Canonical monomial list for a ``prod_parity`` observable (needs ``m`` to expand presets).

    Returns sorted tuples of mode indices: the monomials whose summed coefficient over the ``__``
    segments is *odd*.  Because the score is ``(-1)^P(n)`` only ``P(n) mod 2`` matters, so a
    monomial with an even total cancels -- added twice (``M__M``) or added then subtracted
    (``M__N``) -- and the sign never survives.  Bare ``prod_parity`` defaults to ``full``.
    Deterministic in ``(observable, m)``, so equivalent spellings canonicalise (and hash) alike.
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
    """Order-2 monomials over ``m`` modes: every pair ``i <= j``.

    NOTE the name suggests *consecutive* pairs, but this enumerates **every** pair including the
    diagonal self-terms ``(i, i)`` -- i.e. ``n_i^2``, which makes the set non-square-free.  That is
    what every dataset under ``prod_parity_second`` (and the ``*_second_*`` angle variants) was
    generated with, so it is preserved verbatim.
    """
    m = int(m)
    if m < 2:
        raise ValueError(f"prod_parity_second needs m >= 2 (m={m}): there is no pair product")
    return [(i, j) for i in range(m) for j in range(m) if i <= j]


def consecutive_monomials(m: int, k: int):
    """Consecutive sliding-window monomials of every order ``2..min(k, m)``.

    ``m=6, k=3`` gives all consecutive pairs plus all consecutive triples.  Orders run from 2 (the
    first genuine product) to the photon count ``k`` (a monomial of order > k is 0 on every
    ``k``-photon outcome), capped at ``m``.
    """
    m, k = int(m), int(k)
    top = min(k, m)
    if top < 2:
        raise ValueError(f"prod_parity_consecutive needs min(k, m) >= 2 (m={m}, k={k})")
    monos = {tuple(range(i, i + w)) for w in range(2, top + 1) for i in range(m - w + 1)}
    return sorted(monos)


def prod_family_monomials(observable: str, m: int, k: int):
    """Monomials for any integer-coefficient prod-parity observable."""
    if is_prod_parity_consecutive(observable):
        return consecutive_monomials(m, k)
    if is_prod_parity_second(observable):
        return second_monomials(m)
    return parse_prod_parity(observable, m)


def monomial_product(key, mono) -> int:
    """``prod_{i in mono} n_i`` for one outcome (short-circuits on a zero count)."""
    prod = 1
    for i in mono:
        prod *= int(key[i])
        if prod == 0:
            break
    return prod


def prod_parity_score(key, monomials) -> int:
    """``(-1)^{P(n)}`` for one outcome (``P`` = sum over monomials of the count product)."""
    total = sum(monomial_product(key, mono) for mono in monomials)
    return 1 if total % 2 == 0 else -1


def angle_monomials(observable: str, m: int, k: int, angle_seed: int):
    """``[(theta, monomial), ...]`` for an angle prod_parity observable.

    Base set from :func:`consecutive_monomials` or :func:`second_monomials`.  ``_pi`` gives
    ``theta = pi`` for all (so ``cos(P) = (-1)^P`` exactly); ``_random`` draws
    ``Uniform[0, 2pi]`` from ``angle_seed`` over the monomials in their fixed order, so the draw
    is reproducible and hashable.
    """
    mo = _PROD_ANGLE_RE.match(observable)
    if mo is None:
        raise ValueError(f"{observable!r} is not an angle prod_parity observable "
                         "(expected prod_parity_{consecutive,second}_{pi,random})")
    base_kind, angle_kind = mo.group(1), mo.group(2)
    monos = list(consecutive_monomials(m, k) if base_kind == "consecutive"
                 else second_monomials(m))
    if angle_kind == "random":
        rng = np.random.default_rng(int(angle_seed))
        thetas = [float(t) for t in rng.uniform(0.0, 2 * math.pi, size=len(monos))]
    else:
        thetas = [math.pi] * len(monos)
    return list(zip(thetas, monos))


def prod_parity_angle_score(key, angle_monos) -> float:
    """``Re(exp(i P(n))) = cos(P(n))`` for one outcome; ``P = sum theta . prod n_i``."""
    return math.cos(sum(theta * monomial_product(key, mono) for theta, mono in angle_monos))


class ProdParityFamily(ObservableFamily):
    describe = ("prod_parity[__<preset|M<i>-<j>-...|N<i>-<j>-...>...] / "
                f"{PROD_PARITY_CONSECUTIVE} / {PROD_PARITY_SECOND}")

    def matches(self, name: str) -> bool:
        return is_prod_family(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        monomials = prod_family_monomials(name, ctx.m, ctx.k)
        return Expectation([prod_parity_score(key, monomials) for key in ctx.keys])

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        # Canonicalise to the monomial set, so equivalent spellings (segment order, duplicates,
        # preset/explicit mixes, and prod_parity_consecutive vs an explicit spelling that expands
        # to the same monomials) share one score-cache entry.
        return {"observable": "prod_parity",
                "monomials": [list(mono) for mono in prod_family_monomials(name, ctx.m, ctx.k)]}


class ProdAngleFamily(ObservableFamily):
    describe = "prod_parity_{consecutive,second}_{pi,random}"

    def matches(self, name: str) -> bool:
        return is_prod_parity_angle(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        angle_monos = angle_monomials(name, ctx.m, ctx.k, ctx.angle_seed)
        return Expectation([prod_parity_angle_score(key, angle_monos) for key in ctx.keys])

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        # The resolved phase polynomial IS the identity, so _pi vs _random, a different
        # angle_seed, or a different (m, k) each give a distinct entry.
        return {"observable": "prod_parity_angle",
                "angle_monomials": [[round(float(th), 12), list(mono)] for th, mono
                                    in angle_monomials(name, ctx.m, ctx.k, ctx.angle_seed)]}


# The two patterns are disjoint, so this order is documentation rather than disambiguation:
# `_PROD_PARITY_RE` requires a DOUBLE underscore after "prod_parity", so the single-underscore
# angle names ("prod_parity_consecutive_pi") do not match it, and the geometry-derived names
# ("prod_parity_consecutive") are matched only by their exact-string checks.  Verified, not
# assumed -- but the angle family is still registered first so that a future loosening of
# `_PROD_PARITY_RE` fails toward the more specific family instead of silently swallowing it.
register(ProdAngleFamily())
register(ProdParityFamily())
