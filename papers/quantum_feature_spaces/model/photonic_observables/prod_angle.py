"""``prod_parity_{consecutive,second}_{pi,random}``: ``Re(exp(i·P(n))) = cos(P(n))``.

A *phase* generalisation of the ``(-1)^P(n)`` family in :mod:`.prod_parity`.  Now
``P(n) = Σ_mono θ_mono·Π_{i in mono} n_i`` -- every monomial carries its OWN angle θ -- and the
score is the real part of the phase ``exp(i·P(n))``, i.e. ``cos(P(n)) ∈ [-1, 1]``.  It contains
``(-1)^P`` as the θ=π special case, since ``cos(π·N) = (-1)^N`` for integer ``N``: with every
θ=π and integer counts the score collapses byte-for-byte onto the plain
``prod_parity_consecutive`` / ``prod_parity_second``.

The base monomial set is the same geometry-derived product set as those observables
(:func:`~.prod_parity.consecutive_monomials`, orders 2..min(k,m), or
:func:`~.prod_parity.second_monomials`).  Two suffixes select the angles:

* ``_pi``      -- θ = π for every monomial -> reduces EXACTLY to the plain ``(-1)^P``
  observable of the same geometry.
* ``_random``  -- θ ~ ``Uniform[0, 2π]`` drawn per monomial from ``angle_seed``: a genuinely
  complex phase whose real part spreads continuously over ``[-1, 1]``.

The monomials are enumerated in one fixed, deterministic order, so the ``_random`` draw is
reproducible and hashable in ``(observable, m, k, angle_seed)`` -- which is why the hash spec
canonicalises to the resolved ``(θ, monomial)`` list rather than to the observable string.

NOTE: an earlier docstring claimed ``_random`` also appends the diagonal ``n_i^2`` self-terms
and drew θ from ``[0, π]``.  Neither is what the code does (no diagonals are appended; the draw
is ``[0, 2π]``), and the code is what every existing dataset was generated with, so it is
preserved verbatim.  Diagonal self-terms do reach the ``*_second_*`` variants anyway, since
:func:`~.prod_parity.second_monomials` enumerates every pair ``i <= j`` including ``(i, i)``.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .base import LinearObservable, Observable, ObservableContext, ObservableFamily, register
from .prod_parity import consecutive_monomials, monomial_product, second_monomials

_PROD_ANGLE_RE = re.compile(r"^prod_parity_(consecutive|second)_(pi|random)$")


def is_prod_parity_angle(observable: str) -> bool:
    """True for an angle/phase prod_parity variant (score ``cos(P(n))``; see :func:`angle_monomials`)."""
    return bool(_PROD_ANGLE_RE.match(observable))


def angle_monomials(observable: str, m: int, k: int, angle_seed: int):
    """``[(θ, monomial), ...]`` for an angle prod_parity observable (needs ``(m, k[, angle_seed])``).

    ``monomial`` is a tuple of mode indices (possibly repeated -- ``(i, i)`` = ``n_i^2``) and ``θ``
    its angle.  The base set is :func:`~.prod_parity.consecutive_monomials` (``*_consecutive_*``)
    or :func:`~.prod_parity.second_monomials` (``*_second_*``).  Angles: ``_pi`` -> π for all (so
    ``cos(P) = (-1)^P``); ``_random`` -> ``Uniform[0, 2π]`` drawn from ``angle_seed`` over the
    monomials in their fixed order, so the draw is reproducible and hashable.
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
    else:                                               # "pi" -- the only other spelling
        thetas = [math.pi] * len(monos)
    return list(zip(thetas, monos))


def prod_parity_angle_score(key, angle_monos) -> float:
    """``Re(exp(i·P(n))) = cos(P(n))`` for one Fock outcome; ``P = Σ θ·Π_{i in mono} n_i``."""
    return math.cos(sum(theta * monomial_product(key, mono) for theta, mono in angle_monos))


class ProdAngleFamily(ObservableFamily):
    describe = "prod_parity_{consecutive,second}_{pi,random}"

    def matches(self, name: str) -> bool:
        return is_prod_parity_angle(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        angle_monos = angle_monomials(name, ctx.m, ctx.k, ctx.angle_seed)
        return LinearObservable([prod_parity_angle_score(key, angle_monos) for key in ctx.keys])

    def hash_spec(self, name: str, ctx: ObservableContext) -> dict:
        # Canonicalise to the resolved (angle, monomial) list: the actual phase polynomial IS the
        # identity, so _pi vs _random, a different angle_seed, or a different (m, k) each give a
        # distinct dataset (equal specs -> identical, hashable list).
        return {"observable": "prod_parity_angle",
                "angle_monomials": [[round(float(th), 12), list(mono)] for th, mono
                                    in angle_monomials(name, ctx.m, ctx.k, ctx.angle_seed)]}


register(ProdAngleFamily())