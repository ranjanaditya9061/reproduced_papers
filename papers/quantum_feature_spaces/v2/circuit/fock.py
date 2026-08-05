"""The Fock outcome basis, built combinatorially.

``fock_keys`` is the same *set* of outcome labels a merlin ``QuantumLayer`` reports for
``(m, k)``, computed from stars-and-bars so the classical models need no quantum dependency.

The enumeration order need not match merlin's: every observable scores
``sum_n v(n) phi(p(n))`` with ``v`` and ``p`` indexed by the same enumeration, so the score is
invariant to it.  Align the order only if you want to compare two platforms' ``p`` vectors
entry-by-entry rather than their scores.

Carried from ``model/mlp_fock.py::fock_keys``.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb


def n_fock(m: int, k: int) -> int:
    """``C(m + k - 1, k)`` -- the number of ``k``-photon occupations of ``m`` modes."""
    return comb(int(m) + int(k) - 1, int(k))


def fock_keys(m: int, k: int) -> list[tuple[int, ...]]:
    """The ``C(m+k-1, k)`` ``k``-photon occupation vectors over ``m`` modes, canonically ordered."""
    m, k = int(m), int(k)
    if m < 1 or k < 1:
        raise ValueError(f"need m >= 1 and k >= 1 (got m={m}, k={k})")
    keys = []
    for combo in combinations_with_replacement(range(m), k):
        occ = [0] * m
        for i in combo:
            occ[i] += 1
        keys.append(tuple(occ))
    return keys


def binary_keys(n_outcomes: int = 2) -> list[tuple[int, ...]]:
    """A trivial one-hot outcome basis, for the models with no Fock structure.

    ``[(1, 0), (0, 1)]`` for two outcomes.  This is what lets the non-Fock models
    (:mod:`v2.model.mlp`, :mod:`v2.model.analytical`) satisfy the same
    :class:`~v2.model.base.DistributionModel` interface as the boson sampler: they emit a
    distribution over a 2-element basis instead of over Fock space.

    The pay-off is that ``parity`` on this basis is exactly the signed score.  With
    ``keys = [(1,0), (0,1)]`` the parity of the first ``ceil(2/2) = 1`` mode gives
    ``v = (-1, +1)``, so ``probs @ v = p_1 - p_0``; a model that sets ``p_1 = (1 + s)/2``
    recovers ``s`` exactly.  So no model needs a special-cased scalar path, and the whole
    pipeline has one shape.
    """
    n = int(n_outcomes)
    return [tuple(1 if i == j else 0 for i in range(n)) for j in range(n)]
