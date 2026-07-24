"""Photonic sandwich teacher: W1(Haar) -> phase-encode(x) -> W2(Haar) -> measure.

The multiphoton interference is genuine boson sampling, so this wraps a Merlin
``QuantumLayer`` (perceval/merlin imported lazily, only on construction).

The circuit builder :func:`build_sandwich_circuit` is shared by the teacher and by
:class:`PhotonicFeatureMap` (used by the photonic kernels), so a matched-seed
kernel rebuilds the *identical* ``W1, W2`` (drawn sequentially from one seed, so
``W1 != W2``).  ``PhotonicTeacher.forward`` returns a continuous ``(N, 1)`` score
chosen by ``observable``; ``PhotonicFeatureMap`` exposes the full Fock-state
amplitudes (for the fidelity kernel) and probabilities (for the projected kernel).
"""

from __future__ import annotations

import math
import re
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from .base import Teacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

OBSERVABLES = ("parity", "majority", "bunching", "single_output", "n_first", "max_prob")

#: Base scorers usable under a ``loop_path_<base>`` graph observable (see
#: :func:`_overlay_counts`).  ``loop``/``path`` return the mean loop/path count over
#: the selected subset; the rest are the plain per-Fock-state scores above (minus
#: ``single_output``), averaged over that same subset.
GRAPH_BASES = ("parity", "majority", "bunching", "n_first", "loop", "path")

#: A ``loop_path_<base>`` observable reinterprets each collision-free Fock outcome as
#: an edge set of a fixed graph ``G`` (mode ``i`` <-> edge ``e_i``), keeps only the
#: outcomes that are matchings, overlays them with a fixed reference perfect matching
#: ``M_0`` (``H = E(x) | M_0``, a disjoint union of alternating loops/paths), then
#: pre-selects on the loop/path counts (encoded in the observable string via ``__L`` /
#: ``__P`` suffixes) before scoring ``<base>`` over the renormalised survivors.  See
#: :class:`PhotonicTeacher`.
_LOOP_PATH_RE = re.compile(r"^loop_path_(.+)$")


def _default_input_state(m: int, k: int) -> list[int]:
    """Inject k photons evenly spaced across m modes (no light-cone gaps)."""
    state = [0] * m
    for i in range(k):
        state[round(i * m / k)] = 1
    return state


def build_sandwich_circuit(m: int, n_features: int, seed: int):
    """``W1(Haar) -> PS(x_i) on first n_features modes -> W2(Haar)``.

    ``W1`` and ``W2`` are drawn sequentially from one seed -> reproducible and
    distinct.  Shared by the teacher and the kernel feature map so equal seeds
    give byte-identical circuits.
    """
    import perceval as pcvl

    torch.manual_seed(seed)
    pcvl.random_seed(seed)
    circuit = pcvl.Circuit(m, name="haar_phase_haar")
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W1
    for i in range(n_features):
        circuit.add(i, pcvl.PS(pcvl.P(f"x{i}")))
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W2
    return circuit


def _parity_score(key, parity_modes) -> int:
    n = sum(int(key[i]) for i in parity_modes)
    return 1 if n % 2 == 0 else -1


def _majority_score(key, m: int, k: int) -> float:
    split = m // 2
    n_left = sum(int(key[i]) for i in range(split))
    n_right = sum(int(key[i]) for i in range(split, m))
    return (n_left - n_right) / k


def _bunching_score(key) -> int:
    return 1 if max(int(n) for n in key) <= 1 else -1


def _first_mode_score(key) -> int:
    """Photon count in the first mode; dotted with probs gives ``E[n_0]`` (in [0, k])."""
    return int(key[0]%2)


def _single_output_score(key, input_state) -> int:
    kl = [int(key[i]) for i in range(len(input_state))]
    if kl == list(input_state):
        return 1
    if kl == list(reversed(input_state)):
        return -1
    return 0


# --- prod_parity: (-1)^P(n), P(n) a signed sum of square-free monomials in the counts - #
#
# ``P(n) = Σ_monomials c·Π_{i in monomial} n_i`` over the per-mode photon counts (``c`` an
# integer coefficient), and the score is ``(-1)^P(n) ∈ {+1, -1}`` -- a product-generalisation
# of ``parity`` (which is just ``(-1)^{Σ n_i}``, the sum of all size-1 monomials).  Each mode
# appears at most once per monomial (square-free; "no higher powers"), but a count itself may
# exceed 1.
#
# The monomial set is chosen by ``__``-suffixes on the observable string.  A segment is a
# named preset or an explicit (possibly subtracted) monomial, and the segments are summed
# (this is how ``custom = presetA + presetB`` is expressed):
#
#   preset ``full`` / ``top`` : the single highest monomial, all m modes  (Π_i n_i)
#   preset ``lo<t>``          : "leave-t-out" -- every monomial with t fewer modes than
#                               the highest, i.e. all (m-t)-subsets.  ``lo1`` = every
#                               monomial with exactly one mode dropped.
#   explicit ``M<i>-<j>-...`` : the monomial +n_i·n_j·...  (dash-joined mode indices)
#   explicit ``N<i>-<j>-...`` : the *subtracted* monomial -n_i·n_j·...  (same syntax as ``M``)
#
# Because the score is ``(-1)^P(n)`` it depends only on ``P(n) mod 2``: coefficients are
# summed per monomial (``M`` adds +1, ``N`` adds -1) and only the monomials with an *odd*
# total survive -- the sign itself never affects the score, and a monomial that accumulates
# an even coefficient cancels (added twice ``M__M``, or added then subtracted ``M__N``).
#
# Examples: ``prod_parity`` (== ``prod_parity__full``) = n_0·n_1·…·n_{m-1};
# ``prod_parity__lo1`` = Σ_j Π_{i≠j} n_i; ``prod_parity__full__lo1`` = the sum of both;
# ``prod_parity__M0-1__M2-3`` = n_0·n_1 + n_2·n_3; ``prod_parity__M0-1__N2-3-4`` =
# n_0·n_1 - n_2·n_3·n_4.

#: Named presets for :func:`parse_prod_parity` (``lo<t>`` is a family, any t < m).
PROD_PARITY_PRESETS = ("full", "top", "lo<t>")

_PROD_PARITY_RE = re.compile(r"^prod_parity(?:__.+)?$")
_LO_RE = re.compile(r"^lo(\d+)$")


def is_prod_parity_observable(observable: str) -> bool:
    """True for a ``prod_parity`` observable (bare, or with ``__`` monomial segments)."""
    return bool(_PROD_PARITY_RE.match(observable))


def _parse_prod_segment(seg: str, m: int):
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
        for coeff, mono in _parse_prod_segment(seg, m):
            coeffs[mono] = coeffs.get(mono, 0) + coeff
    monos = {mono for mono, c in coeffs.items() if c % 2 != 0}
    if not monos:
        raise ValueError(f"prod_parity observable {observable!r} has no surviving monomials "
                         "(every term cancelled to an even coefficient mod 2)")
    return sorted(tuple(sorted(mono)) for mono in monos)


#: ``prod_parity`` variants whose monomial set is derived from the problem geometry ``(m[, k])``
#: rather than the observable string (see :func:`consecutive_monomials` / :func:`second_monomials`).
PROD_PARITY_CONSECUTIVE = "prod_parity_consecutive"
PROD_PARITY_SECOND = "prod_parity_second"


def is_prod_parity_consecutive(observable: str) -> bool:
    """True for the ``prod_parity_consecutive`` observable (monomials come from ``(m, k)``)."""
    return observable == PROD_PARITY_CONSECUTIVE


def is_prod_parity_second(observable: str) -> bool:
    """True for the ``prod_parity_second`` observable (consecutive PAIRS only; from ``m``)."""
    return observable == PROD_PARITY_SECOND


def second_monomials(m: int):
    """Consecutive second-order (pair) monomials over ``m`` modes: the order-2 slice.

    ``P(n) = Σ_{i=0}^{m-2} n_i·n_{i+1}`` -- ``n0n1 + n1n2 + n2n3 + …`` -- every neighbouring pair,
    independent of ``k``.  Returns a sorted list of sorted mode-index tuples, matching
    :func:`parse_prod_parity`.
    """
    m = int(m)
    if m < 2:
        raise ValueError(f"prod_parity_second needs m >= 2 (m={m}): there is no pair product")
    return [(i, j) for i in range(m) for j in range(m) if i<=j]


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


def is_prod_family(observable: str) -> bool:
    """True for any prod-parity-family observable (``prod_parity[...]``, ``_consecutive``, ``_second``)."""
    return (is_prod_parity_observable(observable) or is_prod_parity_consecutive(observable)
            or is_prod_parity_second(observable))


def prod_family_monomials(observable: str, m: int, k: int):
    """Monomials for any prod-parity-family observable.

    Dispatches to :func:`consecutive_monomials` (orders 2..k, needs ``k``) for
    ``prod_parity_consecutive``, to :func:`second_monomials` (order-2 pairs, ``k`` unused) for
    ``prod_parity_second``, else to :func:`parse_prod_parity`.  All feed the same ``(-1)^P(n)``
    scorer :func:`_prod_parity_score`.
    """
    if is_prod_parity_consecutive(observable):
        return consecutive_monomials(m, k)
    if is_prod_parity_second(observable):
        return second_monomials(m)
    return parse_prod_parity(observable, m)


def _prod_parity_score(key, monomials) -> int:
    """``(-1)^{P(n)}`` for one Fock outcome (``P`` = Σ over monomials of the count product)."""
    total = 0
    for mono in monomials:
        prod = 1
        for i in mono:
            prod *= int(key[i])
            if prod == 0:
                break                                   # a zero count kills the monomial
        total += prod
    return 1 if total % 2 == 0 else -1


# --- prod_parity angle variants: Re(exp(i P(n))) = cos(P(n)), each monomial its own angle --- #
#
# A *phase* generalisation of the ``(-1)^P(n)`` family above.  Now ``P(n) = Σ_mono θ_mono·Π_{i in
# mono} n_i`` -- every monomial carries its OWN angle θ -- and the score is the real part of the
# phase ``exp(i·P(n))``, i.e. ``cos(P(n)) ∈ [-1, 1]``.  It contains ``(-1)^P`` as the θ=π special
# case, since ``cos(π·N) = (-1)^N`` for integer ``N``: with every θ=π and integer counts the score
# collapses byte-for-byte onto the plain ``prod_parity_consecutive`` / ``prod_parity_second``.
#
# The base monomial set is the same geometry-derived product set as those observables
# (:func:`consecutive_monomials` orders 2..min(k,m), or :func:`second_monomials` consecutive
# pairs); a monomial may now also be a *diagonal self-term* ``n_i^2`` (written as the repeated
# tuple ``(i, i)`` -- the count is multiplied once per occurrence).  Two suffixes select the
# angles:
#
#   ``_pi``     : θ = π for every monomial, products only (NO diagonals) -> reduces EXACTLY to the
#                 plain (-1)^P observable of the same geometry.
#   ``_random`` : θ ~ Uniform[0, π] drawn per monomial from ``angle_seed``, AND the diagonal
#                 ``n_i^2`` self-terms are appended (one per mode) -- a genuinely complex phase
#                 whose real part spreads continuously over [-1, 1].
#
# The monomials are enumerated in one fixed, deterministic order (base products first, then the
# diagonals in mode order) so the ``_random`` draw is reproducible and hashable in
# ``(observable, m, k, angle_seed)``.

_PROD_ANGLE_RE = re.compile(r"^prod_parity_(consecutive|second)_(pi|random)$")


def is_prod_parity_angle(observable: str) -> bool:
    """True for an angle/phase prod_parity variant (score ``cos(P(n))``; see :func:`angle_monomials`)."""
    return bool(_PROD_ANGLE_RE.match(observable))


def angle_monomials(observable: str, m: int, k: int, angle_seed: int):
    """``[(θ, monomial), ...]`` for an angle prod_parity observable (needs ``(m, k[, angle_seed])``).

    ``monomial`` is a tuple of mode indices (possibly repeated -- ``(i, i)`` = ``n_i^2``) and ``θ``
    its angle.  The base set is :func:`consecutive_monomials` (``*_consecutive_*``) or
    :func:`second_monomials` (``*_second_*``); ``_random`` additionally appends the diagonal
    ``n_i^2`` self-term for every mode.  Angles: ``_pi`` -> π for all (so ``cos(P) = (-1)^P``);
    ``_random`` -> ``Uniform[0, π]`` drawn from ``angle_seed`` over the monomials in their fixed
    order (base products first, then diagonals in mode order), so the draw is reproducible and
    hashable.
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
        thetas = [float(t) for t in rng.uniform(0.0, 2*math.pi, size=len(monos))]
    elif angle_kind == "pi":
        thetas = [math.pi] * len(monos)
    else:
        rng = np.random.default_rng(int(angle_seed))
        thetas = [float(t) for t in rng.uniform(0.0, math.pi, size=len(monos))]

    return list(zip(thetas, monos))


def _prod_parity_angle_score(key, angle_monos) -> float:
    """``Re(exp(i·P(n))) = cos(P(n))`` for one Fock outcome; ``P = Σ θ·Π_{i in mono} n_i``."""
    total = 0.0
    for theta, mono in angle_monos:
        prod = 1
        for i in mono:
            prod *= int(key[i])
            if prod == 0:
                break                                   # a zero count kills the monomial
        total += theta * prod
    return math.cos(total)


# --- nonlinear (degree-2) observables: quadratic forms in the Fock probabilities --- #
#
# Every observable above is LINEAR in the output distribution: ``O(x) = Σ_n p_x(n)·P(n) = <P>``,
# a bounded diagonal expectation.  Such a value is additive-error estimable from single-shot samples
# in ~1/eps^2 shots (just average ``P`` over the samples -- no permanents), which is the classically
# "easy" regime and a weak basis for a hardness / quantum-advantage teacher.  The two observables
# here are degree-2 in ``p`` -- collision / 2-copy quantities that are NOT single-shot additive
# estimable (you need two independent samples to hit the same outcome):
#
#   ``sq_<base>``  : ``O(x) = Σ_n <base>(n) · p(n)^2``            -- diagonal (a signed purity term)
#   ``pairprod``   : ``O(x) = Σ_{n1,n2} p(n1) p(n2) (-1)^<n1,n2>`` -- full quadratic form ``p^T K p``
#
# ``pairprod``'s kernel ``K(n1,n2) = (-1)^{Σ_i n1_i·n2_i}`` (parity of the two outcomes' occupation
# OVERLAP) is deliberately NON-SEPARABLE: a separable ``K(n1,n2) = f(n1)·f(n2)`` would collapse to
# ``<f>^2``, a product of two independently additive-estimable expectations -- straight back into the
# easy regime.  The overlap does not factor over the sum, so ``p^T K p`` is a genuine two-point
# function.  Both scores are signed (mixed ±1), so ``sign(O)`` yields two classes with no centering
# or balancing needed.  Both are deterministic in ``(observable, m, k)`` -> reproducible / hashable.

#: Per-Fock-state bases usable under a ``sq_<base>`` observable (the plain ±1/float scorers).
SQ_BASES = ("parity", "majority", "bunching", "n_first")
_SQ_RE = re.compile(r"^sq_(.+)$")

#: Cap on the Fock dimension for ``pairprod``'s dense ``(n_fock, n_fock)`` kernel.  8192 -> a 256 MB
#: fp32 matrix; above this, use a ``sq_<base>`` observable (diagonal, no dense kernel).
PAIRPROD_MAX_FOCK = 8192


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


def is_pairprod_observable(observable: str) -> bool:
    """True for the ``pairprod`` observable (score ``Σ_{n1,n2} p(n1) p(n2) (-1)^<n1,n2>``)."""
    return observable == "pairprod"


def _sq_base_vec(base: str, keys, *, m: int, k: int):
    """Per-Fock-state ``<base>`` score list for a ``sq_<base>`` observable (reuses the plain scorers)."""
    if base == "parity":
        return [_parity_score(key, tuple(range((m + 1) // 2))) for key in keys]
    if base == "majority":
        if m % 2:
            raise ValueError("sq_majority requires even m")
        return [_majority_score(key, m, k) for key in keys]
    if base == "bunching":
        return [_bunching_score(key) for key in keys]
    if base == "n_first":
        return [_first_mode_score(key) for key in keys]
    raise ValueError(f"unknown sq base {base!r}; choose from {SQ_BASES}")


def pairprod_kernel(keys, m: int) -> np.ndarray:
    """Symmetric ±1 pair kernel ``K[a,b] = (-1)^{Σ_i n^a_i · n^b_i}`` for ``a != b`` (``0`` on the diagonal).

    The exponent is the OVERLAP (dot product) of the two outcomes' per-mode occupation vectors, so
    ``K`` does not factor as ``f(a)·f(b)`` -- the quadratic form ``p^T K p = Σ_{a!=b} p(a) p(b) K[a,b]``
    is a genuine two-point (distinct-sample, 2-copy) functional, not a product of two separately
    additive-estimable expectations.  The diagonal is zeroed on purpose: ``K[a,a] = (-1)^{Σ_i (n^a_i)^2}``
    is a *constant* ``(-1)^k`` on the collision-free outcomes, i.e. a fixed-sign purity term that would
    bias the whole score one way (all one class for even ``k``); dropping it leaves the mixed-sign
    off-diagonal part, so ``sign(p^T K p)`` gives two classes with no balancing.  Returns a
    ``(n_fock, n_fock)`` float64 array; deterministic in ``(keys, m)`` -> reproducible / hashable.
    """
    occ = np.array([[int(key[i]+1) for i in range(m)] for key in keys], dtype=np.int64)
    overlap = occ @ occ.T                               # (n_fock, n_fock) integer overlaps
    K = np.where(overlap % 2 == 0, 1.0, -1.0)
    np.fill_diagonal(K, 0.0)                             # distinct-sample two-point functional
    return K


def _occ_matrix(keys, m: int) -> np.ndarray:
    """``(n_fock, m)`` int64 per-mode photon-count matrix for the Fock basis ``keys``."""
    return np.array([[int(key[i]) for i in range(m)] for key in keys], dtype=np.int64)


def _finish_kernel(K: np.ndarray, zero_diagonal: bool) -> np.ndarray:
    """Coerce to float64 and, if requested, zero the diagonal (the distinct-sample form).

    Zeroing drops ``K[a,a]`` -- for the sign/parity kernels a fixed constant on the collision-free
    outcomes (a purity term that biases the whole score one way, all one class for even ``k``); the
    off-diagonal part is the genuine two-point (2-copy) functional and is mixed-sign, so
    ``sign(p^T K p)`` yields two classes with no balancing.
    """
    K = np.asarray(K, dtype=np.float64)
    if zero_diagonal:
        np.fill_diagonal(K, 0.0)
    return K


# --- pairprod kernels: alternative builders for the pair kernel K[a,b] = P(n^a, n^b) ------ #
#
# Each returns a symmetric ``(n_fock, n_fock)`` float64 array and is a drop-in for
# :func:`pairprod_kernel` (call ``f(keys, m)``); the seeded / parameterised ones take extra kwargs
# with defaults.  Governing properties (see the design table): a *separable* ``K = f(a) f(b)``
# collapses to ``<f>^2`` (single-copy, easy) -- all kernels here are non-separable EXCEPT
# ``base_overlay`` with a trivial core and the pure ``rbf`` (a PSD Gram matrix -> ``p^T K p >= 0``
# -> one class).  Only the SYMMETRIC part of a kernel survives ``p^T K p`` (``p`` is real), so every
# builder returns symmetric K.  They use the raw photon counts ``n_i`` (``_occ_matrix``); tweak
# there if you want a shifted count.


def pairprod_random_phase_kernel(keys, m: int, *, seed: int = 0,
                                 zero_diagonal: bool = True) -> np.ndarray:
    """Random bilinear phase ``K[a,b] = cos(n^a^T W n^b)``, ``W`` a seeded symmetric matrix.

    Non-separable, indefinite, and naturally centred: the dense random coupling spreads
    ``n^a^T W n^b`` so the cosine sign is balanced (no sparse-overlap ``+`` bias).  ``W_ij ~ U[0, 2pi]``,
    symmetrised.  Deterministic in ``(keys, m, seed)``.
    """
    occ = _occ_matrix(keys, m)
    A = np.random.default_rng(int(seed)).uniform(0.0, 2 * math.pi, size=(m, m))
    W = 0.5 * (A + A.T)
    return _finish_kernel(np.cos(occ @ W @ occ.T), zero_diagonal)


def pairprod_random_parity_kernel(keys, m: int, *, seed: int = 0,
                                  zero_diagonal: bool = True) -> np.ndarray:
    """Random bilinear parity ``K[a,b] = (-1)^{n^a^T W n^b mod 2}``, ``W`` a seeded symmetric 0/1 matrix.

    The discrete (±1) sibling of :func:`pairprod_random_phase_kernel`.  Non-separable, indefinite,
    roughly centred.  Deterministic in ``(keys, m, seed)``.
    """
    occ = _occ_matrix(keys, m)
    A = np.random.default_rng(int(seed)).integers(0, 2, size=(m, m))
    W = np.triu(A) + np.triu(A, 1).T                    # symmetric 0/1
    return _finish_kernel(np.where(occ @ W @ occ.T % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_weighted_overlap_kernel(keys, m: int, *, seed: int = 0,
                                     zero_diagonal: bool = True) -> np.ndarray:
    """Weighted overlap parity ``K[a,b] = (-1)^{Σ_i w_i n^a_i n^b_i}``, ``w`` seeded integers in [1, 3].

    Non-separable, indefinite.  Non-uniform ``w`` breaks the ``|A∪B| = 2k - |A∩B|`` identity, so this
    is genuinely richer than the unit-weight overlap :func:`pairprod_kernel`.  Deterministic in
    ``(keys, m, seed)``.
    """
    occ = _occ_matrix(keys, m)
    w = np.random.default_rng(int(seed)).integers(1, 4, size=m)
    return _finish_kernel(np.where((occ * w) @ occ.T % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_weighted_or_parity_kernel(keys, m: int, *, seed: int = 0,
                                       zero_diagonal: bool = True) -> np.ndarray:
    """Weighted OR parity ``K[a,b] = (-1)^{Σ_i w_i (n^a_i ∨ n^b_i)}``, ``w`` seeded integers in [1, 3].

    Union (OR) rather than product (AND): ``a_i ∨ b_i = a_i + b_i - a_i b_i``, so this factors as
    ``s(a) s(b) (-1)^{Σ w_i a_i b_i}`` -- a separable ±1 sign overlay ``s(a) = (-1)^{Σ w_i a_i}`` on a
    non-separable weighted-overlap core.  The overlay re-centres (pushes ΣK toward 0) while the core
    keeps it two-copy; equals :func:`pairprod_kernel` at ``w ≡ 1`` (fixed-``k`` collision-free).
    Deterministic in ``(keys, m, seed)``.
    """
    occ = _occ_matrix(keys, m)
    w = np.random.default_rng(int(seed)).integers(1, 4, size=m)
    supp = (occ > 0).astype(np.int64)
    ga = supp @ w                                       # Σ_i w_i supp_a_i
    H = (supp * w) @ supp.T                             # Σ_i w_i supp_a_i supp_b_i
    U = ga[:, None] + ga[None, :] - H                   # Σ_i w_i (a_i ∨ b_i)
    return _finish_kernel(np.where(U % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_distance_sign_kernel(keys, m: int, *, d0: float | None = None,
                                  zero_diagonal: bool = True) -> np.ndarray:
    """Distance sign ``K[a,b] = +1 if ||n^a - n^b||_1 > d0 else -1``.

    Non-separable, indefinite.  ``d0`` defaults to the median off-diagonal L1 distance, which centres
    the split.  Deterministic in ``(keys, m, d0)``.
    """
    occ = _occ_matrix(keys, m)
    n = occ.shape[0]
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(m):                                  # avoid the (n, n, m) broadcast intermediate
        col = occ[:, i]
        D += np.abs(col[:, None] - col[None, :])
    if d0 is None:
        off = D[~np.eye(n, dtype=bool)]
        d0 = float(np.median(off)) if off.size else 0.0
    return _finish_kernel(np.where(D > float(d0), 1.0, -1.0), zero_diagonal)


def pairprod_shared_modes_kernel(keys, m: int, *, t: int = 1,
                                 zero_diagonal: bool = True) -> np.ndarray:
    """Relational ``K[a,b] = +1 if |supp(a) ∩ supp(b)| == t else -1`` (shared-mode count equals ``t``).

    Non-separable, indefinite; the split is tuned by ``t``.  Deterministic in ``(keys, m, t)``.
    """
    supp = (_occ_matrix(keys, m) > 0).astype(np.int64)
    return _finish_kernel(np.where(supp @ supp.T == int(t), 1.0, -1.0), zero_diagonal)


def pairprod_disjoint_kernel(keys, m: int, *, zero_diagonal: bool = True) -> np.ndarray:
    """HOM-like disjointness ``K[a,b] = +1 if a, b share no mode else -1`` (``|supp(a) ∩ supp(b)| == 0``).

    Non-separable, indefinite; biased (two random small supports are usually disjoint -> ``+``).
    The ``t = 0`` case of :func:`pairprod_shared_modes_kernel`.  Deterministic in ``(keys, m)``.
    """
    return pairprod_shared_modes_kernel(keys, m, t=0, zero_diagonal=zero_diagonal)


def pairprod_rbf_kernel(keys, m: int, *, gamma: float = 1.0,
                        zero_diagonal: bool = False) -> np.ndarray:
    """Gaussian/RBF Gram ``K[a,b] = exp(-gamma ||n^a - n^b||^2)``.

    NON-separable but **PSD** -> ``p^T K p >= 0`` -> ONE class (never centred); the definite-kernel
    reference.  ``zero_diagonal=False`` keeps it a true Gram matrix (set ``True`` to force it
    indefinite).  Deterministic in ``(keys, m, gamma)``.
    """
    occ = _occ_matrix(keys, m).astype(np.float64)
    sq = (occ ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (occ @ occ.T)
    return _finish_kernel(np.exp(-float(gamma) * d2), zero_diagonal)


def pairprod_base_overlay_kernel(base_vec, core) -> np.ndarray:
    """Base overlay ``K[a,b] = base(a) base(b) core(a,b)`` (``base_vec`` per-state, ``core`` a kernel).

    Separable **iff** ``core`` is trivial (``core ≡ 1`` -> ``K = v v^T`` -> ``<base>^2``, PSD, one class);
    with a non-separable ``core`` (any builder above) the product stays non-separable.  ``base_vec``
    can be built from the plain scorers (e.g. :func:`_sq_base_vec`).  Diagonal is inherited from ``core``.
    """
    v = np.asarray(base_vec, dtype=np.float64)
    return v[:, None] * v[None, :] * np.asarray(core, dtype=np.float64)


# --- loop_path_<base>: interpret Fock outcomes as edge sets of a fixed graph ---- #

def _parse_var_segment(spec: str):
    """Parse one ``L``/``P`` var spec body: dash-joined non-negative ints (empty = keep all)."""
    spec = spec.strip()
    if not spec:
        return []                                       # e.g. ``__L`` -> [] -> keep all
    try:
        return [int(v) for v in spec.split("-")]
    except ValueError as exc:
        raise ValueError(f"bad loop_path var segment {spec!r}: expected dash-joined "
                         "non-negative ints (empty = keep all)") from exc


def parse_graph_observable(observable: str):
    """Split a graph observable into ``(is_graph, base, loop_vars, path_vars)``.

    Plain observables return ``(False, observable, None, None)``.  A ``loop_path_<base>``
    string may carry filesystem-safe var suffixes ``__L<a>-<b>`` and/or ``__P<a>-<b>`` that
    encode the loop / path selection directly (``__L`` with an empty body means keep-all);
    a *missing* segment yields ``None`` (keep-all on that dimension).  So
    ``loop_path_parity__L0-1__P2-3`` keeps overlays with 0-or-1 loops and 2-or-3 paths, while
    ``loop_path_parity`` keeps every matching.
    """
    mo = _LOOP_PATH_RE.match(observable)
    if mo is None:
        return False, observable, None, None
    parts = mo.group(1).split("__")
    base = parts[0]
    loop_vars = path_vars = None
    for seg in parts[1:]:
        if seg[:1] == "L":
            loop_vars = _parse_var_segment(seg[1:])
        elif seg[:1] == "P":
            path_vars = _parse_var_segment(seg[1:])
        else:
            raise ValueError(f"bad loop_path segment {seg!r} in {observable!r} "
                             "(expected L<ints> or P<ints>)")
    return True, base, loop_vars, path_vars


def is_graph_observable(observable: str) -> bool:
    """True for a well-formed ``loop_path_<base>`` observable (``base`` in :data:`GRAPH_BASES`)."""
    is_graph, base, _, _ = parse_graph_observable(observable)
    return is_graph and base in GRAPH_BASES


def resolve_graph_spec(observable: str, loop_vars, path_vars):
    """``(base, eff_loop_vars, eff_path_vars)`` for a graph observable.

    Vars encoded in the ``observable`` string are authoritative; a dimension left
    unspecified in the string falls back to the passed ``loop_vars`` / ``path_vars`` (a
    programmatic override, normally ``None`` -> keep-all, since the config carries the
    selection only in the observable string).  Single source of truth for the teacher,
    :meth:`PhotonicTeacher.hash_spec` and :func:`score_from_distribution`.
    """
    is_graph, base, s_loop, s_path = parse_graph_observable(observable)
    if not is_graph:
        raise ValueError(f"{observable!r} is not a loop_path_<base> observable")
    eff_loop = s_loop if s_loop is not None else loop_vars
    eff_path = s_path if s_path is not None else path_vars
    return base, eff_loop, eff_path


def _is_connected(edges, n_vertices: int) -> bool:
    """True iff the undirected graph on ``n_vertices`` given by ``edges`` is connected."""
    adj: dict[int, list[int]] = {v: [] for v in range(n_vertices)}
    for u, w in edges:
        adj[u].append(w)
        adj[w].append(u)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == n_vertices


def build_matching_graph(m: int, n_vertices: int, seed: int):
    """A fixed, connected, seeded graph on ``n_vertices`` with ``m`` edges + a matching.

    ``mode i <-> edges[i]``.  ``M_0`` (a *perfect* matching, ``n_vertices // 2`` edges,
    marked by ``m0_mask``) is drawn first, then filled with distinct random edges up to
    ``m`` total; the draw is repeated with a bumped sub-seed until the graph is connected
    (so loop counts are genuinely global, per the expander requirement).  Deterministic
    in ``seed`` -> reproducible and hashable.

    Returns ``(edges, m0_mask)``: ``edges`` a list of ``m`` sorted ``(u, v)`` tuples and
    ``m0_mask`` a ``(m,)`` bool array, ``True`` where the mode's edge belongs to ``M_0``.
    """
    V = int(n_vertices)
    if V < 2 or V % 2:
        raise ValueError(f"n_vertices must be a positive even int (got {V})")
    half = V // 2
    if m < half:
        raise ValueError(f"need m >= V/2={half} to fit the reference matching (m={m})")
    max_edges = V * (V - 1) // 2
    if m > max_edges:
        raise ValueError(f"m={m} exceeds C(V, 2)={max_edges} for V={V}")

    all_edges = [(i, j) for i in range(V) for j in range(i + 1, V)]
    for attempt in range(1024):
        rng = np.random.default_rng(int(seed) + attempt)
        perm = rng.permutation(V)                       # random perfect matching M_0
        m0 = [tuple(sorted((int(perm[2 * t]), int(perm[2 * t + 1])))) for t in range(half)]
        m0_set = set(m0)
        rest = [e for e in all_edges if e not in m0_set]
        rng.shuffle(rest)
        edges = m0 + rest[: m - half]                   # M_0 first, then filler edges
        if not _is_connected(edges, V):
            continue
        order = rng.permutation(m)                      # scatter M_0 across the mode indices
        edges_p = [edges[o] for o in order]
        m0_mask = np.array([edges[o] in m0_set for o in order], dtype=bool)
        return edges_p, m0_mask
    raise RuntimeError(f"could not draw a connected graph (V={V}, m={m}, seed={seed})")


#: Hard cap on any single vertex's degree in :func:`build_vertex_graph`'s ``G``.  An edge whose
#: addition would push either endpoint past this is skipped, so ``G`` stays bounded-degree
#: (expander-like) rather than growing hubs as ``density`` rises; it also caps the achievable
#: edge count at ``V * MAX_VERTEX_DEGREE // 2``.  Set to ``-1`` to disable the cap entirely, i.e.
#: draw exactly ``round(density * C(V, 2))`` edges with no degree limit.
MAX_VERTEX_DEGREE = 4


def build_vertex_graph(m: int, density: float, seed: int):
    """A fixed, seeded, *connected*, bounded-degree graph ``G`` on ``V = m`` vertices.

    The graph builder for the ``connected_<base>`` observable (:func:`is_connected_observable`);
    ``mode i <-> vertex i``.  Each of the ``m`` modes is a *vertex* (not an edge, unlike
    :func:`build_matching_graph`), so ``G`` has ``V = m`` vertices; each Fock outcome scores a
    global property of its *clicked* vertices' (the modes with a non-zero photon count) induced
    subgraph -- ``maxcc``, the size of the largest connected component.  ``G`` itself is drawn
    connected (a random spanning path first, so ``maxcc`` can in principle reach ``V`` when every
    mode clicks), then how large the clicked subsets' components grow is governed by ``G``'s
    density and degree cap.

    ``density`` is the fraction of the ``C(V, 2)`` possible edges targeted, so the edge count is
    ``round(density * C(V, 2))`` -- but it is floored at ``V - 1`` (a connected graph needs a
    spanning tree) and every vertex's degree is capped at :data:`MAX_VERTEX_DEGREE`, so the count
    is also bounded by ``V * MAX_VERTEX_DEGREE // 2``.  ``MAX_VERTEX_DEGREE = -1`` disables the
    cap.  A degree cap below ``2`` can't span ``V > 2`` vertices, so it is rejected.  The spanning
    path plus the seeded filler edges are all drawn from ``seed``, deterministic -> reproducible /
    hashable.

    Returns ``edges``: a sorted list of distinct ``(u, v)`` vertex pairs.
    """
    V = int(m)
    if V < 2:
        raise ValueError(f"need m >= 2 vertices for a graph (m={m})")
    d = float(density)
    if not 0.0 < d <= 1.0:
        raise ValueError(f"graph_density must be in (0, 1] (got {density})")
    max_edges = V * (V - 1) // 2
    capped = MAX_VERTEX_DEGREE >= 0
    if capped and V > 2 and MAX_VERTEX_DEGREE < 2:
        raise ValueError(f"MAX_VERTEX_DEGREE={MAX_VERTEX_DEGREE} can't span V={V} vertices "
                         "(a connected graph needs degree >= 2 on interior vertices)")
    # with a cap, at most V * MAX_VERTEX_DEGREE // 2 edges fit; uncapped, the density target rules
    degree_cap_edges = V * MAX_VERTEX_DEGREE // 2 if capped else max_edges
    n_edges = max(int(round(d * max_edges)), V - 1)      # >= V-1 so G can be connected
    n_edges = min(n_edges, max_edges, degree_cap_edges)

    rng = np.random.default_rng(int(seed))
    degree = [0] * V
    edges = []
    used: set = set()
    # spanning path (max degree 2) -> a connected, degree-legal backbone of V-1 edges
    perm = rng.permutation(V)
    for a, b in zip(perm[:-1], perm[1:]):
        e = (int(a), int(b)) if a < b else (int(b), int(a))
        edges.append(e)
        used.add(e)
        degree[e[0]] += 1
        degree[e[1]] += 1
    # fill with more seeded edges up to n_edges, respecting the degree cap
    all_edges = [(i, j) for i in range(V) for j in range(i + 1, V)]
    order = rng.permutation(len(all_edges))              # seeded scan order
    for o in order:
        if len(edges) >= n_edges:
            break
        u, w = all_edges[int(o)]
        if (u, w) in used:
            continue
        if capped and (degree[u] >= MAX_VERTEX_DEGREE or degree[w] >= MAX_VERTEX_DEGREE):
            continue                                     # keep G bounded-degree
        edges.append((u, w))
        used.add((u, w))
        degree[u] += 1
        degree[w] += 1
    return sorted(edges)


def _overlay_counts(key, edges, m0_edges: set, n_vertices: int):
    """``(valid, n_loops, n_paths)`` for one Fock outcome ``key`` (per-mode counts).

    ``valid`` is ``False`` for a bunched outcome (some mode > 1) or one whose clicked
    edges share a vertex (not a matching).  Otherwise overlays the clicked edges with
    ``M_0`` (set union, so an edge present in *both* becomes a length-1 path) and counts
    cycle components (loops) and path components -- every vertex has degree <= 2 because
    both are matchings, so each component is a simple loop or path.
    """
    counts = [int(c-1) if c>1 else int(c) for c in key]
    if any(c > 1 for c in counts):
        return False, 0, 0                              # bunched -> not collision-free
    used: set[int] = set()
    clicked = []
    for i, c in enumerate(counts):
        if not c:
            continue
        u, w = edges[i]
        if u in used or w in used:
            return False, 0, 0                          # shared vertex -> not a matching
        used.add(u)
        used.add(w)
        clicked.append(edges[i])

    union = m0_edges | set(clicked)                     # set union: shared edges collapse
    adj: dict[int, list[int]] = {v: [] for v in range(n_vertices)}
    for u, w in union:
        adj[u].append(w)
        adj[w].append(u)

    seen = [False] * n_vertices
    n_loops = n_paths = 0
    for s in range(n_vertices):
        if seen[s] or not adj[s]:
            continue                                    # M_0 perfect -> no isolated vertex
        stack = [s]
        seen[s] = True
        is_cycle = True
        while stack:
            x = stack.pop()
            if len(adj[x]) != 2:
                is_cycle = False                        # a degree-1 endpoint -> path
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        n_loops += is_cycle
        n_paths += not is_cycle
    # print(m0_edges, key)
    # print(edges)
    # print(n_loops, n_paths)
    
    return True, n_loops, n_paths


def _graph_tables(keys, edges, m0_mask, n_vertices: int):
    """Per-Fock-state ``(valid, n_loops, n_paths)`` arrays over the fixed basis ``keys``."""
    m0_edges = {edges[i] for i in range(len(edges)) if m0_mask[i]}
    valid = np.zeros(len(keys), dtype=bool)
    loops = np.zeros(len(keys), dtype=np.int64)
    paths = np.zeros(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        v, nl, npth = _overlay_counts(key, edges, m0_edges, n_vertices)
        valid[i], loops[i], paths[i] = v, nl, npth
    print(np.sum(valid), len(valid), np.array2string(np.array(list(set(loops))), separator=', '), np.array2string(np.array(list(set(paths))), separator=', '))
    return valid, loops, paths


def _graph_base_scores(keys, base: str, loops, paths, *, m: int, k: int):
    """Per-Fock-state ``<base>`` score vector for a ``loop_path_<base>`` observable."""
    if base == "loop":
        return loops.astype(np.float64)
    if base == "path":
        return paths.astype(np.float64)
    if base == "parity":
        pm = tuple(range((m + 1) // 2))
        return np.array([_parity_score(key, pm) for key in keys], dtype=np.float64)
    if base == "majority":
        return np.array([_majority_score(key, m, k) for key in keys], dtype=np.float64)
    if base == "bunching":
        return np.array([_bunching_score(key) for key in keys], dtype=np.float64)
    if base == "n_first":
        return np.array([_first_mode_score(key) for key in keys], dtype=np.float64)
    if base == "max_prob":
        return np.array([0.0] * len(keys))
    raise ValueError(f"unknown graph base {base!r}; choose from {GRAPH_BASES}")


def _var_mask(count_arr, vars_) -> np.ndarray:
    """Keep-mask over ``count_arr``: keep where the count is in ``vars_``.

    An empty/``None`` ``vars_`` -- or one containing a negative sentinel -- means "no
    filter on this dimension" (keep every count), so ``loop_path_majority`` with both
    lists empty selects all matchings.
    """
    if not vars_ or any(int(v) < 0 for v in vars_):
        return np.ones_like(count_arr, dtype=bool)
    allowed = {int(v) for v in vars_}
    return np.array([int(c) in allowed for c in count_arr], dtype=bool)


def _graph_selection(keys, *, m, k, base, edges, m0_mask, n_vertices, loop_vars, path_vars):
    """``(keep_mask, base_scores)`` float vectors for a graph observable over ``keys``.

    ``keep_mask`` = matching AND (loop count in ``loop_vars``) AND (path count in
    ``path_vars``); ``base_scores`` is the per-state ``<base>`` value.  Both align to the
    fixed Fock basis, so scoring a distribution is a masked, renormalised dot product.
    """
    valid, loops, paths = _graph_tables(keys, edges, m0_mask, n_vertices)
    keep = valid & _var_mask(loops, loop_vars) & _var_mask(paths, path_vars)
    scores = _graph_base_scores(keys, base, loops, paths, m=m, k=k)
    return keep.astype(np.float64), scores


def _conditional_expectation(probs: torch.Tensor, keep_mask: torch.Tensor,
                             score_vec: torch.Tensor, observable: str) -> torch.Tensor:
    """``E[score | selected]`` per row of ``probs`` ``(N, n_fock)`` (0 where no mass survives)."""
    sel = probs * keep_mask                             # broadcast (n_fock,)
    if observable == "max_prob":
        return sel.max(dim=1).values.clamp(min=1e-10)
    den = sel.sum(dim=1)
    num = sel @ score_vec
    return torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.zeros_like(den))


# --- connected_<base>: interpret Fock outcomes as vertex sets, score a global connectivity ---- #
#
# A sibling of ``loop_path_<base>`` (:func:`parse_graph_observable`) that maps each mode to a
# *vertex* (``mode i <-> vertex i``) of a fixed, seeded, connected, bounded-degree graph ``G`` on
# ``V = m`` vertices (:func:`build_vertex_graph`) rather than to an edge.  For each Fock outcome the
# *clicked* vertices (the modes with a non-zero photon count) induce a subgraph of ``G``, and the
# score is a GLOBAL property of that subgraph: the size (vertex count) of its largest connected
# component (base ``maxcc``).  There is no pre-selection -- the observable's output is the plain
# expectation ``E[<base>]`` over the full Fock distribution (``probs @ score_vec``), so unlike
# ``loop_path_<base>`` it needs no ``keep_mask``.  ``G``'s density (``graph_density``, fraction of
# the ``C(V, 2)`` possible edges) and degree cap set how large those components can grow, so equal
# ``(m, graph_density, graph_seed)`` give the identical edge set.

_CONNECTED_RE = re.compile(r"^connected_(.+)$")

#: Base scorers usable under a ``connected_<base>`` observable.  ``maxcc`` returns the size of the
#: largest connected component of the clicked vertices' induced subgraph of ``G`` (a global
#: connectivity property, 1 when the clicked set is independent); the rest are the plain
#: per-Fock-state scores (:func:`_parity_score` etc.).
CONNECTED_BASES = ("parity", "majority", "bunching", "n_first", "maxcc")


def parse_connected_observable(observable: str):
    """Split a ``connected_<base>`` string into ``(is_conn, base)``.

    Plain observables return ``(False, observable)``.  The ``connected_<base>`` family takes no
    ``__`` suffixes (the score is a single global property, not a selectable subset), so
    ``connected_maxcc`` scores the largest-connected-component size.
    """
    mo = _CONNECTED_RE.match(observable)
    if mo is None:
        return False, observable
    parts = mo.group(1).split("__")
    if len(parts) > 1:
        raise ValueError(f"connected_<base> takes no '__' suffix, got {observable!r}")
    return True, parts[0]


def is_connected_observable(observable: str) -> bool:
    """True for a well-formed ``connected_<base>`` observable (``base`` in :data:`CONNECTED_BASES`)."""
    is_conn, base = parse_connected_observable(observable)
    return is_conn and base in CONNECTED_BASES


def _clicked_max_component(key, edges) -> int:
    """Size of the largest connected component of the clicked vertices' induced subgraph of ``G``.

    A mode is *clicked* iff its photon count is > 0; vertex ``i`` is then present.  Only edges of
    ``G`` with both endpoints clicked join the induced subgraph, whose components are found by
    flood fill; the return is the vertex count of the biggest one (1 when the clicked set is
    independent, 0 only if nothing is clicked -- impossible for k >= 1).
    """
    clicked = {i for i, c in enumerate(key) if int(c) > 0}
    if not clicked:
        return 0
    adj: dict = {v: [] for v in clicked}
    for u, w in edges:
        if u in clicked and w in clicked:
            adj[u].append(w)
            adj[w].append(u)
    seen: set = set()
    best = 0
    for start in clicked:
        if start in seen:
            continue
        size = 0
        stack = [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            size += 1
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        best = max(best, size)
    return best


def _connected_tables(keys, edges):
    """Per-Fock-state largest-connected-component-size array ``maxcc`` over the basis ``keys``."""
    maxcc = np.zeros(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        maxcc[i] = _clicked_max_component(key, edges)
    return maxcc


def _connected_base_scores(keys, base: str, maxcc, *, m: int, k: int):
    """Per-Fock-state ``<base>`` score vector for a ``connected_<base>`` observable."""
    if base == "maxcc":
        return maxcc.astype(np.float64)
    if base == "parity":
        pm = tuple(range((m + 1) // 2))
        return np.array([_parity_score(key, pm) for key in keys], dtype=np.float64)
    if base == "majority":
        return np.array([_majority_score(key, m, k) for key in keys], dtype=np.float64)
    if base == "bunching":
        return np.array([_bunching_score(key) for key in keys], dtype=np.float64)
    if base == "n_first":
        return np.array([_first_mode_score(key) for key in keys], dtype=np.float64)
    raise ValueError(f"unknown connected base {base!r}; choose from {CONNECTED_BASES}")


def _connected_scores(keys, *, m, k, base, edges):
    """Per-Fock-state ``<base>`` score vector for a ``connected_<base>`` observable (no selection).

    Aligns to the fixed Fock basis, so scoring a distribution is the plain expectation
    ``probs @ score_vec`` -- no ``keep_mask`` (the score is a global property of every outcome).
    """
    maxcc = _connected_tables(keys, edges)
    return _connected_base_scores(keys, base, maxcc, m=m, k=k)


class PhotonicFeatureMap(nn.Module):
    """``|psi(x)> = W2 P(x) W1 |in>`` embedding (the W1->P(x)->W2 sandwich).

    ``amplitudes(X)`` -> ``(N, n_fock)`` complex Fock amplitudes (for the fidelity
    kernel); ``probs(X) = |amplitudes|^2`` and ``occ`` (per-Fock-state photon
    counts) feed the projected kernel's occupation moments.
    """

    def __init__(self, m: int, k: int, n_features: int, seed: int):
        super().__init__()
        self.m, self.k, self.seed = m, k, int(seed)
        import merlin as ML
        import perceval as pcvl

        circuit = build_sandwich_circuit(m, n_features, seed)
        self.input_state = _default_input_state(m, k)
        self.layer = ML.QuantumLayer(
            input_size=n_features,
            experiment=pcvl.Experiment(circuit),
            input_state=self.input_state,
            input_parameters=["x"],
            measurement_strategy=ML.MeasurementStrategy.amplitudes(ML.ComputationSpace.FOCK),
        )
        keys = list(self.layer.output_keys)
        occ = torch.tensor([[int(key[i]) for i in range(m)] for key in keys],
                           dtype=torch.float32)
        self.register_buffer("occ", occ)          # (n_fock, m) photon counts

    @torch.no_grad()
    def amplitudes(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer.forward(X)               # (N, n_fock) complex

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        a = self.amplitudes(X)
        return (a.conj() * a).real                 # (N, n_fock)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.amplitudes(X)


class PhotonicTeacher(Teacher):
    name = "photonic_quantum"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0,
                 n_vertices: int | None = None, graph_seed: int | None = None,
                 angle_seed: int | None = None, graph_density: float | None = None):
        super().__init__(n_features)
        self.is_graph = is_graph_observable(observable)
        self.is_prod = is_prod_family(observable)
        self.is_prod_angle = is_prod_parity_angle(observable)
        self.is_connected = is_connected_observable(observable)
        self.is_sq = is_sq_observable(observable)
        self.is_pairprod = is_pairprod_observable(observable)
        if (not self.is_graph and not self.is_prod and not self.is_prod_angle
                and not self.is_connected and not self.is_sq and not self.is_pairprod
                and observable not in OBSERVABLES):
            raise ValueError(f"observable must be one of {OBSERVABLES} "
                             f"(or loop_path_<base>, base in {GRAPH_BASES}; or connected_<base>, "
                             f"base in {CONNECTED_BASES}; or prod_parity"
                             f"[__<preset|M...|N...>...]; or {PROD_PARITY_CONSECUTIVE}"
                             f" / {PROD_PARITY_SECOND}; or an angle variant "
                             f"prod_parity_{{consecutive,second}}_{{pi,random}}; or a nonlinear "
                             f"sq_<base> (base in {SQ_BASES}) / pairprod), got {observable!r}")
        if observable == "majority" and m % 2:
            raise ValueError("observable 'majority' requires even m")
        self.m, self.k, self.observable, self.nsample = m, k, observable, int(nsample)
        self.seed = int(seed)
        self.n_vertices = None if n_vertices is None else int(n_vertices)
        self.graph_density = None if graph_density is None else float(graph_density)
        self.loop_vars = self.path_vars = None   # parsed from the observable string (graph obs)
        self.graph_seed = self.seed if graph_seed is None else int(graph_seed)
        self.angle_seed = self.seed if angle_seed is None else int(angle_seed)
        self._capture = False
        self._dist_probs: list = []       # per-forward (N, n_fock) prob matrices (capture)

        import merlin as ML
        import perceval as pcvl

        circuit = build_sandwich_circuit(m, n_features, seed)
        input_state = _default_input_state(m, k)
        self.input_state = input_state
        self.layer = ML.QuantumLayer(
            input_size=n_features,
            experiment=pcvl.Experiment(circuit),
            input_state=input_state,
            input_parameters=["x"],
            measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        )

        keys = list(self.layer.output_keys)
        self._fock_keys = keys
        if self.is_sq:
            base = parse_sq_observable(observable)[1]
            vec = _sq_base_vec(base, keys, m=m, k=k)
        elif self.is_pairprod:
            n_fock = len(keys)
            if n_fock > PAIRPROD_MAX_FOCK:
                raise ValueError(
                    f"pairprod builds a dense {n_fock}x{n_fock} kernel (n_fock={n_fock} > "
                    f"{PAIRPROD_MAX_FOCK}); lower (m, k) or use a sq_<base> observable")
            self.register_buffer("pair_kernel",
                                 torch.tensor(pairprod_kernel(keys, m), dtype=torch.float32))
            vec = [0.0] * n_fock                        # unused: pairprod scores via pair_kernel
        elif self.is_graph:
            if self.n_vertices is None:
                raise ValueError("loop_path_<base> observables require n_vertices")
            half = self.n_vertices // 2
            if not k <= half <= m:
                raise ValueError(f"loop_path_ needs k <= n_vertices//2 <= m "
                                 f"(k={k}, n_vertices//2={half}, m={m})")
            base, self.loop_vars, self.path_vars = resolve_graph_spec(observable, None, None)
            self.edges, self.m0_mask = build_matching_graph(m, self.n_vertices, self.graph_seed)
            keep, vec = _graph_selection(
                keys, m=m, k=k, base=base, edges=self.edges, m0_mask=self.m0_mask,
                n_vertices=self.n_vertices, loop_vars=self.loop_vars, path_vars=self.path_vars)
            self.register_buffer("keep_mask", torch.tensor(keep, dtype=torch.float32))
        elif self.is_connected:
            if self.graph_density is None:
                raise ValueError("connected_<base> observables require graph_density")
            _, base = parse_connected_observable(observable)
            self.edges = build_vertex_graph(m, self.graph_density, self.graph_seed)
            vec = _connected_scores(keys, m=m, k=k, base=base, edges=self.edges)
        elif self.is_prod:
            self.monomials = prod_family_monomials(observable, m, k)
            vec = [_prod_parity_score(key, self.monomials) for key in keys]
        elif self.is_prod_angle:
            self.angle_monos = angle_monomials(observable, m, k, self.angle_seed)
            vec = [_prod_parity_angle_score(key, self.angle_monos) for key in keys]
        elif observable == "parity":
            pm = tuple(range((m + 1) // 2))
            vec = [_parity_score(key, pm) for key in keys]
        elif observable == "majority":
            vec = [_majority_score(key, m, k) for key in keys]
        elif observable == "bunching":
            vec = [_bunching_score(key) for key in keys]
        elif observable == "n_first":
            vec = [_first_mode_score(key) for key in keys]   # soft = E[n_0]
        elif observable == "max_prob":
            vec = [0.0] * len(keys)
        else:  # single_output
            vec = [_single_output_score(key, input_state) for key in keys]
        self.register_buffer("score_vec", torch.tensor(vec, dtype=torch.float32))
        # Row-batch the forward so peak memory scales with the chunk, not the whole pool: the
        # per-forward (N, n_fock) prob matrix (n_fock = len(keys) = C(m+k-1, k)) blows up for
        # large (m, k) -- e.g. m=14,k=7 -> n_fock=77520, ~3 GB at N=1e4 in fp32, and merlin's
        # complex amplitudes push peak higher -> the sampler gets OOM-killed.  Auto-size the chunk
        # to ~32M fp32 elements (~128 MB) per call; override with ``teacher.forward_batch`` (a
        # larger value if you have RAM, or <= 0 to disable batching).
        self.forward_batch = max(1, 33_554_432 // max(len(keys), 1))

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bs = self.forward_batch
        if bs is None or bs <= 0 or X.shape[0] <= bs:
            return self._forward_chunk(X)
        # Process row-chunks and concatenate; peak memory stays at (bs, n_fock) instead of
        # (N, n_fock).  Row order is preserved, so the result is identical to an unbatched call.
        scores = [self._forward_chunk(X[i:i + bs]) for i in range(0, X.shape[0], bs)]
        return torch.cat(scores, dim=0)

    @torch.no_grad()
    def _forward_chunk(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.layer.forward(X, shots=self.nsample if self.nsample > 0 else None)
        if self._capture:
            self._dist_probs.append(probs.detach().cpu().numpy())
        if self.is_graph:
            # E[base | pre-selection]: renormalised masked mean (loop/path selection).
            score = _conditional_expectation(probs, self.keep_mask, self.score_vec, self.observable)
        elif self.is_sq:
            score = (probs * probs) @ self.score_vec           # Σ_n <base>(n) p(n)^2
        elif self.is_pairprod:
            score = (probs @ self.pair_kernel * probs).sum(dim=1)  # p^T K p (K symmetric ±1)
        else:
            # connected_<base> falls here too: plain E[base] over all outcomes (no keep_mask).
            score = probs @ self.score_vec
            if self.observable == "max_prob":
                score = probs.max(dim=1).values.clamp(min=1e-10)
            elif self.observable == "single_output":
                score = score / probs.max(dim=1).values.clamp(min=1e-10)
        return score.unsqueeze(-1)  # (N_chunk, 1)

    # --- optional full-distribution capture (parity with spoqc_magic) ---------- #

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every forward's full Fock distribution so it can be persisted.

        Off by default; the generator turns it on when ``generation.save_dist`` is set,
        letting the saved ``distributions.npz`` be re-scored offline under a different
        observable / ``loop_vars`` / ``path_vars`` (:func:`score_from_distribution`)
        without re-running the boson sampler.
        """
        self._capture = bool(enable)
        self._dist_probs = []

    def captured_distributions(self) -> dict:
        """Recorded distributions as a dict (same shape as :func:`spoqc_magic.load_distributions`)."""
        if not self._dist_probs:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        keys = np.array([[int(key[i]) for i in range(self.m)] for key in self._fock_keys],
                        dtype=np.int16)
        probs = np.vstack(self._dist_probs)
        return {"keys": keys, "probs": probs, "readout_modes": (),
                "m": self.m, "k": self.k, "observable": self.observable,
                "t_var": None, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        from .spoqc_magic import write_distributions
        return write_distributions(path, self.captured_distributions())

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "PhotonicTeacher":
        p = cfg.problem
        return cls(m=p.m, k=p.k, n_features=cfg.resolved_n_features,
                   observable=p.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample, n_vertices=p.n_vertices,
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed,
                   graph_density=p.graph_density)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        spec = {"observable": cfg.problem.observable, "nsample": cfg.generation.nsample}
        if is_prod_parity_angle(cfg.problem.observable):
            # Canonicalise to the resolved (angle, monomial) list: the actual phase polynomial IS
            # the identity, so _pi vs _random, a different angle_seed, or a different (m, k) each
            # give a distinct dataset (equal specs -> identical, hashable list).
            p = cfg.problem
            aseed = cfg.seeds.teacher_seed if p.angle_seed is None else int(p.angle_seed)
            spec.update(observable="prod_parity_angle",
                        angle_monomials=[[round(float(th), 12), list(mono)]
                                         for th, mono in angle_monomials(p.observable, p.m, p.k, aseed)])
        elif is_prod_family(cfg.problem.observable):
            # Canonicalise to the monomial set: equivalent spellings (segment order,
            # duplicates, preset/explicit mixes -- and prod_parity_consecutive vs the explicit
            # prod_parity that expands to the same monomials) map to one dataset.
            p = cfg.problem
            spec.update(observable="prod_parity",
                        monomials=[list(mono) for mono in
                                   prod_family_monomials(p.observable, p.m, p.k)])
        elif is_graph_observable(cfg.problem.observable):
            p = cfg.problem
            base, eff_loop, eff_path = resolve_graph_spec(p.observable, None, None)
            # Canonicalise: the selection is folded into loop_vars/path_vars below, so
            # ``__L0-1__P2`` and ``__P2__L0-1`` (same selection, different spelling) map to
            # one dataset (identical teacher output -> identical hash).
            spec.update(
                observable=f"loop_path_{base}",
                n_vertices=p.n_vertices,
                loop_vars=None if eff_loop is None else sorted(int(v) for v in eff_loop),
                path_vars=None if eff_path is None else sorted(int(v) for v in eff_path),
                graph_seed=cfg.seeds.teacher_seed if p.graph_seed is None else int(p.graph_seed),
            )
        elif is_connected_observable(cfg.problem.observable):
            p = cfg.problem
            _, base = parse_connected_observable(p.observable)
            spec.update(
                observable=f"connected_{base}",
                graph_density=p.graph_density,
                graph_seed=cfg.seeds.teacher_seed if p.graph_seed is None else int(p.graph_seed),
            )
        return spec


def score_from_distribution(dist, observable: str | None = None, *,
                            n_vertices: int | None = None, loop_vars=None,
                            path_vars=None, graph_seed: int | None = None,
                            angle_seed: int | None = None,
                            graph_density: float | None = None):
    """Re-score a saved photonic distribution (dict from :func:`spoqc_magic.load_distributions`).

    ``observable`` defaults to the stored one.  For a ``loop_path_<base>`` observable the
    graph knobs (``n_vertices``, ``loop_vars``, ``path_vars``, ``graph_seed``) must be
    supplied -- they are not persisted in the ``.npz`` -- and ``graph_seed`` defaults to the
    stored teacher ``seed``.  ``__L``/``__P`` var suffixes encoded in ``observable`` override
    the passed ``loop_vars`` / ``path_vars``, so a sweep can vary the selection purely
    through the observable string.  For a ``connected_<base>`` observable supply ``graph_density``
    (+ ``graph_seed``) instead; it scores E[<base>] over all outcomes (``connected_maxcc`` = the
    largest-connected-component size), no selection.  For an angle prod_parity variant (``*_random``) the
    ``angle_seed`` fixes the drawn angles and likewise defaults to the stored teacher ``seed``
    (``_pi`` is deterministic).  Returns ``(n_rows,)`` scores.
    """
    obs = dist["observable"] if observable is None else observable
    m, k = int(dist["m"]), int(dist["k"])
    keys = [tuple(int(v) for v in row) for row in dist["keys"]]
    probs = torch.as_tensor(np.atleast_2d(np.asarray(dist["probs"])), dtype=torch.float32)

    if is_graph_observable(obs):
        if n_vertices is None:
            raise ValueError("re-scoring a loop_path_<base> observable needs n_vertices "
                             "(+ loop_vars / path_vars / graph_seed)")
        gseed = int(dist["seed"]) if graph_seed is None else int(graph_seed)
        base, eff_loop, eff_path = resolve_graph_spec(obs, loop_vars, path_vars)
        edges, m0_mask = build_matching_graph(m, int(n_vertices), gseed)
        keep, vec = _graph_selection(keys, m=m, k=k, base=base, edges=edges, m0_mask=m0_mask,
                                     n_vertices=int(n_vertices), loop_vars=eff_loop,
                                     path_vars=eff_path)
        print(sum(keep),len(keep))
        # keep = [1 if k < sum(keep) else 0 for k in range(len(keep))]
        score = _conditional_expectation(probs, torch.tensor(keep, dtype=torch.float32),
                                         torch.tensor(vec, dtype=torch.float32), obs)
        return score.numpy()

    if is_connected_observable(obs):
        if graph_density is None:
            raise ValueError("re-scoring a connected_<base> observable needs graph_density "
                             "(+ graph_seed)")
        gseed = int(dist["seed"]) if graph_seed is None else int(graph_seed)
        _, base = parse_connected_observable(obs)
        edges = build_vertex_graph(m, float(graph_density), gseed)
        vec = _connected_scores(keys, m=m, k=k, base=base, edges=edges)
        return (probs @ torch.tensor(vec, dtype=torch.float32)).numpy()

    if is_prod_parity_angle(obs):
        # Fully offline: like the prod family, needs only counts + probs + k (all persisted).
        # angle_seed fixes the _random draw and defaults to the stored teacher seed.
        aseed = int(dist["seed"]) if angle_seed is None else int(angle_seed)
        am = angle_monomials(obs, m, k, aseed)
        vec = [_prod_parity_angle_score(key, am) for key in keys]
        return (probs @ torch.tensor(vec, dtype=torch.float32)).numpy()

    if is_prod_family(obs):
        # Fully offline: the prod family needs only the per-mode counts (keys) + probs, both
        # persisted in the .npz (incl. k, which prod_parity_consecutive needs), so a save_dist
        # dump re-scores with no extra knobs.
        monomials = prod_family_monomials(obs, m, k)
        vec = [_prod_parity_score(key, monomials) for key in keys]
        return (probs @ torch.tensor(vec, dtype=torch.float32)).numpy()

    if is_sq_observable(obs):
        # Fully offline: diagonal degree-2 form, needs only counts + probs (+ k for majority).
        _, base = parse_sq_observable(obs)
        vec = torch.tensor(_sq_base_vec(base, keys, m=m, k=k), dtype=torch.float32)
        return ((probs * probs) @ vec).numpy()

    if is_pairprod_observable(obs):
        # Fully offline: quadratic form p^T K p over the persisted counts + probs.
        n_fock = len(keys)
        if n_fock > PAIRPROD_MAX_FOCK:
            raise ValueError(
                f"pairprod builds a dense {n_fock}x{n_fock} kernel (n_fock={n_fock} > "
                f"{PAIRPROD_MAX_FOCK}); lower (m, k) or use a sq_<base> observable")
        K = torch.tensor(pairprod_kernel(keys, m), dtype=torch.float32)
        return (probs @ K * probs).sum(dim=1).numpy()

    if obs not in OBSERVABLES:
        raise ValueError(f"observable must be one of {OBSERVABLES} "
                         f"(or loop_path_<base>), got {obs!r}")
    if obs == "parity":
        vec = [_parity_score(key, tuple(range((m + 1) // 2))) for key in keys]
    elif obs == "majority":
        vec = [_majority_score(key, m, k) for key in keys]
    elif obs == "bunching":
        vec = [_bunching_score(key) for key in keys]
    elif obs == "n_first":
        vec = [_first_mode_score(key) for key in keys]
    elif obs == "max_prob":
        return probs.max(dim=1).values.clamp(min=1e-10).numpy()
    else:  # single_output has no persisted input_state -> unsupported offline
        raise ValueError(f"observable {obs!r} cannot be re-scored offline")
    return (probs @ torch.tensor(vec, dtype=torch.float32)).numpy()
