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
    print(m0_edges, key)
    print(edges)
    print(n_loops, n_paths)
    
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
                 angle_seed: int | None = None):
        super().__init__(n_features)
        self.is_graph = is_graph_observable(observable)
        self.is_prod = is_prod_family(observable)
        self.is_prod_angle = is_prod_parity_angle(observable)
        if (not self.is_graph and not self.is_prod and not self.is_prod_angle
                and observable not in OBSERVABLES):
            raise ValueError(f"observable must be one of {OBSERVABLES} "
                             f"(or loop_path_<base>, base in {GRAPH_BASES}; or prod_parity"
                             f"[__<preset|M...|N...>...]; or {PROD_PARITY_CONSECUTIVE}"
                             f" / {PROD_PARITY_SECOND}; or an angle variant "
                             f"prod_parity_{{consecutive,second}}_{{pi,random}}), got {observable!r}")
        if observable == "majority" and m % 2:
            raise ValueError("observable 'majority' requires even m")
        self.m, self.k, self.observable, self.nsample = m, k, observable, int(nsample)
        self.seed = int(seed)
        self.n_vertices = None if n_vertices is None else int(n_vertices)
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
        if self.is_graph:
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

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.layer.forward(X, shots=self.nsample if self.nsample > 0 else None)
        if self._capture:
            self._dist_probs.append(probs.detach().cpu().numpy())
        if self.is_graph:
            # E[base | matching & loop/path pre-selection]: renormalised masked mean.
            score = _conditional_expectation(probs, self.keep_mask, self.score_vec, self.observable)
        else:
            score = probs @ self.score_vec
            if self.observable == "max_prob":
                score = probs.max(dim=1).values.clamp(min=1e-10)
            elif self.observable == "single_output":
                score = score / probs.max(dim=1).values.clamp(min=1e-10)
        return score.unsqueeze(-1)  # (N, 1)

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
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed)

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
        return spec


def score_from_distribution(dist, observable: str | None = None, *,
                            n_vertices: int | None = None, loop_vars=None,
                            path_vars=None, graph_seed: int | None = None,
                            angle_seed: int | None = None):
    """Re-score a saved photonic distribution (dict from :func:`spoqc_magic.load_distributions`).

    ``observable`` defaults to the stored one.  For a ``loop_path_<base>`` observable the
    graph knobs (``n_vertices``, ``loop_vars``, ``path_vars``, ``graph_seed``) must be
    supplied -- they are not persisted in the ``.npz`` -- and ``graph_seed`` defaults to the
    stored teacher ``seed``.  ``__L``/``__P`` var suffixes encoded in ``observable`` override
    the passed ``loop_vars`` / ``path_vars``, so a sweep can vary the selection purely
    through the observable string.  For an angle prod_parity variant (``*_random``) the
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
        score = _conditional_expectation(probs, torch.tensor(keep, dtype=torch.float32),
                                         torch.tensor(vec, dtype=torch.float32), obs)
        return score.numpy()

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
