"""Graph-based score vectors: read an outcome as a vertex set, score a global property.

``connected_<base>``: each mode maps to a *vertex* of a fixed, seeded, connected, bounded-degree
graph ``G`` on ``V = m`` vertices.  For each outcome the *clicked* vertices (modes with a non-zero
photon count) induce a subgraph of ``G``, and ``maxcc`` is the vertex count of its largest
connected component -- a global connectivity property of the outcome, not a local count.

**This is an expectation value.**  There is no pre-selection: the score is the plain
``probs @ v`` with ``v(n) = maxcc(n)``, so a graph-based observable is an
:class:`~v2.observable.base.Expectation` like any other.  Only the *construction* of ``v`` is
graph-flavoured.  That is the point of the two-axis split -- the legacy package's ``loop_path``
family conflated a graph reading with a post-selection, and only the selection made it a different
shape.  The selection is dropped in ``v2`` (a post-selected mean is a score vector on a smaller
support), and what remains is this.

Carried from ``model/photonic_observables/{connected,graphs}.py``.
"""

from __future__ import annotations

import re

import numpy as np

from observable.base import (Expectation, Observable, ObservableContext, ObservableFamily, base_score_vec,
                    register)
from observable.scorers.counting import parity_modes, parity_score

#: Base scorers usable under ``connected_<base>``.  ``maxcc`` is the graph reading; ``parity_maxcc``
#: is the elementwise product of ``parity`` and ``maxcc`` (not a selection, not a sub-scoring of
#: parity restricted to the winning component -- both factors are computed the normal whole-outcome
#: way and simply multiplied); the rest are the plain counting scorers, scored (unselected) over the
#: full distribution.
CONNECTED_BASES = ("parity", "majority", "bunching", "n_first", "maxcc", "parity_maxcc")

#: Hard cap on any vertex's degree in ``G``.  An edge whose addition would push either endpoint
#: past this is skipped, so ``G`` stays bounded-degree (expander-like) rather than growing hubs as
#: density rises; it also caps the edge count at ``V * MAX_VERTEX_DEGREE // 2``.  ``-1`` disables.
MAX_VERTEX_DEGREE = 4

_CONNECTED_RE = re.compile(r"^connected_(.+)$")


def is_connected_graph(edges, n_vertices: int) -> bool:
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


def build_vertex_graph(m: int, density: float, seed: int):
    """A fixed, seeded, *connected*, bounded-degree graph ``G`` on ``V = m`` vertices.

    ``mode i <-> vertex i``.  ``G`` is drawn connected (a random spanning path first, so ``maxcc``
    can in principle reach ``V`` when every mode clicks), then filled with seeded edges up to
    ``round(density * C(V, 2))`` -- floored at ``V - 1`` (a connected graph needs a spanning tree)
    and bounded by :data:`MAX_VERTEX_DEGREE`.  Deterministic in ``seed``, hence reproducible and
    hashable.

    Returns a sorted list of distinct ``(u, v)`` vertex pairs.
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
    degree_cap_edges = V * MAX_VERTEX_DEGREE // 2 if capped else max_edges
    n_edges = max(int(round(d * max_edges)), V - 1)
    n_edges = min(n_edges, max_edges, degree_cap_edges)

    rng = np.random.default_rng(int(seed))
    degree = [0] * V
    edges: list[tuple[int, int]] = []
    used: set = set()
    perm = rng.permutation(V)                              # spanning path: connected backbone
    for a, b in zip(perm[:-1], perm[1:]):
        e = (int(a), int(b)) if a < b else (int(b), int(a))
        edges.append(e)
        used.add(e)
        degree[e[0]] += 1
        degree[e[1]] += 1
    all_edges = [(i, j) for i in range(V) for j in range(i + 1, V)]
    for o in rng.permutation(len(all_edges)):              # seeded scan order
        if len(edges) >= n_edges:
            break
        u, w = all_edges[int(o)]
        if (u, w) in used:
            continue
        if capped and (degree[u] >= MAX_VERTEX_DEGREE or degree[w] >= MAX_VERTEX_DEGREE):
            continue                                       # keep G bounded-degree
        edges.append((u, w))
        used.add((u, w))
        degree[u] += 1
        degree[w] += 1
    return sorted(edges)


def clicked_max_component(key, edges) -> int:
    """Largest connected component of the clicked vertices' induced subgraph of ``G``.

    A mode is *clicked* iff its photon count is > 0.  Only edges with both endpoints clicked join
    the induced subgraph, whose components are found by flood fill; the return is the vertex count
    of the biggest (1 when the clicked set is independent, 0 only if nothing clicks).
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
        size, stack = 0, [start]
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


def parity_maxcc_score(key, edges, m: int) -> float:
    """``parity(key) * maxcc(key)`` -- the plain elementwise product of the two existing scorers,
    each computed the normal whole-outcome way (:func:`~observable.scorers.counting.parity_score`
    over the readout half, :func:`clicked_max_component` over the full clicked set); no
    sub-selection of one by the other.  Couples an additive, graph-blind signal (``parity`` is
    linear in occupation, ``+-1``) with a nonlinear, connectivity-mediated one (``maxcc``, which is
    blind to anything outside the winning component) -- genuinely different information from either
    factor alone, without introducing a new selection/support-restriction mechanism.
    """
    return float(parity_score(key, parity_modes(m)) * clicked_max_component(key, edges))


def parse_connected_observable(observable: str):
    """Split a ``connected_<base>`` string into ``(is_conn, base)``."""
    mo = _CONNECTED_RE.match(observable)
    if mo is None:
        return False, observable
    parts = mo.group(1).split("__")
    if len(parts) > 1:
        raise ValueError(f"connected_<base> takes no '__' suffix, got {observable!r}")
    return True, parts[0]


def is_connected_observable(observable: str) -> bool:
    """True for a well-formed ``connected_<base>`` (``base`` in :data:`CONNECTED_BASES`)."""
    is_conn, base = parse_connected_observable(observable)
    return is_conn and base in CONNECTED_BASES


def layered_vertex_index(m: int, k: int, i: int, p: int) -> int:
    """Vertex id of ``g_{i,p}`` ("mode ``i`` has exactly ``p`` photons") in the ``m*(k+1)``-vertex
    layered graph -- mode ``i``'s own ``k+1`` occupation levels (``p = 0..k``) laid out
    contiguously, mode-major: ``vertex = i*(k+1) + p``.
    """
    if not 0 <= p <= k:
        raise ValueError(f"level p={p} outside [0, k={k}]")
    if not 0 <= i < m:
        raise ValueError(f"mode i={i} outside [0, m={m})")
    return i * (k + 1) + p


def _layered_induced_subgraph(key, edges, *, m: int, k: int):
    """``(active_set, adj, n_active_edges)`` for one outcome on the layered graph -- the shared
    setup behind :func:`layered_clicked_max_component`, :func:`layered_product_component`, and
    :func:`layered_num_loops`.  ``clicked = {g_{i,n_i} : i in 0..m-1}``: exactly one active vertex
    per mode (occupied or not -- an empty mode still activates its own ``g_{i,0}``), landing at
    whichever of that mode's ``k+1`` level-vertices matches its actual occupation, which is the
    mechanism that fixes :func:`clicked_max_component`'s bunching-blindness (two outcomes sharing
    an occupied-mode set but differing in bunching depth activate different vertices here; see
    ``GRAPH_OBSERVABLE_PROPOSALS.md`` for the full derivation). ``n_active_edges`` counts each edge
    once, only over pairs both in ``active_set``.
    """
    active_set = {layered_vertex_index(m, k, i, int(n_i)) for i, n_i in enumerate(key)}
    adj: dict = {v: [] for v in active_set}
    n_active_edges = 0
    for u, w in edges:
        if u in active_set and w in active_set:
            adj[u].append(w)
            adj[w].append(u)
            n_active_edges += 1
    return active_set, adj, n_active_edges


def _components(active_set, adj) -> list[int]:
    """Sizes of every connected component of ``adj`` restricted to ``active_set`` (flood fill)."""
    return [len(members) for members in _component_members(active_set, adj)]


def _component_members(active_set, adj) -> list[set]:
    """Vertex-member sets of every connected component of ``adj`` restricted to ``active_set``
    (flood fill) -- the member-set generalisation of :func:`_components`, needed whenever a reading
    wants to look *inside* a component (e.g. the occupation numbers of the modes it contains) rather
    than just its size.
    """
    seen: set = set()
    components = []
    for start in active_set:
        if start in seen:
            continue
        members, stack = set(), [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            members.add(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        components.append(members)
    return components


def layered_clicked_max_component(key, edges, *, m: int, k: int) -> int:
    """``maxcc`` on the ``m*(k+1)``-vertex layered graph -- the size of the largest connected
    component among the ``m`` active vertices.  See :func:`_layered_induced_subgraph` for the
    activation rule.  Always exactly ``m`` active vertices, so the ceiling here is ``m``, not
    ``m*(k+1)``: the extra vertices only ever widen *which* ``m``-subset of the graph gets read.

    **Ceiling is size-independent, hence not "harder" for larger sizes.** ``maxcc``'s range is
    always ``[1, m]`` regardless of ``k`` (the layer count only changes which subset activates, not
    how many vertices can be simultaneously active) -- so this statistic alone does not gain
    dynamic range as the photon count grows, only as the mode count does.  Use
    :func:`layered_product_component` or :func:`layered_num_loops` (both size-*and*-bunching
    sensitive) if the goal is a statistic that keeps growing with ``k`` at fixed ``m``.
    """
    active_set, adj, _ = _layered_induced_subgraph(key, edges, m=m, k=k)
    sizes = _components(active_set, adj)
    return max(sizes) if sizes else 0


def layered_product_component(key, edges, *, m: int, k: int) -> float:
    """``productcc`` on the layered graph: the product of every connected component's size among
    the ``m`` active vertices (:func:`_layered_induced_subgraph`), rather than only the largest.

    Unlike :func:`layered_clicked_max_component`, this is sensitive to the *whole* component-size
    distribution, not just its max: it is largest when the active vertices split into a few,
    roughly-balanced components (by AM-GM, a product of parts summing to a fixed total is maximised
    when the parts are as equal as possible), and grows with how many of the ``m`` active vertices
    land in non-trivial (size > 1) components -- a singleton component contributes a factor of 1,
    diluting the product relative to an equal-size split, but does not zero it out. Ranges up to
    (but is generically far below) ``m^m / m^m``-scale combinatorial ceilings in principle; in
    practice bounded well below that by ``build_vertex_graph``'s ``MAX_VERTEX_DEGREE`` cap limiting
    how large any one component can plausibly get.
    """
    active_set, adj, _ = _layered_induced_subgraph(key, edges, m=m, k=k)
    sizes = _components(active_set, adj)
    prod = 1
    for s in sizes:
        prod *= s
    return float(prod)


def layered_num_loops(key, edges, *, m: int, k: int) -> int:
    """``numloops`` on the layered graph: the **cycle rank** (first Betti number) of the induced
    subgraph on the ``m`` active vertices (:func:`_layered_induced_subgraph`),
    ``|E_active| - |V_active| + (#components)``.

    Counts independent loops -- redundant connectivity (multiple distinct paths between the same
    pair of active vertices) -- rather than mere reachability. A tree-like (or forest-like) induced
    subgraph scores 0 regardless of its size or shape; the count only grows when the active
    vertices are dense enough among themselves to close cycles. This is a genuinely different
    sensitivity from both :func:`layered_clicked_max_component` (reachability only) and
    :func:`layered_product_component` (component-size distribution only) -- it is sensitive to
    *edge density among the active set*, which for a fixed, seeded background graph depends on
    exactly *which* ``m`` vertices activated, i.e. on the full bunching pattern, not just on how
    many components they happen to fall into.
    """
    active_set, adj, n_active_edges = _layered_induced_subgraph(key, edges, m=m, k=k)
    n_components = len(_components(active_set, adj))
    return n_active_edges - len(active_set) + n_components


def _maxcc_members(key, edges, *, m: int, k: int) -> set:
    """Vertex-member set of the (a, on ties) largest connected component among the ``m`` active
    vertices on the layered graph -- the shared lookup behind
    :func:`layered_maxcc_occupation_product` and :func:`layered_maxcc_occupation_sum`. Ties broken
    arbitrarily by iteration order (matches :func:`layered_clicked_max_component`'s ``max`` tie
    handling, which has the same ambiguity on sizes alone).
    """
    active_set, adj, _ = _layered_induced_subgraph(key, edges, m=m, k=k)
    components = _component_members(active_set, adj)
    return max(components, key=len) if components else set()


def layered_maxcc_occupation_product(key, edges, *, m: int, k: int) -> float:
    """``maxcc_occprod`` on the layered graph: the product of occupation numbers ``n_i`` over
    exactly the modes whose vertex ``g_{i,n_i}`` lies in the *largest* connected component
    (:func:`_maxcc_members`), rather than that component's plain vertex count.

    Where :func:`layered_clicked_max_component` answers "how many modes are in the best-connected
    region" and :func:`layered_product_component` answers "how balanced is the split across *all*
    components," this answers "how much bunching is concentrated in that one best-connected region"
    -- connectivity picks *which* modes count, occupation numbers say *how much*. On this layered
    graph every active vertex's level *is* that mode's occupation, so an empty mode (``n_i=0``)
    landing in the winning component zeroes the product outright -- a deliberate all-or-nothing
    sensitivity, unlike the additive version (:func:`layered_maxcc_occupation_sum`), which only
    dilutes in that case.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    if not members:
        return 0.0
    prod = 1
    for v in members:
        i, p = divmod(v, k + 1)
        prod *= p
    return float(prod)


def layered_maxcc_occupation_sum(key, edges, *, m: int, k: int) -> float:
    """``maxcc_occsum`` on the layered graph: the sum of occupation numbers ``n_i`` over exactly the
    modes whose vertex ``g_{i,n_i}`` lies in the *largest* connected component
    (:func:`_maxcc_members`) -- the additive sibling of
    :func:`layered_maxcc_occupation_product`, without that reading's all-or-nothing sensitivity to
    an empty mode landing in the winning component (a zero there only drops one term from the sum,
    it does not zero the whole score). Ranges ``[0, k]`` -- the largest component can in principle
    contain every photon, but never more since occupations across distinct modes sum to ``k``
    exactly.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    total = 0
    for v in members:
        i, p = divmod(v, k + 1)
        total += p
    return float(total)


def layered_maxcc_occupation_indexed_sum(key, edges, *, m: int, k: int) -> float:
    """``maxcc_occisum`` on the layered graph: ``sum_{i in maxcc} (i+1)*(n_i+1)`` over exactly the
    modes whose vertex ``g_{i,n_i}`` lies in the *largest* connected component
    (:func:`_maxcc_members`) -- a mode-index-weighted refinement of
    :func:`layered_maxcc_occupation_sum`.

    ``occsum`` collapses to plain occupation totals, so two components with the same total
    occupation but different *membership* (e.g. modes ``{0,1}`` at occupations ``(2,1)`` vs. modes
    ``{2,3}`` at occupations ``(1,2)``) tie -- measured directly: at ``(m=10,k=5)``, ``occsum`` has
    only 6 distinct values over a 200-outcome sample and every single occsum-tied group collapses
    further under this reading (34 distinct values, same sample). Multiplying occupation by
    ``(mode_index+1)`` breaks that degeneracy because it is injective in *which* modes contributed,
    not just how much they held -- so no two distinct member/occupation combinations that summed
    identically under ``occsum`` continue to coincide here, *unless* they happen to collide under
    this specific linear index-weighting too (unlikely but not structurally excluded, same caveat
    ``occsum`` itself has relative to a fully injective encoding). The ``+1`` offsets on both index
    and occupation are deliberate: an index-0 member or an occupation-0 member (``occsum`` credits an
    unoccupied member with 0; this reading still wants it to contribute a nonzero index-based term)
    must not silently drop out the way :func:`layered_maxcc_occupation_product`'s zero-occupation
    case does -- this reading stays purely additive, so it inherits none of that product's
    all-or-nothing sensitivity.

    Deliberately still bounded well short of a fully injective encoding of the outcome (which would
    need >= ``C(m+k-1,k)`` distinct values and would stop measuring anything about graph structure --
    see the discussion this reading followed from): range grows only linearly in ``m`` and ``k``
    (``max <= m*(k+1)``), not combinatorially in the Fock-basis size.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    total = 0
    for v in members:
        i, p = divmod(v, k + 1)
        total += (i + 1) * (p + 1)
    return float(total)


def layered_parity_maxcc(key, edges, *, m: int, k: int) -> float:
    """``maxcc_paritymaxcc`` on the layered graph: ``parity(key) * maxcc(key)``, the plain
    elementwise product of :func:`~observable.scorers.counting.parity_score` (whole-outcome, over
    the readout half) and :func:`layered_clicked_max_component` (whole-outcome, on the layered
    graph) -- the layered-graph sibling of :func:`parity_maxcc_score`, same product, just reading
    ``maxcc`` off the ``m*(k+1)``-vertex layered graph instead of the plain ``m``-vertex one so it
    inherits the layered construction's bunching-depth sensitivity in the ``maxcc`` factor.
    """
    return float(parity_score(key, parity_modes(m))
                 * layered_clicked_max_component(key, edges, m=m, k=k))


def layered_maxcc_parity(key, edges, *, m: int, k: int) -> float:
    """``maxcc_ccparity`` on the layered graph: **parity computed only over the modes in the
    largest connected component** (:func:`_maxcc_members`), not the fixed readout half -- the
    genuinely different construction from :func:`layered_parity_maxcc`'s plain product. Where
    ``paritymaxcc`` multiplies two whole-outcome scorers together, this restricts parity's *domain*
    to a per-outcome, dynamically-selected mode set: the winning component's own membership decides
    which modes' occupations get summed before the even/odd check, so the same photon count parity
    reads differently depending on which modes the graph happened to connect for this outcome.

    ``+1`` if the summed occupation over the winning component's member modes is even, ``-1`` if
    odd -- same even/odd convention as :func:`~observable.scorers.counting.parity_score`, applied to
    a different, per-outcome mode set instead of the fixed first-``ceil(m/2)`` readout half. An empty
    winning-component set (only possible if ``key`` is the all-zero outcome, excluded by ``k>=1``)
    would sum to 0, i.e. ``+1`` -- not reachable in practice but noted for completeness.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    modes = {divmod(v, k + 1)[0] for v in members}
    total = sum(int(key[i]) for i in modes)
    return 1.0 if total % 2 == 0 else -1.0


def layered_maxcc_bunching(key, edges, *, m: int, k: int) -> float:
    """``maxcc_ccbunching`` on the layered graph: **bunching computed only over the modes in the
    largest connected component** (:func:`_maxcc_members`) -- the domain-restriction sibling of
    :func:`layered_maxcc_parity` (``ccparity``), applied to
    :func:`~observable.scorers.counting.bunching_score` instead of ``parity``.

    ``+1`` if every member mode of the winning component holds at most one photon, ``-1`` if any
    member mode is bunched -- same even/odd-style +-1 convention as the plain ``bunching``
    observable, but read only over the graph-selected mode set rather than every mode in the
    outcome. A mode with zero occupation trivially satisfies "at most one," so an empty member mode
    does not by itself trigger ``-1`` -- only a genuinely bunched (``n_i > 1``) member does.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    modes = {divmod(v, k + 1)[0] for v in members}
    if not modes:
        return 1.0
    return 1.0 if max(int(key[i]) for i in modes) <= 1 else -1.0


def layered_maxcc_majority(key, edges, *, m: int, k: int) -> float:
    """``maxcc_ccmajority`` on the layered graph: **the left/right occupation imbalance computed
    only over the modes in the largest connected component** (:func:`_maxcc_members`) -- the
    domain-restriction sibling of :func:`layered_maxcc_parity` (``ccparity``), applied to
    :func:`~observable.scorers.counting.majority_score` instead of ``parity``.

    ``(n_left - n_right) / k`` where ``n_left``/``n_right`` sum occupation only over member modes
    falling in the first/second half of the mode index range (``m // 2`` split, same convention as
    plain ``majority``) -- so unlike plain ``majority``, both the split point and which modes
    contribute are fixed, but a member mode outside neither half's membership (impossible here since
    every mode falls in exactly one half) never arises; the only per-outcome variation is *which*
    modes from each half are actually in the winning component. Normalised by the *total* photon
    count ``k``, not by the component's own occupation sum, so this stays comparable in scale to
    plain ``majority`` across outcomes with differently sized winning components -- a component that
    is small or occupation-light relative to ``k`` will read close to 0, not because there is no
    imbalance in the modes it contains, but because it is capturing only a fraction of ``k``'s total
    mass; this is a deliberate scale choice, documented rather than silently changed to a
    component-local normalisation.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    modes = {divmod(v, k + 1)[0] for v in members}
    split = m // 2
    n_left = sum(int(key[i]) for i in modes if i < split)
    n_right = sum(int(key[i]) for i in modes if i >= split)
    return (n_left - n_right) / k


def layered_maxcc_first(key, edges, *, m: int, k: int) -> float:
    """``maxcc_ccnfirst`` on the layered graph: **the occupation parity of the lowest-indexed mode
    in the largest connected component** (:func:`_maxcc_members`) -- the domain-restriction sibling
    of :func:`layered_maxcc_parity` (``ccparity``), applied to
    :func:`~observable.scorers.counting.first_mode_score` instead of ``parity``.

    Plain ``n_first`` always reads mode 0, a fixed choice with no ``x``-dependence in *which* mode
    gets read (only in that mode's occupation). Here the graph picks the mode instead: whichever
    member of the winning component has the smallest mode index plays the role mode 0 played, so the
    *identity* of the read-out mode is itself ``x``-dependent (it changes with whichever modes the
    graph happened to connect for this outcome), not just its occupation value -- the same
    "selection becomes part of the signal, not just a fixed lens" idea behind ``ccparity``, ported to
    the one existing counting scorer that names a specific mode rather than aggregating over a set.
    Returns ``min_mode_occupation mod 2``, same convention as plain ``n_first``.
    """
    members = _maxcc_members(key, edges, m=m, k=k)
    modes = {divmod(v, k + 1)[0] for v in members}
    i0 = min(modes)
    return float(int(key[i0]) % 2)


def connected_scores(keys, *, m, k, base, edges):
    """Per-outcome ``<base>`` score vector for ``connected_<base>`` (no selection)."""
    if base == "maxcc":
        return np.array([clicked_max_component(key, edges) for key in keys], dtype=np.float64)
    if base == "parity_maxcc":
        return np.array([parity_maxcc_score(key, edges, m) for key in keys], dtype=np.float64)
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=CONNECTED_BASES, label="connected")


class ConnectedFamily(ObservableFamily):
    describe = f"connected_<base> (base in {CONNECTED_BASES})"

    def matches(self, name: str) -> bool:
        return is_connected_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.graph_density is None:
            raise ValueError("connected_<base> observables require graph_density (+ graph_seed)")
        _, base = parse_connected_observable(name)
        edges = build_vertex_graph(ctx.m, ctx.graph_density, ctx.graph_seed)
        obs = Expectation(connected_scores(ctx.keys, m=ctx.m, k=ctx.k, base=base, edges=edges))
        obs.edges = edges                                  # kept for inspection / debugging
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        _, base = parse_connected_observable(name)
        return {"observable": f"connected_{base}",
                "graph_density": ctx.graph_density,
                "graph_seed": ctx.graph_seed,
                "max_vertex_degree": MAX_VERTEX_DEGREE}


#: ``connected_<reading>_layered`` readings -- all six read the same ``m*(k+1)``-vertex layered
#: graph and the same per-outcome active-vertex set (:func:`_layered_induced_subgraph`), differing
#: only in what they compute over the resulting induced subgraph.  ``maxcc``'s ceiling is ``m``
#: regardless of ``k`` (not "harder" at larger photon counts, see
#: :func:`layered_clicked_max_component`'s docstring); ``productcc``/``numloops`` are both
#: sensitive to the full bunching pattern, not just reachability, so they retain dynamic range as
#: ``k`` grows at fixed ``m`` -- PROVIDED the graph has enough edge density among any outcome's ``m``
#: active vertices to matter.  At :data:`MAX_VERTEX_DEGREE`'s default (4), the induced subgraph on
#: a small active-vertex subset is generically a tree, which makes ``numloops`` identically 0 at
#: every tested size and caps how large ``productcc``'s components can get -- confirmed by raising
#: (or disabling) the module-level cap, which restores real range to both.  This is a shared
#: constant used by every graph observable in this module (including plain ``connected_maxcc``), so
#: changing it is a deliberate, global decision, not something this family overrides for itself.
#: ``occprod``/``occsum``/``occisum`` are a different axis again from the first three: instead of a
#: property of the component-size *distribution* (``maxcc``/``productcc``/``numloops`` all reduce to
#: one number derived purely from how the ``m`` active vertices split into components, blind to
#: which occupation levels those vertices sit at beyond that split), they read the occupation
#: numbers *inside* the single largest component. ``occprod`` was found empirically degenerate at
#: low fill fractions (``k/m`` small): most components then contain at least one empty (``n_i=0``)
#: member, which zeroes the product outright, collapsing ~94% of a measured 100-sample
#: ``(m=10,k=5,density=0.75)`` draw to exactly 0 -- an all-or-nothing gate dressed up as a
#: multiplicative statistic, not real dynamic range; the resulting near-constant target is what
#: drove an MLP learner to diverge (R^2 in the -1e10 range) in an actual sweep. ``occsum`` fixes the
#: degeneracy (additive, no zero-gate) but still ties across components with the same total
#: occupation but different membership. ``occisum`` (:func:`layered_maxcc_occupation_indexed_sum`)
#: breaks those remaining ties by weighting occupation by mode index, while staying strictly bounded
#: (``<= m*(k+1)``, linear in system size) rather than approaching a combinatorial, hash-like
#: encoding of the full outcome -- measured to resolve every occsum-tied group in a sample sweep at
#: both a small size and ``(m=10,k=5)``. ``paritymaxcc`` (:func:`layered_parity_maxcc`) is a
#: different axis again from all of the above: a plain elementwise product of ``parity`` (additive,
#: graph-blind, whole-outcome) and ``maxcc`` (nonlinear, graph-mediated, blind to anything outside
#: the winning component) -- not a selection or sub-scoring of one by the other, just their product.
#: ``ccparity`` (:func:`layered_maxcc_parity`) is different again, and easy to conflate with
#: ``paritymaxcc`` -- it does NOT multiply the two; it restricts *parity's own domain* to the
#: winning component's member modes (a per-outcome, graph-selected mode set) instead of the fixed
#: readout half, so the even/odd check itself runs over a different, ``x``-dependent set of modes
#: per outcome rather than combining two independently-computed whole-outcome scores. Reseed-checked
#: (5 matched ``graph_seed=nn_seed`` runs, mlp, photonic Fock m=6..12): unlike ``paritymaxcc``
#: (which reseeded into noise indistinguishable from plain ``parity``), ``ccparity`` reproduced a
#: clean, monotonically increasing R^2-vs-``m`` trend across both seeds tested
#: (0.853/0.913/0.974/0.968 and 0.853/0.903/0.916/0.937) -- the one reading in this module with a
#: confirmed, reproducible size-dependent learnability signal. ``ccbunching``
#: (:func:`layered_maxcc_bunching`) and ``ccmajority`` (:func:`layered_maxcc_majority`) are the same
#: domain-restriction idea applied to ``bunching``/``majority`` instead of ``parity`` -- untested for
#: a size trend as of this writing. ``ccnfirst`` (:func:`layered_maxcc_first`) is a variant of the
#: same idea for ``n_first``, which names one fixed mode rather than aggregating over a set: instead
#: of restricting a domain, the graph picks *which* mode plays ``n_first``'s role (the winning
#: component's lowest-indexed member) each outcome, so the mode identity itself becomes
#: ``x``-dependent, not just its occupation.
LAYERED_READINGS = {
    "maxcc": layered_clicked_max_component,
    "productcc": layered_product_component,
    "numloops": layered_num_loops,
    "occprod": layered_maxcc_occupation_product,
    "occsum": layered_maxcc_occupation_sum,
    "occisum": layered_maxcc_occupation_indexed_sum,
    "paritymaxcc": layered_parity_maxcc,
    "ccparity": layered_maxcc_parity,
    "ccbunching": layered_maxcc_bunching,
    "ccmajority": layered_maxcc_majority,
    "ccnfirst": layered_maxcc_first,
}

_LAYERED_RE = re.compile(rf"^connected_({'|'.join(LAYERED_READINGS)})_layered$")


class ConnectedLayeredFamily(ObservableFamily):
    """``connected_<reading>_layered``, ``reading`` in :data:`LAYERED_READINGS` -- a graph-property
    reading on one full random graph over ``m*(k+1)`` vertices (``build_vertex_graph`` called at
    the larger size directly, no per-level structure imposed -- edges are free between any pair of
    ``(mode, level)`` vertices, including across different modes' different levels, rather than
    being restricted to same-level copies of a smaller graph). See
    :func:`_layered_induced_subgraph`'s docstring for why the activation rule fixes
    :func:`clicked_max_component`'s bunching-blindness, and ``GRAPH_OBSERVABLE_PROPOSALS.md`` for
    the design discussion.
    """

    describe = f"connected_<reading>_layered (reading in {tuple(LAYERED_READINGS)})"

    def matches(self, name: str) -> bool:
        return bool(_LAYERED_RE.match(name))

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.graph_density is None:
            raise ValueError(f"{name} requires graph_density (+ graph_seed)")
        reading = _LAYERED_RE.match(name).group(1)
        scorer = LAYERED_READINGS[reading]
        edges = build_vertex_graph(ctx.m * (ctx.k + 1), ctx.graph_density, ctx.graph_seed)
        v = [scorer(key, edges, m=ctx.m, k=ctx.k) for key in ctx.keys]
        obs = Expectation(np.array(v, dtype=np.float64))
        obs.edges = edges                                  # kept for inspection / debugging
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        reading = _LAYERED_RE.match(name).group(1)
        return {"observable": f"connected_{reading}_layered",
                "graph_density": ctx.graph_density,
                "graph_seed": ctx.graph_seed,
                "max_vertex_degree": MAX_VERTEX_DEGREE,
                "n_layers": ctx.k + 1}


def pair_vertex_index(m: int, i: int, j: int) -> int:
    """Vertex id of mode-pair ``(i, j)`` (``i <= j``) in the ``m*(m+1)/2``-vertex mode-pair graph --
    pairs laid out in colex order over ``i <= j``, matching
    ``[(a, b) for a in range(m) for b in range(a, m)]``'s own enumeration order (see
    :func:`_pair_vertex_table`) so vertex ids are stable without materialising that list.

    **Includes the diagonal ``i == j``** -- vertex ``(i, i)`` represents mode ``i``'s own *bunching*
    (two-or-more photons in the same mode), the mode-pair graph's analogue of a self-pair. Without
    it, an outcome like ``(2, 0, 0, ...)`` (all photons bunched into one mode) would activate *zero*
    mode-pair vertices under the off-diagonal-only construction (no two *distinct* modes are
    co-occupied), making the graph blind to single-mode bunching in exactly the way §1 of
    ``GRAPH_OBSERVABLE_PROPOSALS.md`` already flagged for the plain mode graph -- the diagonal
    restores that sensitivity here too, at the cost of ``m`` extra vertices (``C(m,2) -> C(m,2)+m =
    m*(m+1)/2``, still quadratic in ``m``).
    """
    if not 0 <= i <= j < m:
        raise ValueError(f"need 0 <= i <= j < m={m} (got i={i}, j={j})")
    #: number of pairs (a, b), a < i, plus offset of (i, j) within first-element-i pairs (b >= i).
    return i * m - i * (i - 1) // 2 + (j - i)


def _pair_vertex_table(m: int) -> list[tuple[int, int]]:
    """``[(i, j), ...]`` indexed by :func:`pair_vertex_index`'s own vertex id -- the inverse of that
    function, built once per ``m`` rather than re-derived per vertex (which an earlier version of
    :func:`pair_maxcc_parity` did via an ``O(m^2)`` scan per active vertex).
    """
    return [(i, j) for i in range(m) for j in range(i, m)]


def _pair_active_set(key, *, m: int) -> set:
    """Mode-pair vertices ``(i, j)`` (``i <= j``) that are "clicked": for ``i < j``, both modes
    co-occupied; for the diagonal ``i == j``, mode ``i`` itself is bunched (``n_i >= 2``, see
    :func:`pair_vertex_index`'s docstring for why the diagonal exists). Unlike
    :func:`_layered_induced_subgraph`'s one-vertex-per-mode activation, this reads pairwise
    co-occupation (plus single-mode bunching via the diagonal) rather than per-mode occupation
    levels, so the vertex set is a genuinely different kind of information about the outcome.
    """
    occupied = [i for i, n_i in enumerate(key) if int(n_i) > 0]
    active = {pair_vertex_index(m, i, i) for i in range(m) if int(key[i]) >= 2}
    for a in range(len(occupied)):
        for b in range(a + 1, len(occupied)):
            i, j = occupied[a], occupied[b]
            active.add(pair_vertex_index(m, i, j))
    return active


def _pair_induced_subgraph(key, edges, *, m: int):
    """``(active_set, adj)`` for one outcome on the mode-pair graph -- the pair-graph analogue of
    :func:`_layered_induced_subgraph`. ``active_set`` is every co-occupied mode pair
    (:func:`_pair_active_set`); size varies per outcome (``C(|occupied|, 2)``), unlike the layered
    graph's fixed ``m`` active vertices per outcome -- an outcome with more occupied modes activates
    more mode-pair vertices, so activation count is itself informative here in a way it structurally
    cannot be on the layered graph.
    """
    active_set = _pair_active_set(key, m=m)
    adj: dict = {v: [] for v in active_set}
    for u, w in edges:
        if u in active_set and w in active_set:
            adj[u].append(w)
            adj[w].append(u)
    return active_set, adj


def pair_max_component(key, edges, *, m: int) -> int:
    """``maxcc`` on the mode-pair graph: size of the largest connected component among the active
    vertices (:func:`_pair_induced_subgraph`) -- co-occupied off-diagonal pairs plus any bunched
    mode's diagonal self-pair. ``0`` only when no mode is occupied at all (impossible for ``k>=1``);
    a single occupied, unbunched mode with no co-occupied partner activates nothing either (one
    occupied mode alone forms neither an off-diagonal pair nor a bunched diagonal), so ``0`` is still
    reachable for a "one photon, no bunching, no partner" outcome.
    """
    active_set, adj = _pair_induced_subgraph(key, edges, m=m)
    sizes = _components(active_set, adj)
    return max(sizes) if sizes else 0


def pair_maxcc_members(key, edges, *, m: int) -> set:
    """Vertex-member set of the largest connected component on the mode-pair graph -- the pair-graph
    analogue of :func:`_maxcc_members`, feeding :func:`pair_maxcc_parity`.
    """
    active_set, adj = _pair_induced_subgraph(key, edges, m=m)
    components = _component_members(active_set, adj)
    return max(components, key=len) if components else set()


def pair_maxcc_parity(key, edges, *, m: int, k: int) -> float:
    """``pair_ccparity``: **parity computed only over the modes that appear in some mode pair
    belonging to the largest connected component** of the mode-pair graph
    (:func:`pair_maxcc_members`) -- the mode-pair-graph analogue of :func:`layered_maxcc_parity`
    (``ccparity``), which reseed-checked as the one construction in the layered-graph family with a
    confirmed, reproducible size-dependent R^2 trend (see :data:`LAYERED_READINGS`'s docstring).
    Here the graph-selected domain is "every mode that co-occurs, in the winning component, with at
    least one other occupied mode" rather than "every mode whose own occupation-level vertex landed
    in the winning component" -- a genuinely different selection rule, since it is driven by
    *pairwise* co-occupation structure rather than single-mode occupation levels.

    ``+1`` if the summed occupation over those modes is even, ``-1`` if odd. ``0`` active vertices
    (fewer than 2 modes occupied) reduces to summing over no modes, i.e. ``+1``.
    """
    members = pair_maxcc_members(key, edges, m=m)
    table = _pair_vertex_table(m)
    modes: set = set()
    for v in members:
        i, j = table[v]
        modes.add(i)
        modes.add(j)
    total = sum(int(key[i]) for i in modes)
    return 1.0 if total % 2 == 0 else -1.0


def pair_parity_maxcc(key, edges, *, m: int) -> float:
    """``pair_paritymaxcc``: ``parity(key) * maxcc(key)`` on the mode-pair graph -- the plain
    elementwise product of :func:`~observable.scorers.counting.parity_score` (whole-outcome, over
    the readout half) and :func:`pair_max_component` (whole-outcome, on the mode-pair graph), NOT a
    domain restriction. The mode-pair-graph sibling of :func:`layered_parity_maxcc`, which reseed-
    checked as noise-indistinguishable from plain ``parity`` (see :data:`LAYERED_READINGS`'s
    docstring) -- included here for the same direct comparison against ``pair_ccparity``
    (:func:`pair_maxcc_parity`) on this differently-selected graph, since the product-vs-domain-
    restriction distinction is exactly what separated a null result (``paritymaxcc``) from a real one
    (``ccparity``) on the layered graph, and it is worth checking whether that same separation holds
    here too rather than assuming it transfers.
    """
    return float(parity_score(key, parity_modes(m)) * pair_max_component(key, edges, m=m))


#: ``connected_<reading>_pair`` readings -- both read the ``m*(m+1)/2``-vertex mode-pair graph
#: (:func:`_pair_induced_subgraph`), whose active set is co-occupied mode *pairs* plus per-mode
#: bunching via the diagonal (:func:`pair_vertex_index`'s docstring), not individual mode occupation
#: levels: a structurally different selection mechanism from every ``*_layered`` reading, which is
#: why this is a new family rather than another :data:`LAYERED_READINGS` entry. ``maxcc`` is the
#: direct size reading; ``ccparity`` is the domain-restriction construction that proved out on the
#: layered graph, ported to this graph's own selection rule; ``paritymaxcc`` is the plain product
#: (not domain restriction) that reseed-checked as a null result on the layered graph, included here
#: for the matching direct comparison against ``ccparity`` on this graph too.
PAIR_READINGS = {
    "maxcc": pair_max_component,
    "ccparity": pair_maxcc_parity,
    "paritymaxcc": pair_parity_maxcc,
}

_PAIR_RE = re.compile(rf"^connected_({'|'.join(PAIR_READINGS)})_pair$")


class ConnectedPairFamily(ObservableFamily):
    """``connected_<reading>_pair``, ``reading`` in :data:`PAIR_READINGS` -- a graph-property
    reading on an ``m*(m+1)/2``-vertex graph over *mode pairs* (including the diagonal ``i==i`` for
    single-mode bunching) rather than modes: off-diagonal vertex ``(i,j)`` is active iff both modes
    ``i`` and ``j`` are occupied, diagonal vertex ``(i,i)`` is active iff mode ``i`` itself is
    bunched (:func:`_pair_active_set`). Grows quadratically in ``m`` with no extra bunching-budget
    mechanism needed (``GRAPH_OBSERVABLE_PROPOSALS.md`` section 2.3), and reads pairwise
    co-occupation correlation rather than single-mode occupation -- the natural graph-connectivity
    complement to ``pairprod``'s signed pairwise-product observable, which reads the same pair
    structure through a different (non-graph) lens.
    """

    describe = f"connected_<reading>_pair (reading in {tuple(PAIR_READINGS)})"

    def matches(self, name: str) -> bool:
        return bool(_PAIR_RE.match(name))

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.graph_density is None:
            raise ValueError(f"{name} requires graph_density (+ graph_seed)")
        reading = _PAIR_RE.match(name).group(1)
        n_vertices = ctx.m * (ctx.m + 1) // 2
        edges = build_vertex_graph(n_vertices, ctx.graph_density, ctx.graph_seed)
        if reading == "maxcc":
            v = [pair_max_component(key, edges, m=ctx.m) for key in ctx.keys]
        elif reading == "paritymaxcc":
            v = [pair_parity_maxcc(key, edges, m=ctx.m) for key in ctx.keys]
        else:
            v = [pair_maxcc_parity(key, edges, m=ctx.m, k=ctx.k) for key in ctx.keys]
        obs = Expectation(np.array(v, dtype=np.float64))
        obs.edges = edges                                  # kept for inspection / debugging
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        reading = _PAIR_RE.match(name).group(1)
        return {"observable": f"connected_{reading}_pair",
                "graph_density": ctx.graph_density,
                "graph_seed": ctx.graph_seed,
                "max_vertex_degree": MAX_VERTEX_DEGREE,
                "n_vertices": ctx.m * (ctx.m + 1) // 2}


def mode_coupling_strengths(m: int, model_seed: int, n_features: int) -> "np.ndarray":
    """``|U_ab|`` for every ``0 <= a <= b < m``, ``(m,m)`` array, symmetric -- the sandwich
    circuit's own mode-to-mode transition amplitude at ``x=0`` (no encoding contribution), used as a
    physically-grounded coupling strength in place of :func:`build_vertex_graph`'s uniform-random
    edge draw.

    Reconstructed via :func:`~circuit.photonic_circuit.sandwich_unitaries` +
    :func:`~circuit.photonic_circuit.sandwich_unitary_at` -- **not** by reading the live model's
    internal state, since the merlin ``QuantumLayer`` this repo wraps has an empty ``state_dict()``
    (``W1``/``W2`` are never exposed as torch parameters) and no ``circuit.pt`` is persisted for the
    photonic model as a result. ``sandwich_unitaries(m, model_seed)`` regenerates the *exact same*
    ``W1``/``W2`` the original circuit used (same two seeded ``perceval`` draws, in the same order;
    verified elsewhere in this codebase to reproduce merlin's own probabilities to ~1e-7), so this
    needs only ``(m, model_seed, n_features)`` -- all already in the artifact's stored metadata, none
    of it merlin-internal.

    The diagonal ``|U_ii|`` (probability amplitude for a photon to stay in its own mode) is included
    deliberately, not filtered out -- it is what lets the mode-pair graph's own diagonal
    (:func:`pair_vertex_index`'s ``i==j`` bunching vertex) get a physically meaningful weight too,
    rather than only the off-diagonal pairs.
    """
    from circuit.photonic_circuit import sandwich_unitaries, sandwich_unitary_at
    import torch as _torch

    W1, W2 = sandwich_unitaries(m, model_seed)
    x0 = _torch.zeros(1, n_features)
    U = sandwich_unitary_at(W1, W2, x0, n_features, encoding="phase")[0]
    return U.abs().numpy()


def build_unitary_weighted_pair_graph(m: int, model_seed: int, n_features: int, *,
                                      threshold: float, seed: int) -> list[tuple[int, int]]:
    """Edge list for the mode-pair graph, weighted by circuit coupling instead of drawn uniformly at
    random (:func:`build_vertex_graph`'s construction) -- the circuit-derived sibling of that
    function, same output shape (a sorted list of distinct ``(u, v)`` vertex-id pairs) so it drops
    straight into :func:`pair_max_component` / :func:`pair_maxcc_parity` / :func:`pair_parity_maxcc`
    unchanged.

    Each vertex ``(a, b)`` (``a <= b``, :func:`pair_vertex_index`'s indexing, diagonal included) gets
    a strength ``s(a,b) = |U_ab|`` from :func:`mode_coupling_strengths`. An edge between vertices
    ``(a,b)`` and ``(a',b')`` is included iff ``s(a,b) * s(a',b') > threshold`` -- the product
    (rather than the sum) so a genuinely weakly-coupled vertex suppresses every edge it would
    otherwise participate in, regardless of how strong its partner is; this is a stricter,
    self-consistently physical notion of "both endpoints matter" than an additive rule would give.
    ``threshold`` is therefore this construction's density-like knob, but its scale is set by the
    circuit's own ``|U|`` values rather than being a unitless fraction of ``C(V,2)`` the way
    :func:`build_vertex_graph`'s ``density`` is -- callers should inspect
    :func:`mode_coupling_strengths`'s own range for a given circuit before picking a threshold.

    Deliberately **not** run through :func:`build_vertex_graph`'s connectivity-guaranteeing spanning
    path -- unlike that function, this graph is allowed to be disconnected (or have isolated
    diagonal-only vertices) when the circuit's own coupling is weak enough, since the point is to let
    the circuit's real structure show through rather than forcing a connected graph regardless of
    what the physics says. ``seed`` is accepted for interface parity with :func:`build_vertex_graph`
    but only used to break ties among edges at exactly ``threshold`` (float equality is otherwise
    seed-independent and deterministic in the circuit alone).
    """
    strengths = mode_coupling_strengths(m, model_seed, n_features)
    table = _pair_vertex_table(m)
    n_vertices = len(table)
    edges: list[tuple[int, int]] = []
    for u in range(n_vertices):
        au, bu = table[u]
        su = float(strengths[au, bu])
        for w in range(u + 1, n_vertices):
            aw, bw = table[w]
            sw = float(strengths[aw, bw])
            if su * sw > threshold:
                edges.append((u, w))
    return sorted(edges)


#: ``connected_<reading>_pairU`` readings -- same readings as :data:`PAIR_READINGS`, on the same
#: ``m*(m+1)/2``-vertex mode-pair graph, but with edges from :func:`build_unitary_weighted_pair_graph`
#: (circuit-derived, via the reconstructed sandwich unitary at ``x=0``) instead of
#: :func:`build_vertex_graph`'s uniform-random draw. Tried specifically because the mode-pair
#: graph's own random-edge version (``connected_paritymaxcc_pair``) was the one construction in this
#: whole module that showed a clean, reseed-confirmed harder-with-size *and* harder-with-density
#: trend (see ``PAIR_READINGS``'s own reseed-check history) -- the natural next question is whether
#: a graph tied to the circuit's real coupling structure, rather than an arbitrary random draw,
#: sustains or improves on that trend.
PAIR_U_READINGS = {
    "maxcc": pair_max_component,
    "ccparity": pair_maxcc_parity,
    "paritymaxcc": pair_parity_maxcc,
}

_PAIR_U_RE = re.compile(rf"^connected_({'|'.join(PAIR_U_READINGS)})_pairU$")


class ConnectedPairUnitaryFamily(ObservableFamily):
    """``connected_<reading>_pairU``, ``reading`` in :data:`PAIR_U_READINGS` -- the circuit-derived
    sibling of :class:`ConnectedPairFamily`: same ``m*(m+1)/2``-vertex mode-pair graph and the same
    scoring functions, but edges come from :func:`build_unitary_weighted_pair_graph` (weighted by
    the reconstructed sandwich unitary's ``|U_ab|`` at ``x=0``) instead of a uniform-random draw.
    ``ctx.graph_density`` is reused as the coupling-product ``threshold`` (see
    :func:`build_unitary_weighted_pair_graph`'s docstring for why its scale differs from the random
    construction's density fraction) so the same ``graph_density`` config knob drives both families,
    just interpreted differently per family.
    """

    describe = f"connected_<reading>_pairU (reading in {tuple(PAIR_U_READINGS)})"

    def matches(self, name: str) -> bool:
        return bool(_PAIR_U_RE.match(name))

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.graph_density is None:
            raise ValueError(f"{name} requires graph_density (used as the coupling threshold)")
        if ctx.n_features is None:
            raise ValueError(f"{name} requires n_features (from the artifact's stored metadata)")
        reading = _PAIR_U_RE.match(name).group(1)
        edges = build_unitary_weighted_pair_graph(
            ctx.m, ctx.seed, ctx.n_features, threshold=ctx.graph_density, seed=ctx.graph_seed)
        if reading == "maxcc":
            v = [pair_max_component(key, edges, m=ctx.m) for key in ctx.keys]
        elif reading == "paritymaxcc":
            v = [pair_parity_maxcc(key, edges, m=ctx.m) for key in ctx.keys]
        else:
            v = [pair_maxcc_parity(key, edges, m=ctx.m, k=ctx.k) for key in ctx.keys]
        obs = Expectation(np.array(v, dtype=np.float64))
        obs.edges = edges                                  # kept for inspection / debugging
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        reading = _PAIR_U_RE.match(name).group(1)
        return {"observable": f"connected_{reading}_pairU",
                "coupling_threshold": ctx.graph_density,
                "model_seed": ctx.seed,
                "n_vertices": ctx.m * (ctx.m + 1) // 2}


def triple_vertex_index(m: int, i: int, j: int, l: int) -> int:
    """Vertex id of mode-triple ``(i, j, l)`` (``i < j < l``) in the mode-triple graph's off-diagonal
    block -- laid out to match ``list(itertools.combinations(range(m), 3))``'s own enumeration order
    (see :func:`_triple_vertex_table`), computed directly rather than looked up.

    This is the **off-diagonal block only** (``C(m,3)`` vertices); it is followed in the full vertex
    space by the doubled-mode block (:func:`triple_doubled_vertex_index`) and the tripled-mode block
    (:func:`triple_tripled_vertex_index`), analogous to how :func:`pair_vertex_index`'s single
    diagonal restored bunching sensitivity to the pair graph -- see :func:`_triple_active_set`'s
    docstring for how the three blocks together fix this construction's earlier bunching-blindness.
    """
    if not 0 <= i < j < l < m:
        raise ValueError(f"need 0 <= i < j < l < m={m} (got i={i}, j={j}, l={l})")
    # offset of (i, j, l) among all triples: count triples with smaller first element, then smaller
    # second element (given the first), then the position within the third.
    off_i = sum((m - 1 - a) * (m - 2 - a) // 2 for a in range(i))
    off_j = sum(m - 1 - b for b in range(i + 1, j))
    off_l = l - j - 1
    return off_i + off_j + off_l


def triple_doubled_vertex_index(m: int, i: int, j: int) -> int:
    """Vertex id of the doubled-mode vertex ``(i, i, j)`` (mode ``i`` bunched at depth >= 2, mode
    ``j != i`` also occupied) in the mode-triple graph's doubled block, immediately following the
    off-diagonal block (:func:`triple_vertex_index`) in the full vertex ordering. **Ordered** in
    ``(i, j)`` -- ``i`` is the bunched mode and ``j`` the plain co-occupied one, so ``(i, i, j)`` and
    ``(j, j, i)`` are distinct vertices (different mode is the one that's bunched); ``m*(m-1)``
    vertices total, laid out row-major over ``i`` then ``j != i``.
    """
    if not (0 <= i < m and 0 <= j < m and i != j):
        raise ValueError(f"need 0 <= i, j < m={m}, i != j (got i={i}, j={j})")
    n_off_diag = m * (m - 1) * (m - 2) // 6
    row = i * (m - 1) + (j if j < i else j - 1)
    return n_off_diag + row


def triple_tripled_vertex_index(m: int, i: int) -> int:
    """Vertex id of the fully-tripled vertex ``(i, i, i)`` (mode ``i`` bunched at depth >= 3) in the
    mode-triple graph's tripled block, the final ``m`` vertices in the full vertex ordering (after
    the off-diagonal and doubled blocks).
    """
    if not 0 <= i < m:
        raise ValueError(f"need 0 <= i < m={m} (got i={i})")
    n_off_diag = m * (m - 1) * (m - 2) // 6
    n_doubled = m * (m - 1)
    return n_off_diag + n_doubled + i


def triple_n_vertices(m: int) -> int:
    """Total vertex count of the full mode-triple graph -- off-diagonal + doubled + tripled blocks:
    ``C(m,3) + m*(m-1) + m``.
    """
    return m * (m - 1) * (m - 2) // 6 + m * (m - 1) + m


def _triple_vertex_table(m: int) -> list[tuple[int, int, int]]:
    """``[(i, j, l), ...]`` indexed by vertex id across all three blocks (off-diagonal, doubled,
    tripled) -- the inverse of :func:`triple_vertex_index` / :func:`triple_doubled_vertex_index` /
    :func:`triple_tripled_vertex_index`, built once per ``m``. Doubled entries are returned as
    ``(i, i, j)`` and tripled entries as ``(i, i, i)``, matching each block's own vertex semantics.
    """
    off_diag = [(i, j, l) for i in range(m) for j in range(i + 1, m) for l in range(j + 1, m)]
    doubled = [(i, i, j) for i in range(m) for j in range(m) if j != i]
    tripled = [(i, i, i) for i in range(m)]
    return off_diag + doubled + tripled


def _triple_active_set(key, *, m: int) -> set:
    """Mode-triple vertices active for one outcome, across all three blocks:

    - off-diagonal ``(i, j, l)``, ``i < j < l``: active iff all three modes occupied
      (:func:`triple_vertex_index`).
    - doubled ``(i, i, j)``, ``i != j``: active iff mode ``i`` is bunched (``n_i >= 2``) and mode
      ``j`` is occupied (``n_j >= 1``) (:func:`triple_doubled_vertex_index`).
    - tripled ``(i, i, i)``: active iff mode ``i`` is bunched at depth >= 3 (``n_i >= 3``)
      (:func:`triple_tripled_vertex_index`).

    Together these restore the bunching sensitivity the earlier off-diagonal-only construction
    lacked (mirroring :func:`_pair_active_set`'s single diagonal, generalised to the two ways three
    photon-slots can pile onto fewer than three distinct modes).
    """
    occupied = [i for i, n_i in enumerate(key) if int(n_i) > 0]
    active = set()
    n = len(occupied)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                i, j, l = occupied[a], occupied[b], occupied[c]
                active.add(triple_vertex_index(m, i, j, l))
    for i in range(m):
        n_i = int(key[i])
        if n_i >= 2:
            for j in occupied:
                if j != i:
                    active.add(triple_doubled_vertex_index(m, i, j))
        if n_i >= 3:
            active.add(triple_tripled_vertex_index(m, i))
    return active


def _triple_induced_subgraph(key, edges, *, m: int):
    """``(active_set, adj)`` for one outcome on the mode-triple graph -- the triple-graph analogue of
    :func:`_pair_induced_subgraph`.
    """
    active_set = _triple_active_set(key, m=m)
    adj: dict = {v: [] for v in active_set}
    for u, w in edges:
        if u in active_set and w in active_set:
            adj[u].append(w)
            adj[w].append(u)
    return active_set, adj


def triple_max_component(key, edges, *, m: int) -> int:
    """``maxcc`` on the mode-triple graph: size of the largest connected component among the active
    vertices across all three blocks -- off-diagonal (three distinct co-occupied modes), doubled (one
    bunched mode plus one more occupied mode), tripled (one mode bunched at depth >= 3)
    (:func:`_triple_active_set`, :func:`_triple_induced_subgraph`). ``0`` only when no mode is
    occupied at all (impossible for ``k>=1``); a single occupied, unbunched mode with no co-occupied
    partners activates nothing (too few modes/photons to trigger any of the three blocks).
    """
    active_set, adj = _triple_induced_subgraph(key, edges, m=m)
    sizes = _components(active_set, adj)
    return max(sizes) if sizes else 0


def triple_parity_maxcc(key, edges, *, m: int) -> float:
    """``triple_paritymaxcc``: ``parity(key) * maxcc(key)`` on the mode-triple graph -- the plain
    elementwise product, mirroring :func:`pair_parity_maxcc`. This reading (not the domain-
    restriction ``ccparity`` shape) is the one that showed the cleanest, most reproducible
    harder-with-size *and* harder-with-density trend on the mode-pair graph (see
    ``PAIR_READINGS``'s comment above for the reseed-check summary), so it is the first reading
    ported to the triple graph -- the direct test of whether a stricter, three-way co-occupation
    selection extends that same trend to higher densities before saturating (the mode-pair graph's
    own trend was found to plateau past density ~0.4 at ``m=12``; see the density-sweep discussion
    in this session's history).
    """
    return float(parity_score(key, parity_modes(m)) * triple_max_component(key, edges, m=m))


#: ``connected_<reading>_triple`` readings -- read the mode-triple graph
#: (:func:`_triple_induced_subgraph`), whose vertex space has three blocks: off-diagonal
#: (``C(m,3)``, three distinct co-occupied modes), doubled (``m*(m-1)``, one bunched mode plus one
#: more occupied mode), tripled (``m``, one mode bunched at depth >= 3) -- see :func:`triple_n_vertices`
#: for the total and :func:`_triple_active_set` for the per-outcome activation rule across all three.
#: This mirrors :func:`pair_vertex_index`'s single diagonal, generalised to the two distinct ways
#: three photon-slots can pile onto fewer than three distinct modes, so this graph is (unlike an
#: earlier off-diagonal-only version of this construction) bunching-sensitive. Empirically (checked
#: at m=12,k=6 against the pair graph's own off-diagonal active-set statistics): the *typical*
#: off-diagonal-only active set is smaller (mean 6.47 vs the pair graph's 8.60) but has substantially
#: higher variance (std 5.15 vs 3.01) and a longer tail (max 20 vs 15) -- so while a typical outcome
#: selects fewer triple-vertices than pair-vertices, outcomes with many co-occupied modes select
#: disproportionately more, since C(n,3) grows faster than C(n,2) for the same occupied-mode count n
#: (the doubled/tripled blocks add further active vertices on top of this baseline whenever bunching
#: occurs). Only ``paritymaxcc`` is implemented initially; a ``ccparity`` domain-restriction sibling
#: could be added the same way :func:`pair_maxcc_parity` was, if this reading's result motivates it.
TRIPLE_READINGS = {
    "paritymaxcc": triple_parity_maxcc,
}

_TRIPLE_RE = re.compile(rf"^connected_({'|'.join(TRIPLE_READINGS)})_triple$")


class ConnectedTripleFamily(ObservableFamily):
    """``connected_<reading>_triple``, ``reading`` in :data:`TRIPLE_READINGS` -- a graph-property
    reading on the mode-triple graph (:func:`triple_n_vertices` vertices, across the off-diagonal /
    doubled / tripled blocks): vertex activation is co-occupation (off-diagonal) or bunching
    (doubled/tripled) (:func:`_triple_active_set`). The three-way-co-occupation escalation of
    :class:`ConnectedPairFamily`'s two-way construction -- see :data:`TRIPLE_READINGS`'s docstring
    for the measured active-set-size comparison that motivated trying it.
    """

    describe = f"connected_<reading>_triple (reading in {tuple(TRIPLE_READINGS)})"

    def matches(self, name: str) -> bool:
        return bool(_TRIPLE_RE.match(name))

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.graph_density is None:
            raise ValueError(f"{name} requires graph_density (+ graph_seed)")
        reading = _TRIPLE_RE.match(name).group(1)
        m = ctx.m
        if m < 3:
            raise ValueError(f"{name} requires m >= 3 (got m={m})")
        n_vertices = triple_n_vertices(m)
        edges = build_vertex_graph(n_vertices, ctx.graph_density, ctx.graph_seed)
        scorer = TRIPLE_READINGS[reading]
        v = [scorer(key, edges, m=m) for key in ctx.keys]
        obs = Expectation(np.array(v, dtype=np.float64))
        obs.edges = edges                                  # kept for inspection / debugging
        return obs

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        reading = _TRIPLE_RE.match(name).group(1)
        return {"observable": f"connected_{reading}_triple",
                "graph_density": ctx.graph_density,
                "graph_seed": ctx.graph_seed,
                "max_vertex_degree": MAX_VERTEX_DEGREE,
                "n_vertices": triple_n_vertices(ctx.m)}


register(ConnectedLayeredFamily())
register(ConnectedFamily())
register(ConnectedPairFamily())
register(ConnectedPairUnitaryFamily())
register(ConnectedTripleFamily())
