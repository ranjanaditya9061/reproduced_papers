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

#: Base scorers usable under ``connected_<base>``.  ``maxcc`` is the graph reading; the rest are
#: the plain counting scorers, scored (unselected) over the full distribution.
CONNECTED_BASES = ("parity", "majority", "bunching", "n_first", "maxcc")

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
    seen: set = set()
    sizes = []
    for start in active_set:
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
        sizes.append(size)
    return sizes


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


def connected_scores(keys, *, m, k, base, edges):
    """Per-outcome ``<base>`` score vector for ``connected_<base>`` (no selection)."""
    if base == "maxcc":
        return np.array([clicked_max_component(key, edges) for key in keys], dtype=np.float64)
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


#: ``connected_<reading>_layered`` readings -- all three read the same ``m*(k+1)``-vertex layered
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
LAYERED_READINGS = {
    "maxcc": layered_clicked_max_component,
    "productcc": layered_product_component,
    "numloops": layered_num_loops,
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


register(ConnectedLayeredFamily())
register(ConnectedFamily())
