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

from ..base import (Expectation, Observable, ObservableContext, ObservableFamily, base_score_vec,
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
    and bounded by the degree cap.  Deterministic in ``seed``, hence reproducible and hashable.

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


register(ConnectedFamily())
