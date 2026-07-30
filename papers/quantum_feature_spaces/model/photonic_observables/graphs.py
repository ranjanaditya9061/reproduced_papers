"""The fixed, seeded graphs the graph observables interpret Fock outcomes against.

Two mode->graph mappings, one per family:

* :func:`build_matching_graph` -- ``mode i <-> edge e_i`` of a graph on ``n_vertices``, plus a
  reference perfect matching ``M_0``.  Used by :mod:`.loop_path`.
* :func:`build_vertex_graph` -- ``mode i <-> vertex i`` of a graph on ``V = m`` vertices.  Used
  by :mod:`.connected`.

Both are deterministic in their seed (and drawn *connected*, so a global property of the
clicked subgraph is genuinely global rather than a sum of local pieces) -> reproducible and
hashable.
"""

from __future__ import annotations

import numpy as np


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
        if not is_connected_graph(edges, V):
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

    The graph builder for the ``connected_<base>`` observable (:mod:`.connected`);
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
