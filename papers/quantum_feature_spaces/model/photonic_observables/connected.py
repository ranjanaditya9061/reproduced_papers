"""``connected_<base>``: read each Fock outcome as a vertex set, score a global connectivity.

A sibling of :mod:`.loop_path` that maps each mode to a *vertex* (``mode i <-> vertex i``) of a
fixed, seeded, connected, bounded-degree graph ``G`` on ``V = m`` vertices
(:func:`~.graphs.build_vertex_graph`) rather than to an edge.  For each Fock outcome the
*clicked* vertices (the modes with a non-zero photon count) induce a subgraph of ``G``, and the
score is a GLOBAL property of that subgraph: base ``maxcc`` is the size (vertex count) of its
largest connected component.

There is no pre-selection -- the output is the plain expectation ``E[<base>]`` over the full Fock
distribution, so unlike ``loop_path_<base>`` this is an ordinary
:class:`~.base.LinearObservable` with no keep mask, and the family takes no ``__`` suffixes.
``G``'s density (``graph_density``, the fraction of the ``C(V, 2)`` possible edges) and its degree
cap set how large those components can grow, so equal ``(m, graph_density, graph_seed)`` give the
identical edge set.
"""

from __future__ import annotations

import re

import numpy as np

from .base import (LinearObservable, Observable, ObservableContext, ObservableFamily,
                   base_score_vec, register)
from .graphs import build_vertex_graph

#: Base scorers usable under a ``connected_<base>`` observable.  ``maxcc`` returns the size of
#: the largest connected component of the clicked vertices' induced subgraph of ``G`` (a global
#: connectivity property, 1 when the clicked set is independent); the rest are the plain
#: per-Fock-state scorers.
CONNECTED_BASES = ("parity", "majority", "bunching", "n_first", "maxcc")

_CONNECTED_RE = re.compile(r"^connected_(.+)$")


def parse_connected_observable(observable: str):
    """Split a ``connected_<base>`` string into ``(is_conn, base)``.

    Plain observables return ``(False, observable)``.  The family takes no ``__`` suffixes (the
    score is a single global property, not a selectable subset), so ``connected_maxcc`` scores
    the largest-connected-component size.
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


def clicked_max_component(key, edges) -> int:
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


def connected_scores(keys, *, m, k, base, edges):
    """Per-Fock-state ``<base>`` score vector for a ``connected_<base>`` observable (no selection).

    Aligns to the fixed Fock basis, so scoring a distribution is the plain expectation
    ``probs @ score_vec``.
    """
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
        obs = LinearObservable(connected_scores(ctx.keys, m=ctx.m, k=ctx.k, base=base, edges=edges))
        obs.edges = edges                                # kept for inspection / debugging
        return obs

    def hash_spec(self, name: str, ctx: ObservableContext) -> dict:
        _, base = parse_connected_observable(name)
        return {"observable": f"connected_{base}",
                "graph_density": ctx.graph_density,
                "graph_seed": ctx.graph_seed}


register(ConnectedFamily())
