"""``loop_path_<base>``: read each Fock outcome as an edge set, pre-select on loop/path counts.

A ``loop_path_<base>`` observable reinterprets each collision-free Fock outcome as an edge set of
a fixed graph ``G`` (``mode i <-> edge e_i``, :func:`~.graphs.build_matching_graph`), keeps only
the outcomes that are matchings, overlays them with a fixed reference perfect matching ``M_0``
(``H = E(x) | M_0``, a disjoint union of alternating loops/paths), then pre-selects on the
loop/path counts before scoring ``<base>`` over the renormalised survivors -- so the value is
``E[<base> | selection]``, a :class:`~.base.SelectiveObservable`.

The selection is encoded in the observable string via filesystem-safe ``__L`` / ``__P``
suffixes: ``loop_path_parity__L0-1__P2-3`` keeps overlays with 0-or-1 loops and 2-or-3 paths,
``__L`` with an empty body means keep-all on that dimension, and a *missing* segment likewise
keeps all -- so bare ``loop_path_parity`` keeps every matching.  Because the selection rides in
the string, a sweep varies it without touching the config, and ``__L0-1__P2`` /
``__P2__L0-1`` canonicalise to one dataset.

``<base>`` is ``loop`` / ``path`` (the mean loop / path count over the selected subset) or any of
the plain per-Fock-state scorers, averaged over that same subset.
"""

from __future__ import annotations

import re

import numpy as np

from .base import (Observable, ObservableContext, ObservableFamily, SelectiveObservable,
                   base_score_vec, register)
from .graphs import build_matching_graph

#: Base scorers usable under a ``loop_path_<base>`` observable.
GRAPH_BASES = ("parity", "majority", "bunching", "n_first", "loop", "path")

_LOOP_PATH_RE = re.compile(r"^loop_path_(.+)$")


# --- observable-string parsing ------------------------------------------------------------ #


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
    a *missing* segment yields ``None`` (keep-all on that dimension).
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
    selection only in the observable string).  Single source of truth for the teacher's
    build, its hash spec, and the offline re-scorer.
    """
    is_graph, base, s_loop, s_path = parse_graph_observable(observable)
    if not is_graph:
        raise ValueError(f"{observable!r} is not a loop_path_<base> observable")
    eff_loop = s_loop if s_loop is not None else loop_vars
    eff_path = s_path if s_path is not None else path_vars
    return base, eff_loop, eff_path


# --- overlaying an outcome with M_0 ------------------------------------------------------- #


def overlay_counts(key, edges, m0_edges: set, n_vertices: int):
    """``(valid, n_loops, n_paths)`` for one Fock outcome ``key`` (per-mode counts).

    ``valid`` is ``False`` for a bunched outcome (some mode > 1) or one whose clicked
    edges share a vertex (not a matching).  Otherwise overlays the clicked edges with
    ``M_0`` (set union, so an edge present in *both* becomes a length-1 path) and counts
    cycle components (loops) and path components -- every vertex has degree <= 2 because
    both are matchings, so each component is a simple loop or path.

    NOTE: a count of ``c > 1`` is first folded down by one (``c -> c - 1``), so a doubly
    occupied mode reads as a single click rather than as a bunching rejection, and only
    ``c >= 3`` is rejected outright.  That is what every existing ``loop_path_`` dataset was
    generated with, so it is preserved verbatim.
    """
    counts = [int(c - 1) if c > 1 else int(c) for c in key]
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
    return True, n_loops, n_paths


def graph_tables(keys, edges, m0_mask, n_vertices: int):
    """Per-Fock-state ``(valid, n_loops, n_paths)`` arrays over the fixed basis ``keys``."""
    m0_edges = {edges[i] for i in range(len(edges)) if m0_mask[i]}
    valid = np.zeros(len(keys), dtype=bool)
    loops = np.zeros(len(keys), dtype=np.int64)
    paths = np.zeros(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        valid[i], loops[i], paths[i] = overlay_counts(key, edges, m0_edges, n_vertices)
    return valid, loops, paths


# --- selection + scoring ------------------------------------------------------------------ #


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


def graph_selection(keys, *, m, k, base, edges, m0_mask, n_vertices, loop_vars, path_vars):
    """``(keep_mask, base_scores)`` float vectors for a graph observable over ``keys``.

    ``keep_mask`` = matching AND (loop count in ``loop_vars``) AND (path count in
    ``path_vars``); ``base_scores`` is the per-state ``<base>`` value.  Both align to the
    fixed Fock basis, so scoring a distribution is a masked, renormalised dot product.
    """
    valid, loops, paths = graph_tables(keys, edges, m0_mask, n_vertices)
    keep = valid & _var_mask(loops, loop_vars) & _var_mask(paths, path_vars)
    if base == "loop":
        scores = loops.astype(np.float64)
    elif base == "path":
        scores = paths.astype(np.float64)
    else:
        scores = base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                                allowed=GRAPH_BASES, label="graph")
    return keep.astype(np.float64), scores


class LoopPathFamily(ObservableFamily):
    describe = f"loop_path_<base>[__L<ints>][__P<ints>] (base in {GRAPH_BASES})"

    def matches(self, name: str) -> bool:
        return is_graph_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.n_vertices is None:
            raise ValueError("loop_path_<base> observables require n_vertices "
                             "(+ optional loop_vars / path_vars / graph_seed)")
        half = ctx.n_vertices // 2
        if not ctx.k <= half <= ctx.m:
            raise ValueError(f"loop_path_ needs k <= n_vertices//2 <= m "
                             f"(k={ctx.k}, n_vertices//2={half}, m={ctx.m})")
        base, loop_vars, path_vars = resolve_graph_spec(name, ctx.loop_vars, ctx.path_vars)
        edges, m0_mask = build_matching_graph(ctx.m, ctx.n_vertices, ctx.graph_seed)
        keep, vec = graph_selection(
            ctx.keys, m=ctx.m, k=ctx.k, base=base, edges=edges, m0_mask=m0_mask,
            n_vertices=ctx.n_vertices, loop_vars=loop_vars, path_vars=path_vars)
        obs = SelectiveObservable(keep, vec)
        obs.edges, obs.m0_mask = edges, m0_mask         # kept for inspection / debugging
        obs.loop_vars, obs.path_vars = loop_vars, path_vars
        return obs

    def hash_spec(self, name: str, ctx: ObservableContext) -> dict:
        # The selection is folded into loop_vars/path_vars, so ``__L0-1__P2`` and
        # ``__P2__L0-1`` (same selection, different spelling) map to one dataset.
        base, eff_loop, eff_path = resolve_graph_spec(name, ctx.loop_vars, ctx.path_vars)
        return {
            "observable": f"loop_path_{base}",
            "n_vertices": ctx.n_vertices,
            "loop_vars": None if eff_loop is None else sorted(int(v) for v in eff_loop),
            "path_vars": None if eff_path is None else sorted(int(v) for v in eff_path),
            "graph_seed": ctx.graph_seed,
        }


register(LoopPathFamily())
