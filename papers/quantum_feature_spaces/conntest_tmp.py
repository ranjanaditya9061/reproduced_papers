"""Smoke test for the connected_<base> (largest-connected-component) observable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from model.photonic import (PhotonicTeacher, parse_connected_observable,
                            is_connected_observable, score_from_distribution,
                            build_vertex_graph, _clicked_max_component, _is_connected,
                            MAX_VERTEX_DEGREE)
from model.sampler import sample_X

# build_vertex_graph: connected G on V=m vertices, edges floored at V-1, capped by MAX_VERTEX_DEGREE
_capped = MAX_VERTEX_DEGREE >= 0
for (mm, dens) in [(6, 0.5), (7, 0.3), (5, 1.0), (8, 1.0), (5, 0.1)]:
    e = build_vertex_graph(mm, dens, 42)
    e2 = build_vertex_graph(mm, dens, 42)
    max_edges = mm * (mm - 1) // 2
    cap_edges = mm * MAX_VERTEX_DEGREE // 2 if _capped else max_edges
    want = min(max(int(round(dens * max_edges)), mm - 1), max_edges, cap_edges)
    assert len(e) == want and len(set(e)) == want, (mm, dens, want, len(e), e)
    assert e == e2, "not deterministic"
    assert all(0 <= u < w < mm for u, w in e), e
    assert _is_connected(e, mm), f"G not connected: {(mm, dens)} {e}"
    if _capped:
        deg = {}
        for u, w in e:
            deg[u] = deg.get(u, 0) + 1
            deg[w] = deg.get(w, 0) + 1
        assert all(v <= MAX_VERTEX_DEGREE for v in deg.values()), (mm, dens, deg)
for bad in [(6, 0.0), (6, 1.5), (1, 0.5)]:               # density <= 0, > 1, m < 2
    try:
        build_vertex_graph(*bad, 42)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass
_cap_txt = "no cap (-1)" if not _capped else f"degree <= {MAX_VERTEX_DEGREE}"
print(f"build_vertex_graph OK (connected, V=m, density in (0,1], {_cap_txt} enforced)\n")

# _clicked_max_component: largest connected component of the clicked-vertex induced subgraph
edges = [(0, 1), (1, 2), (3, 4)]     # path 0-1-2 and edge 3-4
assert _clicked_max_component((1, 1, 1, 0, 0, 0), edges) == 3    # all of 0-1-2 clicked
assert _clicked_max_component((1, 0, 1, 1, 1, 0), edges) == 2    # {0},{2} isolated, {3,4} joined
assert _clicked_max_component((1, 0, 1, 0, 0, 0), edges) == 1    # {0},{2} independent
assert _clicked_max_component((0, 0, 0, 0, 0, 1), edges) == 1    # single clicked vertex
print("_clicked_max_component OK (largest connected component size)\n")

m, k, dens, seed = 6, 3, 0.6, 42
nf = m - 1
X = sample_X(64, nf, seed)

# parser: no suffixes; unknown base / bad suffix rejected
print("parse connected_maxcc :", parse_connected_observable("connected_maxcc"))
print("parse parity          :", parse_connected_observable("parity"))
print("is_connected(connected_maxcc):", is_connected_observable("connected_maxcc"))
print("is_connected(connected_bogus):", is_connected_observable("connected_bogus"))
try:
    parse_connected_observable("connected_maxcc__dep")
    raise AssertionError("expected ValueError for __ suffix")
except ValueError:
    print("connected_maxcc__dep rejected (no suffixes)")
print()

t = PhotonicTeacher(m=m, k=k, n_features=nf, observable="connected_maxcc",
                    seed=seed, graph_density=dens)
y = t(X).squeeze(-1)
print(f"n_fock={len(t.score_vec)}  edges={len(t.edges)}  "
      f"maxcc range over basis=[{int(t.score_vec.min())},{int(t.score_vec.max())}]")
print(f"connected_maxcc : mean={y.mean():+.4f} std={y.std():.4f} "
      f"range=[{y.min():+.3f},{y.max():+.3f}]  (E[largest connected component size])")

# offline re-scoring must match the online teacher
t.enable_distribution_capture()
_ = t(X)
dist = t.captured_distributions()
off = score_from_distribution(dist, "connected_maxcc", graph_density=dens, graph_seed=seed)
online = t(X).squeeze(-1).numpy()
print(f"\noffline re-score matches online: {np.allclose(off, online, atol=1e-5)} "
      f"(max abs diff {np.abs(off - online).max():.2e})")
