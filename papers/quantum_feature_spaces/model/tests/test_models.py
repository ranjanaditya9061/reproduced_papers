"""Tests for the teacher-model layer."""

from __future__ import annotations

import math

import torch

from model import TEACHERS, sample_X
from model.analytical import AnalyticalTeacher
from model.mlp import MLPTeacher
from model.qubit import QubitTeacher


def test_registry_has_all_teachers():
    assert {"analytical", "mlp", "qubit_quantum", "photonic_quantum"} <= set(TEACHERS)


def test_sample_X_prefix_stable_and_ranged():
    a = sample_X(50, 3, seed=1)
    b = sample_X(100, 3, seed=1)
    assert torch.equal(a, b[:50])              # prefix stable
    assert a.shape == (50, 3)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 2 * math.pi


def test_analytical_output_is_signed_score():
    t = AnalyticalTeacher(n_features=3, k=3)
    soft = t(sample_X(64, 3, seed=0))
    assert soft.shape == (64, 1)
    assert float(soft.min()) >= -1.0 and float(soft.max()) <= 1.0


def test_mlp_output_is_prob_simplex():
    t = MLPTeacher(n_features=3, k=2, seed=0)
    soft = t(sample_X(64, 3, seed=0))
    assert soft.shape == (64, 2)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(64), atol=1e-5)


def test_qubit_parity_in_range_and_deterministic():
    soft1 = QubitTeacher(n_features=3, k=2, seed=7)(sample_X(32, 3, seed=0))
    soft2 = QubitTeacher(n_features=3, k=2, seed=7)(sample_X(32, 3, seed=0))
    assert soft1.shape == (32, 1)
    assert float(soft1.min()) >= -1.0001 and float(soft1.max()) <= 1.0001
    assert torch.allclose(soft1, soft2)        # same seed -> identical teacher


def test_photonic_runs_small():
    from model.photonic import PhotonicTeacher  # constructs a Merlin layer
    t = PhotonicTeacher(m=4, k=2, n_features=3, observable="parity", seed=2)
    soft = t(sample_X(8, 3, seed=0))
    assert soft.shape == (8, 1)
    assert torch.isfinite(soft).all()


def test_matching_graph_is_connected_perfect_matching():
    from model.photonic import build_matching_graph

    edges, m0_mask = build_matching_graph(m=10, n_vertices=8, seed=7)
    assert len(edges) == 10 and m0_mask.sum() == 4          # M_0 = V/2 = 4 edges
    assert len({tuple(e) for e in edges}) == 10             # distinct edges
    covered = {v for i, e in enumerate(edges) if m0_mask[i] for v in e}
    assert covered == set(range(8))                         # M_0 is a *perfect* matching


def test_overlay_counts_loops_and_paths():
    from model.photonic import build_matching_graph, _overlay_counts

    edges, m0_mask = build_matching_graph(m=6, n_vertices=4, seed=42)
    m0 = {edges[i] for i in range(6) if m0_mask[i]}
    # bunched outcome (mode 0 has 2 photons) is not a matching -> discarded
    assert _overlay_counts([2, 0, 0, 0, 0, 0], edges, m0, 4) == (False, 0, 0)
    # clicking exactly the two M_0 edges -> two shared (length-1) paths, no loop
    m0_modes = [i for i in range(6) if m0_mask[i]]
    key = [1 if i in m0_modes else 0 for i in range(6)]
    assert _overlay_counts(key, edges, m0, 4) == (True, 0, 2)


def test_photonic_graph_observable_and_capture():
    from model.photonic import PhotonicTeacher, score_from_distribution

    X = sample_X(12, 5, seed=0)
    t = PhotonicTeacher(m=6, k=2, n_features=5, observable="loop_path_loop", seed=3,
                        n_vertices=4)                       # no __L/__P suffix -> keep all
    t.enable_distribution_capture()
    soft = t(X)
    assert soft.shape == (12, 1) and torch.isfinite(soft).all()
    assert float(soft.min()) >= 0.0                         # a mean loop count is non-negative

    # the saved distribution re-scores exactly, and under a *different* base
    dist = t.captured_distributions()
    same = score_from_distribution(dist, "loop_path_loop", n_vertices=4)
    assert torch.allclose(torch.tensor(same), soft.squeeze(-1), atol=1e-6)
    par = score_from_distribution(dist, "loop_path_parity", n_vertices=4)
    assert par.shape == (12,)

    # same seed -> identical teacher
    assert torch.allclose(soft, PhotonicTeacher(
        m=6, k=2, n_features=5, observable="loop_path_loop", seed=3, n_vertices=4)(X))


def test_graph_observable_string_encoding():
    from model.photonic import parse_graph_observable, resolve_graph_spec, PhotonicTeacher

    # __L/__P suffixes encode the selection; a missing segment stays None (keep-all)
    assert parse_graph_observable("loop_path_parity__L0-1__P2-3") == (True, "parity", [0, 1], [2, 3])
    assert parse_graph_observable("loop_path_loop__P2-4") == (True, "loop", None, [2, 4])
    assert parse_graph_observable("loop_path_majority__L__P") == (True, "majority", [], [])
    assert parse_graph_observable("parity") == (False, "parity", None, None)
    # string vars are authoritative; an unspecified dim falls back to the passed override
    assert resolve_graph_spec("loop_path_loop__P2-4", [9], [9]) == ("loop", [9], [2, 4])

    # the __L/__P order is irrelevant: same selection -> identical teacher output
    X = sample_X(10, 5, seed=1)
    lp = PhotonicTeacher(m=6, k=2, n_features=5, observable="loop_path_parity__L0-1__P2",
                         seed=3, n_vertices=4)(X)
    pl = PhotonicTeacher(m=6, k=2, n_features=5, observable="loop_path_parity__P2__L0-1",
                         seed=3, n_vertices=4)(X)
    assert torch.allclose(lp, pl)