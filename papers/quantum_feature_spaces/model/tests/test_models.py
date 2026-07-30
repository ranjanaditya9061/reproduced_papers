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


def test_photonic_nonlinear_observables_run_and_are_deterministic():
    """sq_<base> and pairprod (degree-2 in the probabilities) run and are seed-deterministic."""
    from model.photonic import PhotonicTeacher

    X = sample_X(16, 3, seed=0)
    for obs in ("sq_parity", "sq_bunching", "pairprod"):
        a = PhotonicTeacher(m=4, k=2, n_features=3, observable=obs, seed=2)(X)
        b = PhotonicTeacher(m=4, k=2, n_features=3, observable=obs, seed=2)(X)
        assert a.shape == (16, 1)
        assert torch.isfinite(a).all()
        assert torch.allclose(a, b)                 # same seed -> identical teacher


def test_photonic_pairprod_is_genuinely_nonlinear():
    """pairprod = p^T K p is not a linear functional probs @ v: its value on the average of two
    distributions differs from the average of its two values (a linear functional would not)."""
    from model.photonic import PhotonicTeacher, pairprod_kernel

    t = PhotonicTeacher(m=4, k=2, n_features=3, observable="pairprod", seed=2)
    keys = t._fock_keys
    n = len(keys)
    K = torch.tensor(pairprod_kernel(keys, 4), dtype=torch.float32)

    # Two arbitrary normalised distributions and their mean.
    p = torch.zeros(n); p[0] = 0.7; p[1] = 0.3
    q = torch.zeros(n); q[2 % n] = 0.4; q[3 % n] = 0.6
    quad = lambda d: float((d @ K * d).sum())
    mid = quad((p + q) / 2)
    avg = (quad(p) + quad(q)) / 2
    assert abs(mid - avg) > 1e-6                     # strict convexity -> not linear


def test_photonic_sq_offline_matches_forward():
    """score_from_distribution reproduces the teacher's sq_<base> / pairprod forward score."""
    import numpy as np

    from model.photonic import PhotonicTeacher, score_from_distribution

    X = sample_X(12, 3, seed=1)
    for obs in ("sq_parity", "pairprod"):
        t = PhotonicTeacher(m=4, k=2, n_features=3, observable=obs, seed=3)
        t.enable_distribution_capture()
        online = t(X).squeeze(-1).numpy()
        offline = score_from_distribution(t.captured_distributions(), obs)
        assert np.allclose(online, offline, atol=1e-5)


def test_squared_observable_is_quadratic_with_diagonal_kernel():
    """sq_<base> is p^T K p with K = diag(<base>); it just stores the diagonal instead of K."""
    import numpy as np

    from model.photonic import PhotonicTeacher, QuadraticObservable, SquaredObservable

    t = PhotonicTeacher(m=4, k=2, n_features=3, observable="sq_parity", seed=2)
    assert isinstance(t.obs, SquaredObservable)
    assert isinstance(t.obs, QuadraticObservable)            # the diagonal special case

    v = t.obs.score_vec
    dense = QuadraticObservable(np.diag(v.numpy()))
    probs = t.layer.forward(sample_X(8, 3, seed=0))
    assert torch.allclose(t.obs.score(probs), dense.score(probs), atol=1e-7)
    assert torch.allclose(t.obs.kernel_matrix(), dense.kernel_matrix(), atol=1e-7)
    assert t.obs.score_vec.shape == v.shape                  # (n_fock,), not (n_fock, n_fock)


def test_entropy_observable_is_negative_entropy_and_rescores():
    """ent = Σ p log p = -H(p); ent_<base> weights each surprisal term by <base>."""
    import numpy as np

    from model.photonic import EntropyWeightedObservable, PhotonicTeacher, score_from_distribution

    n = 12
    p = torch.rand(5, n)
    p = p / p.sum(dim=1, keepdim=True)
    plain = EntropyWeightedObservable()
    H = -(p * torch.log(p)).sum(dim=1)
    assert torch.allclose(plain.score(p), -H, atol=1e-6)     # unweighted -> -H(p)
    assert (plain.score(p) <= 0).all()                       # p log p <= 0 termwise

    # p log p -> 0 as p -> 0: a zero-probability outcome must contribute exactly 0, not -inf
    spike = torch.zeros(1, n)
    spike[0, 0] = 1.0
    assert float(plain.score(spike)) == 0.0                  # a point mass has zero entropy
    uniform = torch.full((1, n), 1.0 / n)
    assert abs(float(plain.score(uniform)) + math.log(n)) < 1e-5

    # on the real teacher: signed base -> mixed sign, and it re-scores offline exactly
    X = sample_X(12, 3, seed=1)
    for obs, signed in (("ent", False), ("ent_parity", True)):
        t = PhotonicTeacher(m=4, k=2, n_features=3, observable=obs, seed=3)
        t.enable_distribution_capture()
        online = t(X)
        assert online.shape == (12, 1) and torch.isfinite(online).all()
        assert (online <= 0).all() or signed                 # unweighted ent is non-positive
        offline = score_from_distribution(t.captured_distributions(), obs)
        assert np.allclose(online.squeeze(-1).numpy(), offline, atol=1e-6)


def test_cross_entropy_observable_scores_against_x_zero_reference():
    """xent = Σ_n p_x(n) log q(n) with q the circuit's output at x = 0 -- linear in p."""
    import numpy as np

    from model.photonic import LinearObservable, PhotonicTeacher, score_from_distribution

    t = PhotonicTeacher(m=4, k=2, n_features=3, observable="xent", seed=3)
    q = t.exact_probs_at_zero()
    assert q.shape == (len(t._fock_keys),)
    assert abs(float(q.sum()) - 1.0) < 1e-5                  # q is a distribution
    assert torch.allclose(q, t.layer.forward(torch.zeros(1, 3))[0])

    X = sample_X(12, 3, seed=1)
    probs = t.layer.forward(X)
    manual = (probs * torch.log(q.clamp(min=1e-12))).sum(dim=1)
    assert torch.allclose(t(X).squeeze(-1), manual, atol=1e-5)

    # q is fixed, so the score is LINEAR in p -- unlike pairprod it is not strictly convex
    assert isinstance(t.obs, LinearObservable)
    p, r = probs[0:1], probs[1:2]
    mid = float(t.obs.score((p + r) / 2))
    avg = (float(t.obs.score(p)) + float(t.obs.score(r))) / 2
    assert abs(mid - avg) < 1e-5

    # KL(p || q) = ent(x) - xent(x) >= 0, and vanishes at x = 0 where p == q
    ent = PhotonicTeacher(m=4, k=2, n_features=3, observable="ent", seed=3)
    assert bool((ent(X) - t(X) >= -1e-5).all())               # Gibbs' inequality
    zero = torch.zeros(1, 3)
    assert abs(float(ent(zero)) - float(t(zero))) < 1e-5      # p == q -> KL == 0

    # q is not persisted, so an offline re-score must be handed it back
    t.enable_distribution_capture()
    online = t(X).squeeze(-1).numpy()
    dist = t.captured_distributions()
    assert np.allclose(online, score_from_distribution(dist, "xent", reference_probs=q), atol=1e-6)
    try:
        score_from_distribution(dist, "xent")
        raise AssertionError("xent re-score without reference_probs should raise")
    except ValueError:
        pass


def test_oscillatory_observable_is_damped_and_bounded():
    """osc = Σ_n p(n) sin(1/(p(n)+eps)); the p prefactor bounds it and kills the p=0 terms."""
    import numpy as np

    from model.photonic import (EntropyWeightedObservable, OSC_EPS, OscillatoryObservable,
                                PhotonicTeacher, PointwiseObservable, score_from_distribution)

    # osc and ent are the same shape -- Σ v·φ(p) -- differing only in φ
    assert isinstance(OscillatoryObservable(), PointwiseObservable)
    assert isinstance(EntropyWeightedObservable(), PointwiseObservable)

    osc = OscillatoryObservable()
    zeros = torch.zeros(2, 7)
    assert float(osc.transform(zeros).abs().max()) == 0.0    # φ(0) = 0 exactly, no singularity
    p = torch.rand(4, 20)
    p = p / p.sum(dim=1, keepdim=True)
    assert bool((osc.transform(p).abs() <= p + 1e-7).all())  # |φ(p)| <= p
    assert float(osc.score(p).abs().max()) <= 1.0            # so |O| <= Σ p = 1

    X = sample_X(12, 3, seed=1)
    t = PhotonicTeacher(m=4, k=2, n_features=3, observable="osc", seed=3)
    t.enable_distribution_capture()
    online = t(X)
    probs = t.layer.forward(X)
    manual = (probs * torch.sin(1.0 / (probs + OSC_EPS))).sum(dim=1)
    assert torch.allclose(online.squeeze(-1), manual, atol=1e-6)
    assert torch.isfinite(online).all() and float(online.abs().max()) <= 1.0

    # weighted by a base, and re-scorable offline with no extra knobs
    offline = score_from_distribution(t.captured_distributions(), "osc")
    assert np.allclose(online.squeeze(-1).numpy(), offline, atol=1e-6)
    w = PhotonicTeacher(m=4, k=2, n_features=3, observable="osc_parity", seed=3)(X)
    assert w.shape == (12, 1) and torch.isfinite(w).all()


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