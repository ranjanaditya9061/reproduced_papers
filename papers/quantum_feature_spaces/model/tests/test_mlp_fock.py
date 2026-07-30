"""Tests for the classical control teacher (:mod:`model.mlp_fock`).

The point of ``mlp_fock`` is to be identical to :class:`model.photonic.PhotonicTeacher` in
everything but the map, so these check exactly that: same outcome basis, same scorers, and -- the
one that guards the experiment -- matching ``p`` statistics, since ``osc``'s behaviour is driven by
the small-``p`` tail and a mismatched tail would rig the comparison.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
import torch

from model import TEACHERS, sample_X
from model.mlp_fock import MlpFockTeacher, fock_keys


def test_registered_and_fock_basis_matches_photonic_dimension():
    assert "mlp_fock" in TEACHERS
    for m, k in ((4, 2), (6, 3), (8, 4)):
        keys = fock_keys(m, k)
        assert len(keys) == comb(m + k - 1, k)          # stars and bars
        assert len({tuple(x) for x in keys}) == len(keys)
        assert all(sum(key) == k and len(key) == m for key in keys)


def test_output_is_a_normalised_distribution_and_deterministic():
    t = MlpFockTeacher(m=6, k=3, n_features=5, observable="parity", seed=3)
    X = sample_X(32, 5, seed=0)
    p = t.probs(X)
    assert p.shape == (32, comb(8, 3))
    assert bool((p >= 0).all())
    assert torch.allclose(p.sum(dim=1), torch.ones(32), atol=1e-5)

    soft = t(X)
    assert soft.shape == (32, 1) and torch.isfinite(soft).all()
    same = MlpFockTeacher(m=6, k=3, n_features=5, observable="parity", seed=3)(X)
    assert torch.allclose(soft, same)                   # same seed -> identical teacher
    other = MlpFockTeacher(m=6, k=3, n_features=5, observable="parity", seed=4)(X)
    assert not torch.allclose(soft, other)              # and the seed actually matters


def test_scores_through_the_same_observable_code_as_photonic():
    """Any registry observable runs on the classical distribution, with identical semantics."""
    X = sample_X(16, 5, seed=1)
    for obs in ("parity", "majority", "bunching", "n_first", "max_prob",
                "prod_parity_consecutive", "sq_parity", "ent", "ent_parity", "osc", "osc_parity",
                "xent", "pairprod"):
        t = MlpFockTeacher(m=6, k=3, n_features=5, observable=obs, seed=3)
        soft = t(X)
        assert soft.shape == (16, 1), obs
        assert torch.isfinite(soft).all(), obs

    # the scorer really is the shared one: hand-score a distribution and compare
    t = MlpFockTeacher(m=6, k=3, n_features=5, observable="parity", seed=3)
    p = t.probs(X)
    assert torch.allclose(t(X).squeeze(-1), p @ t.score_vec, atol=1e-6)


def test_p_statistics_match_the_photonic_teacher():
    """The confound guard: classical and quantum p must share their dynamic range / entropy.

    ``osc`` lives in the small-``p`` tail, so if the two platforms' distributions had different
    tails the comparison would measure that instead of the map.  The complex-amplitude
    normalisation is chosen to land on the same Porter-Thomas law the Haar photonic circuit gives;
    a plain softmax would fail this test (it has no mass below 1e-4 at all).
    """
    from model.photonic import PhotonicTeacher

    m, k, nf = 6, 3, comb(8, 3)
    X = sample_X(400, m - 1, seed=0)

    pq = PhotonicTeacher(m=m, k=k, n_features=m - 1, observable="parity", seed=3).layer.forward(X)
    pc = MlpFockTeacher(m=m, k=k, n_features=m - 1, observable="parity", seed=3).probs(X)

    def summary(p):
        p = p.double()
        p = p / p.sum(dim=1, keepdim=True)
        ent = -(p * torch.log(p.clamp(min=1e-30))).sum(dim=1).mean()
        return (float(torch.log10(p.clamp(min=1e-30)).median()), float(ent),
                float((p < 1e-4).double().mean()))

    med_q, ent_q, tail_q = summary(pq)
    med_c, ent_c, tail_c = summary(pc)

    assert abs(med_c - med_q) < 0.5                       # median log10 p within half a decade
    assert abs(ent_c - ent_q) < 0.30                      # entropy within 0.3 nat of ln(56)=4.03
    assert ent_c < math.log(nf)                            # and neither is uniform
    assert tail_c > 1e-3                                   # a real small-p tail exists (softmax: 0)
    assert abs(tail_c - tail_q) < 0.05


def test_shot_sampling_and_offline_rescore():
    """nsample>0 gives a finite-shot empirical distribution; captures re-score exactly."""
    from model.photonic import score_from_distribution

    X = sample_X(12, 5, seed=2)
    t = MlpFockTeacher(m=6, k=3, n_features=5, observable="osc", seed=3)
    t.enable_distribution_capture()
    online = t(X).squeeze(-1).numpy()
    offline = score_from_distribution(t.captured_distributions(), "osc")
    assert np.allclose(online, offline, atol=1e-6)

    shots = MlpFockTeacher(m=6, k=3, n_features=5, observable="osc", seed=3, nsample=500)
    ps = shots.probs(X)
    sampled = shots._shot_sample(ps)
    assert torch.allclose(sampled.sum(dim=1), torch.ones(12), atol=1e-5)
    assert bool(((sampled * 500).round() - sampled * 500).abs().max() < 1e-3)   # integer counts
