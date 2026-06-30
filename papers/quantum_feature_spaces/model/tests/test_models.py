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