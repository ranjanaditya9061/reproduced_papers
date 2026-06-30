"""Tests for the RBF Gram -- the whole kernel stage now (feature-agnostic)."""

from __future__ import annotations

import math

import torch

from kernel import RBFGram, gram_from_features


def _F(n=24, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 2 * math.pi * torch.rand(n, d, generator=g)


def _is_psd(K, tol=1e-4):
    return bool((torch.linalg.eigvalsh(0.5 * (K + K.T)) >= -tol).all())


def test_gram_properties():
    F = _F()
    K = RBFGram().gram_from_features(F)
    assert K.shape == (24, 24)
    assert torch.allclose(K, K.T, atol=1e-6)
    assert torch.allclose(torch.diag(K), torch.ones(24), atol=1e-6)  # exp(0)=1
    assert float(K.min()) >= 0.0 and float(K.max()) <= 1.0 + 1e-6
    assert _is_psd(K)


def test_cross_gram_shape_and_gamma_reuse():
    Ftr, Fte = _F(20, 5, seed=1), _F(7, 5, seed=2)
    rbf = RBFGram()
    Ktr = rbf.gram_from_features(Ftr)          # fits gamma on the train block
    g = rbf._fitted_gamma
    Kcross = rbf.gram_from_features(Ftr, Fte)
    assert Ktr.shape == (20, 20)
    assert Kcross.shape == (20, 7)
    assert g is not None and rbf._fitted_gamma == g   # reused, not refit


def test_explicit_gamma_matches_formula():
    F = _F(d=4)
    K = gram_from_features(F, gamma=0.5)
    sq = torch.cdist(F, F) ** 2
    assert torch.allclose(K, torch.exp(-0.5 * sq), atol=1e-6)
