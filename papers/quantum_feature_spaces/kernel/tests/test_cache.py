"""Tests for gram_from_cache (the pure cache consumer; storage lives in embedding/).

A cached blob holds a feature matrix in ``blob["data"]``; gram_from_cache RBFs it
(optionally selecting train/test row subsets) with no feature recompute.
"""

from __future__ import annotations

import math

import torch

from kernel import RBFGram, gram_from_cache


def _F(n=16, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 2 * math.pi * torch.rand(n, d, generator=g)


def test_blob_reconstructs_full_gram():
    F = _F()
    blob = {"data": F}
    assert torch.allclose(gram_from_cache(blob), RBFGram().gram_from_features(F), atol=1e-5)


def test_blob_slices_cross_block():
    F = _F()
    blob = {"data": F}
    tr, te = torch.arange(0, 10), torch.arange(10, 16)
    # gram_from_cache selects the rows, then RBFs the idx_a x idx_b cross-block.
    expected = RBFGram().gram_from_features(F[tr], F[te])
    got = gram_from_cache(blob, idx_a=tr, idx_b=te)
    assert got.shape == (10, 6)
    assert torch.allclose(got, expected, atol=1e-5)
