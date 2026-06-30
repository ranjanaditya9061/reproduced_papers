"""Tests for kernel-kernel comparison metrics + the end-to-end matrix."""

from __future__ import annotations

import math

import torch

from analyzer import (
    compare_kernels,
    kernel_alignment,
    kernel_kernel_matrix,
)
from kernel import RBFGram, gram_from_features
from model.mlp import fourier_features


def _X(n=20, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 2 * math.pi * torch.rand(n, d, generator=g)


# --- pure metric -------------------------------------------------------- #

def test_self_alignment_is_one():
    K = gram_from_features(_X())
    assert abs(kernel_alignment(K, K) - 1.0) < 1e-5


def test_alignment_symmetric_and_in_unit_interval():
    X = _X(seed=1)
    K1 = gram_from_features(X)
    K2 = gram_from_features(fourier_features(X, 3))
    a, b = kernel_alignment(K1, K2), kernel_alignment(K2, K1)
    assert abs(a - b) < 1e-6
    assert -1e-6 <= a <= 1.0 + 1e-6


def test_kernel_kernel_matrix_properties():
    X = _X(seed=2)
    grams = [
        gram_from_features(X),
        gram_from_features(fourier_features(X, 3)),
        RBFGram(gamma=0.5).gram_from_features(X),
    ]
    M = kernel_kernel_matrix(grams)
    assert M.shape == (3, 3)
    assert torch.allclose(M, M.T, atol=1e-6)
    assert torch.allclose(torch.diag(M), torch.ones(3), atol=1e-5)
    assert float(M.min()) >= -1e-5 and float(M.max()) <= 1.0 + 1e-5


# --- end-to-end on a generated dataset ---------------------------------- #

def _write_cfg(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "problem: {m: 4, k: 3, observable: parity}\n"
        "generation: {generator: analytical, size: 200}\n"
        "split: {test_fraction: 0.25, split_seed: 0}\n"
        "seeds: {sample_seed: 1, teacher_seed: 1}\n"
    )
    return cfg


def test_compare_kernels_end_to_end(tmp_path):
    from embedding import build_embedding
    from Generator import generate, load_config

    cfg_path = _write_cfg(tmp_path)
    dcfg = load_config(cfg_path)
    generate(dcfg, out_root=tmp_path)  # create the artifact

    embeddings = [
        build_embedding({"type": "rbf"}, dcfg),
        build_embedding({"type": "fourier_rbf", "fourier_order": 3}, dcfg),
    ]
    names, M = compare_kernels(cfg_path, embeddings, names=["rbf", "fourier"],
                               out_root=tmp_path)
    assert names == ["rbf", "fourier"]
    assert M.shape == (2, 2)
    assert torch.allclose(M, M.T, atol=1e-6)
    assert torch.allclose(torch.diag(M), torch.ones(2), atol=1e-5)


def test_compare_from_embeddings(tmp_path):
    from analyzer import compare_from_embeddings
    from Generator import generate, load_config

    cfg_path = _write_cfg(tmp_path)
    generate(load_config(cfg_path), out_root=tmp_path)
    embed_cfg = tmp_path / "embed.yaml"
    embed_cfg.write_text(
        f"dataset: {cfg_path.as_posix()}\n"
        "embeddings:\n  - {type: rbf}\n  - {type: fourier_rbf, fourier_order: 3}\n"
    )
    names, M = compare_from_embeddings(embed_cfg, embeddings_root=tmp_path / "emb",
                                       dataset_root=tmp_path)
    assert names == ["rbf", "fourier_rbf"]
    assert M.shape == (2, 2)
    assert torch.allclose(torch.diag(M), torch.ones(2), atol=1e-5)
