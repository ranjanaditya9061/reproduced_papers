"""Tests for the learn stage (RBF-SVR regressing the teacher's soft output, R^2)."""

from __future__ import annotations

from Generator import generate, load_config
from learn import run_svm


def _setup(tmp_path):
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(
        "problem: {m: 4, k: 3, observable: parity}\n"
        "generation: {generator: analytical, size: 800}\n"
        "split: {test_fraction: 0.25, split_seed: 0}\n"
        "seeds: {sample_seed: 1, teacher_seed: 1}\n"
    )
    generate(load_config(data_cfg), out_root=tmp_path)

    embed_cfg = tmp_path / "embed.yaml"
    embed_cfg.write_text(
        f"dataset: {data_cfg.as_posix()}\n"
        "embeddings:\n  - {type: rbf}\n  - {type: fourier_rbf, fourier_order: 3}\n"
    )
    return embed_cfg


def test_run_svm_regresses_soft(tmp_path):
    embed_cfg = _setup(tmp_path)
    rows, meta = run_svm(embed_cfg, n_train=400, n_test=200,
                         embeddings_root=tmp_path / "emb", dataset_root=tmp_path)

    assert {r["name"] for r in rows} == {"rbf", "fourier_rbf"}
    for r in rows:
        assert r["n_train"] == 400 and r["n_test"] == 200
        assert r["test_r2"] <= 1.0 + 1e-9          # R^2 is upper-bounded by 1
        assert r["train_r2"] <= 1.0 + 1e-9

    # the analytical teacher's soft output is a sum of sines -> the Fourier embedding
    # should regress it well (high R^2); deterministic given the fixed seeds.
    fourier = next(r for r in rows if r["name"] == "fourier_rbf")
    assert fourier["test_r2"] > 0.5