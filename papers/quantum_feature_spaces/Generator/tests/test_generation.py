"""Tests for the generation stage / shared pipeline.

Uses the fast `analytical` teacher (no quantum simulation) so the suite runs in
well under a second.
"""

from __future__ import annotations

import copy

import torch

from Generator import (
    ExperimentConfig,
    GenerationConfig,
    ProblemConfig,
    SeedConfig,
    SplitConfig,
    compute_hash,
    generate,
    load_raw,
    load_split,
    prepare,
)
from Generator.prepare import derive_confidence, derive_labels


def make_cfg(generator="analytical", **gen_overrides) -> ExperimentConfig:
    """Small, fast config (analytical teacher by default)."""
    cfg = ExperimentConfig(
        problem=ProblemConfig(m=4, k=3, observable="parity"),
        generation=GenerationConfig(generator=generator, size=200),
        split=SplitConfig(test_fraction=0.25, split_seed=0),
        seeds=SeedConfig(sample_seed=42, teacher_seed=42),
    )
    for key, val in gen_overrides.items():
        setattr(cfg.generation, key, val)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# round-trip + determinism
# --------------------------------------------------------------------------- #

def test_round_trip(tmp_path):
    cfg = make_cfg()
    path = generate(cfg, out_root=tmp_path)
    X, soft, meta = load_raw(path)
    assert X.shape[0] == cfg.generation.size
    assert soft.shape[0] == cfg.generation.size
    assert meta["size"] == cfg.generation.size
    assert {"hash", "format_version", "sample_seed", "teacher_seed", "spec"} <= set(meta)
    assert (tmp_path / path.name / "teacher.pt").exists()


def test_determinism_same_config(tmp_path):
    a = generate(make_cfg(), out_root=tmp_path / "a")
    b = generate(make_cfg(), out_root=tmp_path / "b")
    Xa, softa, ma = load_raw(a)
    Xb, softb, mb = load_raw(b)
    assert ma["hash"] == mb["hash"]
    assert torch.equal(Xa, Xb)
    assert torch.equal(softa, softb)


# --------------------------------------------------------------------------- #
# hash scope
# --------------------------------------------------------------------------- #

def test_hash_excludes_load_time_and_size():
    base = make_cfg()
    assert compute_hash(base) == compute_hash(make_cfg(size=999))
    other = copy.deepcopy(base)
    other.split.split_seed = 123       # the partition is not part of dataset identity
    other.split.test_fraction = 0.5
    assert compute_hash(other) == compute_hash(base)


def test_hash_sensitive_to_generation_fields():
    base = make_cfg()
    changed = copy.deepcopy(base)
    changed.seeds.teacher_seed = 43
    assert compute_hash(changed) != compute_hash(base)
    changed2 = copy.deepcopy(base)
    changed2.problem.k = 4
    assert compute_hash(changed2) != compute_hash(base)
    # sample_seed (the X points) also affects the data -> the hash
    changed3 = copy.deepcopy(base)
    changed3.seeds.sample_seed = 999
    assert compute_hash(changed3) != compute_hash(base)


def test_hash_includes_model_spec():
    # photonic's observable lives in its hash_spec -> changes the artifact identity.
    # (compute_hash only reads config, so no Merlin construction happens here.)
    a = make_cfg(generator="photonic_quantum")
    a.problem.observable = "parity"
    b = copy.deepcopy(a)
    b.problem.observable = "majority"  # m=4 is even, valid
    assert compute_hash(a) != compute_hash(b)


# --------------------------------------------------------------------------- #
# prefix stability + extend
# --------------------------------------------------------------------------- #

def test_prefix_stability_and_extend(tmp_path):
    n = 150
    small = generate(make_cfg(size=n), out_root=tmp_path)
    X_small, soft_small, _ = load_raw(small)

    big = generate(make_cfg(size=2 * n), out_root=tmp_path)
    X_big, soft_big, meta_big = load_raw(big)

    assert big == small  # same artifact directory (size excluded from hash)
    assert meta_big["size"] == 2 * n
    assert torch.equal(X_big[:n], X_small)
    assert torch.equal(soft_big[:n], soft_small)


def test_cache_hit_when_large_enough(tmp_path, capsys):
    generate(make_cfg(size=200), out_root=tmp_path)
    capsys.readouterr()
    generate(make_cfg(size=120), out_root=tmp_path)  # smaller -> reuse
    assert "cache hit" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# labelling + prepare determinism
# --------------------------------------------------------------------------- #

def test_prepare_determinism(tmp_path):
    path = generate(make_cfg(), out_root=tmp_path)
    X, soft, _ = load_raw(path)
    kw = dict(min_margin=0.1, balanced=True, split_seed=0)
    X1, y1, s1 = prepare(X, soft, **kw)
    X2, y2, s2 = prepare(X, soft, **kw)
    assert torch.equal(X1, X2) and torch.equal(y1, y2) and torch.equal(s1, s2)
    assert int((y1 == 0).sum()) == int((y1 == 1).sum())  # balanced


def test_margin_filter_monotone(tmp_path):
    path = generate(make_cfg(), out_root=tmp_path)
    X, soft, _ = load_raw(path)
    n_lo = prepare(X, soft, min_margin=0.0, balanced=False, split_seed=0)[0].shape[0]
    n_hi = prepare(X, soft, min_margin=0.5, balanced=False, split_seed=0)[0].shape[0]
    assert n_hi <= n_lo


def test_labels_inferred_from_shape(tmp_path):
    path = generate(make_cfg(), out_root=tmp_path)
    _, soft, _ = load_raw(path)            # analytical -> (N, 1) score
    assert soft.shape[-1] == 1
    assert torch.equal(derive_labels(soft), (soft[:, 0] >= 0).long())
    assert derive_confidence(soft).min() >= 0.0


# --------------------------------------------------------------------------- #
# the cross-model fairness guarantee
# --------------------------------------------------------------------------- #

def test_identical_X_across_loads(tmp_path):
    path = generate(make_cfg(), out_root=tmp_path)
    kw = dict(test_fraction=0.25, split_seed=0)
    s1 = load_split(path, **kw)
    s2 = load_split(path, **kw)
    assert torch.equal(s1.X_train, s2.X_train)
    assert torch.equal(s1.X_test, s2.X_test)
    assert s1.X_test.shape[0] > 0


def test_split_covers_full_pool_and_indices_align(tmp_path):
    # load_split discards nothing: train + test = the full pool, and the indices
    # map back into the raw load_raw order (so embeddings can be sliced by them).
    path = generate(make_cfg(size=200), out_root=tmp_path)
    X, soft, _ = load_raw(path)
    s = load_split(path, test_fraction=0.25, split_seed=0)
    assert s.X_train.shape[0] + s.X_test.shape[0] == X.shape[0]
    assert torch.equal(X[s.train_idx], s.X_train)
    assert torch.equal(X[s.test_idx], s.X_test)