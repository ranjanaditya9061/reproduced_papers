"""Tests for the explicit-feature linear-regression learnability curve (learn.linreg)."""

from __future__ import annotations

import math

import numpy as np

from learn.linreg import (
    FEATURE_BASES, _cos_base, _expanded_dim, _fit_ridge_gcv, _fourier_base, _monomial_base,
    _poly_expand, run_linreg,
)


def test_feature_bases_registry():
    assert set(FEATURE_BASES) >= {"monomial", "fourier", "cos"}


def test_expanded_dim_matches_polynomialfeatures():
    from sklearn.preprocessing import PolynomialFeatures

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    for d in (1, 2, 3):
        cols = PolynomialFeatures(degree=d, include_bias=False).fit_transform(X).shape[1]
        assert _expanded_dim(4, d) == cols


def test_base_builders_shapes():
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.uniform(0, 2 * np.pi, (20, 3)), rng.uniform(0, 2 * np.pi, (8, 3))

    mtr, mte = _monomial_base(Xtr, Xte, n_features=3, fourier_order=3)
    assert mtr.shape == (20, 3) and mte.shape == (8, 3)          # degree-1 = the components

    ftr, fte = _fourier_base(Xtr, Xte, n_features=3, fourier_order=3)
    assert ftr.shape == (20, 2 * 3 * 3) and fte.shape == (8, 2 * 3 * 3)  # sin+cos [sin,cos]_{j<=3}

    ctr, cte = _cos_base(Xtr, Xte, n_features=3, fourier_order=3)
    assert ctr.shape == (20, 3 * 3) and cte.shape == (8, 3 * 3)   # cos-only [cos(j x)]_{j<=3}

    # cos-only order 1 is param-matched to the monomial base (one feature per component)
    c1, _ = _cos_base(Xtr, Xte, n_features=3, fourier_order=1)
    assert c1.shape == mtr.shape

    ptr, pte = _poly_expand(mtr, mte, 2)                        # cross terms appear at degree 2
    assert ptr.shape[1] == _expanded_dim(3, 2) == 9


def test_ridge_gcv_recovers_linear_target():
    rng = np.random.default_rng(2)
    Xtr, Xte = rng.normal(size=(400, 5)), rng.normal(size=(200, 5))
    w = rng.normal(size=5)
    ytr, yte = Xtr @ w, Xte @ w                                 # exactly linear -> degree 1 fits
    lambdas = np.geomspace(1e-6, 1e2, 20)
    train_r2, test_r2, lam, n_feat = _fit_ridge_gcv(Xtr, ytr, Xte, yte, lambdas=lambdas)
    assert test_r2 > 0.99 and n_feat == 5


def test_max_feat_guard_skips_and_flags(tmp_path):
    _write_analytical_config(tmp_path)
    # a tiny max_feat forces every degree>=2 point to be skipped (nan), not an OOM
    results, degrees = run_linreg(
        tmp_path, bases=["monomial"], degrees=[1, 2, 3], n_fit=300, n_test=150,
        max_feat=6, dataset_root=tmp_path)
    rows = next(iter(results.values()))["monomial"]
    assert not math.isnan(rows[0]["test_r2"])                   # degree 1 (<=6 feats) ran
    assert math.isnan(rows[1]["test_r2"]) and math.isnan(rows[2]["test_r2"])  # skipped


def test_run_linreg_end_to_end(tmp_path):
    _write_analytical_config(tmp_path)
    results, degrees = run_linreg(
        tmp_path, bases=["monomial", "fourier"], degrees=[1, 2], n_fit=400, n_test=200,
        dataset_root=tmp_path)
    assert degrees == [1, 2]
    (per_basis,) = results.values()
    assert set(per_basis) == {"monomial", "fourier"}
    for rows in per_basis.values():
        assert [r["degree"] for r in rows] == [1, 2]
        assert all(np.isfinite(r["test_r2"]) and np.isfinite(r["train_r2"]) for r in rows)


def _write_analytical_config(dirpath):
    (dirpath / "d.yaml").write_text(
        "name: T\n"
        "problem: {m: 4, k: 3, observable: parity}\n"
        "generation: {generator: analytical, size: 900}\n"
        "split: {test_fraction: 0.25, split_seed: 0}\n"
        "seeds: {sample_seed: 1, teacher_seed: 1}\n"
    )
