"""Tests for the min-degree scaling driver (learn.scaling)."""

from __future__ import annotations

from learn.scaling import (
    _summarize_basis, aggregate, load_results, parse_group, run_scaling, save_results,
)


def _base_config(tmp_path):
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "name: T\n"
        "problem: {m: 6, k: 3, observable: parity}\n"       # fields overridden per sweep/average
        "generation: {generator: analytical, size: 900}\n"
        "split: {test_fraction: 0.25, split_seed: 0}\n"
        "seeds: {sample_seed: 1, teacher_seed: 1}\n"
    )
    return cfg


def test_parse_group_zips_covarying_fields():
    points, fields = parse_group(["problem.m=4,6,8", "problem.k=2,3,4"])
    assert fields == ["problem.m", "problem.k"]
    assert points == [{"problem.m": 4, "problem.k": 2}, {"problem.m": 6, "problem.k": 3},
                      {"problem.m": 8, "problem.k": 4}]
    assert parse_group(None) == ([{}], [])                   # no group -> one no-override point

    single, _ = parse_group(["problem.graph_seed=1,2,3"])
    assert [p["problem.graph_seed"] for p in single] == [1, 2, 3]

    try:
        parse_group(["problem.m=4,6", "problem.k=2,3,4"])     # unequal lengths -> error
        assert False
    except ValueError:
        pass


def test_summarize_basis_min_degree_and_capped():
    reached = [{"degree": 1, "test_r2": 0.4}, {"degree": 2, "test_r2": 0.95}]
    assert _summarize_basis(reached, 0.9) == {"min_degree": 2, "r2_at_min": 0.95, "capped": False}

    capped = [{"degree": 1, "test_r2": 0.2}, {"degree": 2, "test_r2": float("nan")}]
    out = _summarize_basis(capped, 0.9)
    assert out["min_degree"] is None and out["capped"] is True

    short = [{"degree": 1, "test_r2": 0.1}, {"degree": 2, "test_r2": 0.3}]
    assert _summarize_basis(short, 0.9) == {"min_degree": None, "r2_at_min": None, "capped": False}


def test_run_scaling_generic_fields_and_roundtrip(tmp_path):
    cfg = _base_config(tmp_path)
    sweep = [{"problem.m": 4, "problem.k": 2}, {"problem.m": 6, "problem.k": 3}]
    average = [{"seeds.teacher_seed": 1}, {"seeds.teacher_seed": 2}]
    payload = run_scaling(cfg, sweep, average, x_field="problem.m", bases=["monomial", "fourier"],
                          threshold=0.3, max_degree=3, n_fit=400, n_test=200, dataset_root=tmp_path)

    assert payload["meta"]["x_field"] == "problem.m" and payload["meta"]["x_label"] == "m"
    assert len(payload["runs"]) == 4                          # 2 sweep x 2 average
    assert {r["x"] for r in payload["runs"]} == {4, 6}
    for run in payload["runs"]:
        assert set(run["per_basis"]) == {"monomial", "fourier"}

    out = save_results(payload, str(tmp_path / "r.json"))
    assert load_results(out)["runs"] == payload["runs"]       # JSON round-trip (nan cleaned)


def test_run_scaling_fixed_m_vary_other_field(tmp_path):
    cfg = _base_config(tmp_path)
    # fix m via base config; sweep k only, average over teacher_seed
    payload = run_scaling(cfg, [{"problem.k": 2}, {"problem.k": 3}],
                          [{"seeds.teacher_seed": 1}], x_field="problem.k",
                          threshold=0.3, max_degree=3, n_fit=400, n_test=200, dataset_root=tmp_path)
    agg = aggregate(payload)
    assert payload["meta"]["x_label"] == "k"
    for b in ("monomial", "fourier"):
        assert {agg[b][i]["x"] for i in agg[b]} == {2, 3}
        for i in agg[b]:
            a = agg[b][i]
            assert a["n_total"] == 1
            assert a["mean"] != a["mean"] or a["mean"] >= 1.0  # nan (none) or a real degree
