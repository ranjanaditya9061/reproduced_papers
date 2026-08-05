"""The load-bearing invariants, so the metrics can be trusted rather than hoped over.

    python -m pytest v2/tests -q

Every assertion here is one the design notes call out as the check that would catch a silent
wrong number.  In particular the two exactness tests -- ``psi`` against autograd, and
``eta in [0, 1]`` -- are what make analysis B's output believable: the bound is Cauchy-Schwarz on
the discrete distribution, so a violation is unambiguously a bug and never a sampling artifact.
"""

from __future__ import annotations

import math

import pytest
import torch

from v2.config import ExperimentConfig, ModelConfig, ProblemConfig
from v2.pipeline.artifact import load_meta
from v2.metrics.distribution import (conditional_sqrt_jacobian, finite_difference_jacobian,
                                     fisher_spectrum, input_fisher, phase_eigenvalue,
                                     probs_and_jacobian, project_physical, shared_support,
                                     spectrum_from_jacobian, sqrt_jacobian)
from v2.metrics.observable import (ade_summaries, efficiency, eta, fisher_at, influence_terms,
                                   project_vec, rho2_per_direction)
from v2.model import MODELS, build_model, sample_X
from v2.observable import ObservableContext, resolve_observable

OBSERVABLES = ["parity", "majority", "n_first", "bunching", "xent_parity", "single_output",
               "sq_parity", "pairprod", "ent", "ent_parity", "osc", "osc_parity",
               "prod_parity_consecutive", "connected_maxcc"]


def cfg_for(kind="photonic", m=6, k=3, **model_kw):
    return ExperimentConfig(problem=ProblemConfig(n_features=6, m=m, k=k),
                            model=ModelConfig(kind=kind, **model_kw))


@pytest.fixture(scope="module")
def setup():
    cfg = cfg_for()
    model = build_model(cfg)
    X = sample_X(8, 6, 42)
    model.probs(X[:1])              # photonic populates input_state lazily on first evaluation
    ctx = ObservableContext(m=6, k=3, keys=model.outcome_keys(), seed=42, graph_density=0.5,
                            input_state=model.input_state(),
                            reference_probs=model.probs_at_zero().numpy())
    return model, X, ctx


# --- the artifact must not depend on the readout ------------------------------------------------ #


def test_artifact_is_observable_independent(tmp_path):
    """Two observables on one circuit -> ONE dataset directory, TWO score files."""
    from v2.pipeline.generate import generate_exact
    from v2.pipeline.score import EXACT_SOURCE, load_soft

    cfg = cfg_for()
    cfg.generation.size = 16
    path = generate_exact(cfg, out_root=tmp_path / "datasets_v2")
    scores = tmp_path / "scores_v2"
    for name in ("parity", "ent"):
        load_soft(path, name, scores_root=scores)

    assert len(list((tmp_path / "datasets_v2").iterdir())) == 1
    assert len(list((scores / path.name.split("_")[-1] / EXACT_SOURCE).glob("*.pt"))) == 2


def test_shot_budget_is_not_part_of_the_circuit_identity(tmp_path):
    """The fix for the costliest defect: 20k and 30k must share ONE directory and ONE simulation.

    Previously ``nsample`` was hashed, so they were separate artifacts -- the second re-ran the exact
    simulation (76% of the work, bit-identical output), redrew from shot zero, and still did not
    produce an extension of the first.
    """
    from v2.pipeline.artifact import artifact_path
    from v2.pipeline.generate import generate_exact, generate_shots
    from v2.pipeline.shots import BLOCK, load_shots

    out_root, shots_root = tmp_path / "datasets_v2", tmp_path / "shots_v2"
    cfg = cfg_for()
    cfg.generation.size = 16
    generate_exact(cfg, out_root=out_root)
    exact_dir = artifact_path(cfg, build_model(cfg), out_root)

    cfg.generation.shots = 2 * BLOCK
    p2 = generate_shots(cfg, shots_root=shots_root)
    s2, m2 = load_shots(p2)

    cfg.generation.shots = 3 * BLOCK
    p3 = generate_shots(cfg, shots_root=shots_root)
    s3, m3 = load_shots(p3)

    assert p2 == p3                                        # same store: n_blocks is NOT hashed
    assert (m2["n_blocks"], m3["n_blocks"]) == (2, 3)
    assert sum(s2[0].values()) == 2 * BLOCK
    assert sum(s3[0].values()) == 3 * BLOCK                # exactly one block added
    # nested BY CONSTRUCTION: the stored counts are kept and added to, not redrawn, so every count
    # in the smaller budget survives in the larger one without the draw being reproducible.
    for a, b in zip(s2, s3):
        assert all(b.get(key, 0) >= n for key, n in a.items())
    # and the exact branch was simulated once, for both budgets
    assert len(list((tmp_path / "datasets_v2").iterdir())) == 1
    assert exact_dir.name.endswith(load_meta(exact_dir)["hash"])


def test_shot_draw_never_builds_the_distribution():
    """The separation, asserted: sampling must not call probs().

    Shots go through CliffordClifford2017, which samples the interferometer without forming a
    distribution.  A ``probs()`` call here would mean the branch had silently become a readout of the
    exact one again -- which is exactly what it looked like before.
    """
    model = build_model(cfg_for())
    calls = []
    original = type(model).probs
    type(model).probs = lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1]
    try:
        model.shot_counts(sample_X(2, 6, 42), blocks=[0], shot_seed=0)
    finally:
        type(model).probs = original
    assert calls == []


def test_shot_draw_returns_one_dict_per_requested_row():
    """The row contract: ``rows=`` picks which rows to draw and the result is aligned to *it*.

    Not padded out to the pool with empty rows -- that is what makes a pool extension a plain list
    concatenation in ``generate_shots``, with no index bookkeeping.
    """
    from v2.pipeline.shots import BLOCK

    model = build_model(cfg_for())
    X = sample_X(6, 6, 42)
    whole = model.shot_counts(X, blocks=[0], shot_seed=0)
    assert len(whole) == 6
    assert all(sum(row.values()) == BLOCK for row in whole)
    # keys are plain occupation tuples over m modes conserving k photons, discovered from the draw
    assert all(len(key) == 6 and sum(key) == 3 for row in whole for key in row)

    subset = model.shot_counts(X, blocks=[0], rows=[3, 5], shot_seed=0)
    assert len(subset) == 2
    assert all(sum(row.values()) == BLOCK for row in subset)


def test_shot_blocks_are_additive():
    """Integer counts, so extending a budget is ``Counter`` addition and 10k stays a prefix of 20k.

    Only the arithmetic is asserted, not bit-equality against a single two-block draw: exqalibur is
    seeded once per block and ``backend.samples`` draws in bulk, so a redraw is not reproducible.
    Extension does not need it to be -- the stored counts are kept and added to, never recomputed.
    """
    from v2.pipeline.shots import BLOCK, merge_shots

    model = build_model(cfg_for())
    X = sample_X(4, 6, 42)
    b0 = model.shot_counts(X, blocks=[0], shot_seed=0)
    b1 = model.shot_counts(X, blocks=[1], shot_seed=0)
    merged = merge_shots(b0, b1)

    assert all(isinstance(n, int) for row in merged for n in row.values())
    assert all(sum(row.values()) == 2 * BLOCK for row in merged)
    for row, a, b in zip(merged, b0, b1):
        assert set(row) == set(a) | set(b)                  # a new block may hit unseen outcomes
        assert all(row[key] == a.get(key, 0) + b.get(key, 0) for key in row)


def test_shot_seed_gives_independent_realisations_at_a_fixed_circuit():
    """Impossible before: the noise seed was ``model_seed + 13``, so a new realisation meant a new
    circuit.  Error bars on the SNR -> R^2 ceiling need exactly this."""
    model = build_model(cfg_for())
    X = sample_X(2, 6, 42)
    draws = [model.shot_counts(X, blocks=[0], shot_seed=s) for s in range(3)]
    assert draws[0] != draws[1] and draws[1] != draws[2]


def test_clifford_sampling_converges_to_the_exact_distribution():
    """Cross-validation across two independent code paths: CC2017 sampling vs SLOS/merlin all_prob.

    Total-variation distance falls as ``1/sqrt(S)`` -- measured 0.0255 / 0.0118 / 0.0060 at
    10k / 40k / 160k shots.
    """
    model = build_model(cfg_for())
    model.forward_batch = 0
    X = sample_X(2, 6, 42)
    exact = model.probs(X).double()
    index = {tuple(int(c) for c in key): i for i, key in enumerate(model.outcome_keys())}
    tvs = []
    for nb in (1, 4):
        rows = model.shot_counts(X, blocks=range(nb), shot_seed=0)
        emp = torch.zeros_like(exact)
        for i, row in enumerate(rows):                      # align onto the exact basis to compare
            for key, n in row.items():
                emp[i, index[key]] = float(n)
        emp = emp / emp.sum(1, keepdim=True)
        assert torch.allclose(emp.sum(1), torch.ones(2, dtype=torch.float64), atol=1e-6)
        tvs.append(float(0.5 * (emp - exact).abs().sum(1).mean()))
    assert tvs[0] < 0.06                                    # converging to the same distribution
    assert tvs[1] < tvs[0]                                  # and improving with shots


def test_noisy_scores_do_not_share_a_cache_key_with_exact(tmp_path):
    """One file standing for several label sets is the original bug in a new place."""
    from v2.pipeline.generate import generate_exact, generate_shots
    from v2.pipeline.score import EXACT_SOURCE, load_soft, score_path
    from v2.pipeline.shots import BLOCK

    roots = {"out_root": tmp_path / "datasets_v2", "shots_root": tmp_path / "shots_v2"}
    cfg = cfg_for()
    cfg.generation.size, cfg.generation.shots = 16, BLOCK
    path = generate_exact(cfg, out_root=roots["out_root"])
    sdir = generate_shots(cfg, shots_root=roots["shots_root"])
    scores = tmp_path / "scores_v2"

    from v2.pipeline.distribution import load_dist
    from v2.pipeline.score import context_for

    exact = load_soft(path, "parity", scores_root=scores)
    noisy = load_soft(path, "parity", scores_root=scores, shots_dir=sdir)
    assert not torch.equal(exact, noisy)
    assert score_path(path, "parity", context_for(load_dist(path)), scores, EXACT_SOURCE).exists()
    assert len(list(scores.rglob("*.pt"))) == 2             # two sources, two files


def test_circuit_spec_rejects_an_observable():
    from v2.pipeline.artifact import _check_spec
    with pytest.raises(ValueError, match="circuit_spec"):
        _check_spec({"observable": "parity"}, "photonic")


def test_parity_on_a_two_outcome_basis_is_the_signed_score():
    """The 2-outcome basis is what removes the need for a special-cased scalar path."""
    cfg = cfg_for(kind="analytical")
    model = build_model(cfg)
    X = sample_X(32, 6, 42)
    probs = model.probs(X)
    keys = model.outcome_keys()
    assert tuple(keys) == ((1, 0), (0, 1))
    # ctx.m is the OUTCOME BASIS's mode count (2 here), not the circuit's: parity over the first
    # ceil(2/2) = 1 mode gives v = (-1, +1), so probs @ v = p_1 - p_0 exactly.
    ctx = ObservableContext(m=2, k=1, keys=keys, seed=42)
    soft = resolve_observable("parity", ctx).score(probs)
    assert torch.allclose(probs[:, 1] - probs[:, 0], soft, atol=1e-6)


def test_parameter_counts_scale_as_2m2_minus_1():
    for m, k in [(6, 3), (8, 4), (10, 5)]:
        for kind in ("photonic", "fermion"):
            assert build_model(cfg_for(kind, m, k)).n_model_parameters() == 2 * m * m - 1
    # mlp_fock is capacity-unbounded by construction -- its head scales with n_fock, not with
    # n_features -- so it must grow with (m, k) and dwarf the circuit's count.  Measured
    # 35_328 -> 105_472 at m = 6 -> 8, i.e. ~3x, against the circuit's 71 -> 127.
    counts = [build_model(cfg_for("mlp_fock", m, k)).n_model_parameters() for m, k in [(6, 3), (8, 4)]]
    assert counts[1] > 2 * counts[0]
    assert counts[0] > 100 * (2 * 6 * 6 - 1)

    # quadratic_fock(param_matched) resolves a rank landing near the circuit's count.  With the
    # default fourier order the closest achievable is 90 against 71, so the bound is a factor,
    # not equality -- and the resolved rank is recorded in circuit_spec.
    matched = build_model(cfg_for("quadratic_fock", param_matched=True)).n_model_parameters()
    assert 0.5 * 71 < matched < 2.0 * 71


def test_n_features_is_a_study_invariant():
    with pytest.raises(ValueError, match="STUDY INVARIANT"):
        ExperimentConfig(problem=ProblemConfig(n_features=5)).validate()


# --- analysis A ---------------------------------------------------------------------------------- #


def test_fisher_is_symmetric_and_psd(setup):
    model, X, _ = setup
    F = input_fisher(model, X[:3])
    for Fi in F:
        assert torch.allclose(Fi, Fi.T, atol=1e-12)
        assert float(torch.linalg.eigvalsh(Fi).min()) > -1e-10


def test_jacobian_matches_finite_differences(setup):
    """Cross-check of the autograd path.  Limited by the model's float32 internals, so the
    tolerance is 1e-2 *relative*, not the 1e-5 an all-float64 model would give."""
    model, X, _ = setup
    _, dp = probs_and_jacobian(model, X[0])
    fd = finite_difference_jacobian(model, X[0], eps=1e-3)
    assert float((dp - fd).abs().max() / dp.abs().max()) < 1e-2


def test_probability_derivatives_sum_to_zero(setup):
    """``sum_n d_i p_n = 0`` -- the exact identity behind ``E_p[s] = 0``, on which every
    ``Cov(psi, s)`` in analysis B depends."""
    model, X, _ = setup
    _, dp = probs_and_jacobian(model, X[0])
    assert float(dp.sum(dim=0).abs().max()) < 1e-6


def test_svdvals_beats_gram_on_known_ground_truth():
    """The SVD-vs-Gram claim is only assertable against *known* singular values."""
    g = torch.Generator().manual_seed(0)
    sig = torch.tensor([1e0, 1e-2, 1e-4, 1e-5, 1e-7, 1e-8], dtype=torch.float64)
    A = torch.linalg.qr(torch.randn(40, 6, generator=g, dtype=torch.float64))[0]
    B = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))[0]
    J = A @ torch.diag(sig) @ B.T
    true = sig ** 2
    sv = torch.linalg.svdvals(J) ** 2
    ev = torch.linalg.eigvalsh(J.T @ J).flip(0).clamp(min=0)
    sv_err = float(((sv - true).abs() / true).max())
    ev_err = float(((ev - true).abs() / true).max())
    assert sv_err < 1e-8            # the Gram path loses the small end entirely
    assert ev_err > 1e-3
    assert ev_err > 1e4 * sv_err


def test_global_phase_mode_and_the_projection_invariant():
    """At ``n_f == m`` one exact null direction, the rest alive; at ``n_f < m`` none -- and the
    projected spectrum is ``n_f - 1`` dimensional at BOTH, which is what makes the sweep stackable."""
    X = sample_X(2, 6, 42)
    for m, k, has_null in [(6, 3, True), (8, 4, False)]:
        model = build_model(cfg_for("fermion", m, k, flavours=2))
        J = sqrt_jacobian(model, X[0], project=False)
        e = spectrum_from_jacobian(J)
        if has_null:
            assert float(e[-1] / e[0]) < 1e-10          # by RATIO, never matrix_rank
            assert float(e[-2] / e[0]) > 1e-6
            assert float(phase_eigenvalue(J)) < 1e-10
        else:
            assert float(e[-1] / e[0]) > 1e-6
            assert float(phase_eigenvalue(J)) > 1e-4    # genuinely informative, reported separately
        proj = spectrum_from_jacobian(project_physical(J))
        assert int((proj > 1e-10 * proj[0]).sum()) == 5


def test_conditioning_centering_is_exactly_the_rank_one_correction():
    """Omitting the centering overestimates ``F^(C)`` by ``(d log Z_C)(d log Z_C)^T``, exactly."""
    X = sample_X(4, 6, 42)
    boson, fermi = build_model(cfg_for("photonic")), build_model(cfg_for("fermion", flavours=1))
    mask = shared_support([boson, fermi], X[:2])
    assert int(mask.sum()) == math.comb(6, 3)           # 20 of 56: the collision-free sector

    x = X[0]
    Jc = conditional_sqrt_jacobian(boson, x, mask, project=False, center=True).double()
    Ju = conditional_sqrt_jacobian(boson, x, mask, project=False, center=False).double()
    p, dp = probs_and_jacobian(boson, x)
    pc, dpc = p[mask].double(), dp[mask].double()
    q = pc / pc.sum()
    s = dpc / pc.unsqueeze(1)
    dlogZ = (q.unsqueeze(1) * s).sum(0)
    resid = (Ju.T @ Ju - Jc.T @ Jc) - torch.outer(dlogZ, dlogZ)
    assert float(resid.abs().max()) < 1e-5 * float(torch.outer(dlogZ, dlogZ).abs().max().clamp(min=1e-12)) + 1e-6


def test_fermion_flavours_open_the_bunched_sector():
    """Exactly 0 bunched mass at ``r=1`` (Pauli), non-negligible at ``r>=2``: the contrast the
    flavoured mod exists to create, and what makes the shared-support comparison honest."""
    X = sample_X(4, 6, 42)
    keys = build_model(cfg_for("fermion", flavours=1)).outcome_keys()
    bunched = torch.tensor([max(kk) > 1 for kk in keys])
    masses = []
    for r in (1, 2, 3):
        probs = build_model(cfg_for("fermion", flavours=r)).probs(X)
        masses.append(float(probs[:, bunched].sum(dim=1).mean()))
    assert masses[0] < 1e-9
    assert masses[1] > 0.1
    assert masses[2] > masses[1]


# --- analysis B ---------------------------------------------------------------------------------- #


def test_influence_matches_autograd_on_p(setup):
    """One check that validates the entire efficiency framework."""
    _, _, ctx = setup
    model, X, _ = setup
    p, _, _ = fisher_at(model, X[0])
    pd = p.double()
    for name in OBSERVABLES:
        obs = resolve_observable(name, ctx).double()
        pp = pd.detach().clone().requires_grad_(True)
        psi_auto, = torch.autograd.grad(obs.score(pp.unsqueeze(0))[0], pp)
        psi = obs.influence(pd.unsqueeze(0))[0]
        assert float((psi - psi_auto).abs().max()) < 1e-12, name


def test_g_equals_the_gradient_of_the_label(setup):
    """``g_i = Cov(psi, s_i) = d<T>/dx_i`` -- verified against autograd through the CIRCUIT."""
    model, X, ctx = setup
    x = X[0]
    p, dp, _ = fisher_at(model, x)
    for name in ("parity", "ent", "sq_parity", "pairprod"):
        obs = resolve_observable(name, ctx)
        g, _ = influence_terms(obs, p, dp)
        xx = x.detach().clone().requires_grad_(True)
        auto, = torch.autograd.grad(obs.score(model._probs(xx.unsqueeze(0)))[0], xx)
        rel = float((g - project_vec(auto.double())).abs().max() / g.abs().max())
        assert rel < 1e-5, name


def test_eta_in_unit_interval_and_above_the_local_reading(setup):
    """Exact by Cauchy-Schwarz on the discrete distribution, so a violation is a bug -- and
    ``eta >= max_i rho^2`` (the inequality only, never strictness)."""
    model, X, ctx = setup
    for x in X[:4]:
        p, dp, F = fisher_at(model, x)
        for name in OBSERVABLES:
            obs = resolve_observable(name, ctx)
            g, V = influence_terms(obs, p, dp)
            e = eta(g, V, F)
            assert -1e-9 <= e <= 1.0 + 1e-9, (name, e)
            assert e >= float(rho2_per_direction(g, V, F).max()) - 1e-9, name


def test_eta_is_one_when_the_observable_is_the_score(setup):
    """The tight case: equality iff ``psi`` is the score.  This validates the whole identity."""
    model, X, _ = setup
    for x in X[:3]:
        p, dp, F = fisher_at(model, x)
        u = project_vec(torch.eye(6, dtype=torch.float64)[1])
        s = (dp.double() / p.double().clamp(min=1e-12).unsqueeze(1)) @ u
        g = project_vec(dp.double().T @ s)
        pd = p.double()
        mean = (pd * s).sum()
        V = float((pd * s * s).sum() - mean ** 2)
        assert abs(eta(g, V, F) - 1.0) < 1e-6


def test_max_prob_is_the_only_exclusion(setup):
    """It raises for **non-differentiability**, not for lacking a variance."""
    model, X, ctx = setup
    obs = resolve_observable("max_prob", ctx)
    assert obs.is_differentiable is False
    p, _, _ = fisher_at(model, X[0])
    with pytest.raises(NotImplementedError):
        obs.influence(p.unsqueeze(0))


def test_quadratics_are_not_degenerate(setup):
    """``zeta_1/zeta_2 ~ 0`` is a FLAG, not an exclusion -- and neither quadratic is degenerate."""
    model, X, ctx = setup
    p, _, _ = fisher_at(model, X[0])
    for name in ("sq_parity", "pairprod"):
        z = float(resolve_observable(name, ctx).u_statistic_degeneracy(p.unsqueeze(0))[0])
        assert z > 1e-3, name


def test_ade_summaries_are_in_unit_interval_and_carry_the_1_over_k(setup):
    model, X, ctx = setup
    F_sum, FO_sum = None, None
    for x in X[:4]:
        p, dp, F = fisher_at(model, x)
        g, V = influence_terms(resolve_observable("parity", ctx), p, dp)
        FO = torch.outer(g, g) / V
        F_sum = F if F_sum is None else F_sum + F
        FO_sum = FO if FO_sum is None else FO_sum + FO
    F_bar, FO_bar = F_sum / 4, FO_sum / 4
    s = ade_summaries(FO_bar, F_bar)
    assert s["k"] == 5
    for key in ("A", "D", "E"):
        assert 0.0 <= s[key] <= 1.0
    # efficiency() must NOT carry the 1/k -- with k=5 the two differ by a factor of 5.
    assert abs(s["trace_ratio"] - s["A"] * s["k"]) < 1e-9
    assert abs(efficiency(F_bar, FO_bar) - s["trace_ratio"]) < 1e-6


def test_mispaired_averages_can_exceed_the_bound():
    """A **constructed** negative test, so the range assertions cannot pass vacuously.  Sampling a
    random ``x_0`` might come in under the bound with no bug present, so the counterexample is
    built from prescribed eigenvalues instead."""
    F_O = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert abs(efficiency(torch.eye(2, dtype=torch.float64), F_O) - 1.0) < 1e-9
    mispaired = efficiency(torch.diag(torch.tensor([0.1, 1.0], dtype=torch.float64)), F_O)
    assert mispaired > 1.0          # no bug present -- just the wrong pairing


# --- learner -------------------------------------------------------------------------------------- #


def test_verdict_table():
    from v2.learner.compare import verdict
    assert verdict(0.9, 0.1) == "INFORMATIVE"
    assert verdict(0.9, 0.9).startswith("no separation")
    assert verdict(0.1, 0.1).startswith("VOID")
    assert verdict(0.1, 0.9).startswith("INVESTIGATE")


def test_split_is_deterministic_and_partitions_the_pool():
    from v2.pipeline.split import split_indices
    tr, te = split_indices(100, test_fraction=0.2, split_seed=0)
    tr2, te2 = split_indices(100, test_fraction=0.2, split_seed=0)
    assert torch.equal(tr, tr2) and torch.equal(te, te2)
    assert len(te) == 20 and len(tr) == 80
    assert sorted(torch.cat([tr, te]).tolist()) == list(range(100))


# --- shots are opt-in per model; derivatives live outside the sampler ---------------------------- #


def test_only_pure_boson_sampling_supports_shots():
    """Shots are a capability a model earns.  Everything else is a distribution model and REFUSES,
    rather than wrapping a multinomial around its exact probs and implying a scaling route it does
    not have."""
    X = sample_X(2, 6, 42)
    for kind in sorted(MODELS):
        model = build_model(cfg_for(kind))
        if kind == "photonic":
            assert model.supports_shots is True
            rows = model.shot_counts(X, shots=10_000)
            assert all(isinstance(n, int) for row in rows for n in row.values())
            assert sum(rows[0].values()) == 10_000
            # sparse by construction: the basis is discovered from the draw, never enumerated
            assert all(len(row) <= 10_000 for row in rows)
        else:
            assert model.supports_shots is False
            with pytest.raises(NotImplementedError, match="probability-distribution model"):
                model.shot_counts(X, shots=10_000)

    # spoqc preps refuse too: HybridProcessor returns a distribution and exposes no sampler
    for prep in ("spin", "spin_magic"):
        model = build_model(cfg_for("photonic", prep=prep, cx_pairs=[[0, 1]]))
        assert model.supports_shots is False
        with pytest.raises(NotImplementedError, match="finite-shot"):
            model.shot_counts(X, shots=10_000)


def test_generate_refuses_shots_for_distribution_only_models(tmp_path):
    from v2.pipeline.generate import generate_exact, generate_shots

    cfg = cfg_for("fermion")
    cfg.generation.size, cfg.generation.shots = 16, 10_000
    generate_exact(cfg, out_root=tmp_path / "d")
    with pytest.raises(NotImplementedError, match="probability-distribution model"):
        generate_shots(cfg, shots_root=tmp_path / "s")


def test_fd_wrapper_differentiates_any_sampler(setup):
    """One wrapper, two forward evaluations per direction, no autograd required of the model."""
    from v2.metrics.fd import fd_jacobian, probs_and_fd_jacobian, sampler

    model, X, _ = setup
    x = X[0]
    _, dp_exact = probs_and_jacobian(model, x)

    # exact sampler: FD tracks autograd
    fn = sampler(model)
    dp_fd = fd_jacobian(fn, x)
    assert dp_fd.shape == dp_exact.shape
    assert float((dp_fd - dp_exact).abs().max() / dp_exact.abs().max()) < 1e-2

    # and the Fisher spectrum survives it, which is what analysis A actually consumes
    def spec(dp, p):
        keep = p > 1e-12
        return spectrum_from_jacobian(project_physical(dp[keep] / p[keep].sqrt().unsqueeze(1)))
    p, _ = probs_and_jacobian(model, x)
    ea, ef = spec(dp_exact, p), spec(dp_fd, p)
    assert float(((ef[:5] - ea[:5]).abs() / ea[:5]).max()) < 1e-2

    # same signature as the autograd path, so the two are interchangeable
    p2, dp2 = probs_and_fd_jacobian(model, x)
    assert p2.shape == p.shape and dp2.shape == dp_exact.shape


def test_fd_on_a_shot_sampler_is_noise_dominated(setup):
    """Documented, asserted, so nobody differentiates a shot sampler and trusts the number.

    Shot noise on p is ~sqrt(p/S); dividing by 2*eps amplifies it ~50x at eps=1e-2.  Common random
    numbers help (the substream is keyed on (shot_seed, block, row), never on x) but do not rescue it.
    """
    from v2.metrics.fd import fd_jacobian, sampler

    model, X, _ = setup
    x = X[0]
    _, dp_exact = probs_and_jacobian(model, x)
    dp_shot = fd_jacobian(sampler(model, shots=50_000, shot_seed=0), x)
    rel = float((dp_shot - dp_exact).abs().max() / dp_exact.abs().max())
    assert rel > 0.05          # noise-dominated: measured ~0.32
    # exact path, same wrapper, three orders better
    assert float((fd_jacobian(sampler(model), x) - dp_exact).abs().max()
                 / dp_exact.abs().max()) < 1e-2


# --- finite-sample (partial basis) path --------------------------------------------------------- #


def test_key_scorers_are_the_single_source_of_the_dense_tables():
    """The table is derived from the per-key function, so there is no second implementation."""
    from v2.observable import BASE_SCORERS, KEY_SCORERS

    assert set(KEY_SCORERS) == set(BASE_SCORERS)
    model = build_model(cfg_for())
    ctx = ObservableContext(m=6, k=3, keys=model.outcome_keys(), seed=42)
    for name, fn in KEY_SCORERS.items():
        dense = list(BASE_SCORERS[name](ctx))
        assert dense == [fn(key, ctx) for key in ctx.keys], name


def test_score_on_a_partial_basis_equals_the_full_basis(setup):
    """The finite-sample readout: evaluate v on the OBSERVED keys, never over the full basis.

    Exercised on ``fermion(flavours=1)``, whose 36 structurally-zero outcomes give a natural partial
    basis, and on a real shot draw.  Every implemented shape agrees because an unobserved outcome
    contributes nothing -- the quadratics are homogeneous in ``p`` and both pointwise transforms
    vanish at 0.
    """
    import dataclasses

    from v2.observable import observable_on_keys
    from v2.pipeline.shots import score_sparse, to_sparse

    model, X, _ = setup
    fermi = build_model(cfg_for("fermion", flavours=1))
    fermi.forward_batch = 0
    fp = fermi.probs(X)
    keys_full = tuple(fermi.outcome_keys())
    hit = (fp > 0).any(0)
    keys_sub = tuple(k for k, t in zip(keys_full, hit.tolist()) if t)
    assert len(keys_sub) == math.comb(6, 3) and len(keys_sub) < len(keys_full)

    ctx = ObservableContext(m=6, k=3, keys=keys_full, seed=42, graph_density=0.5,
                            input_state=fermi.input_state(),
                            reference_probs=fermi.probs_at_zero().numpy())
    ctx_sub = dataclasses.replace(ctx, keys=keys_sub,
                                  reference_probs=fermi.probs_at_zero().numpy()[hit.numpy()])
    for name in ["parity", "majority", "bunching", "n_first", "prod_parity_consecutive",
                 "connected_maxcc", "single_output", "sq_parity", "pairprod", "ent", "osc",
                 "max_prob"]:
        full = resolve_observable(name, ctx).score(fp)
        part = observable_on_keys(name, ctx_sub, keys_sub).score(fp[:, hit])
        assert float((full - part).abs().max()) < 1e-5, name

    # and on an actual shot draw, through the observed-keys view
    model.forward_batch = 0
    p = model.probs(X)
    mctx = ObservableContext(m=6, k=3, keys=model.outcome_keys(), seed=42, graph_density=0.5,
                             input_state=model.input_state(),
                             reference_probs=model.probs_at_zero().numpy())
    rows = model.shot_counts(X, blocks=[0], shot_seed=0)
    keys, emp = to_sparse(rows)
    for name in ("parity", "ent", "osc", "max_prob", "pairprod"):
        a = observable_on_keys(name, mctx, keys).score(emp)
        assert torch.allclose(a, score_sparse(name, mctx, rows), atol=1e-6), name


def test_partial_basis_guard_fires_when_phi_does_not_vanish_at_zero():
    """The one way the partial-basis path could break silently, guarded rather than discovered."""
    from v2.observable import ProbFunction

    class NonVanishing(ProbFunction):
        def transform(self, probs):
            return torch.sin(1.0 / (probs + 1e-3))       # phi(0) = sin(1000) != 0

    assert NonVanishing().partial_basis_safe is False
    for name in ("ent", "osc"):                          # both implemented transforms DO vanish
        model = build_model(cfg_for())
        ctx = ObservableContext(m=6, k=3, keys=model.outcome_keys(), seed=42)
        assert resolve_observable(name, ctx).partial_basis_safe is True


def test_max_prob_is_partial_basis_safe_but_still_excluded_from_analysis_B(setup):
    """Two independent facts that were previously conflated.

    ``max_prob`` computes fine on a partial basis (a zero can never be the max) -- what rules it out
    of analysis B is non-differentiability, in every regime.  Its shot plug-in is separately biased
    upward: ``+0.018`` at 100 shots against a value of ``0.094``.
    """
    model, X, ctx = setup
    obs = resolve_observable("max_prob", ctx)
    assert obs.partial_basis_safe is True
    assert obs.is_differentiable is False
    with pytest.raises(NotImplementedError):
        obs.influence(model.probs(X[:1]))


def test_merlin_perm_matches_the_analytic_permanent_on_the_FISHER_spectrum():
    """What licenses using merlin for the headline ``Perm`` arm instead of a hand-rolled permanent.

    ``boson_probs_reference`` exists so ``Perm`` and ``det`` could share ``sandwich_unitary_at`` --
    the plan warns that "going through merlin instead would compare two different code paths".  This
    pins that they are the *same* path numerically, at the level the comparison actually consumes:
    not just ``p`` (which agrees to 1.1e-8) but the Fisher eigenvalues built from ``dp/sqrt(p)``,
    where a small-``p`` relative error could in principle amplify.  It does not -- 7.3e-8 across all
    five eigenvalues, ``tr F`` to 4.5e-8.

    So the analytic permanent stays a *verification* tool, and no Ryser implementation is needed: it
    is ``O(k!)`` and dies past ``k = 4`` (434 s for one ``jacrev`` at ``m=10, k=5``) where merlin is
    166 ms.
    """
    from torch.func import jacrev

    from v2.circuit.photonic import (default_input_state, sandwich_unitaries,
                                     sandwich_unitary_at)
    from v2.model.fermion import boson_probs_reference
    from v2.model.fock import fock_keys

    m, k = 6, 3
    x = sample_X(1, 6, 42)[0]

    def spec(p, dp):
        keep = p > 1e-12
        return spectrum_from_jacobian(project_physical(dp[keep] / p[keep].sqrt().unsqueeze(1)),
                                      n_f=6)

    model = build_model(cfg_for("photonic", m, k))
    model.forward_batch = 0
    e_merlin = spec(*probs_and_jacobian(model, x))

    W1, W2 = sandwich_unitaries(m, 42)
    keys = fock_keys(m, k)
    state = default_input_state(m, k)
    s_modes = [i for i, n in enumerate(state) for _ in range(int(n))]

    def fn(z):
        return boson_probs_reference(sandwich_unitary_at(W1, W2, z.unsqueeze(0), 6), s_modes, keys)[0]

    e_perm = spec(fn(x).detach(), jacrev(fn)(x).detach())

    for i in range(5):
        assert abs(float(e_merlin[i]) - float(e_perm[i])) / float(e_merlin[i]) < 1e-5, i
    assert abs(float(e_merlin.sum()) - float(e_perm.sum())) / float(e_merlin.sum()) < 1e-5


def test_combinatorial_fock_basis_matches_merlin_exactly():
    """``fock_keys`` is a second implementation of merlin's ``output_keys`` -- pin the agreement.

    merlin exposes ``output_keys`` without a forward pass, so the photonic path could read the basis
    from the simulator instead.  The combinatorial version is kept because ``quadratic_fock`` and
    ``mlp_fock`` are pure torch and define ``p`` over this basis: making them construct a merlin
    ``QuantumLayer`` to learn their own output basis would add a quantum dependency to a classical
    model.  Both are ``O(n_out)``, so the choice does not move the memory wall -- and now that shot
    draws discover their own keys, the basis is only needed by the exact branch, where it is
    enumerable by definition.

    What this test removes is the hedge: the docstring said the order "need not match", which left a
    latent inconsistency.  It matches, and now it has to keep matching.
    """
    from v2.circuit.photonic import build_quantum_layer
    from v2.model.fock import fock_keys

    for m, k in [(6, 3), (8, 4)]:
        layer, _ = build_quantum_layer(m, k, 6, 42)
        merlin = [tuple(int(c) for c in key) for key in layer.output_keys]
        combinatorial = [tuple(int(c) for c in key) for key in fock_keys(m, k)]
        assert merlin == combinatorial, (m, k)


def test_learner_fourier_map_is_independent_of_the_teacher_map():
    """They must not be one import: sharing let the learner reach the labels' own featurisation."""
    from v2.learner.embedding import fourier_features as learner_map
    from v2.model.features import fourier_features as teacher_map
    assert learner_map is not teacher_map
