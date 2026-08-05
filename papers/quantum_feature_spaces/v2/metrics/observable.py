"""Analysis B -- how much of the distribution's input-information a readout captures.

    python -m v2.metrics.observable --config v2/configs/photonic.yaml --all-observables

Same differentiation target as analysis A (the input ``x``), all exact from the shared score
matrix and the registry's influence functions.  Nothing here samples.

**One identity organises everything.**  For any smooth functional ``T(p)`` with influence function
``psi_n = dT/dp_n``, and using ``E_p[s] = 0`` (since ``sum_n d_i p_n = 0``):

    dT/dx_i  =  sum_n psi_n d_i p_n  =  Cov_p(psi, s_i)  =:  g_i
    V_eff    =  Var_p(psi)                        [exact single-shot variance for an Expectation]
    eta_T    =  g^T F^+ g / V_eff   in  [0, 1]

Note ``g`` needs **no** division by ``p`` and so no support mask -- it is ``dp^T psi`` exactly.

**All three functional shapes are in scope.**  An earlier design gated this on
``isinstance(obs, Expectation)``, claiming the others "have no variance".  That conflated two
claims: there is no single-shot unbiased estimator with variance ``p.v^2 - (p.v)^2`` (**true**),
and there is no finite asymptotic variance (**false**).  For a :class:`Quadratic`, ``psi = 2Kp`` is
the Hajek projection of the order-2 U-statistic and the factors of 4 cancel in the ratio, so its
efficiency equals that of a linear observable with the ``x``-dependent score vector ``w = K p_x``.
For entropy, ``psi = log p + 1`` gives ``V_eff = Var(log p)``, finite and exact.  The **only**
exclusion is non-differentiability: ``max_prob`` has ``psi_n = 1[n = argmax]``, undefined at a tie,
and raises via ``is_differentiable = False``.

**``eta in [0, 1]`` is exact, not asymptotic** -- Cauchy-Schwarz on the discrete distribution:
``u^T F_T u = Cov(psi, u^T s)^2/V_eff <= Var(psi)Var(u^T s)/V_eff = u^T F u``, so
``F_T = gg^T/V_eff <= F``; then ``v = F^+ g`` gives ``(g^T F^+ g)^2/V_eff <= g^T F^+ g``.  A
violation is therefore unambiguously a code bug (Jacobian or pseudo-inverse projection), never a
sampling artifact -- which is what makes the assertion worth having.

Equality holds iff ``psi`` **is** the score.  So the information-optimal readout is the score
itself -- and for a boson sampler the score's evaluation cost *is* the ``#P``-hardness.  The
optimal readout is precisely the unusable one, and every implementable observable is inefficient
by construction: a statement about cost, not information.

**Rank-1 caveat, scoped as a caveat.**  ``F_O(x) = gg^T/V_eff`` is rank 1, so with any other input
component treated as a nuisance the Schur complement is identically zero: one scalar readout cannot
tell "``x_1`` moved" from "``x_2`` moved".  **This study never poses that inverse problem** -- the
dataset is ``(x, <O>_x)`` and the learner *receives* ``x``, so there is no nuisance parameter and
``F_eff = 0`` is a true fact about a problem the pipeline does not pose.  Reporting it as the
headline would replace a useful signal measure with an identically-zero one.  Hence:
per-direction ``F_{O,ii}`` and ``shots_required`` are reported as **signal-carrying** measures, and
the A/D/E summaries are computed on the ``x``-**averaged** ``F_bar_O``, which is generically full
rank on ``range(F)``.

**Ranges differ by case -- asserted per case, never one blanket bound:**

    pointwise rank-1:             tr(F(x)^+ F_O(x))  in [0, 1]
    averaged / multi-observable:  tr(F_bar^+ F_bar_O) in [0, k]     since tr(F_bar^+ F_bar) = k
    every A/D/E summary:                              in [0, 1]

That ``[0, k]`` is exactly why A-type carries the ``1/k``.  :func:`efficiency` therefore returns
``tr(F^+ F_O)`` **without** the ``1/k``, and :func:`ade_summaries` divides at its own call site --
with ``k = 5`` the two differ by a factor of 5.  And for rank-1 ``F_O``,
``tr(F^+ F_O) = g^T F^+ g / V_eff`` exactly, so the joint ``eta`` and A-type at ``R = 1`` are one
quantity, implemented once.
"""

from __future__ import annotations

import math

import torch

from observable import Expectation, ProbFunction, Quadratic, resolve_observable
from .distribution import SUPPORT_TOL, probs_and_jacobian, project_physical, sqrt_jacobian

#: Relative cut for the pseudo-inverse / ``range(F)`` projection.  ``F`` is rank ``n_f - 1`` by
#: construction (the global-phase mode), so this is structural, not a tolerance guess.
RCOND = 1e-10


def project_vec(g: torch.Tensor) -> torch.Tensor:
    """The vector form of :func:`~v2.metrics.distribution.project_physical`."""
    return g - g.mean(dim=-1, keepdim=True)


def fisher_at(model, x: torch.Tensor):
    """``(p, dp, F)`` at one input, with ``F`` on the physical subspace.

    ``dp`` is returned unprojected because ``g = dp^T psi`` is a *directional derivative* that
    gets projected afterwards; ``F`` is projected because every spectrum and every ``F^+`` in this
    module lives on the ``(n_f - 1)``-dimensional complement of ``1/sqrt(n_f)``.
    """
    p, dp = probs_and_jacobian(model, x)
    keep = p > SUPPORT_TOL
    J = project_physical(dp[keep] / p[keep].sqrt().unsqueeze(1)).double()
    return p, dp, J.T @ J


def influence_terms(obs, p: torch.Tensor, dp: torch.Tensor):
    """``(g, V_eff)`` for one observable at one input: ``g (n_f,)`` and the scalar ``V_eff``.

    ``g_i = Cov_p(psi, s_i) = sum_n psi_n d_i p_n``, i.e. ``dp^T psi`` -- which is also
    ``d<T>/dx_i``, so this doubles as the exact gradient of the label map.
    """
    psi = obs.influence(p.unsqueeze(0))[0].double()
    pd = p.double()
    g = project_vec(dp.double().T @ psi)
    mean = (pd * psi).sum()
    V = float(((pd * psi * psi).sum() - mean * mean).clamp(min=0.0))
    return g, V


def observable_fisher(g: torch.Tensor, V: float) -> torch.Tensor:
    """``F_O = g g^T / V_eff`` -- **rank 1** per ``x``.  See the rank-1 caveat in the docstring."""
    return torch.outer(g, g) / max(V, 1e-300)


def efficiency(F: torch.Tensor, F_O: torch.Tensor) -> float:
    """``tr(F^+ F_O)`` -- **without** the ``1/k``.  See the per-case ranges in the docstring."""
    return float((torch.linalg.pinv(F, rcond=RCOND) @ F_O).diagonal().sum())


def eta(g: torch.Tensor, V: float, F: torch.Tensor) -> float:
    """``g^T F^+ g / V_eff``: the joint (multiple-correlation) efficiency, exactly in ``[0, 1]``.

    Identical to ``efficiency(F, observable_fisher(g, V))`` for rank-1 ``F_O``; kept as the direct
    form because it is one quadratic form rather than a matrix product and trace.
    """
    gp = project_vec(g).double()
    return float(gp @ torch.linalg.pinv(F, rcond=RCOND) @ gp) / max(V, 1e-300)


def rho2_per_direction(g: torch.Tensor, V: float, F: torch.Tensor) -> torch.Tensor:
    """``rho^2(O, s_i) = F_{O,ii}/F_ii = g_i^2/(V_eff F_ii)`` -- the **local** reading, ``(n_f,)``.

    A ratio of diagonal entries, answering "how much of direction ``i``", where :func:`eta` answers
    "how much of ``p``'s total input-information".  ``eta >= max_i rho^2``, generically strict once
    the ``s_i`` are correlated, with equality only when ``O`` lies along a single score component.
    Report both; only the joint form is congruence-invariant.
    """
    diag = torch.diagonal(F).clamp(min=1e-300)
    return (project_vec(g).double() ** 2) / (max(V, 1e-300) * diag)


def shots_required(g: torch.Tensor, V: float, delta: float = 0.1) -> torch.Tensor:
    """``V_eff / (d<O>/dx_i . delta)^2`` per direction, ``(n_f,)``.

    **A signal proxy, not an estimation bound** -- by the rank-1 caveat the genuine per-direction
    estimation statement needs ``R >= n_f`` observables with a full-rank ``(R x n_f)`` Jacobian.
    It is the right quantity here for a different reason: ``x`` is *controlled, not estimated* (we
    draw it via ``sample_X`` and hand it to the model), so this measures how strongly the readout
    responds to a known input perturbation against its own shot noise -- which is what bounds
    learnability of the ``x -> label`` map.  Do not write "which input directions are resolvable".
    """
    return max(V, 1e-300) / (project_vec(g).double() * float(delta)).pow(2).clamp(min=1e-300)


def ade_summaries(F_bar_O: torch.Tensor, F_bar: torch.Tensor) -> dict:
    """A/D/E-type efficiencies on ``range(F_bar)``, all in ``[0, 1]``.

    Evaluated on the range because all three are **ill-defined on the raw ``F``** at the base
    config: ``rank F = 5 < 6``, so ``F^{-1/2}`` does not exist, D-type is ``0/0`` and E-type is
    ``lambda_min = 0`` *identically for every observable set*.  So restrict to the ``k = rank F``
    physical subspace -- the same projection analysis A applies to the sweep, one shared invariant
    rather than two local patches.

    **Both sides must be averaged over the same pool.**  ``F_O(x) <= F(x)`` pointwise and linearity
    preserves the PSD order, so ``E_x[F_O] <= E_x[F]``.  Pairing ``F_bar_O`` with a single-point
    ``F(x_0)`` is ill-formed and **can exceed 1 with no bug present**.
    """
    evals, evecs = torch.linalg.eigh(F_bar.double())
    keep = evals > RCOND * evals.max()
    k = int(keep.sum())
    U, lam = evecs[:, keep], evals[keep]
    inv_sqrt = U @ torch.diag(lam.rsqrt())                       # F^{-1/2} on range(F)
    M = inv_sqrt.T @ F_bar_O.double() @ inv_sqrt                 # (k, k), PSD, <= I
    m_eigs = torch.linalg.eigvalsh(M).clamp(min=0.0)
    return {
        "k": k,
        "trace_ratio": float(m_eigs.sum()),                      # in [0, k] -- NOT [0, 1]
        "A": float(m_eigs.sum() / k),                            # the 1/k lives HERE
        "D": float(torch.exp(torch.log(m_eigs.clamp(min=1e-300)).sum() / k)),
        "E": float(m_eigs.min()),                                # lead with this one
    }


def set_fisher(G: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """``F_O = Cov(psi, s)^T Sigma^+ Cov(psi, s)`` for a vector of ``R`` observables.

    ``Sigma^+`` automatically discounts redundant observables, so this is the joint ``R^2`` of
    regressing the score on the set rather than a sum of individual efficiencies.
    """
    return G.double().T @ torch.linalg.pinv(Sigma.double(), rcond=RCOND) @ G.double()


def combination_weights(G: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """``beta = Sigma^+ Cov(psi, s)``, ``(R, n_f)`` -- the projection of the score onto ``span(O)``.

    Influence functions are linear in the functional, so ``psi* = sum_r beta_r psi_r`` and the whole
    machinery works in ``psi``-space regardless of class.  When every member is an
    :class:`~v2.observable.Expectation`, ``O*`` is *itself* an ``Expectation`` with score vector
    ``sum_r beta_r v_r``, constructible inside the existing class; for a mixed-class set it is a
    combination of estimators rather than one observable -- an implementation note, not an
    obstruction.
    """
    return torch.linalg.pinv(Sigma.double(), rcond=RCOND) @ G.double()


def greedy_selection(G: torch.Tensor, Sigma: torch.Tensor, F: torch.Tensor, names,
                     max_r: int | None = None) -> list[tuple[str, float]]:
    """Forward-select observables by the joint ``tr(F^+ F_O)`` they achieve together.

    Returns ``[(name, cumulative_efficiency)]`` -- a ranked, non-redundant subset.
    """
    remaining, chosen, out = list(range(len(names))), [], []
    budget = len(names) if max_r is None else int(max_r)
    while remaining and len(chosen) < budget:
        best, best_val = None, -1.0
        for r in remaining:
            idx = chosen + [r]
            val = efficiency(F, set_fisher(G[idx], Sigma[idx][:, idx]))
            if val > best_val:
                best, best_val = r, val
        chosen.append(best)
        remaining.remove(best)
        out.append((names[best], best_val))
    return out


# --- gradient-free screen (B6) ----------------------------------------------------------------- #


def g_ratio(mu: torch.Tensor, sigma_sq: torch.Tensor) -> float:
    """``G_O = Var_x(mu_x) / E_x[sigma_x^2]`` -- an ANOVA F-statistic, no gradients needed.

    The law of total variance ``Var(O) = E_x[sigma_x^2] + Var_x(mu_x)`` is **exact**, so the two
    halves are complementary, not interchangeable: at fixed total, one large forces the other
    small, and neither is a figure of merit alone.

    **A prioritiser, never a filter.**  The relation to ``eta_O`` is one-directional, by Poincare
    on ``x`` uniform over an interval of length ``D``: ``Var_x(mu_x) <= (D^2/pi^2) E_x[(mu')^2]``.
    So ``G_O`` large **implies** ``F_O`` large somewhere, but ``F_O`` large does **not** imply
    ``G_O`` large -- an oscillating ``mu_x`` has large ``E[(mu')^2]`` and small ``Var_x(mu_x)``
    from cancellation over ``U[0, 2pi]``.  The failure case is in-repo: the ``exp_poly`` scorers
    have rapidly alternating-sign score vectors, so their means are the likeliest to oscillate and
    screen out while being locally informative.  **Rule: always run the gradient path on
    ``exp_poly`` regardless of its ``G_O``.**

    Also **grid-dependent**: changing the spacing or range changes the number without changing
    ``mu_x``, so state the grid.  ``eta_O(x)`` has no such dependence.
    """
    return float(mu.double().var(unbiased=False) / sigma_sq.double().mean().clamp(min=1e-300))


def snr_r2_ceiling(G_O: float, shots: int) -> float:
    """``R^2 <~ SNR/(1+SNR)`` with ``SNR = S . G_O``.

    **Only valid against SHOT-NOISY test labels.**  With ``generation.shots = 0`` the stored
    labels are exact, the ceiling is 1, and the comparison is vacuous -- so generate at
    ``generation.shots = S``, score with ``--from-shots``, and
    score against those labels, or the prediction is untestable.  Also excluded for
    :class:`~v2.observable.ProbFunction` labels, whose ``O(1/S)`` plug-in bias violates the
    zero-mean-noise assumption this bound rests on.
    """
    snr = float(shots) * float(G_O)
    return snr / (1.0 + snr)


def monotonicity(mu_grid: torch.Tensor) -> float:
    """Fraction of sign changes in ``diff(mu)`` along a 1-D sweep: a **non-injectivity** flag.

    ``G_O`` has two further blind spots beyond oscillation -- *dead zones* (a sharp jump in one
    sliver plus flatness elsewhere gives a healthy ``G_O`` while local estimation is hopeless) and
    *non-injectivity* (``mu_{x1} = mu_{x2}`` for well-separated inputs breaks global
    identifiability while ``F_O`` looks fine).  Both show up as many sign changes here.  ``0`` is
    monotone; ``~0.5`` is strongly oscillatory.
    """
    d = mu_grid.double().diff()
    if d.numel() < 2:
        return 0.0
    return float((d[1:].sign() != d[:-1].sign()).double().mean())


# --- pool-level analysis ------------------------------------------------------------------------ #


def analyse(model, X: torch.Tensor, names, ctx, *, delta: float = 0.1,
            shots: int | None = None) -> dict:
    """Analysis B over an input pool for a list of observable names.

    Per observable: ``eta`` (joint) and ``rho^2`` (local) with the per-case ranges asserted,
    ``V_eff``, ``shots_required``, ``G_O``, and ``zeta_1/zeta_2`` for the quadratics.
    For the set: the joint ``R^2``, the constructed ``O*``, greedy forward selection, and the
    A/D/E summaries on ``F_bar_O`` paired with ``F_bar`` over the *same* pool.
    """
    obs = {}
    for n in names:
        o = resolve_observable(n, ctx)
        if not o.is_differentiable:
            obs[n] = None                        # max_prob: recorded as excluded, with the reason
        else:
            obs[n] = o

    live = [n for n, o in obs.items() if o is not None]
    per = {n: {"eta": [], "rho2": [], "V_eff": [], "shots": [], "mu": [], "sigma_sq": [],
               "zeta": []} for n in live}
    F_sum = None
    FO_sum = {n: None for n in live}
    G_rows, S_rows = [], []                       # per-x (R, n_f) and (R, R), accumulated

    for x in X:
        p, dp, F = fisher_at(model, x)
        F_sum = F if F_sum is None else F_sum + F
        gs, psis = {}, {}
        for n in live:
            o = obs[n]
            g, V = influence_terms(o, p, dp)
            gs[n] = g
            psis[n] = o.influence(p.unsqueeze(0))[0].double()
            FO = observable_fisher(g, V)
            FO_sum[n] = FO if FO_sum[n] is None else FO_sum[n] + FO
            e = eta(g, V, F)
            per[n]["eta"].append(e)
            per[n]["rho2"].append(rho2_per_direction(g, V, F))
            per[n]["V_eff"].append(V)
            per[n]["shots"].append(shots_required(g, V, delta))
            s = float(o.score(p.unsqueeze(0))[0])
            per[n]["mu"].append(s)
            # within-x shot variance: exact single-shot for an Expectation, asymptotic otherwise
            per[n]["sigma_sq"].append(V)
            if isinstance(o, Quadratic):
                per[n]["zeta"].append(float(o.u_statistic_degeneracy(p.unsqueeze(0))[0]))

        pd = p.double()
        G_rows.append(torch.stack([gs[n] for n in live]))
        P = torch.stack([psis[n] for n in live])                  # (R, n_out)
        means = (P * pd).sum(dim=1, keepdim=True)
        S_rows.append(((P - means) * pd) @ (P - means).T)         # Cov_p(psi_r, psi_r')

    n_x = X.shape[0]
    F_bar = F_sum / n_x
    G_bar = torch.stack(G_rows).mean(dim=0)
    Sigma_bar = torch.stack(S_rows).mean(dim=0)

    rows = {}
    for n in live:
        d = per[n]
        rho2 = torch.stack(d["rho2"])
        rows[n] = {
            "eta_mean": float(torch.tensor(d["eta"]).mean()),
            "eta_min": float(torch.tensor(d["eta"]).min()),
            "eta_max": float(torch.tensor(d["eta"]).max()),
            "rho2_max_mean": float(rho2.max(dim=1).values.mean()),
            "V_eff": float(torch.tensor(d["V_eff"]).mean()),
            "shots_required_best": float(torch.stack(d["shots"]).min(dim=1).values.median()),
            "G_O": g_ratio(torch.tensor(d["mu"]), torch.tensor(d["sigma_sq"])),
            "monotonicity": monotonicity(torch.tensor(d["mu"])),
            "zeta_ratio": (float(torch.tensor(d["zeta"]).mean()) if d["zeta"] else None),
            "is_prob_function": isinstance(obs[n], ProbFunction),
            "ade": ade_summaries(FO_sum[n] / n_x, F_bar),
        }
        if shots:
            rows[n]["r2_ceiling"] = (None if rows[n]["is_prob_function"]
                                     else snr_r2_ceiling(rows[n]["G_O"], shots))

    joint_F_O = set_fisher(G_bar, Sigma_bar)
    return {
        "n_x": n_x,
        "excluded": {n: "non-differentiable (psi undefined at a tie)"
                     for n, o in obs.items() if o is None},
        "rows": rows,
        "F_bar": F_bar,
        "joint": {"efficiency": efficiency(F_bar, joint_F_O),
                  "ade": ade_summaries(joint_F_O, F_bar)},
        "beta": combination_weights(G_bar, Sigma_bar),
        "greedy": greedy_selection(G_bar, Sigma_bar, F_bar, live),
    }


# --- CLI ---------------------------------------------------------------------------------------- #


def main(argv=None) -> None:
    import argparse

    from config import load_config
    from model import build_model, sample_X
    from observable import PLAIN_OBSERVABLES
    from pipeline.artifact import artifact_path
    from pipeline.distribution import load_dist
    from pipeline.score import context_for

    DEFAULT = ["parity", "majority", "bunching", "n_first", "prod_parity_consecutive",
               "prod_parity_second", "connected_maxcc", "connected_parity", "single_output",
               "xent_parity", "sq_parity", "pairprod", "ent", "ent_parity", "osc", "osc_parity",
               "max_prob"]

    ap = argparse.ArgumentParser(description="Analysis B: observable efficiency")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+")
    ap.add_argument("--all-observables", action="store_true")
    ap.add_argument("--n-x", type=int, default=48)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--out-root", default="datasets_v2")
    ap.add_argument("--graph-density", type=float, default=0.5)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg)
    names = args.observables or (DEFAULT if args.all_observables else list(PLAIN_OBSERVABLES))

    path = artifact_path(cfg, model, args.out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run v2.pipeline.generate first")
    ctx = context_for(load_dist(path, size=1), graph_density=args.graph_density)

    X = sample_X(args.n_x, cfg.problem.n_features, cfg.seeds.sample_seed)
    res = analyse(model, X, names, ctx, delta=args.delta, shots=args.shots)

    print(f"Analysis B: {path.name}, n_x={res['n_x']}, delta={args.delta}")
    print("eta is EXACT (Cauchy-Schwarz on the discrete distribution), so eta > 1 is a bug.")
    print("G_O grid: x ~ U[0,2pi]^6 from sample_X -- a PRIORITISER, never a filter.\n")
    hdr = (f"{'observable':<26}{'eta':>8}{'max rho2':>10}{'V_eff':>11}{'E-type':>9}"
           f"{'A-type':>8}{'G_O':>9}{'shots':>10}{'mono':>7}{'z1/z2':>8}")
    print(hdr + ("  R2ceil" if args.shots else ""))
    print("-" * len(hdr))
    for n, r in sorted(res["rows"].items(), key=lambda kv: -kv[1]["eta_mean"]):
        zeta = r["zeta_ratio"]
        zeta_txt = "-" if zeta is None else f"{zeta:.3f}"
        line = (f"{n:<26}{r['eta_mean']:>8.4f}{r['rho2_max_mean']:>10.4f}{r['V_eff']:>11.4g}"
                f"{r['ade']['E']:>9.4f}{r['ade']['A']:>8.4f}{r['G_O']:>9.4f}"
                f"{r['shots_required_best']:>10.3g}{r['monotonicity']:>7.2f}{zeta_txt:>8}")
        if args.shots:
            line += (f"  {r['r2_ceiling']:.4f}" if r.get("r2_ceiling") is not None else "   (excl)")
        print(line)
    for n, why in res["excluded"].items():
        print(f"{n:<26}  EXCLUDED: {why}")

    j = res["joint"]
    print(f"\nSET  (R={len(res['rows'])}):  joint tr(F+ F_O) = {j['efficiency']:.4f}  in [0, k]")
    print(f"     k = rank F = {j['ade']['k']}   E-type = {j['ade']['E']:.4f}   "
          f"A-type = {j['ade']['A']:.4f}   D-type = {j['ade']['D']:.4f}")
    print("     greedy forward selection (cumulative efficiency):")
    for i, (n, val) in enumerate(res["greedy"][:8], 1):
        print(f"       {i}. {n:<28} {val:.4f}")
    if args.shots:
        print("\nR2 ceiling applies ONLY against shot-noisy labels; ProbFunction rows excluded "
              "(O(1/S) plug-in bias breaks the zero-mean-noise assumption).")


if __name__ == "__main__":
    main()
