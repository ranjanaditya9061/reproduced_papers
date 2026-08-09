"""Analysis A -- the input Fisher matrix of ``p_x`` alone.  **No observable.**

    J    = sqrt_jacobian(model, x)          # (n_supp, n_f)   J_ni = d_i p_n / sqrt(p_n)
    eigs = spectrum_from_jacobian(J)        # svdvals(J)**2   -- never eigvalsh(J^T J)
    stats = fisher_spectrum(eigs, n=N)

    python -m v2.metrics.distribution --compare photonic fermion --shared-support
    python -m v2.metrics.distribution --configs-dir v2/configs/size_sweep

**What this measures, and what it does not.**  The Fisher spectrum measures *incompressibility of
the parametrised family at a given sample resolution, and directional estimability*.  It does
**not** measure the computational hardness of evaluating or sampling ``p``.  Boson sampling is the
decisive counterexample and it is our own model: the family is indexed by ``U``, i.e. ``O(m^2)``
numbers -- polynomially describable, maximally compressible -- while each
``p(n) = |Perm(U_{s,n})|^2`` is ``#P``-hard.  Knowing the parametrisation exactly buys nothing
computationally.  **So the boson-vs-determinant comparison here is a learnability /
effective-input-dimension statement and must not be presented as evidence about the ``Perm``/``det``
separation.**

Three design points that are load-bearing rather than stylistic:

1. **The differentiation target is the input ``x``**, not the circuit weights.  So ``F`` is
   ``n_features x n_features`` -- ``6 x 6`` for every model and every ``(m, k)`` -- and spectra are
   stackable across sizes and families.  A weight-space FIM would be ``72 x 72`` at ``m=6`` and
   ``392 x 392`` at ``m=14``: different spaces, no comparison.  It also fixes the gauge, since ``x``
   is the shared physical input in identical radian units for every model (the same rows from
   ``sample_X``).  Only *rank* is reparametrisation-invariant in general -- ``F -> J_r^T F J_r`` is a
   congruence, not a similarity -- so cross-model eigenvalue comparison is legitimate only in a
   fixed gauge, which this setup has by construction.  Keep ``x`` in radians: ``x -> cx`` scales
   ``F`` by ``c^2``.
2. **Gram factorisation, never a matrix product.**  Build ``J`` and take ``svdvals(J)**2``.  Forming
   ``J^T J`` squares the condition number: a resolvable singular-value ratio of ``1e-7`` becomes an
   eigenvalue ratio of ``1e-14``, at the edge of double precision.  :func:`input_fisher` returns the
   matrix (analysis B needs ``F`` itself, for ``F^+``), but every *spectrum* goes through
   :func:`spectrum_from_jacobian`.
3. **The spectrum is always reported on the ``(n_f - 1)``-dimensional complement of
   ``1/sqrt(n_f)``** (:func:`project_physical`).  When ``n_features == m`` a constant added to every
   ``x_i`` is a global phase on ``D(x)``, exactly unobservable, so that direction is an exact null
   mode and ``rank F = n_f - 1``.  Without the projection the sweep ``(6,3), (8,4), (10,5)`` would
   report rank 5 at the first point and 6 at the rest, and ``tr F`` / ``r_eff`` / ``exp(H)`` would
   inherit a discontinuity that is an artifact of the parametrisation, not of the circuit growing.
   At ``m > 6`` the projection discards a genuinely informative direction, so its eigenvalue is
   reported separately (``lambda_phase``).  The same projection is what makes analysis B's A/D/E
   summaries well-defined, so it is one shared invariant rather than two local patches.

This does **not** settle the ``2m^2 - 1`` weight count -- that needs a weight-space rank.  Different
gauge question; do not cross-reference the two.
"""

from __future__ import annotations

import math

import torch
from torch.func import jacrev

#: Outcomes with ``p <= tol`` are treated as off-support and **masked**, not clamped, since
#: ``(d_i p)^2 / p`` at ``p = 0`` is a genuine ``0/0`` and clamping would return the right number
#: while hiding a support mismatch.  The motivating case was strict free fermions, where a repeated
#: column kills the determinant and ``p_n`` is identically 0 in ``x`` on the whole bunched sector;
#: ``fermion`` now has full support by construction (see :mod:`v2.model.fermion`), but its bunched
#: outcomes still reach ``1e-8``-ish, and the shots path still produces never-observed outcomes.
SUPPORT_TOL = 1e-12


def _probs_fn(model):
    """``x (n_f,) -> p (n_out,)``, differentiable: the un-chunked, un-no-grad path."""
    return lambda x: model._probs(x.unsqueeze(0))[0]


def probs_and_jacobian(model, x: torch.Tensor):
    """``(p, dp)`` at one input: ``p (n_out,)`` and ``dp (n_out, n_f)``.

    Reverse mode: ``n_out`` outputs vs ``n_f`` inputs favours forward mode in principle, but
    ``jacfwd`` through merlin measures ~300x slower than ``jacrev`` (5.3 s vs 17 ms at ``m=6, k=3``),
    so reverse mode is the default everywhere.  ``has_aux`` returns ``p`` from the same pass.
    """
    fn = _probs_fn(model)
    dp, p = jacrev(lambda z: (fn(z),) * 2, has_aux=True)(x.double() if x.dtype == torch.float64 else x)
    return p.detach(), dp.detach()


def finite_difference_jacobian(model, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Central-difference ``dp (n_out, n_f)`` -- the cross-check for the autograd path."""
    fn = _probs_fn(model)
    cols = []
    for i in range(x.shape[0]):
        h = torch.zeros_like(x)
        h[i] = eps
        cols.append((fn(x + h) - fn(x - h)) / (2 * eps))
    return torch.stack(cols, dim=1).detach()


def project_physical(J: torch.Tensor) -> torch.Tensor:
    """``J -> J (I - 11^T/n_f)``: drop the global-phase direction.  See docstring point 3.

    Implemented as a row-mean subtraction, which is that projector exactly.
    """
    return J - J.mean(dim=-1, keepdim=True)


def phase_eigenvalue(J: torch.Tensor) -> torch.Tensor:
    """``u^T F u`` for ``u = 1/sqrt(n_f)``: the eigenvalue of the direction :func:`project_physical`
    removes.  Exactly 0 (to round-off) when ``n_features == m``; genuinely informative when
    ``n_features < m``, which is why it is reported instead of silently dropped."""
    n_f = J.shape[-1]
    return (J.sum(dim=-1) / math.sqrt(n_f)).pow(2).sum()


def sqrt_jacobian(model, x: torch.Tensor, *, project: bool = True,
                  tol: float = SUPPORT_TOL) -> torch.Tensor:
    """``J_ni = d_i p_n / sqrt(p_n)`` over the model's **own** support: ``(n_supp, n_f)``.

    ``F = J^T J`` is the input FIM, and ``J = d(2 sqrt p)/dx`` up to the factor absorbed here.
    Off-support rows are dropped (see :data:`SUPPORT_TOL`), so ``J.shape[0]`` reports the support
    size -- the number the honest boson-vs-fermion comparison has to disclose.
    """
    p, dp = probs_and_jacobian(model, x)
    keep = p > tol
    J = dp[keep] / p[keep].sqrt().unsqueeze(1)
    return project_physical(J) if project else J


def conditional_sqrt_jacobian(model, x: torch.Tensor, support: torch.Tensor, *,
                              project: bool = True, center: bool = True) -> torch.Tensor:
    """``J`` for the distribution **conditioned** on a shared support ``C``: ``(|C|, n_f)``.

    Restricting to ``C`` means ``q = p/Z_C``, whose score is the **centered** score:

        d_i log q = d_i log p - E_q[d_i log p]
        F^(C)_ij  = Cov_q(d_i log p, d_j log p)          <- a COVARIANCE, not E_q[. .]

    So ``q``-weight *and* ``q``-mean-center the score rows before the SVD.  Omitting the centering
    overestimates ``F^(C)`` by the rank-1 outer product ``(d log Z_C)(d log Z_C)^T``, which is not
    small here: at ``m=6, k=3`` the collision-free sector is 20 of 56 outcomes, so ``Z_C`` is far
    from 1 and strongly ``x``-dependent.  ``center=False`` exists only so a test can measure that
    difference.
    """
    p, dp = probs_and_jacobian(model, x)
    p_c, dp_c = p[support], dp[support]
    Z = p_c.sum()
    q = p_c / Z
    s = dp_c / p_c.clamp(min=SUPPORT_TOL).unsqueeze(1)          # d_i log p on C
    if center:
        s = s - (q.unsqueeze(1) * s).sum(dim=0, keepdim=True)    # subtract E_q[d_i log p]
    J = q.sqrt().unsqueeze(1) * s
    return project_physical(J) if project else J


def spectrum_from_jacobian(J: torch.Tensor, n_f: int | None = None) -> torch.Tensor:
    """Eigenvalues of ``F = J^T J``, **descending**, as ``svdvals(J)**2``.

    Padded to ``n_f`` entries so a projected (rank-deficient) spectrum still stacks against an
    unprojected one.  Never ``eigvalsh(J.T @ J)`` -- see docstring point 2.
    """
    eigs = torch.linalg.svdvals(J.double()).pow(2)
    n_f = J.shape[-1] if n_f is None else n_f
    if eigs.numel() < n_f:
        eigs = torch.cat([eigs, eigs.new_zeros(n_f - eigs.numel())])
    return eigs[:n_f]


def input_fisher(model, X: torch.Tensor, *, project: bool = True,
                 average: bool = False) -> torch.Tensor:
    """``F = J^T J`` per input: ``(N, n_f, n_f)``, or the pool average when ``average=True``.

    The **matrix**, for the consumers that need ``F`` itself rather than its spectrum -- analysis
    B's ``F^+`` and the A/D/E sandwich.  For spectra use :func:`spectrum_from_jacobian` on ``J``.
    """
    mats = []
    for x in X:
        J = sqrt_jacobian(model, x, project=project).double()
        mats.append(J.T @ J)
    F = torch.stack(mats)
    return F.mean(dim=0) if average else F


def tau_N(n: int, gamma: float = 1.0) -> float:
    """``tau_N ~ 2 pi log N / (gamma N)``: the resolution a pool of ``N`` samples can support.

    "Above threshold" means above ``~1/N`` in **absolute** terms, not relative to ``lambda_max``:
    a spectrum with a fine condition number can have every eigenvalue below ``1/N``.
    """
    n = max(int(n), 2)
    return 2.0 * math.pi * math.log(n) / (float(gamma) * n)


def r_eff(eigs: torch.Tensor, tau: float) -> int:
    """Number of eigenvalues above an **absolute** threshold ``tau``."""
    return int((eigs > float(tau)).sum())


def description_cost(eigs: torch.Tensor, eps_sq: float) -> float:
    """``(1/2) sum_i log(1 + lambda_i/eps^2)`` -- reverse water-filling.

    Strictly increasing in ``r_eff`` at fixed trace, so the ordering
    ``r_eff=1 (cheapest) -> gapped -> log-uniform -> flat (most expensive)`` is proved, not
    asserted.  Magnitude enters only logarithmically while the count enters linearly.
    """
    return float(0.5 * torch.log1p(eigs / float(eps_sq)).sum())


def effective_dimension(F: torch.Tensor, n: int, gamma: float = 1.0) -> float:
    """Abbas et al. effective dimension of a **single** (already ``x``-averaged) ``F``.

    The sample-complexity reading of the same spectrum, with ``F`` trace-normalised to ``n_f``.
    """
    n_f = F.shape[-1]
    tr = torch.diagonal(F).sum().clamp(min=1e-30)
    F_hat = F * (n_f / tr)
    kappa = float(gamma) * int(n) / (2.0 * math.pi * math.log(max(int(n), 2)))
    logdet = torch.logdet(torch.eye(n_f, dtype=F_hat.dtype) + kappa * F_hat)
    return float(2.0 * (0.5 * logdet) / math.log(kappa)) if kappa > 1 else float("nan")


def fisher_spectrum(eigs: torch.Tensor, *, n: int, gamma: float = 1.0,
                    rank_tol: float = 1e-10) -> dict:
    """Spread statistics for one spectrum.  Absolute **and** shape, always both.

    ``tr F`` (absolute, in matched physical units) reads as sample complexity; the normalised
    spectrum reads as compressibility and is what makes cross-model comparison legitimate.
    **Normalising away the trace destroys the plateau signal**: if ``lambda_i(S) = 2^-S mu_i`` the
    factor cancels exactly and the normalised spectrum is ``S``-independent, so it cannot see a
    uniform collapse.

    ``rank`` is by **eigenvalue ratio**, not ``matrix_rank``: with a Porter-Thomas spectrum an
    exact rank assertion is not meaningful at float64.  And do **not** read rank as a
    barren-plateau test -- a plateau is typically full rank with every eigenvalue uniformly tiny,
    so only magnitudes detect it, while an exact null direction means the parametrisation is
    redundant and learning is *easier* there.
    """
    eigs = eigs.double()
    tr = float(eigs.sum())
    lam_max = float(eigs.max()) if eigs.numel() else 0.0
    hat = eigs / max(tr, 1e-300)
    nz = hat[hat > 0]
    H = float(-(nz * nz.log()).sum()) if nz.numel() else 0.0
    t = tau_N(n, gamma)
    live = eigs[eigs > rank_tol * max(lam_max, 1e-300)]
    return {
        "trace": tr,                                        # ABSOLUTE -- sample complexity
        "eigenvalues": [float(v) for v in eigs],
        "normalised": [float(v) for v in hat],              # SHAPE -- cross-model comparison
        "lambda_max": lam_max,
        # The raw minimum is a structural zero once project_physical has run (the global-phase
        # direction), so it is ~1e-15 at every (m, k) and carries no information.  The quantity
        # sample complexity actually scales with is the smallest *live* eigenvalue.
        "lambda_min": float(eigs.min()) if eigs.numel() else 0.0,
        "lambda_min_live": float(live.min()) if live.numel() else 0.0,
        "rank": int((eigs > rank_tol * max(lam_max, 1e-300)).sum()),
        "spectral_entropy": H,
        "exp_entropy": math.exp(H),                         # in [1, n_f]
        "participation_ratio": float(tr ** 2 / max(float((eigs ** 2).sum()), 1e-300)),
        "tau_N": t,
        "r_eff": r_eff(eigs, t),
        "description_cost": description_cost(eigs, t),
    }


def r_eff_curve(eigs: torch.Tensor, taus=None) -> list[tuple[float, int]]:
    """``[(tau, r_eff(tau))]`` -- a **diagnostic only**.

    At ``n_features = 6`` the curve has at most 6 steps, which cannot distinguish gapped from
    log-uniform from flat.  Since ``n_features`` is a study invariant this is a permanent
    limitation, not a deferred one: emit the curve, print **no shape label**, and do not fit a
    slope to 6 points.
    """
    if taus is None:
        taus = [10.0 ** e for e in range(-12, 3)]
    return [(float(t), r_eff(eigs, t)) for t in taus]


# --- pool-level summary ---------------------------------------------------------------------- #


def _quantiles(vals: list[float]) -> dict:
    t = torch.tensor(vals, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}


def analyse(model, X: torch.Tensor, *, support: torch.Tensor | None = None,
            gamma: float = 1.0, n_samples: int | None = None) -> dict:
    """Analysis A over an input pool: per-``x`` spectra plus median/IQR summaries.

    One spectrum is only ``n_f`` numbers, so the discriminating power comes from the
    **distribution over ``x``** -- hence median + IQR of every statistic across the pool, and the
    ``x``-averaged ``F``, rather than a single point.

    ``support`` (a boolean mask over the outcome basis) switches to the **conditioned** ``F^(C)``
    of :func:`conditional_sqrt_jacobian`, which is the apples-to-apples boson-vs-determinant
    comparison; without it each model is measured on its own support and the support sizes are
    reported so the inequality is disclosed.

    ``n_samples`` is the **dataset** size that sets the resolution ``tau_N``, and defaults to
    ``len(X)`` only because that is the sole number available here.  They are different quantities:
    ``len(X)`` is how many points the spectrum is averaged over (a precision-of-the-estimate knob),
    while ``tau_N`` asks what resolution a *training pool* of ``N`` rows can support.  Pass the
    config's ``generation.size``, or ``r_eff(tau_N)`` reports the resolution of the metric's own
    subsample rather than of the experiment.
    """
    n = int(n_samples) if n_samples else X.shape[0]
    per_x, supports, phases, mats = [], [], [], []
    for x in X:
        if support is None:
            J = sqrt_jacobian(model, x, project=False)
        else:
            J = conditional_sqrt_jacobian(model, x, support, project=False)
        supports.append(int(J.shape[0]))
        phases.append(float(phase_eigenvalue(J)))
        Jp = project_physical(J)
        per_x.append(fisher_spectrum(spectrum_from_jacobian(Jp), n=n, gamma=gamma))
        mats.append((Jp.double().T @ Jp.double()))

    F_bar = torch.stack(mats).mean(dim=0)
    keys = ("trace", "rank", "r_eff", "exp_entropy", "participation_ratio",
            "lambda_max", "lambda_min_live", "description_cost")
    return {
        "n_x": int(X.shape[0]),
        "n_samples": n,
        "n_features": int(X.shape[1]),
        "conditioned": support is not None,
        "n_support": _quantiles([float(v) for v in supports]),
        "lambda_phase": _quantiles(phases),          # the direction project_physical removes
        "tau_N": tau_N(n, gamma),
        "summary": {k: _quantiles([s[k] for s in per_x]) for k in keys},
        "mean_normalised": [float(v) for v in
                            torch.tensor([s["normalised"] for s in per_x]).mean(dim=0)],
        "F_bar": F_bar,
        "effective_dimension": effective_dimension(F_bar, n, gamma),
        # eigvalsh, not svdvals: an *average* of Grams has no single J to factor, so the
        # squared-condition-number objection of docstring point 2 does not apply (and there is no
        # alternative).  Every per-x spectrum above does go through svdvals(J).
        "r_eff_curve": r_eff_curve(torch.linalg.eigvalsh(F_bar).flip(0).clamp(min=0.0)),
        "per_x": per_x,
    }


def shared_support(models, X: torch.Tensor, *, tol: float = SUPPORT_TOL) -> torch.Tensor:
    """Boolean mask of outcomes on which **every** model has mass, over the whole pool.

    The shared support the conditioned comparison of §3.2 runs on.  Requires the models to agree
    on the outcome basis, which is the point of ``fermion`` reusing the photonic Fock keys.
    """
    keys = [tuple(m.outcome_keys()) for m in models]
    if len({k for k in keys}) != 1:
        raise ValueError("models do not share an outcome basis, so there is no shared support")
    mask = torch.ones(len(keys[0]), dtype=torch.bool)
    for model in models:
        acc = torch.zeros(len(keys[0]), dtype=torch.bool)
        for x in X:
            acc |= (probs_and_jacobian(model, x)[0] > tol)
        mask &= acc
    return mask


# --- CLI ------------------------------------------------------------------------------------- #


def _fmt(q: dict, prec: int = 4) -> str:
    return f"{q['median']:.{prec}g} [{q['q25']:.{prec}g}, {q['q75']:.{prec}g}]"


def _report(label: str, res: dict) -> None:
    s = res["summary"]
    print(f"\n=== {label}   (n_x={res['n_x']}, n_features={res['n_features']}, "
          f"{'CONDITIONED on shared support' if res['conditioned'] else 'own support'})")
    print(f"  support size          {_fmt(res['n_support'], 3)}")
    print(f"  tr F      (ABSOLUTE)  {_fmt(s['trace'])}")
    print(f"  lambda_max            {_fmt(s['lambda_max'])}")
    print(f"  lambda_min (live)     {_fmt(s['lambda_min_live'])}"
          f"   <- smallest of the {int(s['rank']['median'])} live directions")
    print(f"  rank (ratio 1e-10)    {_fmt(s['rank'], 3)}")
    print(f"  r_eff(tau_N={res['tau_N']:.3g})  {_fmt(s['r_eff'], 3)}"
          f"   [tau_N from N={res['n_samples']}]")
    print(f"  exp(H)  in [1, n_f-1] {_fmt(s['exp_entropy'])}")
    print(f"  participation ratio   {_fmt(s['participation_ratio'])}")
    print(f"  description cost      {_fmt(s['description_cost'])}")
    print(f"  lambda_phase (excl.)  {_fmt(res['lambda_phase'])}   "
          f"<- exactly 0 iff n_features == m")
    print(f"  effective dimension   {res['effective_dimension']:.4g}")
    print("  normalised spectrum   " + "  ".join(f"{v:.3g}" for v in res["mean_normalised"]))
    print("  r_eff(tau) curve      " + " ".join(f"{t:.0e}:{r}" for t, r in res["r_eff_curve"])
          + "   [NO shape label at n_f=6 -- see module docstring]")


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from config import load_config
    from model import build_model, sample_X

    ap = argparse.ArgumentParser(description="Analysis A: input Fisher spectrum of p_x alone")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--configs", nargs="+", help="explicit config paths")
    src.add_argument("--configs-dir")
    ap.add_argument("--n-x", type=int, default=64, help="input points to average over")
    ap.add_argument("--shared-support", action="store_true",
                    help="also report F conditioned on the support shared by all models")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="dataset size setting tau_N (default: the config's generation.size)")
    args = ap.parse_args(argv)

    paths = ([Path(p) for p in args.configs] if args.configs else
             sorted(p for p in Path(args.configs_dir).glob("*.yaml") if not p.name.startswith("_")))
    cfgs = [load_config(p) for p in paths]
    from config import check_commensurable
    check_commensurable(cfgs)

    models = [build_model(c) for c in cfgs]
    X = sample_X(args.n_x, cfgs[0].problem.n_features, cfgs[0].seeds.sample_seed)
    print(f"Analysis A: {len(models)} model(s), n_x={args.n_x}, "
          f"n_features={cfgs[0].problem.n_features} (study invariant)")
    print("F is evaluated on the (n_f-1)-dim complement of 1/sqrt(n_f) at every (m,k).")

    for path, cfg, model in zip(paths, cfgs, models):
        ns = args.n_samples or cfg.generation.size
        _report(f"{path.stem}  m={cfg.problem.m} k={cfg.problem.k}",
                analyse(model, X, gamma=args.gamma, n_samples=ns))

    if args.shared_support and len(models) > 1:
        try:
            mask = shared_support(models, X[:min(8, len(X))])
        except ValueError as e:
            print(f"\n[shared-support] skipped: {e}")
            return
        print(f"\n### shared support: {int(mask.sum())} of {mask.numel()} outcomes "
              f"(q-weighted, q-mean-CENTERED scores)")
        for path, model in zip(paths, models):
            _report(f"{path.stem}  [conditioned]",
                    analyse(model, X, support=mask, gamma=args.gamma,
                            n_samples=args.n_samples or cfgs[0].generation.size))


if __name__ == "__main__":
    main()
