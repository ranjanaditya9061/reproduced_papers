"""The input Fisher matrix of ``p_x`` at **one** input, by finite differences.

    python -m metrics.fisher --config configs/photonic.yaml --index 0
    python -m metrics.fisher --config configs/photonic.yaml --index 0 --shots 10000

One job: take a config and a row index, perturb that input, and return

    F_ij = sum_n (d_i p_n)(d_j p_n) / p_n        the n_f x n_f input Fisher matrix

No pool averaging, no summary statistics, no projection -- those live in :mod:`.distribution`.

**Only first derivatives are needed.**  The score outer product above is an exact identity.  The
other familiar form, ``F_ij = -E[d_i d_j log p]``, equals it only in expectation under a
correctly-specified model; the outer product needs half the evaluations, is positive semi-definite by
construction, and avoids squaring the finite-difference error.  So ``2 n_f`` perturbed evaluations,
12 at ``n_features = 6``.

**Why F is finite.**  ``(d_i p)(d_j p) / p`` looks divergent as ``p_n -> 0``, and it is *not* finite
for the reason ``p log p`` is.  Write ``p_n = a_n^2`` with ``a_n`` the amplitude modulus::

    (d_i p)(d_j p) / p  =  (2a d_i a)(2a d_j a) / a^2  =  4 (d_i a)(d_j a)

The ``p`` cancels exactly, leaving ``F_ij = 4 sum_n d_i sqrt(p_n) d_j sqrt(p_n)`` -- finite whenever
the amplitude is smooth, which it is (``p_n = |Perm(U_{s,n})|^2 / prod n_j!`` is analytic in ``x``).
A vanishing outcome contributes a finite and generally **nonzero** amount; nothing is suppressed by
``p -> 0``.  That is why :func:`fisher` builds ``J_ni = d_i p_n / sqrt(p_n) = 2 d_i sqrt(p_n)`` and
never forms the ratio outcome-by-outcome.

**The cancellation does not survive shots.**  Numerically ``d_i p_n`` carries an error that knows
nothing about ``p_n``.  On the exact path that is float round-off and :data:`SUPPORT_TOL` handles it.
On the shots path ``Var(p_hat) ~ p/S``, so every outcome adds ``2 Var(p_hat)/p/(2 eps)^2 =
1/(2 S eps^2)`` to ``F_ii`` *regardless of how small ``p`` is* -- a floor of ``n_supp/(2 S eps^2)``,
about 28 at ``S=1e4, eps=1e-2, n_supp=56``, which swamps the true ``F``.  Shot Fisher is biased **up**
by that noise and **down** by outcomes never observed.  :func:`fisher` returns the floor so it is
reported beside the trace rather than mistaken for a measurement.

Measured at ``m=6, k=3``, ``eps=1e-2``, common random numbers, against an exact ``tr F = 5.61``:

=========  =======  ==========  ==========  ================
shots      support  ``tr F``    excess      predicted floor
=========  =======  ==========  ==========  ================
1e3        51       185.1       179.5       255
1e4        56       50.03       44.42       28
1e5        56       12.16       6.54        2.8
exact      56       5.614       --          --
=========  =======  ==========  ==========  ================

The excess decays with ``S`` as the floor predicts but sits ~1.5-2.3x **above** it, because the floor
is only the leading term: the estimator divides by the *noisy* ``p_hat``, and ``E[1/p_hat] > 1/p`` for
rare outcomes -- an outcome seen once contributes ``1/p_hat = S``.  So the floor is a lower bound on
the bias, not an estimate of it.  At ``1e5`` shots the trace is still **2.2x** the true value, and at
``1e3`` the support itself is short by 5 outcomes.  The practical reading: this path is a diagnostic
of the estimator, and a plug-in Fisher from shots should not be reported as a measurement of ``F``
at any budget reachable here.
"""

from __future__ import annotations

import torch

#: Central-difference step.  Large deliberately: the error is ``O(eps^2)`` truncation plus
#: ``O(float_eps/eps)`` round-off and these models are float32-rooted, so shrinking it makes the
#: exact path *worse*.  On the shots path the noise floor goes as ``1/eps^2``, so shrinking it there
#: is actively harmful.
EPS = 1e-2

#: Outcomes with ``p <= tol`` are **masked, not clamped**.  ``(d_i p)^2 / p`` is a genuine ``0/0``
#: there -- for ``fermion`` at ``flavours=1`` whole bunched outcomes are identically zero in ``x``.
#: On the shots path this is also what drops never-observed outcomes.
SUPPORT_TOL = 1e-12


def probs_fn(model, *, shots: int = 0, shot_seed: int = 0):
    """``x (n_f,) -> p (n_out,)`` on the model's **declared** basis.  ``shots=0`` is exact.

    The basis has to be declared rather than observed: a shot draw reports only the outcomes it saw
    and that set moves with ``x``, so ``p(x+h)`` and ``p(x-h)`` would otherwise be vectors over
    different outcomes and their difference would be meaningless.

    Every shot leg uses the same ``shot_seed`` with no offset, so the legs are drawn from the same
    stream -- common random numbers, which cancels much of the sampling noise in the difference.
    That reduces the variance of the estimate; it does not remove the bias.
    """
    if shots <= 0:
        return lambda x: model.probs(x.unsqueeze(0), grad=False)[0].double()

    index = {tuple(int(c) for c in key): i for i, key in enumerate(model.outcome_keys())}

    def fn(x: torch.Tensor) -> torch.Tensor:
        seq = model.shot_counts(x.unsqueeze(0), shots=int(shots), shot_seed=shot_seed)[0]
        counts = torch.zeros(len(index), dtype=torch.float64)
        for key in seq:
            counts[index[key]] += 1.0
        return counts / counts.sum().clamp(min=1.0)

    return fn


def probs_and_jacobian(fn, x: torch.Tensor, *, eps: float = EPS):
    """``(p, dp)``: ``p (n_out,)`` at ``x`` and ``dp (n_out, n_f)`` by central differences.

    ``2 n_f + 1`` calls to ``fn``, one leg at a time.  Not batched into a single ``model.probs``
    call, because the shots path has no such call and comparing the two is the point.
    """
    p = fn(x)
    cols = []
    for i in range(int(x.shape[0])):
        h = torch.zeros_like(x)
        h[i] = float(eps)
        cols.append((fn(x + h) - fn(x - h)) / (2.0 * float(eps)))
    return p.detach(), torch.stack(cols, dim=1).detach()


def fisher(fn, x: torch.Tensor, *, eps: float = EPS, tol: float = SUPPORT_TOL,
           shots: int = 0) -> tuple[torch.Tensor, dict]:
    """``(F, detail)``: the ``(n_f, n_f)`` input Fisher matrix at ``x``, and what qualifies it.

    ``F = J^T J`` from ``J_ni = d_i p_n / sqrt(p_n)``, built as a Gram rather than a sum of outer
    products so it is symmetric PSD to round-off with no clean-up step.
    """
    p, dp = probs_and_jacobian(fn, x, eps=eps)
    keep = p > float(tol)
    J = dp[keep] / p[keep].sqrt().unsqueeze(1)
    F = J.T @ J

    n_supp = int(keep.sum())
    detail = {
        "n_support": n_supp,
        "n_outcomes": int(p.numel()),
        "trace": float(torch.diagonal(F).sum()),
        "eigenvalues": torch.linalg.svdvals(J)**2,
        # 1^T F 1 / n_f.  When n_features == m this is the global-phase direction and must come out
        # ~0; it is the cheapest check that the Jacobian is right.
        "phase_direction": float(F.sum() / F.shape[0]),
        # Sampling noise contributes ~1/(2 S eps^2) per surviving outcome to each diagonal entry,
        # independent of p.  If this is not far below the trace, the trace is noise.
        "noise_floor": (n_supp / (2.0 * float(shots) * float(eps) ** 2)) if shots > 0 else 0.0,
    }
    return F, detail


def fisher_for_config(cfg, index: int = 0, *, eps: float = EPS, shots: int = 0):
    """``(F, detail, x)`` for row ``index`` of the config's input pool.

    ``x`` comes from the same ``sample_X`` draw the dataset and shots branches use, so this is the
    Fisher info at a real training input and is reproducible from the config alone.
    """
    from model import build_model, sample_X

    model = build_model(cfg)
    X = sample_X(int(cfg.generation.size), int(cfg.problem.n_features), int(cfg.seeds.sample_seed))
    if not 0 <= int(index) < X.shape[0]:
        raise IndexError(f"index {index} outside the pool of {X.shape[0]} rows "
                         f"(generation.size); lower it or raise generation.size")
    x = X[int(index)]
    fn = probs_fn(model, shots=shots, shot_seed=int(cfg.seeds.shot_seed))
    F, detail = fisher(fn, x, eps=eps, shots=shots)
    return F, detail, x


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    if __package__ in (None, ""):                    # allow `python metrics/fisher.py`
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0, help="row of the input pool to evaluate at")
    ap.add_argument("--eps", type=float, default=EPS, help="central-difference step")
    ap.add_argument("--shots", type=int, default=0, help="0 = exact; >0 = finite-shot estimate")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    F, d, x = fisher_for_config(cfg, args.index, eps=args.eps, shots=args.shots)

    src = "exact" if args.shots <= 0 else f"{args.shots} shots"
    print(f"\n=== {Path(args.config).stem}  m={cfg.problem.m} k={cfg.problem.k}  "
          f"x=row {args.index}  ({src}, eps={args.eps:g})")
    print("  x                 " + "  ".join(f"{float(v):+.4f}" for v in x))
    print(f"  support           {d['n_support']} of {d['n_outcomes']} outcomes")
    print("\n  F =")
    for row in F:
        print("      " + "  ".join(f"{float(v):+.6g}".rjust(13) for v in row))
    print("\n  eigenvalues       " + "  ".join(f"{float(v):.6g}" for v in d["eigenvalues"]))
    print(f"  trace             {d['trace']:.6g}")
    if args.shots > 0:
        floor = d["noise_floor"]
        verdict = "DOMINATED BY NOISE" if floor > 0.1 * abs(d["trace"]) else "ok"
        print(f"  shot noise floor  {floor:.6g}   <- n_supp/(2 S eps^2), {verdict}")
    print(f"  phase direction   {d['phase_direction']:.6g}   "
          f"<- 1^T F 1/n_f, ~0 iff n_features == m")


if __name__ == "__main__":
    main()
