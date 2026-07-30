"""Is the nonlinear-observable hardness in the functional, or in the quantum feature map?

Runs the matched comparison: the SAME observables through the SAME scoring code on three maps
that share the Fock basis and differ only in ``x -> p``, so any difference is the map's --

===========  ====================  ==============================================================
label        teacher               map, parameter count
===========  ====================  ==============================================================
``quantum``  ``photonic_quantum``  ``W2 P(x) W1`` boson sampling; ``2m^2`` params, #P-hard amps
``ebm``      ``ebm_fock``          ``p ~ exp(theta(x).phi(n))``; ``O(m^2)`` -- POLY-SIZE control
``mlp``      ``mlp_fock``          dense ``2*n_fock`` output; params exponential in m -- upper bound
===========  ====================  ==============================================================

``ebm`` is the control that can answer "is a small classical model enough".  ``mlp`` cannot: its
parameter count grows exponentially in ``m`` (19.9M against the photonic 392 at ``m=14``), which
hands the classical side the very resource the quantum claim is about, so it only upper-bounds what
an unconstrained classical map could do.

Three stages, deliberately in this order:

1. ``p`` diagnostics.  ``osc``/``ent`` live in the small-``p`` tail, so if the maps' distributions
   had different tails we would be measuring that instead of the map.  Flagged per map.
2. ``parity`` calibration.  A map with too much high-frequency content is hard to learn *whatever*
   observable sits on top, so ``parity`` must be easy on a map before its nonlinear rows mean
   anything.
3. Learnability (test R^2 of an RBF kernel ridge on ``x``) and shot-estimability, per observable.

    python compare_classical_control.py [--m 6] [--n 10000]
"""

from __future__ import annotations

import argparse
import math
from math import comb

import numpy as np
import torch

from model.ebm_fock import EbmFockTeacher
from model.mlp_fock import MlpFockTeacher
from model.photonic import PhotonicTeacher
from model.sampler import sample_X

MAPS = ("quantum", "ebm", "mlp")
TEACHERS = {"quantum": PhotonicTeacher, "ebm": EbmFockTeacher, "mlp": MlpFockTeacher}
OBSERVABLES = ("parity", "majority", "sq_parity", "ent", "ent_parity", "osc", "osc_parity")

#: R^2 above which we call a target "learnable" by the RBF kernel ridge.
EASY = 0.5


def build(kind: str, m: int, k: int, obs: str, seed: int):
    return TEACHERS[kind](m=m, k=k, n_features=m - 1, observable=obs, seed=seed)


def raw_probs(t, X):
    """The teacher's full ``(N, n_fock)`` distribution, before the observable."""
    return t.layer.forward(X) if isinstance(t, PhotonicTeacher) else t.probs(X)


def p_summary(p):
    p = p.double()
    p = p / p.sum(dim=1, keepdim=True)
    lp = torch.log10(p.clamp(min=1e-30))
    ent = -(p * torch.log(p.clamp(min=1e-30))).sum(dim=1)
    return {"log10_min": float(lp.min()), "log10_med": float(lp.median()),
            # the 0.1st percentile of log10 p measures how DEEP the small-p tail runs, which is
            # what osc/ent actually see.  An absolute threshold on "fraction below 1e-4" is
            # useless here -- the quantity itself is ~0.008, so a map with a ZERO tail passes it.
            "log10_p001": float(np.percentile(lp.numpy(), 0.1)),
            "entropy": float(ent.mean()), "tail_1e4": float((p < 1e-4).double().mean())}


#: Rows used to *fit* the kernel ridge.  The pool stays at ``--n`` (the configs generate 10k) but
#: an exact KRR solve is O(n_fit^3), so fitting on all 8000 would take hours per target.  The
#: kernel depends only on ``(X, gamma)`` -- never on the observable -- so :class:`RbfRidgeBank`
#: eigendecomposes once and every one of the 21 (observable, map) targets is then a matvec.
KRR_FIT_CAP = 3000
KRR_GAMMAS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
KRR_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2)


def _sqdist(A, B):
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)


class RbfRidgeBank:
    """Pre-factorised RBF kernel ridge: eigendecompose once per gamma, reuse for every target.

    ``K = Q diag(w) Q^T`` makes the ridge solution ``Q diag(1/(w+alpha)) Q^T y`` free in ``alpha``,
    so the whole hyperparameter grid costs one ``eigh`` per gamma regardless of how many targets
    are scored against it.
    """

    def __init__(self, Xtr, Xte, gammas=KRR_GAMMAS):
        d2_tr, d2_te = _sqdist(Xtr, Xtr), _sqdist(Xte, Xtr)
        self.bank = []
        for g in gammas:
            w, Q = np.linalg.eigh(np.exp(-g * d2_tr))
            self.bank.append((w, Q, np.exp(-g * d2_te)))

    def best_r2(self, ytr, yte, alphas=KRR_ALPHAS):
        """Best test R^2 over the (gamma, alpha) grid for one target."""
        mu, sd = ytr.mean(), ytr.std() + 1e-12
        z = (ytr - mu) / sd
        denom = ((yte - yte.mean()) ** 2).sum() + 1e-30
        best = -np.inf
        for w, Q, Kte in self.bank:
            qty = Q.T @ z
            for a in alphas:
                pred = Kte @ (Q @ (qty / (w + a))) * sd + mu
                best = max(best, 1.0 - ((yte - pred) ** 2).sum() / denom)
        return best


def shot_r(t, X, shots, seed=0):
    """Pearson r between the score from exact p and from a ``shots``-shot empirical p."""
    p = raw_probs(t, X).double()
    p = (p / p.sum(dim=1, keepdim=True)).numpy()
    rng = np.random.default_rng(seed)
    ps = torch.tensor(np.stack([rng.multinomial(shots, r) / shots for r in p]),
                      dtype=torch.float32)
    a = t.obs.score(raw_probs(t, X)).numpy()
    b = t.obs.score(ps).numpy()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    m, k = args.m, args.m // 2
    nf, n_fock = m - 1, comb(m + k - 1, k)
    n_train = int(0.8 * args.n)
    X = sample_X(args.n, nf, seed=args.seed)
    Xn = X.numpy()

    n_fit = min(n_train, KRR_FIT_CAP)
    print(f"m={m}  k=m//2={k}  n_fock={n_fock}  n_features={nf}  "
          f"N={args.n} ({n_train} train / {args.n - n_train} test; KRR fits on {n_fit})")
    ebm = build("ebm", m, k, "parity", args.seed)
    print(f"model size: photonic 2m^2={2 * m * m}   ebm theta={ebm.n_model_parameters():,}"
          f"   mlp={sum(p.numel() for p in build('mlp', m, k, 'parity', args.seed).net.parameters()):,}\n")

    # --- 1. distribution diagnostics ----------------------------------------------------- #
    print("=" * 84)
    print("1. p diagnostics -- a map whose tail differs from the quantum one is confounded")
    print("=" * 84)
    print(f"  {'map':10s} {'log10 p med':>12s} {'log10 p @0.1%':>14s} {'log10 p min':>12s}"
          f" {'entropy':>9s} {'/ ln(n_fock)':>13s} {'frac p<1e-4':>12s}")
    summ = {}
    for kind in MAPS:
        summ[kind] = p_summary(raw_probs(build(kind, m, k, "parity", args.seed), X))
        s = summ[kind]
        print(f"  {kind:10s} {s['log10_med']:12.2f} {s['log10_p001']:14.2f} "
              f"{s['log10_min']:12.2f} {s['entropy']:9.3f} {math.log(n_fock):13.3f} "
              f"{s['tail_1e4']:12.4f}")
    print(f"\n  {'vs quantum':10s} {'d median':>10s} {'d entropy':>10s} {'d tail':>9s}   verdict")
    tail_ok = {}
    for kind in MAPS[1:]:
        dm = abs(summ["quantum"]["log10_med"] - summ[kind]["log10_med"])
        de = abs(summ["quantum"]["entropy"] - summ[kind]["entropy"])
        dt = abs(summ["quantum"]["tail_1e4"] - summ[kind]["tail_1e4"])
        tail_ok[kind] = dm < 0.5 and de < 0.3 and dt < 0.05
        print(f"  {kind:10s} {dm:10.3f} {de:10.3f} {dt:9.4f}   "
              f"{'MATCH' if tail_ok[kind] else 'DIVERGE -> this column is CONFOUNDED'}")
    print("  (thresholds: d median < 0.5 decade, d entropy < 0.3 nat, d tail < 0.05)")

    # --- 2 + 3. learnability and shot-estimability ---------------------------------------- #
    print("\n" + "=" * 84)
    print("2/3. learnability (test R^2, RBF kernel ridge on x) and shot-estimability")
    print("=" * 84)
    print("  factorising the RBF kernel bank (once, shared by every target) ...")
    bank = RbfRidgeBank(Xn[:n_fit], Xn[n_train:])
    print(f"  {'observable':13s} {'map':9s} {'std(y)':>8s} {'R2':>7s} {'r@1e3':>7s} {'r@1e5':>7s}")
    res = {}
    for obs in OBSERVABLES:
        for kind in MAPS:
            t = build(kind, m, k, obs, args.seed)
            y = t(X).squeeze(-1).numpy().astype(np.float64)
            if y.std() < 1e-9:
                print(f"  {obs:13s} {kind:9s} {y.std():8.1e}   DEGENERATE (constant output)")
                res[(obs, kind)] = float("nan")
                continue
            res[(obs, kind)] = bank.best_r2(y[:n_fit], y[n_train:])
            print(f"  {obs:13s} {kind:9s} {y.std():8.4f} {res[(obs, kind)]:7.3f} "
                  f"{shot_r(t, X[:300], 1000):7.3f} {shot_r(t, X[:300], 100000):7.3f}")
        print()

    # --- verdict ------------------------------------------------------------------------- #
    print("=" * 84)
    print("verdict")
    print("=" * 84)
    cal = {kind: res[("parity", kind)] for kind in MAPS}
    print("  calibration (parity R2): " + "   ".join(f"{k}={v:.3f}" for k, v in cal.items()))
    bad = [k for k, v in cal.items() if not np.isfinite(v) or v < EASY]
    if bad:
        print(f"    !! parity is NOT easy on {bad} -- too much high-frequency content in the map.")
        print("       Lower that map's *_FOURIER_ORDER and re-run; its nonlinear rows below are")
        print("       NOT interpretable until this is fixed.")
    else:
        print("    parity easy on every map -> the maps are benign, nonlinear rows are meaningful.")
    print()
    for obs in OBSERVABLES:
        if obs in ("parity", "majority"):
            continue
        rq, rebm, rmlp = (res[(obs, "quantum")], res[(obs, "ebm")], res[(obs, "mlp")])
        if not all(np.isfinite(v) for v in (rq, rebm, rmlp)):
            continue
        if cal["ebm"] < EASY or not tail_ok["ebm"]:
            msg = "ebm column not interpretable (calibration or tail failed)"
        elif rebm >= EASY and rq < EASY:
            msg = "hardness NEEDS THE QUANTUM MAP (a poly-size classical model is easy)"
        elif rebm < EASY and rq < EASY:
            msg = "hard on BOTH -> hardness is in the p->score FUNCTIONAL, not the map"
        elif rebm >= EASY and rq >= EASY:
            msg = "easy on both -> not a hardness source at all"
        else:
            msg = "easy on quantum, hard for a poly-size classical model"
        print(f"  {obs:13s} R2  q={rq:6.3f}  ebm={rebm:6.3f}  mlp={rmlp:6.3f}  -> {msg}")


if __name__ == "__main__":
    main()
