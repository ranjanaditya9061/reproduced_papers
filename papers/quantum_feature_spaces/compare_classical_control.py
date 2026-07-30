"""Is the nonlinear-observable hardness in the functional, or in the quantum feature map?

Runs the matched 2 x N comparison: the SAME observables through the SAME scoring code on two
maps -- ``photonic_quantum`` (``W2 P(x) W1`` boson sampling) and ``mlp_fock`` (a random MLP over
the identical Fock basis, classical) -- so any difference is attributable to the map alone.

Three stages, deliberately in this order:

1. ``p`` diagnostics.  ``osc`` lives in the small-``p`` tail, so if the two maps' distributions
   had different tails we would be measuring that instead of the map.  Bail out if they diverge.
2. ``parity`` calibration.  A random MLP with too much high-frequency content is hard to learn
   *whatever* observable sits on top, so ``parity`` must be easy on both maps before any
   nonlinear number means anything.
3. Learnability (test R^2 of an RBF kernel ridge on ``x``) and shot-estimability, per observable.

    python compare_classical_control.py [--m 6] [--n 3000]
"""

from __future__ import annotations

import argparse
import math
from math import comb

import numpy as np
import torch

from model.mlp_fock import MlpFockTeacher
from model.photonic import PhotonicTeacher
from model.sampler import sample_X

OBSERVABLES = ("parity", "majority", "sq_parity", "ent", "ent_parity", "osc", "osc_parity")


def build(kind: str, m: int, k: int, obs: str, seed: int):
    nf = m - 1
    if kind == "quantum":
        return PhotonicTeacher(m=m, k=k, n_features=nf, observable=obs, seed=seed)
    return MlpFockTeacher(m=m, k=k, n_features=nf, observable=obs, seed=seed)


def raw_probs(t, X):
    return t.layer.forward(X) if isinstance(t, PhotonicTeacher) else t.probs(X)


def p_summary(p):
    p = p.double()
    p = p / p.sum(dim=1, keepdim=True)
    lp = torch.log10(p.clamp(min=1e-30))
    ent = -(p * torch.log(p.clamp(min=1e-30))).sum(dim=1)
    return {"log10_min": float(lp.min()), "log10_med": float(lp.median()),
            "entropy": float(ent.mean()), "tail_1e4": float((p < 1e-4).double().mean())}


def krr_r2(X, y, n_train, gammas=(0.05, 0.1, 0.25, 0.5, 1.0), alphas=(1e-6, 1e-4, 1e-2)):
    """Best test R^2 over an RBF kernel-ridge hyperparameter grid (a strong classical learner)."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    mu, sd = ytr.mean(), ytr.std() + 1e-12
    best = -np.inf
    for g in gammas:
        for a in alphas:
            model = KernelRidge(kernel="rbf", gamma=g, alpha=a).fit(Xtr, (ytr - mu) / sd)
            best = max(best, r2_score(yte, model.predict(Xte) * sd + mu))
    return best


def shot_r(t, X, shots, seed=0):
    """Pearson r between the score from exact p and from an ``shots``-shot empirical p."""
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
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    m = args.m
    k = m // 2
    nf, n_fock = m - 1, comb(m + k - 1, k)
    n_train = int(0.8 * args.n)
    X = sample_X(args.n, nf, seed=args.seed)
    Xn = X.numpy()

    print(f"m={m}  k=m//2={k}  n_fock={n_fock}  n_features={nf}  "
          f"N={args.n} ({n_train} train / {args.n - n_train} test)\n")

    # --- 1. distribution diagnostics ----------------------------------------------------- #
    print("=" * 78)
    print("1. p diagnostics -- the two maps must share their tail, else the rest is confounded")
    print("=" * 78)
    print(f"  {'map':12s} {'log10 p min':>12s} {'log10 p med':>12s} {'entropy':>9s}"
          f" {'/ ln(n_fock)':>13s} {'frac p<1e-4':>12s}")
    summ = {}
    for kind in ("quantum", "classical"):
        t = build(kind, m, k, "parity", args.seed)
        summ[kind] = p_summary(raw_probs(t, X))
        s = summ[kind]
        print(f"  {kind:12s} {s['log10_min']:12.2f} {s['log10_med']:12.2f} "
              f"{s['entropy']:9.3f} {math.log(n_fock):13.3f} {s['tail_1e4']:12.4f}")
    dmed = abs(summ["quantum"]["log10_med"] - summ["classical"]["log10_med"])
    dent = abs(summ["quantum"]["entropy"] - summ["classical"]["entropy"])
    dtail = abs(summ["quantum"]["tail_1e4"] - summ["classical"]["tail_1e4"])
    ok = dmed < 0.5 and dent < 0.3 and dtail < 0.05
    print(f"\n  delta median log10 p = {dmed:.3f} (<0.5), delta entropy = {dent:.3f} (<0.3), "
          f"delta tail = {dtail:.4f} (<0.05)")
    print(f"  => distributions {'MATCH -- comparison is valid' if ok else 'DIVERGE -- results below are CONFOUNDED'}")

    # --- 2 + 3. learnability and shot-estimability ---------------------------------------- #
    print("\n" + "=" * 78)
    print("2/3. learnability (test R^2, RBF kernel ridge on x) and shot-estimability")
    print("=" * 78)
    print(f"  {'observable':13s} {'map':10s} {'std(y)':>8s} {'R2':>7s} "
          f"{'r@1e3':>7s} {'r@1e5':>7s}")
    res = {}
    for obs in OBSERVABLES:
        for kind in ("quantum", "classical"):
            t = build(kind, m, k, obs, args.seed)
            y = t(X).squeeze(-1).numpy().astype(np.float64)
            if y.std() < 1e-9:
                print(f"  {obs:13s} {kind:10s} {y.std():8.1e}  DEGENERATE (constant output)")
                res[(obs, kind)] = (y.std(), float("nan"))
                continue
            r2 = krr_r2(Xn, y, n_train)
            res[(obs, kind)] = (y.std(), r2)
            print(f"  {obs:13s} {kind:10s} {y.std():8.4f} {r2:7.3f} "
                  f"{shot_r(t, X[:300], 1000):7.3f} {shot_r(t, X[:300], 100000):7.3f}")

    # --- verdict ------------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("verdict")
    print("=" * 78)
    pq, pc = res[("parity", "quantum")][1], res[("parity", "classical")][1]
    print(f"  calibration: parity R2 = {pq:.3f} (quantum) / {pc:.3f} (classical)")
    if pc < 0.5:
        print("    !! parity is NOT easy on the classical map -- the MLP is too high-frequency.")
        print("       Lower MLP_FOURIER_ORDER / MLP_WEIGHT_GAIN and re-run; nonlinear numbers")
        print("       below are not interpretable until this is fixed.")
        return
    print("    parity easy on both -> the maps are benign, so the nonlinear rows are meaningful.")
    for obs in OBSERVABLES:
        if obs in ("parity", "majority"):
            continue
        rq, rc = res[(obs, "quantum")][1], res[(obs, "classical")][1]
        if not np.isfinite(rq) or not np.isfinite(rc):
            continue
        if rc >= 0.5 and rq < 0.5:
            msg = "hardness needs the QUANTUM MAP (classical control is easy)"
        elif rc < 0.5 and rq < 0.5:
            msg = "hard on BOTH -> hardness is in the p->score functional, not the map"
        elif rc >= 0.5 and rq >= 0.5:
            msg = "easy on both -> this observable is not a hardness source at all"
        else:
            msg = "easy on quantum but hard classically -> the MLP map is the obstacle here"
        print(f"  {obs:13s} R2 {rq:6.3f} (q) vs {rc:6.3f} (c)  -> {msg}")


if __name__ == "__main__":
    main()
