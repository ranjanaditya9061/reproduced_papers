"""Ad-hoc: variance / N_eff sweep of the photonic teacher output over (m, k).

For each (m, k):  X ~ U[0,2pi]^(m-1);  vals = probs @ score_vec  (parity observable,
so Var(score_vec) ~ 1 across configs -> Var(vals) reflects concentration only).
Reports n_fock, Var(vec), Var(vals), and N_eff two ways:
  N_eff_w = 1 / sum(pbar^2)          (inverse participation ratio of the mean dist)
  N_eff_v = Var(vec) / Var(vals)     (backed out from the observed shrinkage)
"""
import sys, time, math, os, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from model.photonic import PhotonicTeacher
from model.sampler import sample_X

N = 1000
SEED = 42
CONFIGS = [(6, 3), (8, 4), (10, 5), (12, 6)]

print(f"{'m,k':>7} {'n_fock':>8} {'Var(vec)':>9} {'Var(vals)':>11} "
      f"{'N_eff_w':>9} {'N_eff_v':>9} {'std(vals)':>9} {'range':>16}  {'t(s)':>6}")
for m, k in CONFIGS:
    t0 = time.time()
    nf = m - 1
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        teacher = PhotonicTeacher(m=m, k=k, n_features=nf, observable="parity", seed=SEED)
    X = sample_X(N, nf, SEED)
    probs = teacher.layer.forward(X)                    # (N, n_fock)
    vec = teacher.score_vec                             # (n_fock,)
    vals = (probs @ vec)                                # (N,)

    pbar = probs.mean(0)
    n_fock = probs.shape[1]
    var_vec = float(vec.var(unbiased=False))
    var_vals = float(vals.var(unbiased=False))
    neff_w = float(1.0 / (pbar**2).sum())
    neff_v = var_vec / var_vals if var_vals > 0 else float("nan")
    rng = (float(vals.min()), float(vals.max()))
    print(f"{f'{m},{k}':>7} {n_fock:>8d} {var_vec:>9.4f} {var_vals:>11.6f} "
          f"{neff_w:>9.1f} {neff_v:>9.1f} {math.sqrt(var_vals):>9.4f} "
          f"[{rng[0]:+.3f},{rng[1]:+.3f}]  {time.time()-t0:>6.1f}", flush=True)
