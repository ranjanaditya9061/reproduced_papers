"""Photonic vs iid variance sweep with the REAL continuous score vector.

observable = prod_parity_consecutive_random -> score_vec[j] = cos(P(n_j)),
continuous in [-1, 1].  For each (m, k) we take the exact same continuous vec and
feed it (a) the real boson-sampler probs and (b) iid Porter-Thomas (Dirichlet(1))
probs, so the ONLY difference is structured-vs-iid distributions.
"""
import sys, os, math, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from model.photonic import PhotonicTeacher
from model.sampler import sample_X

N = 1000
SEED = 42
OBS = "prod_parity_consecutive_random"
CONFIGS = [(6, 3), (8, 4), (10, 5), (12, 6)]

g = torch.Generator().manual_seed(SEED)
hdr = (f"{'m,k':>7} {'n_fock':>7} {'kind':>5} {'Var(vec)':>9} {'Var(vals)':>11} "
       f"{'N_eff_w':>9} {'N_eff_v':>9} {'std(vals)':>9} {'nVar/Vv':>8} {'vec[min,max,mean]':>22}")
print(hdr)
for m, k in CONFIGS:
    nf = m - 1
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        teacher = PhotonicTeacher(m=m, k=k, n_features=nf, observable=OBS, seed=SEED)
    vec = teacher.score_vec                             # (n_fock,) continuous cos(P(n))
    n_fock = vec.shape[0]
    var_vec = float(vec.var(unbiased=False))
    vinfo = f"[{float(vec.min()):+.2f},{float(vec.max()):+.2f},{float(vec.mean()):+.2f}]"

    X = sample_X(N, nf, SEED)
    probs_real = teacher.layer.forward(X)               # (N, n_fock)

    e = -torch.log(torch.rand(N, n_fock, generator=g))  # Dirichlet(1) iid
    probs_iid = e / e.sum(1, keepdim=True)

    for kind, probs in (("phot", probs_real), ("iid", probs_iid)):
        vals = probs @ vec
        pbar = probs.mean(0)
        var_vals = float(vals.var(unbiased=False))
        neff_w = float(1.0 / (pbar**2).sum())
        neff_v = var_vec / var_vals if var_vals > 0 else float("nan")
        print(f"{f'{m},{k}':>7} {n_fock:>7d} {kind:>5} {var_vec:>9.4f} {var_vals:>11.6f} "
              f"{neff_w:>9.1f} {neff_v:>9.1f} {math.sqrt(var_vals):>9.4f} "
              f"{n_fock*var_vals/var_vec:>8.3f} {vinfo:>22}", flush=True)
