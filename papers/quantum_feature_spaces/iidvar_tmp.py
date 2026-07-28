"""IID control for the photonic variance sweep.

Instead of the boson sampler, each input draws a genuinely independent random
distribution over the same n_fock Fock states: Porter-Thomas = Dirichlet(1,...,1)
= normalized iid Exp(1) (the Haar-random-state null). Score vector is random +-1
so Var(vec)=1, matching the `parity` photonic run. This is the idealized
"branch A": independent fluctuations, full delocalization -> Var(vals) ~ 1/n_fock.
"""
import math
import torch

N = 1000
SEED = 42
# (label m,k) -> n_fock = C(m+k-1, k), same basis sizes as the photonic sweep
CONFIGS = [((6, 3), 56), ((8, 4), 330), ((10, 5), 2002), ((12, 6), 12376)]

g = torch.Generator().manual_seed(SEED)
print(f"{'m,k':>7} {'n_fock':>8} {'Var(vec)':>9} {'Var(vals)':>11} "
      f"{'N_eff_w':>9} {'N_eff_v':>9} {'std(vals)':>9} {'n*Var':>7} {'range':>16}")
for (m, k), n_fock in CONFIGS:
    # Dirichlet(1): normalize iid Exp(1) = -log(U)
    e = -torch.log(torch.rand(N, n_fock, generator=g))
    probs = e / e.sum(1, keepdim=True)                 # (N, n_fock), rows sum to 1
    vec = (torch.randint(0, 2, (n_fock,), generator=g).float() * 2 - 1)  # +-1
    vals = probs @ vec

    pbar = probs.mean(0)
    var_vec = float(vec.var(unbiased=False))
    var_vals = float(vals.var(unbiased=False))
    neff_w = float(1.0 / (pbar**2).sum())
    neff_v = var_vec / var_vals if var_vals > 0 else float("nan")
    print(f"{f'{m},{k}':>7} {n_fock:>8d} {var_vec:>9.4f} {var_vals:>11.6f} "
          f"{neff_w:>9.1f} {neff_v:>9.1f} {math.sqrt(var_vals):>9.4f} "
          f"{n_fock*var_vals:>7.3f} [{float(vals.min()):+.3f},{float(vals.max()):+.3f}]", flush=True)
