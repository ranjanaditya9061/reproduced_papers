"""MKL learnability curve: does a *linear combination* of classical kernels learn the
target, and how much kernel dictionary does it take?  One line per dataset.

The complexity axis is the **kernel dictionary** (its size / max interaction degree),
*not* the sample count.  At each dictionary size we:

1. build the candidate kernels (:func:`kernel_dictionary`),
2. find the best nonnegative linear combination ``K = sum_i mu_i K_i`` in closed form by
   **centered kernel-target alignment** (:func:`alignf`, Cortes-Mohri-Rostamizadeh 2012) --
   no training in the loop,
3. evaluate that single combined kernel with **closed-form KRR** (:func:`krr_eval`),
   reading off test ``R^2``, the RKHS-norm certificate ``||f||_H^2 = alpha^T K alpha``
   (small = learnable, exponential = not), and the effective dimension.

The point (see the module docstring in :mod:`learn.capacity` for the model-size cousin):

- ``dict_kind='rbf'`` -- a grid of RBF bandwidths.  Every kernel is *rotation-invariant*,
  so is every combination, so this obeys the degree lower bound: it learns low-degree
  targets and **stays flat on a high-degree one (qubit parity) no matter how many
  bandwidths you add**.
- ``dict_kind='poly'`` -- polynomial kernels of degree ``1..d``.  These *carry
  interactions*, so as ``d`` grows the dictionary can represent higher-degree targets --
  but a degree-``n`` target (parity) needs ``d=n``, i.e. a dictionary of feature dimension
  ``~2^n``.  So the qubit curve only lifts at exponential dictionary size: the
  classical/quantum separation, quantified.

    python -m learn.mkl --configs-dir configs/Datasets --dict-kind poly --degrees 1 2 3 4 5 6

Pass ``--gcv`` to choose the KRR ridge per degree by closed-form GCV -- the fair way to
compare across capacities.  At a *fixed* ``--lam`` a higher-degree kernel overfits and its
test R^2 dives (the extra capacity fits noise); GCV matches the ridge to the capacity, so
the curve reflects genuine learnability and only stays flat/low when the target really is
not learnable in that dictionary.
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.mkl` and `python learn/mkl.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import numpy as np

from Generator import generate, load_config

from .grid import discover_configs
from .svm import _split_indices, load_target


# --- kernel dictionary ------------------------------------------------------ #

def kernel_dictionary(Xa, Xb, *, kind: str, degree: int, n_features: int, fourier_order: int = 3):
    """Candidate (cross-)Grams ``[K_1(Xa, Xb), ...]`` for a dictionary of size ``degree``.

    Each returned matrix is ``(len(Xa), len(Xb))`` so the same call builds the train Gram
    (``Xa=Xb=Xtr``) and the test cross-Gram (``Xa=Xte, Xb=Xtr``).

    - ``kind='rbf'``: ``degree`` RBF kernels at geometrically spaced bandwidths (all
      rotation-invariant -> combination cannot beat the degree lower bound).
    - ``kind='poly'``: polynomial kernels of degree ``1..degree`` (carry interactions ->
      degree-``d`` targets become representable at ``d``, dictionary cost ``~n^d``).
    - ``kind='fourier'``: interaction-degree ladder in the **Fourier** basis -- polynomial
      kernels of degree ``1..degree`` on the Fourier features ``[sin(jx), cos(jx)]_{j<=order}``.
      The periodic basis natural to angle inputs, and the *matched* one for the (band-limited,
      cross-qubit) quantum targets; degree ``d`` = interaction order among Fourier modes.
      **Feed raw angles here, not standardized ones** (periodicity lives in ``x``, not ``z``).
    """
    from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

    if kind == "rbf":
        gammas = np.geomspace(1.0 / (4 * n_features), 4.0 / n_features, degree)
        return [rbf_kernel(Xa, Xb, gamma=float(g)) for g in gammas]
    if kind == "poly":
        # degree-d homogeneous-ish polynomial kernels; coef0=1 keeps all lower orders too.
        return [polynomial_kernel(Xa, Xb, degree=int(d), gamma=1.0 / n_features, coef0=1.0)
                for d in range(1, degree + 1)]
    if kind == "fourier":
        import torch

        from model.mlp import fourier_features
        Fa = fourier_features(torch.as_tensor(Xa, dtype=torch.float32), fourier_order).numpy()
        Fb = fourier_features(torch.as_tensor(Xb, dtype=torch.float32), fourier_order).numpy()
        # interaction-order ladder among the Fourier modes (cross-qubit parity terms appear
        # at high degree, as in `poly`, but in the periodic basis matched to the target).
        return [polynomial_kernel(Fa, Fb, degree=int(d), gamma=1.0 / Fa.shape[1], coef0=1.0)
                for d in range(1, degree + 1)]
    raise ValueError(f"dict_kind must be 'rbf', 'poly', or 'fourier', got {kind!r}")


# --- closed-form combination (ALIGNF) --------------------------------------- #

def _center(K):
    """Center a square Gram: ``H K H`` with ``H = I - 11^T/n`` (Mercer wrt empirical measure)."""
    n = K.shape[0]
    Kr = K - K.mean(0, keepdims=True)
    return Kr - Kr.mean(1, keepdims=True)


def alignf(Ks, y):
    """Nonnegative weights ``mu`` maximizing centered alignment of ``sum_i mu_i K_i`` with
    ``yy^T`` -- the closed-form MKL of Cortes-Mohri-Rostamizadeh (2012).

    Solves ``argmin_{mu>=0} || sum mu_i K_i^c - yy^T ||_F^2 = mu^T M mu - 2 mu^T a`` (a tiny
    nonnegative least-squares), with ``a_i = y^T K_i^c y`` and ``M_ij = <K_i^c, K_j^c>``.
    ``Ks`` are the (square, train) Grams; ``y`` is the centered target.
    """
    from scipy.optimize import nnls

    Kc = [_center(K) for K in Ks]
    # a = np.array([float(y @ Kc_i @ y) for Kc_i in Kc])
    # M = np.array([[float((Ki * Kj).sum()) for Kj in Kc] for Ki in Kc])
    # L = np.linalg.cholesky(M + 1e-8 * np.trace(M) / len(M) * np.eye(len(M)))  # M = L L^T
    # mu, _ = nnls(L.T, np.linalg.solve(L, a))               # argmin_{mu>=0} ||L^T mu - L^{-1}a||^2
    # 
    n = Kc[0].shape[0]
    A = np.stack([Ki.flatten() for Ki in Kc], axis = 1)
    c = np.outer(y,y).flatten()
    mu, _ = nnls(A,c)

    s = mu.sum()
    return mu / s if s > 0 else mu


# --- closed-form KRR evaluation --------------------------------------------- #

def krr_eval(Ktr, Kte, ytr, yte, *, lam):
    """KRR on a *precomputed* combined kernel: ``(test_r2, rkhs_norm2, d_eff, lam)``, closed form.

    ``rkhs_norm2 = alpha^T K alpha`` (``alpha = (K+lam I)^{-1} y`` = ``KernelRidge.dual_coef_``)
    is the learnability certificate: O(1) when the target lives in the kernel's top modes,
    blowing up when it needs exponentially-suppressed modes.  ``d_eff = tr[K(K+lam I)^{-1}]``.
    """
    from sklearn.kernel_ridge import KernelRidge

    reg = KernelRidge(kernel="precomputed", alpha=lam).fit(Ktr, ytr)
    alpha = np.asarray(reg.dual_coef_).ravel()
    rkhs_norm2 = float(alpha @ Ktr @ alpha)
    r2 = float(reg.score(Kte, yte))
    n = Ktr.shape[0]
    d_eff = float(np.trace(np.linalg.solve(Ktr + lam * np.eye(n), Ktr)))
    return r2, rkhs_norm2, d_eff, lam


def krr_eval_gcv(Ktr, Kte, ytr, yte, *, lambdas):
    """KRR with the ridge ``lam`` chosen by **closed-form GCV**; returns
    ``(test_r2, rkhs_norm2, d_eff, lam*)``.

    Growing the dictionary at *fixed* ``lam`` overfits (test R^2 dives) because the
    regularization isn't matched to the growing capacity.  GCV picks ``lam`` per kernel in
    closed form -- no CV loop -- so the curve reflects genuine learnability, not fixed-``lam``
    overfitting.  One eigendecomposition of ``Ktr = V diag(sigma) V^T`` makes every ``lam``
    on the grid O(N): with ``z = V^T y`` and filter ``lam/(sigma+lam)`` (the eigenvalues of
    ``I - K(K+lam I)^{-1}``),

        GCV(lam) = N * ||(I-S)y||^2 / tr(I-S)^2 = N * sum_i (f_i z_i)^2 / (sum_i f_i)^2.
    """
    sigma, V = np.linalg.eigh(Ktr)                     # Ktr symmetric PSD
    sigma = sigma.clip(min=0.0)
    z = V.T @ ytr                                      # target in the kernel eigenbasis
    N = len(ytr)

    best_lam, best_gcv = None, np.inf
    for lam in lambdas:
        f = lam / (sigma + lam)                        # (I - S) eigenvalues
        gcv = N * float(np.sum((f * z) ** 2)) / (float(np.sum(f)) ** 2 + 1e-30)
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)

    coef = z / (sigma + best_lam)                      # alpha in eigenbasis
    alpha = V @ coef
    rkhs_norm2 = float(np.sum(sigma * coef ** 2))      # alpha^T K alpha
    d_eff = float(np.sum(sigma / (sigma + best_lam)))
    yhat = Kte @ alpha
    ss_res = float(np.sum((yte - yhat) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)
    return r2, rkhs_norm2, d_eff, best_lam


# --- driver ----------------------------------------------------------------- #

def run_mkl(configs_dir, *, degrees=(1, 2, 3, 4, 5), dict_kind="poly", fourier_order=3,
            n_fit=2000, n_test=1000, lam=1e-2, gcv=False, lambdas=None,
            dataset_root="datasets", use_cache=True, observable=None):
    """Return ``(results, degrees)`` with ``results[label] = [{degree, r2, rkhs_norm2, d_eff, lam}]``.

    For each dataset, sweep the dictionary size ``degree``; at each size pick the best
    linear kernel combination by :func:`alignf` and score it with KRR.  Grams are built on
    an ``n_fit``-row train subsample and an ``n_test`` test subsample (both from the
    dataset's own split) to keep the ``O(N^2)`` Grams tractable.

    ``gcv=True`` chooses the ridge per degree by closed-form GCV (:func:`krr_eval_gcv`) over
    ``lambdas`` -- the fair way to compare across capacities, since a *fixed* ``lam`` makes
    higher-degree kernels overfit (test R^2 dives).  ``gcv=False`` uses the fixed ``lam``.
    """
    if gcv and lambdas is None:
        lambdas = np.geomspace(1e-6, 1e2, 25)
    from sklearn.preprocessing import StandardScaler

    dataset_map = discover_configs(configs_dir)
    degrees = list(degrees)
    results = {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                        # ensure the artifact exists
        nfeat = dcfg.resolved_n_features
        t = load_target(dcfg, dataset_root, observable=observable)
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_fit], te[:n_test]

        from Generator import artifact_path, load_raw
        X = load_raw(artifact_path(dcfg, dataset_root))[0].numpy()
        xtr_raw, xte_raw = X[tr.numpy()], X[te.numpy()]
        if dict_kind == "fourier":
            Xtr, Xte = xtr_raw, xte_raw                              # Fourier: keep raw angles (periodic)
        else:
            xs = StandardScaler().fit(xtr_raw)
            Xtr, Xte = xs.transform(xtr_raw), xs.transform(xte_raw)
        ytr, yte = t[tr].numpy(), t[te].numpy()
        yc = ytr - ytr.mean()                                        # centered target for alignment

        rows = []
        for d in degrees:
            Ks_tr = kernel_dictionary(Xtr, Xtr, kind=dict_kind, degree=d, n_features=nfeat,
                                      fourier_order=fourier_order)
            Ks_te = kernel_dictionary(Xte, Xtr, kind=dict_kind, degree=d, n_features=nfeat,
                                      fourier_order=fourier_order)
            mu = alignf(Ks_tr, yc)                                    # closed-form best combo
            Ktr = sum(m * K for m, K in zip(mu, Ks_tr))
            Kte = sum(m * K for m, K in zip(mu, Ks_te))
            if gcv:
                r2, rkhs, deff, lam_used = krr_eval_gcv(Ktr, Kte, ytr, yte, lambdas=lambdas)
            else:
                r2, rkhs, deff, lam_used = krr_eval(Ktr, Kte, ytr, yte, lam=lam)
            rows.append({"degree": d, "r2": r2, "rkhs_norm2": rkhs, "d_eff": deff, "lam": lam_used})
        results[label] = rows
    return results, degrees


def _lineplot(results, degrees, axis_label, save_path, *, metric="r2", show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylabel = {"r2": "Test R²", "rkhs_norm2": "||f||_H^2  (log)", "d_eff": "effective dim",
              "lam": "GCV-chosen λ  (log)"}[metric]
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, rows in results.items():
        ys = [r[metric] for r in rows]
        ax.plot(degrees, ys, marker="o", label=label)
    if metric in ("rkhs_norm2", "lam"):
        ax.set_yscale("log")
    elif metric == "r2":
        ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("kernel dictionary size / max degree")
    ax.set_ylabel(ylabel)
    ax.set_title(f"MKL (alignment + KRR) {metric} across {axis_label}")
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend(title=axis_label)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[mkl] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.mkl",
        description="MKL learnability curve: best linear kernel combo (alignment) + KRR, per dataset.")
    ap.add_argument("--configs-dir", default="configs/Datasets")
    ap.add_argument("--dict-kind", default="poly", choices=["poly", "rbf", "fourier"],
                    help="'poly' (interactions on raw x), 'rbf' (bandwidth grid), or "
                         "'fourier' (interactions on Fourier features -- matched basis for angle inputs)")
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                    help="dictionary sizes / max degrees on the x-axis")
    ap.add_argument("--fourier-order", type=int, default=3,
                    help="Fourier band [sin(jx),cos(jx)]_{j<=order} for --dict-kind fourier")
    ap.add_argument("--metric", default="r2", choices=["r2", "rkhs_norm2", "d_eff", "lam"],
                    help="which closed-form quantity to plot")
    ap.add_argument("--n-fit", type=int, default=8000, help="train subsample (O(N^2) Grams)")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=1e-2, help="KRR ridge (fixed; ignored under --gcv)")
    ap.add_argument("--gcv", action="store_true",
                    help="choose the ridge per degree by closed-form GCV (fair across capacities)")
    ap.add_argument("--axis-label", default=None)
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable (needs "
                         "generation.save_dist). Photonic graph observables may encode the "
                         "selection, e.g. loop_path_parity__L0-1__P2-3")
    ap.add_argument("--save-dir", default="img")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute datasets (skip cache)")
    args = ap.parse_args(argv)

    axis_label = args.axis_label or Path(args.configs_dir).name
    results, degrees = run_mkl(args.configs_dir, degrees=args.degrees, dict_kind=args.dict_kind,
                               fourier_order=args.fourier_order, n_fit=args.n_fit, n_test=args.n_test,
                               lam=args.lam, gcv=args.gcv, use_cache=not args.force,
                               observable=args.observable)

    w = max(len(lbl) for lbl in results)
    ridge = "GCV" if args.gcv else f"lam={args.lam}"
    print(f"  dict_kind={args.dict_kind}  ridge={ridge}  n_fit={args.n_fit}\n")
    print("  " + f"{'dataset':<{w}}  " + "  ".join(f"d={d:>2}" for d in degrees) + "   metric=" + args.metric)
    for label, rows in results.items():
        vals = "  ".join(f"{r[args.metric]:>6.2f}" for r in rows)
        print(f"  {label:<{w}}  {vals}")
    if args.gcv:                                                  # show the ridge GCV picked per degree
        print("\n  GCV-chosen lambda:")
        for label, rows in results.items():
            vals = "  ".join(f"{r['lam']:>6.0e}" for r in rows)
            print(f"  {label:<{w}}  {vals}")

    obs_tag = f"_{args.observable}" if args.observable else ""
    save = Path(args.save_dir) / f"mkl_{args.dict_kind}_{args.metric}_{axis_label}{obs_tag}.png"
    _lineplot(results, degrees, axis_label, save, metric=args.metric, show=args.show)


if __name__ == "__main__":
    main()
