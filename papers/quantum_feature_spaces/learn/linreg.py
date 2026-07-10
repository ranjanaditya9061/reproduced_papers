"""Explicit-feature linear-regression learnability curve: R^2 vs interaction degree.

The *primal* counterpart to :mod:`learn.mkl`'s poly/fourier kernels.  Instead of a kernel
sum, build the explicit feature map ``phi_d(x)`` of a single datapoint's components up to
interaction degree ``d`` and fit a plain **linear ridge** regressor.  Sweeping ``d`` gives a
direct "classical learnability vs interaction order" curve, and -- unlike a kernel -- the
fitted coefficients live on named monomials / Fourier modes, so the degree at which the
target becomes representable is read straight off the curve.

The quantum teacher maps ONE datapoint at a time, so the target is a function of the
*intra-point* component interactions only (no cross-datapoint structure) -- exactly what a
degree-``d`` multivariate polynomial spans, cross terms ``x_i x_j`` included.  So this is the
matched, interpretable classical baseline.

Overfitting is real and expected: the explicit feature dimension ``C(n_in + d, d)`` grows
fast and eventually exceeds ``n_fit``, at which point an unregularised fit interpolates
noise.  So the ridge penalty is chosen per degree by efficient leave-one-out GCV
(:class:`sklearn.linear_model.RidgeCV`) and BOTH train and test R^2 are reported -- a
widening train/test gap *is* the overfitting, shown rather than assumed away.  Any
``(basis, degree)`` whose explicit expansion would exceed ``--max-feat`` columns is skipped
and logged (never silently dropped).

Add a basis by writing a base-feature builder and registering it in :data:`FEATURE_BASES`;
the polynomial-degree expansion, standardisation, dimension guard and ridge fit are shared.

    python -m learn.linreg --configs-dir configs/Datasets --bases monomial fourier --degrees 1 2 3 4 5
"""

from __future__ import annotations

import math
from pathlib import Path

# Support `python -m learn.linreg` and `python learn/linreg.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import numpy as np

from Generator import generate, load_config

from .grid import discover_configs
from .svm import _split_indices, load_target

#: ridge grid searched per degree by leave-one-out GCV.
LAMBDAS = np.geomspace(1e-6, 1e3, 25)


# --- feature bases (extensible registry) ------------------------------------ #
#
# A base builder maps the raw inputs to the degree-1 feature block ``(Btr, Bte)``; the shared
# machinery below raises it to interaction degree ``d`` (all cross terms), standardises, and
# fits.  To add a basis, write one such builder and register it in FEATURE_BASES.

def _standardize(Ftr, Fte):
    """Standardise columns on train statistics (no test leakage)."""
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(Ftr)
    return xs.transform(Ftr), xs.transform(Fte)


def _monomial_base(Xtr_raw, Xte_raw, *, n_features, fourier_order):
    """Standardised raw components ``x_i`` (degree-1 monomials)."""
    return _standardize(Xtr_raw, Xte_raw)


def _fourier_base(Xtr_raw, Xte_raw, *, n_features, fourier_order):
    """Fourier modes ``[sin(j x_i), cos(j x_i)]_{j<=order}`` on the *raw* angles.

    Periodicity lives in the raw angle ``x`` (inputs are in ``[0, 2pi]``), so these are built
    from unstandardised inputs; the expanded columns are standardised downstream.
    """
    import torch

    from model.mlp import fourier_features
    Btr = fourier_features(torch.as_tensor(Xtr_raw, dtype=torch.float32), fourier_order).numpy()
    Bte = fourier_features(torch.as_tensor(Xte_raw, dtype=torch.float32), fourier_order).numpy()
    return Btr, Bte


#: name -> base builder ``(Xtr_raw, Xte_raw, *, n_features, fourier_order) -> (Btr, Bte)``.
FEATURE_BASES = {
    "monomial": _monomial_base,
    "fourier": _fourier_base,
}


def _expanded_dim(n_in: int, degree: int) -> int:
    """Column count of degree-1..``degree`` monomials in ``n_in`` inputs (no bias term)."""
    return math.comb(n_in + degree, degree) - 1


def _poly_expand(Btr, Bte, degree: int):
    """All monomials of degree ``1..degree`` over the base features (cross terms included)."""
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=int(degree), include_bias=False).fit(Btr)
    return poly.transform(Btr), poly.transform(Bte)


# --- linear ridge with leave-one-out GCV ------------------------------------ #

def _fit_ridge_gcv(Ftr, ytr, Fte, yte, *, lambdas):
    """Linear ridge, lambda picked by efficient LOO-GCV; ``(train_r2, test_r2, lam, n_feat)``."""
    from sklearn.linear_model import RidgeCV

    reg = RidgeCV(alphas=lambdas).fit(Ftr, ytr)          # closed-form leave-one-out GCV
    return (float(reg.score(Ftr, ytr)), float(reg.score(Fte, yte)),
            float(reg.alpha_), int(Ftr.shape[1]))


# --- driver ----------------------------------------------------------------- #

def run_linreg(configs_dir, *, bases=("monomial", "fourier"), degrees=(1, 2, 3, 4, 5),
               fourier_order=3, n_fit=4000, n_test=2000, lambdas=None, max_feat=20000,
               dataset_root="datasets", use_cache=True, observable=None):
    """Return ``(results, degrees)`` with ``results[label][basis] = [{degree, train_r2,
    test_r2, lam, n_feat}]`` (a skipped/too-wide degree has ``test_r2 = nan``).

    For each dataset, each basis' degree-1 block is built once and re-expanded per degree;
    ridge lambda is chosen per degree by LOO-GCV so the comparison across degrees is fair.
    """
    unknown = [b for b in bases if b not in FEATURE_BASES]
    if unknown:
        raise ValueError(f"unknown basis {unknown}; choose from {sorted(FEATURE_BASES)}")
    lambdas = LAMBDAS if lambdas is None else np.asarray(lambdas, dtype=float)

    from Generator import artifact_path, load_raw

    dataset_map = discover_configs(configs_dir)
    degrees = list(degrees)
    results = {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                    # ensure the artifact exists
        nfeat = dcfg.resolved_n_features
        t = load_target(dcfg, dataset_root, observable=observable)
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_fit], te[:n_test]
        X = load_raw(artifact_path(dcfg, dataset_root))[0].numpy()
        xtr, xte = X[tr.numpy()], X[te.numpy()]
        ytr, yte = t[tr].numpy(), t[te].numpy()

        per_basis = {}
        for basis in bases:
            Btr, Bte = FEATURE_BASES[basis](xtr, xte, n_features=nfeat, fourier_order=fourier_order)
            n_in = Btr.shape[1]
            rows = []
            for d in degrees:
                dim = _expanded_dim(n_in, d)
                if dim > max_feat:                               # guard: never OOM silently
                    print(f"[linreg] skip {label} [{basis}] degree {d}: "
                          f"{dim} features > max_feat={max_feat}")
                    rows.append({"degree": d, "train_r2": float("nan"),
                                 "test_r2": float("nan"), "lam": float("nan"), "n_feat": dim})
                    continue
                Ftr, Fte = _poly_expand(Btr, Bte, d)
                Ftr, Fte = _standardize(Ftr, Fte)
                tr_r2, te_r2, lam, nf = _fit_ridge_gcv(Ftr, ytr, Fte, yte, lambdas=lambdas)
                rows.append({"degree": d, "train_r2": tr_r2, "test_r2": te_r2,
                             "lam": lam, "n_feat": nf})
            per_basis[basis] = rows
        results[label] = per_basis
    return results, degrees


def _lineplot(results, degrees, axis_label, save_path, *, show=False, show_train=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, per_basis in results.items():
        for basis, rows in per_basis.items():
            te = [r["test_r2"] for r in rows]
            line, = ax.plot(degrees, te, marker="o", label=f"{label} [{basis}]")
            if show_train:
                trn = [r["train_r2"] for r in rows]
                ax.plot(degrees, trn, marker=".", ls="--", color=line.get_color(), alpha=0.45)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("interaction degree d")
    ax.set_ylabel("R²   (solid = test, dashed = train)")
    ax.set_title(f"Explicit-feature linear-regression learnability across {axis_label}")
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[linreg] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _fmt_row(rows, key):
    return "  ".join(("  -  " if math.isnan(r[key]) else f"{r[key]:>5.2f}") for r in rows)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.linreg",
        description="Explicit-feature linear-regression learnability (R^2 vs interaction degree).")
    ap.add_argument("--configs-dir", default="configs/Datasets")
    ap.add_argument("--bases", nargs="+", default=["monomial", "fourier"],
                    choices=sorted(FEATURE_BASES),
                    help="explicit feature bases to expand and fit (one line each)")
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="polynomial interaction degrees on the x-axis")
    ap.add_argument("--fourier-order", type=int, default=1,
                    help="Fourier band [sin(jx),cos(jx)]_{j<=order} for the 'fourier' basis")
    ap.add_argument("--n-fit", type=int, default=8000, help="train subsample")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--lambdas", type=float, nargs="+", default=None,
                    help="ridge grid for LOO-GCV (default: geomspace(1e-6, 1e3, 25))")
    ap.add_argument("--max-feat", type=int, default=20000,
                    help="skip a (basis, degree) whose explicit expansion exceeds this width")
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable instead of the "
                         "stored soft (needs generation.save_dist). Photonic graph observables "
                         "may encode the selection, e.g. loop_path_parity__L0-1__P2-3")
    ap.add_argument("--axis-label", default=None)
    ap.add_argument("--save-dir", default="img")
    ap.add_argument("--no-train", action="store_true", help="hide the dashed train-R^2 lines")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute (skip cache)")
    args = ap.parse_args(argv)

    axis_label = args.axis_label or Path(args.configs_dir).name
    obs_tag = f"_{args.observable}" if args.observable else ""
    results, degrees = run_linreg(
        args.configs_dir, bases=args.bases, degrees=args.degrees,
        fourier_order=args.fourier_order, n_fit=args.n_fit, n_test=args.n_test,
        lambdas=args.lambdas, max_feat=args.max_feat, use_cache=not args.force,
        observable=args.observable)

    hdr = "  ".join(f"d={d:>2}" for d in degrees)
    for label, per_basis in results.items():
        print(f"\n  {label}")
        print(f"    {'basis':<10} {'':<5} {hdr}")
        for basis, rows in per_basis.items():
            print(f"    {basis:<10} test  {_fmt_row(rows, 'test_r2')}")
            print(f"    {basis:<10} train {_fmt_row(rows, 'train_r2')}   "
                  f"(n_feat {rows[-1]['n_feat']})")

    _lineplot(results, degrees, axis_label,
              Path(args.save_dir) / f"linreg_{axis_label}{obs_tag}.png",
              show=args.show, show_train=not args.no_train)


if __name__ == "__main__":
    main()
