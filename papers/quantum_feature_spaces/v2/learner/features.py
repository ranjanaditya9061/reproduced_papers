"""Explicit feature maps shared by the embedding-based (:mod:`.embedding`) and kernel-based
(:mod:`.kernel`) learners.

**Deliberately a separate implementation from** :mod:`model.features`, which is *teacher-side* and
helps define the labels.  Sharing one implementation let the learner see the classical models' own
encoding, which inflated their scores and depressed the photonic ones -- measured on ``parity``,
``0.689 -> 0.957`` for the bilinear model against ``0.785 -> 0.606`` for photonic.  So the
duplication is the point: a learner experiment must not be able to reach the labels' own
featurisation.  Keep the two in sync only by intent, never by import.
"""

from __future__ import annotations

import math

import torch


def raw_features(X: torch.Tensor) -> torch.Tensor:
    return X


def fourier_features(X: torch.Tensor, order: int) -> torch.Tensor:
    """``[cos(jx), sin(jx)]_{j=1..order}`` -> ``(N, 2*order*d)``.  Learner-side; see the docstring.

    **Additive across coordinates -- no interaction terms.**  A ridge readout on this map is a sum
    of univariate functions, so it cannot represent the *product* structure of a Fock-basis
    probability (a permanent is multilinear in the columns, so ``p(n)`` carries cross terms in
    ``e^{i x_i}``).  Measured on ``fermion``/``parity``: ``R^2 = -0.03`` at order 3 and ``-0.06`` at
    order 8, against ``+0.79`` for the MLP.  A tensor-product basis would be ``order^{n_f}`` wide,
    so use ``rff`` (interactions implicit, needs ``gamma`` tuning) or the ``mlp`` learner instead.
    This map is a genuine baseline, not a broken one -- but do not read its failure as evidence
    about the labels.
    """
    parts = []
    for j in range(1, int(order) + 1):
        parts.append(torch.cos(j * X))
        parts.append(torch.sin(j * X))
    return torch.cat(parts, dim=1)


def rff_features(X: torch.Tensor, n_components: int, gamma: float, seed: int) -> torch.Tensor:
    """Random Fourier features approximating an RBF kernel: ``sqrt(2/D) cos(Wx + b)``."""
    g = torch.Generator().manual_seed(int(seed))
    d = X.shape[1]
    W = torch.randn(d, int(n_components), generator=g) * math.sqrt(2.0 * gamma)
    b = 2.0 * math.pi * torch.rand(int(n_components), generator=g)
    return math.sqrt(2.0 / int(n_components)) * torch.cos(X @ W + b)


def fourier_poly_features(X: torch.Tensor, order: int, degree: int) -> torch.Tensor:
    """``fourier_features`` (order ``order``) expanded with degree-``1..degree`` monomials.

    ``degree=1`` reproduces plain :func:`fourier_features`; ``degree=2`` adds every pairwise
    product of Fourier features, the minimal fix for the additive-map ceiling documented on
    :func:`fourier_features` -- a multilinear label (e.g. a Fock-basis permanent/determinant)
    needs cross terms a degree-1 (additive) ridge readout structurally cannot represent.  Same
    ``PolynomialFeatures`` construction as :func:`learner.auto.sweep_degree_grid`'s per-cell
    expansion, just fixed at one ``(order, degree)`` cell instead of swept.  Feature count is
    ``C(2*order*d_x + degree, degree) - 1`` in the Fourier base width -- at ``order=3, degree=2``
    and ``d_x=5`` (this study's default ``n_features``) that is ``495``, still solvable in closed
    form; ``degree=3`` at the same order/``d_x`` is ``5455`` -- large enough that ``alpha`` is
    doing real regularisation work rather than a negligible ridge nudge.

    **Needs `n_train` comfortably above the feature count to show its advantage.**  Measured: at
    ``n_train=150`` (feature/row ratio ~3:1) this basis scored *below* plain ``fourier`` at the
    default ``alpha=1e-3`` (0.41 vs 0.58 R^2 on a target with a genuine cross term) -- pure
    overfitting from under-regularising 495 features on too few rows, not a correctness issue; at
    ``n_train=1600`` the ranking flips as expected (0.76 vs 0.67).  The study's own size-sweep
    configs use ``n_train~8000`` by default, safely past this regime, but any explicit
    ``n_train`` override well below that should either raise ``alpha`` or expect the comparison
    against plain ``fourier`` to be biased by sample size rather than by expressivity.
    """
    from sklearn.preprocessing import PolynomialFeatures
    base = fourier_features(X, order)
    poly = PolynomialFeatures(degree=int(degree), include_bias=False)
    return torch.from_numpy(poly.fit_transform(base.double().numpy())).to(base.dtype)


def build_features(X: torch.Tensor, basis: str, *, order: int = 3, degree: int = 2,
                   n_components: int = 512, gamma: float = 0.5, seed: int = 0) -> torch.Tensor:
    if basis == "raw":
        return raw_features(X)
    if basis == "fourier":
        return fourier_features(X, order)
    if basis == "fourier_poly":
        return fourier_poly_features(X, order, degree)
    if basis == "rff":
        return rff_features(X, n_components, gamma, seed)
    if basis == "combo":
        return torch.cat([X, fourier_features(X, order)], dim=1)
    raise ValueError(f"unknown basis {basis!r}; choose raw | fourier | fourier_poly | rff | combo")
