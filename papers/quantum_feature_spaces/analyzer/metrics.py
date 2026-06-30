"""Label-independent kernel-vs-kernel comparison (centered kernel alignment).

Centered kernel alignment between two Gram matrices::

    CKA(K1, K2) = <K1c, K2c>_F / (||K1c||_F ||K2c||_F),   Kc = H K H,  H = I - 1/n

is in ``[0, 1]`` for PSD kernels (1 = identical up to scale), symmetric, and uses
no labels.  The **kernel-kernel matrix** is the ``M x M`` of pairwise CKAs across a
set of kernels (classical vs quantum, matched seed vs the unmatched ensemble) --
a first look at how the kernels' geometries relate before bringing in labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def center_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-centering ``H K H`` with ``H = I - 1/n`` (PSD-preserving)."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def _fro(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A * B).sum()


def kernel_alignment(K1: torch.Tensor, K2: torch.Tensor, centered: bool = True) -> float:
    """Centered (default) kernel alignment between two Gram matrices."""
    if centered:
        K1, K2 = center_gram(K1), center_gram(K2)
    denom = torch.sqrt((_fro(K1, K1) * _fro(K2, K2)).clamp_min(1e-24))
    return float(_fro(K1, K2) / denom)


def kernel_kernel_matrix(grams: Sequence[torch.Tensor], centered: bool = True) -> torch.Tensor:
    """``M x M`` pairwise centered-kernel-alignment matrix over ``grams``."""
    cs = [center_gram(K) if centered else K for K in grams]
    norms = [torch.sqrt(_fro(c, c).clamp_min(1e-24)) for c in cs]
    m = len(cs)
    out = torch.empty(m, m)
    for i in range(m):
        for j in range(i, m):
            val = _fro(cs[i], cs[j]) / (norms[i] * norms[j])
            out[i, j] = out[j, i] = val
    return out


# --------------------------------------------------------------------------- #
# kernel-target metrics
# --------------------------------------------------------------------------- #

def target_alignment(K: torch.Tensor, y: torch.Tensor) -> float:
    """Centered kernel-target alignment between Gram ``K`` and labels ``y``.

    Labels are mapped to +-1; the comparison kernel is ``y y^T``.  Higher (-> 1)
    means the kernel's geometry matches the label structure -- predictive of how
    learnable the data is for that kernel.
    """
    y_pm = (2 * y.reshape(-1).float() - 1.0) if y.min() >= 0 else y.reshape(-1).float()
    return kernel_alignment(K, torch.outer(y_pm, y_pm), centered=True)


def _normalize_trace(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    return K * (n / K.trace().clamp_min(1e-12))


def _matrix_sqrt(K: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(0.5 * (K + K.T))
    vals = vals.clamp_min(0.0)
    return (vecs * vals.sqrt()) @ vecs.T


def geometric_difference(K1: torch.Tensor, K2: torch.Tensor, reg: float = 1e-6) -> float:
    """Huang geometric difference ``g(K1 || K2) = sqrt(|| sqrt(K2) K1^-1 sqrt(K2) ||_inf)``.

    Both kernels are trace-normalized (``Tr K = N``) first.  Pass
    ``(K_classical, K_quantum)``: a **large** ``g`` means there is a function the
    quantum kernel can fit with few samples that the classical kernel cannot --
    the precondition for a potential quantum advantage on this data.
    ``g(K, K) = 1``.
    """
    n = K1.shape[0]
    K1, K2 = _normalize_trace(K1), _normalize_trace(K2)
    s2 = _matrix_sqrt(K2)
    K1reg = K1 + reg * torch.eye(n, dtype=K1.dtype, device=K1.device)
    A = s2 @ torch.linalg.solve(K1reg, s2)
    lam = torch.linalg.eigvalsh(0.5 * (A + A.T)).max().clamp_min(0.0)
    return float(lam.sqrt())
