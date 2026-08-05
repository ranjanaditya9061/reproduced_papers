"""Teacher-side input featurisation.

**This helps DEFINE LABELS.**  Every classical model here calls it inside its map, so editing it
changes every dataset those models produce.  The artifact identity records
:data:`TEACHER_FOURIER_VERSION`, not the formula, so bump the version if you alter the expansion --
otherwise cached datasets would be silently reused across the change.

For *learner* featurisation use :mod:`v2.learner.embedding`'s Fourier map instead, kept separate so
learner experiments cannot reach the labels.  Sharing one implementation previously let the learner
see the classical models' own encoding, which inflated their scores and depressed the photonic ones
(measured on ``parity``: 0.689 -> 0.957 for ``ebm_fock``, 0.785 -> 0.606 for photonic), so the
separation is deliberate.

Carried from ``model/mlp.py``.
"""

from __future__ import annotations

import torch

#: Bump if the formula below changes; it rides in every classical model's ``circuit_spec``.
TEACHER_FOURIER_VERSION = 1


def fourier_features(X: torch.Tensor, order: int) -> torch.Tensor:
    """Expand angles ``(N, d)`` into ``[sin(jx), cos(jx)]_{j=1..order}`` -> ``(N, 2*order*d)``."""
    parts = []
    for j in range(1, int(order) + 1):
        parts.append(torch.sin(j * X))
        parts.append(torch.cos(j * X))
    return torch.cat(parts, dim=1)


def fourier_dim(order: int, n_features: int) -> int:
    """``2 * order * n_features`` -- the expansion's width."""
    return 2 * int(order) * int(n_features)
