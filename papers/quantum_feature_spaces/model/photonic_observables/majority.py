"""``majority``: the normalised left-minus-right photon imbalance across the mode split."""

from __future__ import annotations

from .base import base_observable


def require_even_m(m: int, label: str) -> None:
    """Reject an odd ``m``: the left/right halves would not be the same size."""
    if m % 2:
        raise ValueError(f"{label} requires even m (m={m})")


def majority_score(key, m: int, k: int) -> float:
    """``(n_left - n_right) / k`` over the ``m // 2`` split -- in ``[-1, 1]`` for k photons."""
    split = m // 2
    n_left = sum(int(key[i]) for i in range(split))
    n_right = sum(int(key[i]) for i in range(split, m))
    return (n_left - n_right) / k


@base_observable("majority", plain_check=lambda ctx: require_even_m(ctx.m, "observable 'majority'"))
def majority_vec(ctx):
    """Per-Fock-state normalised photon imbalance.

    The even-``m`` guard is attached to the *plain* observable only (``plain_check``): the
    composite families that score over ``majority`` (``loop_path_``, ``connected_``) have never
    enforced it, and ``sq_majority`` enforces it itself.
    """
    return [majority_score(key, ctx.m, ctx.k) for key in ctx.keys]