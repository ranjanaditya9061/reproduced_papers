"""``parity``: ``(-1)^{Σ_i n_i}`` over the first half of the modes."""

from __future__ import annotations

from .base import base_observable


def parity_modes(m: int) -> tuple[int, ...]:
    """The modes ``parity`` sums over: the first ``ceil(m/2)`` (the readout half)."""
    return tuple(range((m + 1) // 2))


def parity_score(key, modes) -> int:
    """``+1`` if the photon count over ``modes`` is even, ``-1`` if odd."""
    n = sum(int(key[i]) for i in modes)
    return 1 if n % 2 == 0 else -1


@base_observable("parity")
def parity_vec(ctx):
    """Per-Fock-state ``±1`` parity of the readout half's photon count."""
    modes = parity_modes(ctx.m)
    return [parity_score(key, modes) for key in ctx.keys]