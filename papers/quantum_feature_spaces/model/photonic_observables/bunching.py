"""``bunching``: ``+1`` on a collision-free outcome, ``-1`` once any mode holds >1 photon."""

from __future__ import annotations

from .base import base_observable


def bunching_score(key) -> int:
    """``+1`` if every mode holds at most one photon, else ``-1``."""
    return 1 if max(int(n) for n in key) <= 1 else -1


@base_observable("bunching")
def bunching_vec(ctx):
    """Per-Fock-state ``±1`` collision-free indicator."""
    return [bunching_score(key) for key in ctx.keys]