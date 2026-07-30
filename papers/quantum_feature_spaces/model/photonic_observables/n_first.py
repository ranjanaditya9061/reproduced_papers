"""``n_first``: the parity of the first mode's photon count.

NOTE: the name (and the original docstring) says "photon count in the first mode", i.e.
``E[n_0] in [0, k]``, but the implementation returns ``n_0 mod 2`` and so scores ``P(n_0 odd)
in [0, 1]``.  The ``mod 2`` is deliberate -- it is what every dataset under this observable was
generated with -- so it is preserved verbatim; changing it would silently redefine those
datasets.  Drop the ``% 2`` if you actually want the mean occupation.
"""

from __future__ import annotations

from .base import base_observable


def first_mode_score(key) -> int:
    """``n_0 mod 2`` -- dotted with the probs this gives ``P(n_0 odd)`` in ``[0, 1]``."""
    return int(key[0] % 2)


@base_observable("n_first")
def n_first_vec(ctx):
    """Per-Fock-state parity of the first mode's photon count."""
    return [first_mode_score(key) for key in ctx.keys]