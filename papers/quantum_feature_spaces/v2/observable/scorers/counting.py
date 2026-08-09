"""Elementary counting scorers: ``v(n)`` from the photon counts alone.

The four plain per-outcome scorers, carried verbatim from ``model/photonic_observables/``.  Each is registered as a **per-outcome** function via
:func:`~v2.observable.base.key_scorer`, which derives three things from that one definition: the
plain :class:`Expectation` under its own name, the ``BASE_SCORERS`` table so the composite shapes
(``sq_<base>``, ``ent_<base>``, ``osc_<base>``, ``xent_<base>``) can score over it, and the
``KEY_SCORERS`` entry that lets the vector be built over a partial basis for the finite-sample path.
"""

from __future__ import annotations

from observable.base import key_scorer


def parity_modes(m: int) -> tuple[int, ...]:
    """The modes ``parity`` sums over: the first ``ceil(m/2)`` (the readout half)."""
    return tuple(range((m + 1) // 2))


def parity_score(key, modes) -> int:
    """``+1`` if the photon count over ``modes`` is even, ``-1`` if odd."""
    n = sum(int(key[i]) for i in modes)
    return 1 if n % 2 == 0 else -1


@key_scorer("parity")
def parity_key(key, ctx):
    """``+-1`` parity of the readout half's photon count, for one outcome."""
    return parity_score(key, parity_modes(ctx.m))


def require_even_m(m: int, label: str) -> None:
    """Reject an odd ``m``: the left/right halves would not be the same size."""
    if m % 2:
        raise ValueError(f"{label} requires even m (m={m})")


def majority_score(key, m: int, k: int) -> float:
    """``(n_left - n_right) / k`` over the ``m // 2`` split -- in ``[-1, 1]`` for ``k`` photons."""
    split = m // 2
    n_left = sum(int(key[i]) for i in range(split))
    n_right = sum(int(key[i]) for i in range(split, m))
    return (n_left - n_right) / k


@key_scorer("majority",
            plain_check=lambda ctx: require_even_m(ctx.m, "observable 'majority'"))
def majority_key(key, ctx):
    """Per-outcome normalised photon imbalance.

    The even-``m`` guard is attached to the *plain* observable only (``plain_check``): the
    composite shapes that score over ``majority`` never enforced it, and enforcing it here would
    silently change which datasets they can produce.
    """
    return majority_score(key, ctx.m, ctx.k)


def bunching_score(key) -> int:
    """``+1`` if every mode holds at most one photon, else ``-1``."""
    return 1 if max(int(n) for n in key) <= 1 else -1


@key_scorer("bunching")
def bunching_key(key, ctx):
    """``+-1`` collision-free indicator (HOM interference), for one outcome."""
    return bunching_score(key)


def first_mode_score(key) -> int:
    """``n_0 mod 2`` -- dotted with the probs this gives ``P(n_0 odd)`` in ``[0, 1]``."""
    return int(key[0] % 2)


@key_scorer("n_first")
def n_first_key(key, ctx):
    """Per-outcome parity of the first mode's photon count.

    NOTE the name says "photon count in the first mode", i.e. ``E[n_0] in [0, k]``, but the
    implementation returns ``n_0 mod 2`` and so scores ``P(n_0 odd)``.  The ``mod 2`` is what
    every dataset under this observable was generated with, so it is preserved verbatim; drop it
    only if you want the mean occupation, and rename when you do.
    """
    return first_mode_score(key)
