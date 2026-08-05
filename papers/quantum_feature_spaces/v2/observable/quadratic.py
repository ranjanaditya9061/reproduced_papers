"""Degree-2 observables: ``sq_<base>`` (diagonal ``K``) and ``pairprod`` (dense ``K``).

``O(x) = p^T K p``.  Every :class:`~v2.observable.base.Expectation` is additive-error estimable
from single-shot samples in ``~1/eps^2`` shots (just average ``v`` over them -- no permanents),
which is the classically easy regime and a weak basis for a hardness claim.  These are degree 2 in
``p``: collision / 2-copy quantities (signed purity terms), estimable only from *pairs* of shots
that hit the same outcome.

``sq_<base>`` is ``K = diag(<base>)`` and stores only the diagonal, dropping the cost from
``O(n_out^2)`` to ``O(n_out)`` -- which is why it carries no dimension cap where ``pairprod``
does.  ``pairprod`` is a dense, non-separable ``+-1`` kernel.

**No single-shot variance.**  Both raise from
:meth:`~v2.observable.base.Observable.variance`, so the utility metrics reject them rather than
applying the ``p @ v^2`` formula, which would be wrong (not merely unavailable) for a 2-copy
quantity.  A U-statistic estimator would be the right tool; it is deliberately out of scope.

Carried from ``model/photonic_observables/{sq,pairprod}.py``.
"""

from __future__ import annotations

import re

import numpy as np

from .base import (DiagonalQuadratic, Observable, ObservableContext, ObservableFamily, Quadratic,
                   base_score_vec, register)
from .scorers.counting import require_even_m

#: Bases usable under ``sq_<base>`` (the plain counting scorers).
SQ_BASES = ("parity", "majority", "bunching", "n_first")

#: Cap on the outcome dimension for ``pairprod``: the dense kernel is ``(n_out, n_out)`` float32,
#: so 4096 outcomes is already 64 MB and the per-row score is a full matvec.  Beyond this, use
#: ``sq_<base>``, which is the diagonal case and has no cap.
PAIRPROD_MAX_OUT = 4096

_SQ_RE = re.compile(r"^sq_(.+)$")


def parse_sq_observable(observable: str):
    """Split ``sq_<base>`` into ``(is_sq, base)`` (plain observables -> ``(False, observable)``)."""
    mo = _SQ_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_sq_observable(observable: str) -> bool:
    """True for a well-formed ``sq_<base>`` (``base`` in :data:`SQ_BASES`)."""
    is_sq, base = parse_sq_observable(observable)
    return is_sq and base in SQ_BASES


def sq_base_vec(base: str, keys, *, m: int, k: int):
    """Per-outcome ``<base>`` score list for a ``sq_<base>`` observable.

    Reuses the plain observables' scorers via ``BASE_SCORERS``, so the two paths cannot drift.
    """
    if base == "majority":
        require_even_m(m, "sq_majority")
    return base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                          allowed=SQ_BASES, label="sq")


def occ_matrix(keys, m: int) -> np.ndarray:
    """``(n_out, m)`` per-outcome photon counts."""
    return np.asarray([[int(key[i]) for i in range(m)] for key in keys], dtype=np.float64)


def pairprod_kernel(keys, m: int) -> np.ndarray:
    """``K[n1, n2] = (-1)^{<n1, n2>}`` -- the dense, non-separable ``+-1`` pair kernel.

    ``<n1, n2>`` is the integer inner product of the two occupation vectors, so the sign flips on
    the parity of the overlap.  Non-separable: it does not factor as ``u(n1) v(n2)``, which is what
    makes ``pairprod`` genuinely different from ``sq_<base>`` rather than a re-weighting of it.
    """
    occ = occ_matrix(keys, m)
    overlap = occ @ occ.T
    return np.where(np.mod(overlap, 2) == 0, 1.0, -1.0)


class SqFamily(ObservableFamily):
    describe = f"sq_<base> (base in {SQ_BASES})"

    def matches(self, name: str) -> bool:
        return is_sq_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_sq_observable(name)[1]
        return DiagonalQuadratic(sq_base_vec(base, ctx.keys, m=ctx.m, k=ctx.k))

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        return {"observable": "sq", "base": parse_sq_observable(name)[1]}


class PairProdFamily(ObservableFamily):
    describe = "pairprod"

    def matches(self, name: str) -> bool:
        return name == "pairprod"

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.n_out > PAIRPROD_MAX_OUT:
            raise ValueError(
                f"pairprod builds a dense ({ctx.n_out}, {ctx.n_out}) kernel, over the "
                f"PAIRPROD_MAX_OUT={PAIRPROD_MAX_OUT} cap for (m={ctx.m}, k={ctx.k}). Use "
                "sq_<base> instead -- it is the diagonal case of the same functional and has no "
                "cap -- or lower m/k."
            )
        return Quadratic(pairprod_kernel(ctx.keys, ctx.m))

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        return {"observable": "pairprod", "kernel": "overlap_parity"}


register(SqFamily())
register(PairProdFamily())
