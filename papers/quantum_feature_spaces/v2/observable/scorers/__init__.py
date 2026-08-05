"""Score-vector builders: how ``v(n)`` is constructed from an outcome.

**Axis B of the observable taxonomy.**  Every module here produces an
:class:`~v2.observable.base.Expectation` -- ``probs @ v`` -- so all of these are genuine
expectation values of a diagonal observable.  They differ only in how ``v`` is built:

* :mod:`.counting` -- ``parity``, ``majority``, ``bunching``, ``n_first``: the photon counts alone.
  These also populate ``BASE_SCORERS``, so the composite shapes score over them by name.
* :mod:`.graph` -- ``connected_<base>``: read the outcome as a vertex set of a fixed seeded graph
  and score a global connectivity property (``maxcc``).
* :mod:`.exp_poly` -- ``prod_parity[...]`` and its angle variants: ``(-1)^P(n)`` / ``cos(P(n))``
  for a polynomial ``P`` in the counts.
* :mod:`.marked` -- ``single_output`` (marked-outcome contrast) and ``xent`` (weighted ``log q``
  against the circuit's output at ``x = 0``).

The point of separating this axis from the functional shape (:mod:`v2.observable.base`) is that
graph-based and ``exp(poly)`` observables are *not* different kinds of measurement -- they are
expectation values with an elaborately-constructed ``v``.  The legacy package listed them as
sibling "families" alongside genuinely different shapes like ``ent``, which obscured that.
"""

from __future__ import annotations

# Imported for their registration side effects.
from . import counting, graph, exp_poly, marked        # noqa: F401

__all__ = ["counting", "graph", "exp_poly", "marked"]
