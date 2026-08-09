"""Observables: ``outcome distribution -> score``.

    from observable import ObservableContext, resolve_observable

    ctx    = ObservableContext(m=6, k=3, keys=model.outcome_keys(), seed=42)
    obs    = resolve_observable("prod_parity_consecutive", ctx)   # build tables over the basis
    scores = obs.score(probs)                                    # (N, n_out) -> (N,)

**Organised on two orthogonal axes**, which is the change from the legacy package's flat list of
families:

*Axis A -- functional shape* (:mod:`.base`), which decides shot-estimability and which metrics
apply:

- :class:`~.base.Expectation` ``probs @ v`` -- a diagonal ``<P>``.  Single-shot estimable, so its
  ``V_eff`` is the *exact* single-shot variance.
- :class:`~.base.Quadratic` ``p^T K p`` (:mod:`.quadratic`) -- 2-copy / collision.
- :class:`~.base.ProbFunction` ``sum v phi(p)`` (:mod:`.prob_fn`) -- not the expectation of any
  fixed observable.

**All three are in scope for the efficiency metrics**, through the influence function
``psi_n = dT/dp_n`` (``dT/dx = Cov(psi, s)``, ``V_eff = Var_p(psi)``); ``Expectation`` is the case
``psi = v``.  The only exclusion is non-differentiability -- ``max_prob``, whose
``psi_n = 1[n = argmax]`` is undefined at a tie.  See :meth:`~.base.Observable.influence`.

*Axis B -- score-vector builders* (:mod:`.scorers`), all of which feed ``Expectation``: counting,
graph readings, ``exp(poly)`` forms, marked outcomes.  **Graph-based and ``exp(poly)`` observables
are expectation values** -- they are ``probs @ v`` with an elaborate ``v``, not different kinds of
measurement.  Those builders are also reusable as the *base* of the other two shapes
(``sq_parity``, ``ent_parity``, ...), which is what ``BASE_SCORERS`` is for.

**Dropped from the legacy package**: ``SelectiveObservable``, the ``loop_path_*__L/__P``
pre-selection, and the spoqc ``match{N}_`` prefix.  A post-selected mean is a score vector on a
smaller support, so it earns no class of its own.

An observable is **not** part of a dataset's identity -- it is a readout applied to a saved
distribution and cached separately (:mod:`v2.pipeline.score`), keyed by
:func:`~.base.observable_spec_hash`.
"""

from __future__ import annotations

from .base import (BASE_SCORERS, KEY_SCORERS, DiagonalQuadratic, Expectation, Observable,
                   ObservableContext, ObservableFamily, ProbFunction, Quadratic, as_vec,
                   base_score_vec, find_family, is_known_observable, key_scorer, observable_help,
                   observable_on_keys, observable_spec, observable_spec_hash, register,
                   resolve_observable)

# Family modules, imported for their registration side effects.  Registration order sets match
# order in `find_family` (first match wins); the patterns are disjoint, so this is documentation.
from . import scorers                                   # noqa: E402  (counting/graph/exp_poly/marked)
from . import quadratic, prob_fn                        # noqa: E402

from .prob_fn import (ENT_BASES, ENT_EPS, OSC_BASES, OSC_EPS, EntropyWeighted, MaxProb,
                      Oscillatory, is_ent_observable, is_osc_observable, parse_ent_observable,
                      parse_osc_observable)
from .quadratic import (PAIRPROD_MAX_OUT, SQ_BASES, is_sq_observable, occ_matrix, pairprod_kernel,
                        parse_sq_observable, sq_base_vec)
from .scorers.counting import (bunching_score, first_mode_score, majority_score, parity_modes,
                               parity_score, require_even_m)
from .scorers.exp_poly import (PROD_PARITY_CONSECUTIVE, PROD_PARITY_PRESETS, PROD_PARITY_SECOND,
                               angle_monomials, consecutive_monomials, is_prod_family,
                               is_prod_parity_angle, is_prod_parity_observable,
                               prod_family_monomials, prod_parity_angle_score, prod_parity_score,
                               second_monomials)
from .scorers.graph import (CONNECTED_BASES, MAX_VERTEX_DEGREE, build_vertex_graph,
                            clicked_max_component, is_connected_graph, is_connected_observable,
                            parse_connected_observable)
from .scorers.marked import (XENT_BASES, XENT_EPS, is_xent_observable, parse_xent_observable,
                             reference_log_probs, single_output_score, xent_score_vec)

#: The plain single-name observables, for help text and default sweeps.
PLAIN_OBSERVABLES = ("parity", "majority", "bunching", "n_first", "single_output", "max_prob")

__all__ = [
    # framework / shapes
    "Observable", "ObservableContext", "ObservableFamily",
    "Expectation", "Quadratic", "DiagonalQuadratic", "ProbFunction",
    "EntropyWeighted", "Oscillatory", "MaxProb",
    "BASE_SCORERS", "KEY_SCORERS", "as_vec", "key_scorer", "base_score_vec",
    "observable_on_keys", "find_family",
    "is_known_observable", "observable_help", "observable_spec", "observable_spec_hash",
    "register", "resolve_observable",
    # name sets
    "PLAIN_OBSERVABLES", "SQ_BASES", "ENT_BASES", "OSC_BASES", "XENT_BASES", "CONNECTED_BASES",
    "PROD_PARITY_PRESETS", "PROD_PARITY_CONSECUTIVE", "PROD_PARITY_SECOND",
    "ENT_EPS", "OSC_EPS", "XENT_EPS", "PAIRPROD_MAX_OUT", "MAX_VERTEX_DEGREE",
    # predicates / parsers
    "is_sq_observable", "parse_sq_observable", "is_ent_observable", "parse_ent_observable",
    "is_osc_observable", "parse_osc_observable", "is_xent_observable", "parse_xent_observable",
    "is_connected_observable", "parse_connected_observable",
    "is_prod_family", "is_prod_parity_observable", "is_prod_parity_angle",
    # scorers / tables
    "parity_score", "parity_modes", "majority_score", "require_even_m", "bunching_score",
    "first_mode_score", "single_output_score", "prod_parity_score", "prod_parity_angle_score",
    "sq_base_vec", "xent_score_vec", "reference_log_probs", "clicked_max_component",
    "prod_family_monomials", "consecutive_monomials", "second_monomials", "angle_monomials",
    "build_vertex_graph", "is_connected_graph", "occ_matrix", "pairprod_kernel",
]
