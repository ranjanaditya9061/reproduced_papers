"""Photonic observables: ``full Fock distribution -> score``.

One module per family, all speaking the framework in :mod:`.base`::

    from model.photonic_observables import ObservableContext, resolve_observable

    ctx    = ObservableContext(m=6, k=3, keys=layer.output_keys, seed=1234)
    obs    = resolve_observable("loop_path_parity__L0-1", ctx)   # build tables over the basis
    scores = obs.score(probs)                                    # (N, n_fock) -> (N,)

Adding an observable = adding a module here that registers an
:class:`~.base.ObservableFamily`; nothing downstream grows a branch.
:class:`model.photonic.PhotonicTeacher` (live) and
:func:`model.photonic.score_from_distribution` (offline, from a saved ``distributions.npz``)
both go through :func:`~.base.resolve_observable`, so the two paths cannot drift.

The families, by score shape:

**Linear** ``probs @ v`` -- a diagonal expectation ``<P>``; additive-error estimable from
single-shot samples, i.e. the classically easy regime:

* :mod:`.parity`, :mod:`.majority`, :mod:`.bunching`, :mod:`.n_first` -- the elementary
  per-Fock-state scorers.  These also register in :data:`~.base.BASE_SCORERS`, so the
  composite families score over them by name.
* :mod:`.prod_parity` -- ``(-1)^P(n)``, ``P`` a signed sum of count monomials chosen by the
  observable string or by the ``(m, k)`` geometry.
* :mod:`.prod_angle` -- ``cos(P(n))`` with a per-monomial angle; contains the above at θ=π.
* :mod:`.connected` -- ``E[<base>]`` where ``<base>`` may be a global property (``maxcc``) of
  the clicked *vertices*' induced subgraph of a fixed seeded graph.
* :mod:`.single_output` -- the marked-outcome contrast, peak-normalised (needs ``input_state``,
  so it is the one family that cannot be re-scored offline).

**Selective** ``E[v | keep]``:

* :mod:`.loop_path` -- reads an outcome as an *edge* set, overlays it with a reference perfect
  matching, pre-selects on the resulting loop/path counts.

**Degree-2** in ``p`` -- collision / 2-copy quantities, NOT single-shot additive estimable:

* :mod:`.sq` -- ``Σ_n <base>(n) p(n)^2``, i.e. ``p^T K p`` with ``K`` diagonal.
* :mod:`.pairprod` -- ``p^T K p`` for a dense, non-separable ±1 pair kernel.

**Non-polynomial** in ``p`` -- not the expectation of any fixed observable:

* :mod:`.entropy` -- ``Σ_n <base>(n) p(n) log p(n)``; bare ``ent`` is ``-H(p)``.

**Other**:

* :mod:`.max_prob` -- ``max_n p(n)``, a concentration probe with no score vector.

Support modules: :mod:`.base` (the framework) and :mod:`.graphs` (the fixed seeded graphs the
two graph families interpret outcomes against).
"""

from __future__ import annotations

from .base import (BASE_SCORERS, EntropyWeightedObservable, LinearObservable, Observable,
                   ObservableContext, ObservableFamily, QuadraticObservable,
                   SelectiveObservable, SquaredObservable, base_observable, base_score_vec,
                   find_family, is_known_observable, observable_hash_spec, observable_help,
                   register, resolve_observable)

# Family modules, imported for their registration side effect.  Import order = match order in
# `find_family` (first match wins) and the order families are listed in the "unknown observable"
# error.  The plain single-name families come first: their patterns are exact-match and so are
# disjoint from the prefixed composites, which makes the order documentation rather than
# disambiguation.
from . import parity, majority, bunching, n_first, single_output, max_prob       # noqa: E402
from . import prod_parity, prod_angle, sq, pairprod, entropy, loop_path, connected  # noqa: E402
from . import graphs                                                            # noqa: E402

from .bunching import bunching_score
from .connected import (CONNECTED_BASES, clicked_max_component, connected_scores,
                        is_connected_observable, parse_connected_observable)
from .entropy import ENT_BASES, ent_base_vec, is_ent_observable, parse_ent_observable
from .graphs import (MAX_VERTEX_DEGREE, build_matching_graph, build_vertex_graph,
                     is_connected_graph)
from .loop_path import (GRAPH_BASES, graph_selection, graph_tables, is_graph_observable,
                        overlay_counts, parse_graph_observable, resolve_graph_spec)
from .majority import majority_score, require_even_m
from .max_prob import MaxProbObservable
from .n_first import first_mode_score
from .pairprod import (PAIRPROD_MAX_FOCK, finish_kernel, is_pairprod_observable, occ_matrix,
                       pairprod_base_overlay_kernel, pairprod_disjoint_kernel,
                       pairprod_distance_sign_kernel, pairprod_kernel,
                       pairprod_random_parity_kernel, pairprod_random_phase_kernel,
                       pairprod_rbf_kernel, pairprod_shared_modes_kernel,
                       pairprod_weighted_or_parity_kernel, pairprod_weighted_overlap_kernel)
from .parity import parity_modes, parity_score
from .prod_angle import angle_monomials, is_prod_parity_angle, prod_parity_angle_score
from .prod_parity import (PROD_PARITY_CONSECUTIVE, PROD_PARITY_PRESETS, PROD_PARITY_SECOND,
                          consecutive_monomials, is_prod_family, is_prod_parity_consecutive,
                          is_prod_parity_observable, is_prod_parity_second, parse_prod_parity,
                          parse_prod_segment, prod_family_monomials, prod_parity_score,
                          second_monomials)
from .single_output import single_output_score
from .sq import SQ_BASES, is_sq_observable, parse_sq_observable, sq_base_vec

#: The plain single-name observables, in the order they are reported in errors.  Mirrors the
#: ``_PlainFamily`` registrations above (:mod:`.parity` … :mod:`.max_prob`).
OBSERVABLES = ("parity", "majority", "bunching", "single_output", "n_first", "max_prob")

__all__ = [
    # framework
    "Observable", "ObservableContext", "ObservableFamily", "LinearObservable",
    "SelectiveObservable", "SquaredObservable", "QuadraticObservable",
    "EntropyWeightedObservable", "MaxProbObservable",
    "BASE_SCORERS", "base_observable", "base_score_vec", "find_family", "is_known_observable",
    "observable_help", "observable_hash_spec", "register", "resolve_observable",
    # name sets
    "OBSERVABLES", "GRAPH_BASES", "CONNECTED_BASES", "SQ_BASES", "ENT_BASES",
    "PROD_PARITY_PRESETS",
    "PROD_PARITY_CONSECUTIVE", "PROD_PARITY_SECOND", "PAIRPROD_MAX_FOCK", "MAX_VERTEX_DEGREE",
    # predicates / parsers
    "is_graph_observable", "parse_graph_observable", "resolve_graph_spec",
    "is_connected_observable", "parse_connected_observable",
    "is_prod_family", "is_prod_parity_observable", "is_prod_parity_consecutive",
    "is_prod_parity_second", "is_prod_parity_angle", "parse_prod_parity", "parse_prod_segment",
    "is_sq_observable", "parse_sq_observable", "is_pairprod_observable",
    "is_ent_observable", "parse_ent_observable",
    # per-Fock-state scorers and table builders
    "parity_score", "parity_modes", "majority_score", "require_even_m", "bunching_score",
    "first_mode_score", "single_output_score", "prod_parity_score", "prod_parity_angle_score",
    "sq_base_vec", "ent_base_vec", "connected_scores", "clicked_max_component",
    "graph_selection", "graph_tables", "overlay_counts",
    # monomials / graphs / kernels
    "prod_family_monomials", "consecutive_monomials", "second_monomials", "angle_monomials",
    "build_matching_graph", "build_vertex_graph", "is_connected_graph", "occ_matrix",
    "finish_kernel", "pairprod_kernel", "pairprod_random_phase_kernel", "pairprod_random_parity_kernel",
    "pairprod_weighted_overlap_kernel", "pairprod_weighted_or_parity_kernel",
    "pairprod_distance_sign_kernel", "pairprod_shared_modes_kernel", "pairprod_disjoint_kernel",
    "pairprod_rbf_kernel", "pairprod_base_overlay_kernel",
]
