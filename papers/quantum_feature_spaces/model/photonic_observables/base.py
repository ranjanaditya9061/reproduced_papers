"""The general observable framework: build tables from the Fock basis, then score a distribution.

An *observable* maps a full Fock output distribution to one score::

    obs    = resolve_observable("loop_path_parity__L0-1", ctx)   # build: precompute tables
    scores = obs.score(probs)                                    # (N, n_fock) -> (N,)

Every family lives in its own module of this package and registers one
:class:`ObservableFamily`, so adding an observable means adding a module -- both the live
teacher (:class:`model.photonic.PhotonicTeacher`) and the offline re-scorer
(:func:`model.photonic.score_from_distribution`) go through :func:`resolve_observable` and
never special-case a name.

**Build / score split.**  A family is built from an :class:`ObservableContext` -- ``m``, ``k``,
the Fock basis ``keys``, and the seeds / graph knobs -- and precomputes every per-Fock-state
table it needs (a score vector, a keep mask, a pair kernel).  ``score(probs)`` is then pure
tensor algebra over the distribution and never re-reads ``keys``.  Because the tables are
keyed to a fixed basis, one family object scores a live merlin forward and a distribution
reloaded from disk identically.

**Score shapes.**  Most observables are LINEAR in the distribution, and the blocks below cover
every shape in use, ordered by how hard the value is to estimate from samples:

================================  =========================  ==================================
class                             score                      character
================================  =========================  ==================================
:class:`LinearObservable`         ``probs @ v``              diagonal expectation ``<P>``
:class:`SelectiveObservable`      ``E[v | keep]``            renormalised masked mean
:class:`SquaredObservable`        ``(probs * probs) @ v``    degree-2, ``K`` diagonal
:class:`QuadraticObservable`      ``probs^T K probs``        degree-2, ``K`` dense
:class:`EntropyWeightedObservable` ``Σ v·p log p``           non-polynomial in ``p``
================================  =========================  ==================================

A linear score is additive-error estimable from single-shot samples in ~1/eps^2 shots (just
average ``P`` over them), which is the classically easy regime.  The degree-2 rows are 2-copy /
collision quantities and are not; ``SquaredObservable`` *is* ``QuadraticObservable`` with
``K = diag(v)`` and subclasses it -- it exists only because storing the diagonal alone drops the
cost from O(n_fock^2) to O(n_fock).  The last row is not a polynomial in ``p`` at all, so it is
not the expectation of any fixed observable.  A family needing none of these subclasses
:class:`Observable` directly (see :mod:`.max_prob`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ObservableContext:
    """Everything a family needs to build itself, independent of where the probs come from.

    ``keys`` is the fixed Fock basis (per-mode photon counts, one entry per outcome) the
    tables are aligned to.  It is empty for the *hash* path -- :meth:`ObservableFamily.hash_spec`
    is called before any Merlin layer exists -- so a ``hash_spec`` may only use ``m``, ``k``
    and the knobs below.

    ``graph_seed`` / ``angle_seed`` fall back to the teacher ``seed`` when unset, so a config
    that pins only ``seed`` still gets a reproducible graph / angle draw.  ``loop_vars`` and
    ``path_vars`` are *programmatic* overrides for the ``loop_path_`` selection; normally the
    selection rides in the observable string instead (``__L0-1__P2``) and these stay ``None``.
    ``input_state`` is the teacher's photon injection -- available live, but not persisted in a
    saved distribution, so an observable that needs it cannot be re-scored offline.
    """

    m: int
    k: int
    keys: Sequence = field(default=())
    seed: int = 0
    graph_seed: int | None = None
    angle_seed: int | None = None
    n_vertices: int | None = None
    graph_density: float | None = None
    loop_vars: Sequence[int] | None = None
    path_vars: Sequence[int] | None = None
    input_state: Sequence[int] | None = None

    def __post_init__(self):
        set_ = object.__setattr__                       # frozen -> resolve defaults in place
        set_(self, "m", int(self.m))
        set_(self, "k", int(self.k))
        set_(self, "seed", int(self.seed))
        set_(self, "graph_seed", self.seed if self.graph_seed is None else int(self.graph_seed))
        set_(self, "angle_seed", self.seed if self.angle_seed is None else int(self.angle_seed))
        if self.n_vertices is not None:
            set_(self, "n_vertices", int(self.n_vertices))
        if self.graph_density is not None:
            set_(self, "graph_density", float(self.graph_density))

    @property
    def n_fock(self) -> int:
        """Number of Fock outcomes, i.e. the length of every per-state table."""
        return len(self.keys)

    def require(self, **knobs):
        """Return the named knobs, raising if any is ``None`` (a family's missing-input guard)."""
        missing = [n for n, v in knobs.items() if v is None]
        if missing:
            raise ValueError(f"this observable requires {', '.join(missing)}")
        return tuple(knobs.values()) if len(knobs) > 1 else next(iter(knobs.values()))


# --- score shapes ------------------------------------------------------------------------ #


def _as_vec(vec) -> torch.Tensor:
    """Coerce a per-Fock-state score list/array to a ``(n_fock,)`` float32 tensor."""
    return torch.as_tensor(np.asarray(vec, dtype=np.float64), dtype=torch.float32)


class Observable(nn.Module):
    """Scores a full Fock output distribution: ``(N, n_fock)`` probs -> ``(N,)`` scores.

    An ``nn.Module`` so the precomputed tables ride along as buffers (device moves,
    ``state_dict``).  Subclasses implement :meth:`score`.
    """

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.score(probs)


class LinearObservable(Observable):
    """``O(x) = <P> = probs @ score_vec`` -- the plain diagonal expectation."""

    def __init__(self, score_vec):
        super().__init__()
        self.register_buffer("score_vec", _as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.score_vec


class SelectiveObservable(Observable):
    """``E[score | selected]``: the renormalised mean over the kept Fock states.

    Rows whose selected mass is numerically zero score ``0`` rather than dividing by ~0.
    """

    def __init__(self, keep_mask, score_vec):
        super().__init__()
        self.register_buffer("keep_mask", _as_vec(keep_mask))
        self.register_buffer("score_vec", _as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        sel = probs * self.keep_mask                     # broadcast over (n_fock,)
        den = sel.sum(dim=1)
        num = sel @ self.score_vec
        return torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.zeros_like(den))


class QuadraticObservable(Observable):
    """``O(x) = p^T K p`` -- the degree-2 form for a symmetric pair kernel ``K``.

    ``K`` is stored dense here, so ``score`` costs O(n_fock^2) per row -- which is why
    ``pairprod`` caps the Fock dimension (:data:`~.pairprod.PAIRPROD_MAX_FOCK`).
    :class:`SquaredObservable` is the *diagonal* case of this same functional and stores only
    ``diag(K)``, making it O(n_fock) with no cap.

    A subclass that keeps ``K`` in a cheaper form passes ``pair_kernel=None`` and overrides
    :meth:`score` and :meth:`kernel_matrix`.
    """

    def __init__(self, pair_kernel=None):
        super().__init__()
        if pair_kernel is not None:
            self.register_buffer("pair_kernel", torch.as_tensor(
                np.asarray(pair_kernel, dtype=np.float64), dtype=torch.float32))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs @ self.pair_kernel * probs).sum(dim=1)

    def kernel_matrix(self) -> torch.Tensor:
        """The dense ``(n_fock, n_fock)`` ``K`` this observable is equivalent to."""
        return self.pair_kernel


class SquaredObservable(QuadraticObservable):
    """``O(x) = Σ_n v_n p(n)^2`` -- :class:`QuadraticObservable` with ``K = diag(v)``.

    Exactly ``p^T diag(v) p``, but only the ``(n_fock,)`` diagonal ``v`` is stored (as
    ``score_vec``), so it costs O(n_fock) per row instead of O(n_fock^2) and carries no
    Fock-dimension cap -- which is precisely why ``sq_<base>`` is the fallback when
    ``pairprod``'s dense kernel will not fit.  :meth:`kernel_matrix` materialises the
    equivalent ``K`` on demand, so the two forms can be checked against each other.
    """

    def __init__(self, score_vec):
        super().__init__()                       # no dense kernel: the diagonal is the whole K
        self.register_buffer("score_vec", _as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs * probs) @ self.score_vec

    def kernel_matrix(self) -> torch.Tensor:
        return torch.diag(self.score_vec)


class EntropyWeightedObservable(Observable):
    """``O(x) = Σ_n v_n · p(n) log p(n)`` -- a base-weighted (negative) Shannon entropy.

    With ``v`` omitted (weight ``1`` everywhere) this is ``Σ_n p log p = -H(p)``, the plain
    negative entropy of the outcome distribution: a pure concentration measure, like
    :class:`~.max_prob.MaxProbObservable` but reading the whole distribution rather than its
    peak.  A general ``v`` re-weights each outcome's surprisal contribution by that outcome's
    base score, so the result reports *where* the mass concentrates and not merely how much.

    Unlike every other family here this is neither linear nor polynomial in ``p``: ``p log p``
    has no Taylor expansion at ``p = 0``, so the value is not the expectation of any fixed
    diagonal observable and cannot be estimated to additive error from a bounded number of
    single-shot samples -- entropy estimation over ``n_fock`` outcomes needs ~``n_fock /
    log n_fock`` of them.  That places it further from the classically easy regime than the
    degree-2 forms above, which at least need only two copies.

    Sign: ``p log p <= 0`` for every term, so ``v`` decides entirely.  Unweighted -- or weighted
    by any *non-negative* ``v`` -- the result is non-positive and ``sign(O)`` gives ONE class, so
    threshold it (e.g. at the median) instead.  Only a ``v`` that actually takes both signs (the
    ±1 base scorers) yields a mixed-sign score; see :mod:`.entropy` for which bases do.

    ``p log p -> 0`` as ``p -> 0`` is the correct limit, so the log is clamped at ``eps`` and a
    zero-probability outcome contributes exactly ``0 * log(eps) = 0``.
    """

    def __init__(self, score_vec=None, *, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("score_vec", None if score_vec is None else _as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        plogp = probs * torch.log(probs.clamp(min=self.eps))
        return plogp.sum(dim=1) if self.score_vec is None else plogp @ self.score_vec


# --- families and the registry ------------------------------------------------------------ #


class ObservableFamily:
    """One observable family: the names it owns, how to build it, how to hash it.

    ``describe`` is the spelling shown in the "unknown observable" error, so it should read as
    a *pattern* (``"loop_path_<base>"``) rather than as a single name.
    """

    describe: str = ""

    def matches(self, name: str) -> bool:
        """True iff this family owns the observable string ``name``."""
        raise NotImplementedError

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        """Precompute this family's per-Fock-state tables over ``ctx.keys``."""
        raise NotImplementedError

    def hash_spec(self, name: str, ctx: ObservableContext) -> dict:
        """Extra dataset-identity fields, canonicalised so equivalent spellings collide.

        Called with a ``keys``-less ``ctx`` (the hash is computed before the Merlin layer
        exists), so it may only use ``m``, ``k`` and the seeds / graph knobs.
        """
        return {}


#: Registered families, in match order (see :func:`find_family`).
_FAMILIES: list[ObservableFamily] = []

#: ``base -> (ctx) -> score vector``: the elementary per-Fock-state scorers, populated by the
#: plain observable modules via :func:`base_observable`.  The composite families
#: (``sq_<base>``, ``loop_path_<base>``, ``connected_<base>``) score *over* one of these, so
#: they look it up here instead of each re-implementing the same if-chain.
BASE_SCORERS: dict[str, Callable[[ObservableContext], Sequence[float]]] = {}


def register(family: ObservableFamily) -> ObservableFamily:
    """Register an observable family (called at module import; see this package's ``__init__``)."""
    _FAMILIES.append(family)
    return family


def find_family(name: str) -> ObservableFamily | None:
    """The family owning ``name``, or ``None``.

    First match wins, so the composite families are registered before the plain ones (the
    package ``__init__`` fixes the import order).  The patterns are disjoint in practice, so
    the order is documentation rather than disambiguation.
    """
    for family in _FAMILIES:
        if family.matches(name):
            return family
    return None


def is_known_observable(name: str) -> bool:
    """True iff some family owns ``name`` -- a cheap validity check that builds nothing."""
    return find_family(name) is not None


def observable_help() -> str:
    """The supported spellings, joined for an error message."""
    return "; ".join(f.describe for f in _FAMILIES if f.describe)


def resolve_observable(name: str, ctx: ObservableContext) -> Observable:
    """Build the observable named ``name`` over the Fock basis in ``ctx``."""
    family = find_family(name)
    if family is None:
        raise ValueError(f"unknown observable {name!r}; expected one of: {observable_help()}")
    return family.build(name, ctx)


def observable_hash_spec(name: str, ctx: ObservableContext) -> dict:
    """Canonical dataset-identity fields for ``name`` (``{}`` for a plain or unknown one)."""
    family = find_family(name)
    return {} if family is None else family.hash_spec(name, ctx)


def base_score_vec(base: str, ctx: ObservableContext, *, allowed, label: str) -> np.ndarray:
    """Per-Fock-state vector for the elementary scorer ``base``, gated on ``allowed``.

    The composite families call this for the bases they share with the plain observables,
    after intercepting their own extras (``loop``/``path``/``maxcc``).  ``label`` names the
    family in the error, so a typo reports e.g. ``unknown graph base 'majorty'``.
    """
    if base not in allowed or base not in BASE_SCORERS:
        raise ValueError(f"unknown {label} base {base!r}; choose from {tuple(allowed)}")
    return np.asarray(BASE_SCORERS[base](ctx), dtype=np.float64)


class _PlainFamily(ObservableFamily):
    """A single-name observable that is just a :class:`LinearObservable` over a score vector."""

    def __init__(self, name: str, vec, check):
        self.name, self.vec, self.check = name, vec, check
        self.describe = name

    def matches(self, name: str) -> bool:
        return name == self.name

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if self.check is not None:
            self.check(ctx)
        return LinearObservable(self.vec(ctx))


def base_observable(name: str, *, plain_check=None):
    """Register a plain per-Fock-state observable from its score-vector builder.

    The decorated ``vec(ctx) -> sequence`` becomes both

    * the ``name`` observable itself -- a :class:`LinearObservable` over that vector -- and
    * an entry in :data:`BASE_SCORERS`, so ``sq_<name>`` / ``loop_path_<name>`` /
      ``connected_<name>`` can score over it without duplicating the scorer.

    ``plain_check(ctx)`` runs only on the first path, i.e. when ``name`` *is* the observable.
    That keeps a constraint that the plain form has always enforced (``majority`` needs even
    ``m``) from silently spreading to the composites, which never enforced it.
    """
    def decorate(vec):
        BASE_SCORERS[name] = vec
        register(_PlainFamily(name, vec, plain_check))
        return vec
    return decorate