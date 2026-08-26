"""The observable framework: three functional shapes, and a registry of score-vector builders.

An *observable* maps a full outcome distribution to one score::

    obs    = resolve_observable("prod_parity_consecutive", ctx)   # build: precompute tables
    scores = obs.score(probs)                                     # (N, n_out) -> (N,)

**Two orthogonal axes.**  The legacy package organised observables as one flat list of families,
which obscured the fact that most of them are the same functional with a different score vector.
Here:

*Axis A -- the functional shape* (this module).  What decides shot-estimability, and which
metrics are even defined:

=====================  =====================  ========================  =====================================
class                  score ``T(p)``         influence ``psi = dT/dp``  ``V_eff = Var_p(psi)``
=====================  =====================  ========================  =====================================
:class:`Expectation`   ``probs @ v``          ``v_n``                   ``Var(v)`` -- **exact** single-shot
:class:`Quadratic`     ``p^T K p``            ``2 (K p)_n``             ``4 Var(K p)`` -- asymptotic
:class:`ProbFunction`  ``sum v_n phi(p_n)``   ``v_n phi'(p_n)``         ``Var(v phi'(p))`` -- asymptotic
=====================  =====================  ========================  =====================================

**Every class is in scope for the efficiency metrics, through its influence function.**  An earlier
version of this framework gated those metrics on ``isinstance(obs, Expectation)`` and said the other
shapes "have no variance".  That conflated two claims: there is no single-shot unbiased estimator
whose variance is ``p.v^2 - (p.v)^2`` (true), and the estimator has no finite asymptotic variance
(false).  For any smooth ``T(p)``, writing ``psi_n = dT/dp_n``:

* ``dT/dx_i = sum_n psi_n d_i p_n = Cov(psi, s_i)``, using ``E[s] = 0``.  (``psi`` is defined only
  up to an additive constant, harmlessly, since ``sum_n d_i p_n = 0``.)
* the plug-in estimator's leading-order variance under multinomial sampling is
  ``Var(T_hat) ~ Var_p(psi) / S``, from ``Cov(p_hat_n, p_hat_n') = (p_n delta_nn' - p_n p_n')/S``.

So with ``g_i = Cov(psi, s_i)`` and ``V_eff = Var_p(psi)`` the Cauchy-Schwarz bound is unchanged --
``(u^T g)^2 = Cov(psi, u^T s)^2 <= Var(psi) Var(u^T s) = V_eff . u^T F u`` -- hence
``F_T = g g^T / V_eff <= F`` and ``eta_T in [0, 1]``, with equality iff ``psi`` is the score.  That
upgrades "the information-optimal readout is the score" from observables to arbitrary functionals:
it is the semiparametric efficiency bound.  :class:`Expectation` is simply the case ``psi = v``,
where ``V_eff`` is *additionally* the exact single-shot variance.

Verified numerically: ``dT/dx`` matches autograd to 8 decimals, and ``Var(T_hat) . S`` matches
``Var_p(psi)`` for a dense quadratic (0.0693 vs 0.0685) and for entropy (0.197 vs 0.187) -- the
residual being the ``O(1/S)`` correction that drives the bias caveat on :meth:`Observable.influence`.

*Axis B -- the score-vector builders* (:mod:`v2.observable.scorers`).  How ``v(n)`` is built from
the outcome: plain counting, a graph reading, ``exp``/``cos`` of a count polynomial, or a marked
outcome.  **All of these produce Expectations** -- a graph-based observable and an ``exp(poly)``
observable are both ``probs @ v`` and so are genuine expectation values; they differ only in how
``v`` is constructed.  That is why they are builders here rather than shapes.

Dropped from the legacy package: ``SelectiveObservable`` and the ``loop_path`` ``__L``/``__P``
pre-selection, plus the spoqc ``match{N}_`` prefix.  A post-selected mean is a score vector on a
smaller support, so it earns no class of its own.

**Build / score split.**  A family is built from an :class:`ObservableContext` -- ``m``, ``k``, the
outcome ``keys``, seeds and graph knobs -- and precomputes every per-outcome table it needs.
``score(probs)`` is then pure tensor algebra and never re-reads ``keys``.  Because the tables are
keyed to a fixed basis, one family object scores a live forward pass and a distribution reloaded
from disk identically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ObservableContext:
    """Everything a family needs to build itself, independent of where the probs come from.

    ``keys`` is the fixed outcome basis (per-mode occupation tuples) the tables align to.  It is
    empty on the *hash* path -- :meth:`ObservableFamily.spec` is called before any circuit exists --
    so a ``spec`` may only use ``m``, ``k`` and the knobs below.

    ``input_state`` and ``reference_probs`` are the two pieces of circuit context some families
    need (``marked`` and ``xent`` respectively).  In the legacy pipeline neither was persisted, so
    those observables could not be re-scored offline; ``v2`` stores both in the artifact
    (:mod:`v2.pipeline.artifact`), so every observable is offline-re-scorable and these are
    normally populated.
    """

    m: int
    k: int
    keys: Sequence = field(default=())
    seed: int = 0
    graph_seed: int | None = None
    angle_seed: int | None = None
    n_vertices: int | None = None
    graph_density: float | None = None
    input_state: Sequence[int] | None = None
    reference_probs: Sequence | Callable[[], Sequence] | None = None
    #: Modes reserved for a post-selection readout (``spin_magic``); ``()`` otherwise.
    readout_modes: Sequence[int] = field(default=())
    #: Encoded-input dimension, needed only by circuit-derived graph constructions
    #: (``connected_<reading>_pairU``) that reconstruct the sandwich unitary via
    #: :func:`~circuit.photonic_circuit.sandwich_unitaries`/``sandwich_unitary_at`` at ``x=0`` --
    #: ``None`` for every other family, which never needs it.
    n_features: int | None = None

    def __post_init__(self):
        set_ = object.__setattr__                    # frozen -> resolve defaults in place
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
    def n_out(self) -> int:
        """Number of outcomes, i.e. the length of every per-outcome table."""
        return len(self.keys)

    def require(self, **knobs):
        """Return the named knobs, raising if any is ``None`` (a family's missing-input guard)."""
        missing = [n for n, v in knobs.items() if v is None]
        if missing:
            raise ValueError(f"this observable requires {', '.join(missing)}")
        return tuple(knobs.values()) if len(knobs) > 1 else next(iter(knobs.values()))

    def resolve_reference_probs(self, label: str) -> np.ndarray:
        """The fixed reference distribution ``q`` as a ``(n_out,)`` float64 array.

        May be the vector itself or a zero-argument callable returning it, so the extra circuit
        evaluation happens only for the families that actually want ``q``.
        """
        q = self.reference_probs
        if q is None:
            raise ValueError(
                f"{label} needs reference_probs -- the output distribution q at x = 0. v2 stores "
                "it in the artifact as 'probs_at_zero', so this means the artifact predates that "
                "field or was built by hand; pass reference_probs=<(n_out,) vector> explicitly."
            )
        q = np.asarray(q() if callable(q) else q, dtype=np.float64).ravel()
        if self.keys and q.shape[0] != self.n_out:
            raise ValueError(f"reference_probs has {q.shape[0]} entries but the outcome basis "
                             f"has {self.n_out}")
        return q


# --- the three functional shapes ----------------------------------------------------------- #


def as_vec(vec) -> torch.Tensor:
    """Coerce a per-outcome score list/array to a ``(n_out,)`` float32 tensor."""
    return torch.as_tensor(np.asarray(vec, dtype=np.float64), dtype=torch.float32)


class Observable(nn.Module):
    """Scores a full outcome distribution: ``(N, n_out)`` probs -> ``(N,)`` scores.

    An ``nn.Module`` so precomputed tables ride along as buffers (device moves, ``state_dict``).
    Subclasses implement :meth:`score`.
    """

    #: True only for :class:`Expectation`: the score is the mean of a per-shot random variable, so
    #: :meth:`effective_variance` is the *exact* single-shot variance rather than an asymptotic one.
    #: Not a gate on the efficiency metrics -- those work for every differentiable shape.
    is_expectation: bool = False

    #: False when ``T`` is not differentiable in ``p`` everywhere on the support (``max_prob`` at a
    #: tie).  This -- not the shape -- is what excludes an observable from analysis B.
    is_differentiable: bool = True

    #: True when outcomes carrying zero mass contribute nothing, so the score may be computed over a
    #: **partial** basis (the observed support of a shot draw).  Holds for every implemented shape;
    #: see :func:`observable_on_keys` for the one way it could fail.  Note this is independent of
    #: :attr:`is_differentiable`: ``max_prob`` is partial-basis safe (a zero can never be the max)
    #: yet still excluded from analysis B, and its shot plug-in is biased upward by ``+0.018`` at
    #: 100 shots against a value of ``0.094``.
    partial_basis_safe: bool = True

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.score(probs)

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi_n = dT/dp_n``, the influence function: ``(N, n_out)``.

        The object every efficiency quantity is built from, since ``dT/dx_i = Cov(psi, s_i)`` and
        ``Var(T_hat) ~ Var_p(psi)/S``.  It is ``x``-dependent for every shape except
        :class:`Expectation` (where it is the constant ``v``), which is why this takes ``probs``.

        **Three genuine caveats, none of them "there is no variance":**

        1. *Bias, not variance, is what breaks nonlinear functionals at finite shots.* The plug-in
           ``T(p_hat)`` is biased at ``O(1/S)``; for entropy-like functionals the bias runs as
           ``~(support - 1)/(2S)``, i.e. ``-0.275`` nats at ``n_out = 56, S = 100``.  Against bare
           ``ent ~ -3.5`` that is ~8%, but it can exceed the *weighted* variants outright
           (``ent_parity ~ -0.18``).  This gates the shot-budget and ``R^2``-ceiling readings in
           :mod:`v2.metrics.observable`, which assume **zero-mean** label noise: a biased label adds
           a term that does not vanish with more training data.  It does **not** affect the exact
           ``eta_T`` computed here, which never samples.
        2. *Degenerate U-statistics.*  If ``zeta_1 = Var(K p) = 0`` for a :class:`Quadratic` (i.e.
           ``(K p)_n`` is constant on the support), the ``1/S`` term vanishes, convergence becomes
           ``O(1/S)`` with a weighted-``chi^2`` limit, and ``V_eff`` is not the operative scale.
           Flagged by :meth:`Quadratic.u_statistic_degeneracy`, not excluded.
        3. *Non-smooth functionals.*  ``max_prob`` has ``psi_n = 1[n = argmax]``, defined only away
           from ties, and its plug-in is badly biased upward.  It raises -- for
           non-differentiability, via :attr:`is_differentiable`, not for lacking a variance.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define an influence function. If T is differentiable "
            "in p, implement influence(); if it is not (a max, a median, an argmax), set "
            "is_differentiable = False so analysis B excludes it for the right reason."
        )

    def effective_variance(self, probs: torch.Tensor) -> torch.Tensor:
        """``V_eff = Var_p(psi)`` per row -- the ``S = 1`` scale of the plug-in estimator.

        Exact single-shot variance for :class:`Expectation` (where ``psi = v``, giving
        ``p.v^2 - (p.v)^2``); the leading-order asymptotic variance otherwise.  Either way this is
        the denominator of ``eta_T``, and the ``[0, 1]`` bound holds by Cauchy-Schwarz in both cases.

        Clamped at 0 only against float round-off on a deterministic distribution, never to mask a
        real negative.
        """
        psi = self.influence(probs)
        mean = (probs * psi).sum(dim=1)
        second = (probs * psi * psi).sum(dim=1)
        return (second - mean * mean).clamp(min=0.0)


class Expectation(Observable):
    """``O(x) = <P> = probs @ v`` -- the plain diagonal expectation.

    Every score-vector builder in :mod:`v2.observable.scorers` produces one of these, whatever
    the construction of ``v``: counting, graph readings and ``exp(poly)`` forms are all genuine
    expectation values of a diagonal observable.

    The one shape with a genuine *single-shot* estimator: draw ``n ~ p`` and average ``v(n)``,
    giving additive-error estimation in ``~1/eps^2`` shots.  So its influence function is the
    constant ``psi = v`` and :meth:`effective_variance` is the exact single-shot variance, not an
    asymptotic one.  The other shapes are still in scope for the efficiency metrics -- see
    :meth:`Observable.influence`.
    """

    is_expectation = True

    def __init__(self, score_vec):
        super().__init__()
        self.register_buffer("score_vec", as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.score_vec

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi_n = v_n`` -- constant in ``p``, broadcast to ``(N, n_out)`` for a uniform API."""
        return self.score_vec.unsqueeze(0).expand(probs.shape[0], -1)


class Quadratic(Observable):
    """``O(x) = p^T K p`` -- the degree-2 form for a symmetric pair kernel ``K``.

    ``K`` is stored dense, so ``score`` costs ``O(n_out^2)`` per row -- which is why ``pairprod``
    caps the outcome dimension.  :class:`DiagonalQuadratic` is the diagonal case and stores only
    ``diag(K)``.

    A 2-copy / collision quantity: estimable from *pairs* of shots via a U-statistic rather than
    from single shots.  Its influence function is ``psi = 2 K p`` -- the Hajek projection of the
    order-2 U-statistic -- and ``V_eff = 4 Var(K p) = 4 zeta_1`` is the leading term of
    ``Var(T_hat) = 4(S-2)/(S(S-1)) zeta_1 + 2/(S(S-1)) zeta_2``.

    The factors of 4 cancel in the efficiency ratio, so a quadratic observable's ``eta`` equals
    that of a **linear** observable with the ``x``-dependent score vector ``w = K p_x``.  No new
    machinery is needed; ``w`` is just recomputed per row, which is free because ``p`` is stored.
    """

    def __init__(self, pair_kernel=None):
        super().__init__()
        if pair_kernel is not None:
            self.register_buffer("pair_kernel", torch.as_tensor(
                np.asarray(pair_kernel, dtype=np.float64), dtype=torch.float32))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        kernel = self.pair_kernel.to(probs.dtype)
        return (probs @ kernel * probs).sum(dim=1)

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi_n = 2 (K p)_n`` -- ``x``-dependent, unlike an Expectation's constant ``v``.

        ``pair_kernel`` is stored ``float32`` (see ``__init__``); cast to ``probs``'s own dtype
        before the matmul rather than the other way around, since ``torch.matmul`` requires exact
        dtype agreement (no auto-promotion the way elementwise ops get) and callers such as
        :mod:`metrics.gradient`'s finite-difference path pass ``float64`` probs for precision --
        forcing probs down to ``float32`` here would silently degrade that.
        """
        kernel = self.pair_kernel.to(probs.dtype)
        return 2.0 * (probs @ kernel)

    def u_statistic_degeneracy(self, probs: torch.Tensor) -> torch.Tensor:
        """``zeta_1 / zeta_2`` per row -- a **flag**, not an exclusion.

        ``zeta_1 = Var_p(K p)`` and ``zeta_2 = Var_{p x p}(K)``.  When ``zeta_1 / zeta_2 ~ 0`` the
        U-statistic is *degenerate*: the ``1/S`` term vanishes, convergence becomes ``O(1/S)`` with
        a weighted-``chi^2`` limit rather than Gaussian, and ``V_eff`` is no longer the operative
        scale -- so ``eta`` is still well-defined but the shot-count reading is not.  Report it
        alongside; do not drop the cell.
        """
        K = self.pair_kernel.to(probs.dtype)                            # see influence()'s docstring
        w = probs @ K                                                   # (N, n_out)
        m1 = (probs * w).sum(dim=1, keepdim=True)
        zeta1 = (probs * (w - m1) ** 2).sum(dim=1)
        # zeta_2 = Var of K[n1, n2] over independent draws n1, n2 ~ p.
        mean_k = (probs @ K * probs).sum(dim=1)
        second = (probs @ (K * K) * probs).sum(dim=1)
        zeta2 = (second - mean_k * mean_k).clamp(min=0.0)
        return zeta1 / zeta2.clamp(min=1e-30)

    def kernel_matrix(self) -> torch.Tensor:
        """The dense ``(n_out, n_out)`` ``K`` this observable is equivalent to."""
        return self.pair_kernel


class DiagonalQuadratic(Quadratic):
    """``O(x) = sum_n v_n p(n)^2`` -- :class:`Quadratic` with ``K = diag(v)``.

    Exactly ``p^T diag(v) p``, but only the ``(n_out,)`` diagonal is stored, so it costs
    ``O(n_out)`` per row instead of ``O(n_out^2)`` and carries no dimension cap -- which is why
    ``sq_<base>`` is the fallback when ``pairprod``'s dense kernel will not fit.
    """

    def __init__(self, score_vec):
        super().__init__()                     # no dense kernel: the diagonal is the whole K
        self.register_buffer("score_vec", as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs * probs) @ self.score_vec

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi_n = 2 v_n p_n`` -- the diagonal case of ``2 (K p)_n``, without forming ``K``."""
        return 2.0 * self.score_vec.unsqueeze(0) * probs

    def u_statistic_degeneracy(self, probs: torch.Tensor) -> torch.Tensor:
        w = self.score_vec.unsqueeze(0) * probs                        # (K p) with K = diag(v)
        m1 = (probs * w).sum(dim=1, keepdim=True)
        zeta1 = (probs * (w - m1) ** 2).sum(dim=1)
        # K = diag(v): E[K] = sum_n p_n^2 v_n over independent draws hitting the same outcome.
        mean_k = (probs * probs * self.score_vec.unsqueeze(0)).sum(dim=1)
        second = (probs * probs * self.score_vec.unsqueeze(0) ** 2).sum(dim=1)
        zeta2 = (second - mean_k * mean_k).clamp(min=0.0)
        return zeta1 / zeta2.clamp(min=1e-30)

    def kernel_matrix(self) -> torch.Tensor:
        return torch.diag(self.score_vec)


class ProbFunction(Observable):
    """``O(x) = sum_n v_n . phi(p(n))`` -- a weighted sum of an elementwise nonlinearity of ``p``.

    The shared shape of the non-polynomial families: choose the scalar ``phi`` by overriding
    :meth:`transform` and the rest -- the optional per-outcome weight ``v`` (omitted = ``1``),
    the reduction, the buffer plumbing -- comes for free.  A *linear* ``phi`` would just
    reproduce :class:`Expectation`; the point is the ones that are not, whose value is therefore
    not the expectation of any fixed diagonal observable and cannot be estimated to additive
    error from a bounded number of single shots.

    ``phi`` is a method rather than a stored callable so subclasses stay picklable and their
    parameters (an ``eps``, say) are plain attributes.
    """

    def __init__(self, score_vec=None):
        super().__init__()
        self.register_buffer("score_vec", None if score_vec is None else as_vec(score_vec))

    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        """``phi`` applied elementwise to an ``(N, n_out)`` probability matrix."""
        raise NotImplementedError

    @property
    def partial_basis_safe(self) -> bool:
        """True iff ``phi(0) == 0``, so an unobserved outcome contributes nothing.

        Checked rather than assumed: both implemented transforms vanish at 0 (``ent``'s ``u log u``,
        and ``osc``'s ``p sin(1/(p + eps))``, whose ``p`` prefactor exists precisely for this), but a
        transform like a bare ``sin(1/(p + eps))`` would give every unobserved outcome a
        contribution of ``v_n sin(1/eps)`` and silently break the finite-sample path.
        """
        with torch.no_grad():
            return bool(self.transform(torch.zeros(1, 1)).abs().max() < 1e-9)

    def transform_grad(self, probs: torch.Tensor) -> torch.Tensor:
        """``phi'`` applied elementwise -- needed for the influence function ``psi = v phi'(p)``.

        Analytic rather than autograd, so it works on a distribution loaded from disk (no graph) and
        so the expression is inspectable next to ``phi``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement transform_grad (phi') to take part in the "
            "efficiency metrics; psi_n = v_n phi'(p_n)."
        )

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        t = self.transform(probs)
        return t.sum(dim=1) if self.score_vec is None else t @ self.score_vec

    def influence(self, probs: torch.Tensor) -> torch.Tensor:
        """``psi_n = v_n phi'(p_n)`` (or ``phi'(p_n)`` when ``v`` is omitted)."""
        d = self.transform_grad(probs)
        return d if self.score_vec is None else d * self.score_vec.unsqueeze(0)


# --- families and the registry -------------------------------------------------------------- #


class ObservableFamily:
    """One observable family: the names it owns, how to build it, how to identify it.

    ``describe`` is the spelling shown in the "unknown observable" error, so it should read as a
    *pattern* (``"sq_<base>"``) rather than a single name.
    """

    describe: str = ""

    def matches(self, name: str) -> bool:
        """True iff this family owns the observable string ``name``."""
        raise NotImplementedError

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        """Precompute this family's per-outcome tables over ``ctx.keys``."""
        raise NotImplementedError

    def spec(self, name: str, ctx: ObservableContext) -> dict:
        """Identity fields, canonicalised so equivalent spellings collide.

        Called with a ``keys``-less ``ctx`` (identity is computed before the circuit exists), so
        it may only use ``m``, ``k`` and the seeds / graph knobs.  This is the score-cache key,
        NOT part of the dataset identity -- the distribution does not depend on the observable.
        """
        return {}


#: Registered families, in match order (see :func:`find_family`).
_FAMILIES: list[ObservableFamily] = []

#: ``base -> (ctx) -> score vector``: the elementary per-outcome scorers.  The composite shapes
#: (``sq_<base>``, ``ent_<base>``, ...) score *over* one of these, so they look it up here
#: instead of each re-implementing the same if-chain.
BASE_SCORERS: dict[str, Callable[[ObservableContext], Sequence[float]]] = {}

#: name -> ``v(key, ctx) -> float``: the per-OUTCOME scorer, evaluated on one occupation tuple.
#:
#: This is the primitive; :data:`BASE_SCORERS`' dense tables are *derived* from it by mapping over a
#: key list, so each scorer is defined once instead of as a table and a function that must agree.
#: It is what makes the **finite-sample path** possible: a shot draw observes only the outcomes it
#: lands on, so scoring it means evaluating ``v`` on the observed keys rather than indexing a table
#: built over a basis that may be too large to enumerate (``m=20, k=10`` is ``C(29,10)`` outcomes).
KEY_SCORERS: dict[str, Callable[[Sequence[int], ObservableContext], float]] = {}


def register(family: ObservableFamily) -> ObservableFamily:
    """Register an observable family (called at module import; see this package's ``__init__``)."""
    _FAMILIES.append(family)
    return family


def find_family(name: str) -> ObservableFamily | None:
    """The family owning ``name``, or ``None``.  First match wins."""
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
    """Build the observable named ``name`` over the outcome basis in ``ctx``."""
    family = find_family(name)
    if family is None:
        raise ValueError(f"unknown observable {name!r}; expected one of: {observable_help()}")
    return family.build(name, ctx)


def observable_spec(name: str, ctx: ObservableContext | None = None) -> dict:
    """Canonical identity fields for ``name`` -- the score-cache key.

    Always includes the name itself, so two observables never collide even when a family adds no
    extra fields.
    """
    family = find_family(name)
    if family is None:
        raise ValueError(f"unknown observable {name!r}; expected one of: {observable_help()}")
    ctx = ctx or ObservableContext(m=0, k=0)
    return {"observable": name, **family.spec(name, ctx)}


def observable_spec_hash(name: str, ctx: ObservableContext | None = None) -> str:
    """8-char hash of :func:`observable_spec` -- the score-cache filename."""
    blob = json.dumps(observable_spec(name, ctx), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def base_score_vec(base: str, ctx: ObservableContext, *, allowed, label: str,
                   keys=None) -> np.ndarray:
    """Per-outcome vector for the elementary scorer ``base``, gated on ``allowed``.

    ``label`` names the family in the error, so a typo reports e.g. ``unknown sq base 'majorty'``.

    ``keys`` defaults to ``ctx.keys`` -- the full basis.  Pass an explicit list to build the vector
    over a **partial** basis, which is the finite-sample path: a shot draw only observes the outcomes
    it lands on, and at large ``(m, k)`` the full basis cannot be enumerated at all.  The scorer is
    evaluated per key either way, so the two agree exactly on the keys they share.
    """
    if base not in allowed or base not in KEY_SCORERS:
        raise ValueError(f"unknown {label} base {base!r}; choose from {tuple(allowed)}")
    fn = KEY_SCORERS[base]
    return np.asarray([fn(key, ctx) for key in (ctx.keys if keys is None else keys)],
                      dtype=np.float64)


class _PlainFamily(ObservableFamily):
    """A single-name observable that is just an :class:`Expectation` over a score vector."""

    def __init__(self, name: str, vec, check):
        self.name, self.vec, self.check = name, vec, check
        self.describe = name

    def matches(self, name: str) -> bool:
        return name == self.name

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if self.check is not None:
            self.check(ctx)
        return Expectation(self.vec(ctx))


def key_scorer(name: str, *, plain_check=None):
    """Register an elementary scorer from its **per-outcome** function ``v(key, ctx) -> float``.

    One definition yields three things:

    * the ``name`` observable itself -- an :class:`Expectation` over the mapped vector;
    * an entry in :data:`BASE_SCORERS`, so ``sq_<name>`` / ``ent_<name>`` / ``osc_<name>`` /
      ``xent_<name>`` score over it without duplicating the scorer;
    * an entry in :data:`KEY_SCORERS`, which is what lets the vector be built over a partial basis
      for the finite-sample path.

    Registering the per-key function rather than the dense table is the point: the table is a
    derived artifact, so there is no second implementation to drift.

    ``plain_check(ctx)`` runs only when ``name`` *is* the observable, so a constraint the plain form
    enforces (``majority`` needs even ``m``) does not silently spread to the composites, which never
    enforced it.
    """
    def decorate(fn):
        KEY_SCORERS[name] = fn
        vec = (lambda ctx, _fn=fn: [_fn(key, ctx) for key in ctx.keys])
        BASE_SCORERS[name] = vec
        register(_PlainFamily(name, vec, plain_check))
        return fn
    return decorate


def observable_on_keys(name: str, ctx: ObservableContext, keys) -> Observable:
    """Build ``name`` against an arbitrary outcome basis -- the finite-sample entry point.

    Every family builds its per-outcome tables from ``ctx.keys``, so restricting the basis to the
    outcomes actually observed is just a different context.  Verified on the fermion model's
    collision-free sector, whose 36 absent outcomes give a natural partial basis: all eleven
    families tested agree with the full-basis score to ``<= 4.8e-7`` (float32 round-off).

    That works because an unobserved outcome contributes nothing to any implemented shape --
    ``Expectation`` and the quadratics are homogeneous in ``p``, and both pointwise transforms
    vanish at 0 (``ent``'s ``u log u``, and ``osc``'s ``p sin(1/(p+eps))``, whose ``p`` prefactor is
    there precisely for this).  A future ``ProbFunction`` with ``phi(0) != 0`` would break it
    silently, since every unobserved outcome would then contribute ``v_n phi(0)``, so
    :attr:`Observable.partial_basis_safe` guards it rather than leaving it to be discovered.
    """
    import dataclasses
    obs = resolve_observable(name, dataclasses.replace(ctx, keys=tuple(keys)))
    if not obs.partial_basis_safe:
        raise ValueError(
            f"observable {name!r} cannot be scored on a partial outcome basis: its transform does "
            "not vanish at p = 0, so every UNOBSERVED outcome contributes v_n * phi(0) and dropping "
            "them changes the value. Score it against the full basis, or give it a transform with "
            "phi(0) = 0."
        )
    return obs
