"""Free-fermion readout: the SAME sandwich circuit with ``det`` in place of ``Perm``.

Boson sampling gives outcome ``n`` the probability
``p_boson(n) = |Perm(U_{s,n})|^2 / prod_j n_j!``, where ``U(x) = (W2 D(x) W1)^T``.  This model
changes exactly one thing::

    p_fermion(n) = |det(U_{s,n})|^2

**Why that one swap.**  It is the canonical classical/quantum dividing line: a permanent is
``#P``-hard (Valiant) while a determinant is in ``P`` (an ``O(k^3)`` LU), and correspondingly boson
sampling is hard to simulate while fermion sampling is classically efficient (Valiant,
Terhal-DiVincenzo).  Everything else -- the circuit, its ``2m^2 - 1`` parameters, the seed, the
input state, the outcome basis, the observable scoring it -- is held fixed, so a difference between
this and :class:`v2.model.photonic.PhotonicModel` is attributable to the matrix function alone.

**But note what that does and does not license.**  The Fisher-spectrum comparison in
:mod:`v2.metrics.distribution` measures describability and estimability of the labelling function,
**not** computational hardness -- boson sampling is itself the counterexample, being indexed by
``O(m^2)`` numbers (polynomially describable) while each ``p(n)`` is ``#P``-hard.  So a spectral
difference here is a learnability statement, not evidence about the ``Perm``/``det`` separation.

**Flavoured fermions: the bunching mod, and the subtlety that makes it non-trivial.**  Strict free
fermions (``flavours = 1``) put *exactly zero* mass on any bunched outcome, since a repeated column
makes a determinant vanish -- Pauli exclusion.  That leaves the support at ``C(m, k)`` of the
``C(m+k-1, k)`` basis states (20 of 56 at ``m=6, k=3``), so a naive comparison against the boson
model conflates *support size* with *matrix function*.  Giving each mode ``r`` internal states
(spinful fermions) lets up to ``r`` photons share a spatial mode.

The subtlety, which a first implementation here got wrong: the interferometer is **flavour-blind**
(a beamsplitter does not act on spin), so the lifted unitary is ``U kron I_r`` -- *flavour-diagonal*,
hence **flavour is conserved**.  Lifting alone therefore changes nothing: with every input photon in
flavour 0, any bunched outcome needs a photon in another flavour, which is unreachable, so the
determinant still vanishes and ``r > 1`` merely reproduces the ``r = 1`` physics inside flavour 0.
Two things are required:

1. the input photons must be **distributed across flavours** (photon ``j`` takes flavour
   ``j mod r``), so two photons of *different* flavour can share a spatial mode; and
2. only the *spatial* occupation is observed, so the probability must **sum over every
   flavour-conserving assignment** of that occupation::

       p(n) = sum_{assignments} prod_f |det(U[S_f, T_f])|^2

   where ``S_f`` are the input modes carrying flavour ``f`` and ``T_f`` the output modes assigned
   to it.  The product form is exactly *because* ``U kron I_r`` is flavour-diagonal: the ``k x k``
   determinant block-factorises, so the lifted matrix is never needed -- only the spatial ``U``.

Still an ``O(k^3)``-per-block determinant, and still classically efficient: the assignment set
depends only on ``(n, flavour counts)``, so it is enumerated once at construction and the per-batch
cost is a fixed number of small determinants.

**What support this actually reaches.**  An outcome is reachable only if its occupation can be
flavour-assigned within the input's per-flavour counts, so the gap *narrows* with ``r`` rather than
closing at ``r = 2``.  Measured at ``m=6, k=3`` (input state ``[1,0,1,0,1,0]``):

====  ==================  ===============  ======================
r     reachable / 56      assignments      mean bunched mass
====  ==================  ===============  ======================
1     20  (= C(6,3))      20               0.000000  (exactly)
2     50                  90               0.317
3     56  (all)           216              0.403
====  ==================  ===============  ======================

The 6 outcomes missing at ``r = 2`` are exactly the triply-occupied ones (``(3,0,0,0,0,0)`` and its
permutations): three photons in one mode need three distinct flavours.  At ``r = k`` every input
photon has its own flavour, so the support is complete.  :attr:`FermionModel.n_reachable` reports
this at runtime, and the metrics' shared-support comparison
(:mod:`v2.metrics.distribution`) remains the honest one to read.

Normalisation: at ``r = 1`` Cauchy-Binet already gives
``sum_{|S|=k} |det(U_{R,S})|^2 = det(U_R U_R^dagger) = 1``, so the explicit renormalisation only
removes float residue; at ``r > 1`` the assignment sum changes the total and it is genuine.

Carried from the untracked ``model/fermion.py``, with the flavour construction added.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import torch

from circuit.encoding import build_encoding
from circuit.photonic_circuit import default_input_state, sandwich_unitaries, sandwich_unitary_at
from .base import DistributionModel
from ..circuit.fock import fock_keys

if TYPE_CHECKING:
    from config import ExperimentConfig


def occupied_modes(occ) -> list[int]:
    """Mode indices of an occupation vector, each repeated by its count (length ``k``)."""
    out: list[int] = []
    for i, c in enumerate(occ):
        out += [i] * int(c)
    return out


def input_flavours(input_state, flavours: int) -> list[int]:
    """Flavour of each input photon: photon ``j`` (in mode order) takes flavour ``j mod r``.

    Spreading the input across flavours is what makes spatial bunching reachable at all -- with
    every photon in flavour 0 and a flavour-conserving interferometer, no output mode can hold two
    photons.  Round-robin keeps the per-flavour counts as even as possible, which maximises the
    reachable support for a given ``r``.
    """
    r = int(flavours)
    photons = sum(int(c) for c in input_state)
    return [j % r for j in range(photons)]


def flavour_mode_sets(input_state, flavours: int):
    """``S_f``: the input *spatial* modes carrying each flavour, as a list of ``r`` index lists."""
    r = int(flavours)
    modes = occupied_modes(input_state)
    fl = input_flavours(input_state, r)
    sets: list[list[int]] = [[] for _ in range(r)]
    for mode, f in zip(modes, fl):
        sets[f].append(mode)
    return sets


def _assignments(key, capacities: list[int]):
    """Every flavour-conserving assignment of occupation ``key``, as a list of ``r`` mode lists.

    Walks the modes in order and, for each, chooses which ``n_i`` of the ``r`` flavours serve it
    (distinct flavours -- Pauli within a ``(mode, flavour)`` pair), respecting the remaining
    per-flavour capacity.  Yields ``T`` where ``T[f]`` is the sorted list of modes assigned to
    flavour ``f`` and ``len(T[f]) == capacities[f]``.

    Depends only on ``(key, capacities)``, never on ``x``, so this is enumerated once at
    construction.
    """
    r = len(capacities)
    occ = [int(c) for c in key]
    out: list[list[list[int]]] = []

    def walk(i: int, remaining: list[int], acc: list[list[int]]):
        if i == len(occ):
            if all(v == 0 for v in remaining):
                out.append([sorted(a) for a in acc])
            return
        n_i = occ[i]
        if n_i == 0:
            walk(i + 1, remaining, acc)
            return
        for combo in combinations(range(r), n_i):     # distinct flavours for this mode
            if any(remaining[f] == 0 for f in combo):
                continue
            for f in combo:
                remaining[f] -= 1
                acc[f].append(i)
            walk(i + 1, remaining, acc)
            for f in combo:
                remaining[f] += 1
                acc[f].pop()

    walk(0, list(capacities), [[] for _ in range(r)])
    return out


def build_assignment_tables(keys, input_state, flavours: int):
    """Precompute the flavour-block index tables for every reachable outcome.

    Returns ``(positions, blocks, assign_index)``:

    * ``positions`` -- indices into ``keys`` of the reachable outcomes;
    * ``blocks`` -- per flavour ``f``, ``(rows_f, cols_f)`` where ``rows_f`` is the fixed
      ``(c_f,)`` input-mode index and ``cols_f`` an ``(n_sets_f, c_f)`` array of the distinct
      output-mode sets appearing anywhere;
    * ``assign_index`` -- ``(n_assignments, r)`` int array; row ``a`` gives, per flavour, which row
      of ``cols_f`` that assignment uses.  Paired with ``assign_owner`` (``(n_assignments,)``,
      the index into ``positions``) so the per-assignment products scatter-add into outcomes.

    Factorising this way means each distinct ``(flavour, mode set)`` determinant is computed **once
    per batch** and then looked up, instead of once per assignment.
    """
    r = int(flavours)
    S = flavour_mode_sets(input_state, r)
    capacities = [len(s) for s in S]

    set_ids: list[dict[tuple, int]] = [{} for _ in range(r)]
    set_lists: list[list[tuple]] = [[] for _ in range(r)]
    positions: list[int] = []
    assign_index: list[list[int]] = []
    assign_owner: list[int] = []

    for idx, key in enumerate(keys):
        if max(int(c) for c in key) > r:
            continue                                  # Pauli exclusion at r internal states
        found = _assignments(key, capacities)
        if not found:
            continue                                  # occupation not flavour-assignable
        own = len(positions)
        positions.append(idx)
        for T in found:
            row = []
            for f in range(r):
                t = tuple(T[f])
                if t not in set_ids[f]:
                    set_ids[f][t] = len(set_lists[f])
                    set_lists[f].append(t)
                row.append(set_ids[f][t])
            assign_index.append(row)
            assign_owner.append(own)

    blocks = []
    for f in range(r):
        cols = (np.asarray(set_lists[f], dtype=np.int64).reshape(len(set_lists[f]), capacities[f])
                if set_lists[f] else np.zeros((0, capacities[f]), dtype=np.int64))
        blocks.append((np.asarray(S[f], dtype=np.int64), cols))
    return (np.asarray(positions, dtype=np.int64), blocks,
            np.asarray(assign_index, dtype=np.int64).reshape(len(assign_index), r),
            np.asarray(assign_owner, dtype=np.int64))


def determinant_probs(U: torch.Tensor, positions, blocks, assign_index, assign_owner,
                      n_out: int) -> torch.Tensor:
    """``(N, n_out)`` fermion distribution, summed over flavour-conserving assignments.

    ``p(n) = sum_a prod_f |det(U[S_f, T_f^{(a)}])|^2``.  Each flavour's distinct determinants are
    evaluated once and looked up, so the cost is the number of *distinct* blocks, not of
    assignments.

    Differentiable in ``U`` (hence in ``x``): ``torch.linalg.det`` has a gradient, which is what
    lets :mod:`v2.metrics` take exact input-Jacobians of this readout.
    """
    N = U.shape[0]
    per_flavour = []
    for rows_np, cols_np in blocks:
        c_f = cols_np.shape[1]
        if c_f == 0:                                  # an empty flavour block contributes 1
            per_flavour.append(torch.ones(N, max(cols_np.shape[0], 1), dtype=U.real.dtype,
                                          device=U.device))
            continue
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=U.device)
        cols = torch.as_tensor(cols_np, dtype=torch.long, device=U.device)
        Ur = U[:, rows, :]                                            # (N, c_f, m)
        sub = Ur[:, :, cols].permute(0, 2, 1, 3)                      # (N, n_sets, c_f, c_f)
        per_flavour.append(torch.linalg.det(sub).abs() ** 2)          # (N, n_sets)

    ai = torch.as_tensor(assign_index, dtype=torch.long, device=U.device)      # (n_assign, r)
    weight = torch.ones(N, ai.shape[0], dtype=per_flavour[0].dtype, device=U.device)
    for f, vals in enumerate(per_flavour):
        weight = weight * vals[:, ai[:, f]]
    out = torch.zeros(N, n_out, dtype=weight.dtype, device=U.device)
    owner = torch.as_tensor(positions[assign_owner], dtype=torch.long, device=U.device)
    out.index_add_(1, owner, weight)                                  # sum over assignments
    # At r = 1 Cauchy-Binet already gives sum = 1, so this only removes float residue; at r > 1 the
    # assignment sum changes the total and the renormalisation is genuine.
    return out / out.sum(dim=1, keepdim=True).clamp(min=1e-30)


def boson_probs_reference(U: torch.Tensor, s_modes, keys) -> torch.Tensor:
    """``|Perm(U_{s,n})|^2 / prod_j n_j!`` -- the boson distribution, in pure torch.

    Brute force over ``k!`` permutations, so this is for small ``k``.  **Load-bearing, not merely a
    check**: it is differentiable in ``x`` and built from the *same*
    :func:`~v2.circuit.photonic.sandwich_unitary_at` as the determinant path, so the headline
    ``Perm`` vs ``det`` Fisher comparison is a one-line swap with every other factor held fixed.
    Going through merlin instead would compare two different code paths.

    Also the verification that :func:`~v2.circuit.photonic.sandwich_unitaries` reproduces merlin's
    circuit -- the two agree to ~1e-7.
    """
    from itertools import permutations

    rows = torch.as_tensor(np.asarray(s_modes), dtype=torch.long, device=U.device)
    Ur = U[:, rows, :]
    out_cols = []
    for key in keys:
        cols = torch.as_tensor(occupied_modes(key), dtype=torch.long, device=U.device)
        A = Ur[:, :, cols]                                                    # (N, k, k)
        kk = A.shape[1]
        tot = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
        for p in permutations(range(kk)):
            term = torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
            for i in range(kk):
                term = term * A[:, i, p[i]]
            tot = tot + term
        denom = math.prod(math.factorial(int(c)) for c in key)
        out_cols.append(tot.abs() ** 2 / denom)
    return torch.stack(out_cols, dim=1)


class FermionModel(DistributionModel):
    """``X -> probs``: the sandwich circuit read out fermionically (``det``).

    Matched to :class:`~v2.model.photonic.PhotonicModel` at the same
    ``(m, k, n_features, seed, encoding)`` -- same circuit, same input state, same Fock basis, same
    scoring code -- with ``|det|^2`` replacing ``|Perm|^2``.  Every registry observable applies.
    """

    name = "fermion"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 flavours: int = 1, encoding="phase"):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        r = int(flavours)
        if r < 1:
            raise ValueError(f"flavours must be >= 1 (got {r})")
        if k > r * m:
            raise ValueError(f"fermions need k <= flavours*m by Pauli exclusion "
                             f"(m={m}, k={k}, flavours={r})")
        self.flavours = r
        self.encoding = build_encoding(encoding) if isinstance(encoding, str) else encoding

        self._keys = fock_keys(self.m, self.k)
        self._input_state = default_input_state(self.m, self.k)
        if max(self._input_state) > r:
            raise ValueError(
                f"fermion needs an input state with occupations <= flavours; "
                f"default_input_state({m}, {k}) = {self._input_state} exceeds flavours={r}"
            )
        self.flavour_sets = flavour_mode_sets(self._input_state, r)
        self.positions, self.blocks, self.assign_index, self.assign_owner = (
            build_assignment_tables(self._keys, self._input_state, r))
        if len(self.positions) == 0:
            raise ValueError(f"no outcome is reachable at m={m}, k={k}, flavours={r}")
        self.W1, self.W2 = sandwich_unitaries(self.m, self.seed)

        # Peak intermediate is (chunk, n_sets, c_f, c_f) complex64 per flavour, plus the
        # (chunk, n_assignments) weight matrix; size the chunk on the larger of the two.
        n_sets = max((int(c.shape[0]) for _, c in self.blocks), default=1)
        cost = max(n_sets * self.k * self.k, self.assign_index.shape[0], 1)
        self.forward_batch = max(1, 16_777_216 // cost)

    @property
    def n_reachable(self) -> int:
        """Outcomes with non-zero mass -- the honest support size.

        ``C(m, k)`` at ``flavours=1`` (collision-free only, by Pauli exclusion).  Larger ``r``
        admits bunched outcomes, but only those whose occupation is flavour-assignable within the
        input's per-flavour counts, so the gap *narrows* with ``r` rather than closing -- see the
        module docstring.  Reported so the metrics' shared-support comparison can be stated
        honestly rather than assumed.
        """
        return len(self.positions)

    @property
    def n_assignments(self) -> int:
        """Total flavour-conserving assignments summed over outcomes (the per-batch det budget)."""
        return int(self.assign_index.shape[0])

    def unitary(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, m, m)`` sandwich unitary ``U(x)`` -- the same one the photonic model realises."""
        return sandwich_unitary_at(self.W1, self.W2, X, self.n_features, self.encoding)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        # The spatial U is enough: the lifted U kron I_r is flavour-diagonal, so the k x k
        # determinant block-factorises per flavour (see the module docstring).
        return determinant_probs(self.unitary(X), self.positions, self.blocks,
                                 self.assign_index, self.assign_owner, len(self._keys))

    def boson_probs(self, X: torch.Tensor) -> torch.Tensor:
        """The ``|Perm|^2`` distribution of the *same* circuit -- ``k!`` cost, differentiable.

        Only meaningful at ``flavours = 1`` (bosons have no internal states here); it uses the
        unlifted ``U``.
        """
        return boson_probs_reference(self.unitary(X), occupied_modes(self._input_state),
                                     self._keys)

    def shot_counts(self, X, *, shots: int, shot_seed: int = 0):
        """Not implemented: the fermion model is a probability-distribution model.

        A multinomial wrapper over the stored ``p`` would run, but it would be the wrong sampler to
        build on.  ``p(n) = |det(U[S, T])|^2`` at ``flavours=1`` is a **determinantal projection
        process**, exactly samplable in ``O(m k^2)`` per sample by sequential conditional sampling
        (Hough-Krishnapur-Peres-Virag) -- without materialising any distribution.  That is both
        cheaper than sampling the boson distribution and the natural route past an enumerable basis
        for this family, so it is the implementation this hook should get.

        It is also not needed for the metrics: derivatives are taken outside the sampler by finite
        differences (:mod:`v2.metrics.fd`), and FD on a shot sampler is noise-dominated anyway
        (measured 32% relative error at 50k shots against 1.4e-4 on the exact distribution).
        """
        raise NotImplementedError(
            "fermion is a probability-distribution model; finite-shot draws are not implemented. "
            "The right sampler is the determinantal point process (O(m k^2) per sample, no "
            "distribution needed), not a multinomial over the stored probs. Use probs()."
        )

    def outcome_keys(self):
        return self._keys

    def input_state(self):
        return self._input_state

    def n_model_parameters(self) -> int:
        """``2m^2 - 1`` -- identical to the photonic model, by construction."""
        from circuit.photonic_circuit import n_circuit_parameters

        return n_circuit_parameters(self.m)

    def circuit_spec(self) -> dict:
        # `readout` keeps det and Perm datasets from ever colliding even at a matched seed.
        return {"model": self.name, "readout": "determinant_free_fermion",
                "flavours": self.flavours, **self.encoding.spec()}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "FermionModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed,
                   flavours=cfg.model.flavours, encoding=cfg.model.encoding)

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        r = int(cfg.model.flavours)
        if r < 1:
            raise ValueError(f"model.flavours must be >= 1 (got {r})")
        if cfg.problem.k > r * cfg.problem.m:
            raise ValueError(f"fermion needs k <= flavours*m (k={cfg.problem.k}, "
                             f"m={cfg.problem.m}, flavours={r})")
