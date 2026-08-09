"""Free-fermion teacher: the SAME sandwich circuit, but ``det`` in place of ``Perm``.

Boson sampling gives each outcome ``n`` (per-mode photon counts) the probability

    p_boson(n) = |Perm(U_{s,n})|^2 / prod_j n_j!

where ``U(x) = (W2 D(x) W1)^T`` is the sandwich unitary at input ``x`` and ``U_{s,n}`` is the
``k x k`` submatrix picking the input modes ``s`` as rows and the output modes ``n`` as columns
(each repeated by its occupation).  This teacher changes exactly one thing:

    p_fermion(n) = |det(U_{s,n})|^2

**Why that one swap is the interesting one.**  It is the canonical classical/quantum dividing
line: computing a permanent is ``#P``-hard (Valiant) while a determinant is in ``P`` (an ``O(k^3)``
LU), and correspondingly boson sampling is hard to simulate while fermion sampling is classically
efficient (Valiant, Terhal-DiVincenzo).  Everything else -- the circuit, its ``2m^2`` parameters,
the seed, the input state, the outcome basis, the observable scoring it -- is held fixed, so a
difference between this teacher and :class:`~model.photonic.PhotonicTeacher` is attributable to
the matrix function alone.  Unlike :mod:`model.ebm_fock` / :mod:`model.mlp_fock` this is not a
generic classical model standing in for the quantum one; it is the same physics with the
hardness ingredient removed.

**It is photonics-specific, and input-state- and k-specific**, exactly as you would expect: the
score needs ``U(x)`` (the classical Fock controls have no unitary at all), and ``U_{s,n}``'s shape
and content depend on the injected state ``s`` and the photon number ``k``.

**Pauli exclusion falls out.**  A repeated row or column makes a determinant vanish identically, so
``p_fermion`` is supported only on *collision-free* outcomes -- ``C(m, k)`` of the
``C(m+k-1, k)`` basis states (20 of 56 at ``m=6, k=3``), the rest being exactly 0.  Fermions cannot
bunch.  It is still a normalised distribution: by Cauchy-Binet
``sum_{|S|=k} |det(U_{R,S})|^2 = det(U_R U_R^dagger) = 1`` for a unitary ``U`` and collision-free
``s``, so no renormalisation is applied or needed.

The zeros are harmless to the observables: ``parity``/``majority`` just weight them by 0, and both
pointwise families vanish there too (``p log p -> 0``, ``p sin(1/(p+eps)) -> 0``).  The basis is
the same as the photonic teacher's, so ``parity`` on this teacher is literally the determinant
analogue of ``parity`` on that one.
"""

from __future__ import annotations

import math
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .mlp_fock import fock_keys
from .photonic_observables import (ObservableContext, is_known_observable, observable_hash_spec,
                                   observable_help, resolve_observable)

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


def sandwich_unitaries(m: int, seed: int):
    """``(W1, W2)``: the two Haar unitaries :func:`model.photonic_circuit.build_sandwich_circuit` draws.

    Reproduces that function's draw order exactly (``torch.manual_seed`` + ``pcvl.random_seed``,
    then two sequential ``random_unitary(m)`` calls), so a matched seed gives the *same* circuit the
    photonic teacher uses -- which is what makes the ``Perm`` vs ``det`` comparison controlled.
    """
    import perceval as pcvl

    torch.manual_seed(int(seed))
    pcvl.random_seed(int(seed))
    W1 = np.array(pcvl.Matrix.random_unitary(m), dtype=np.complex128)
    W2 = np.array(pcvl.Matrix.random_unitary(m), dtype=np.complex128)
    return W1, W2


def occupied_modes(occ) -> list[int]:
    """Mode indices of an occupation vector, each repeated by its count (length ``k``)."""
    out: list[int] = []
    for i, c in enumerate(occ):
        out += [i] * int(c)
    return out


def sandwich_unitary_at(W1, W2, X: torch.Tensor, n_features: int) -> torch.Tensor:
    """``(N, m, m)`` complex ``U(x) = (W2 D(x) W1)^T`` for a batch of inputs.

    ``D(x) = diag(e^{i x_0}, .., e^{i x_{n_features-1}}, 1, .., 1)`` is the phase encoding on the
    first ``n_features`` modes.  Transposing folds in as ``U = W1^T D W2^T`` (``D`` is diagonal);
    the transpose is perceval/merlin's convention -- verified by reproducing that layer's
    probabilities from ``|Perm|^2`` to ~1e-7 (see :func:`boson_probs_reference`).
    """
    m = W1.shape[0]
    A = torch.tensor(W1.T, dtype=torch.complex64)                 # (m, m)
    B = torch.tensor(W2.T, dtype=torch.complex64)                 # (m, m)
    ph = torch.ones(X.shape[0], m, dtype=torch.complex64)
    ph[:, :n_features] = torch.exp(1j * X[:, :n_features].to(torch.complex64))
    return torch.einsum("il,nl,lj->nij", A, ph, B)


def collision_free_index(keys):
    """``(cf_positions, cf_columns)`` for the collision-free outcomes of a Fock basis.

    ``cf_positions`` are their indices in ``keys``; ``cf_columns`` is a ``(n_cf, k)`` int array of
    the occupied mode indices.  Bunched outcomes are excluded because a repeated column makes the
    determinant identically 0 (Pauli exclusion), so they need no submatrix at all.
    """
    pos, cols = [], []
    for i, key in enumerate(keys):
        if max(int(c) for c in key) <= 1:
            pos.append(i)
            cols.append(occupied_modes(key))
    return np.asarray(pos, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def determinant_probs(U: torch.Tensor, s_modes, cf_positions, cf_columns,
                      n_fock: int) -> torch.Tensor:
    """``(N, n_fock)`` free-fermion distribution ``|det(U_{s,n})|^2``, zero on bunched outcomes.

    Already normalised (Cauchy-Binet), so nothing is divided out; the tiny residual from float32
    is left visible rather than hidden by a renormalisation.
    """
    rows = torch.as_tensor(np.asarray(s_modes), dtype=torch.long)
    cols = torch.as_tensor(cf_columns, dtype=torch.long)          # (n_cf, k)
    Ur = U[:, rows, :]                                            # (N, k, m)
    sub = Ur[:, :, cols].permute(0, 2, 1, 3)                      # (N, n_cf, k, k)
    w = torch.linalg.det(sub).abs() ** 2                          # (N, n_cf)
    out = torch.zeros(U.shape[0], n_fock, dtype=w.dtype)
    out[:, torch.as_tensor(cf_positions, dtype=torch.long)] = w
    return out


def boson_probs_reference(U: torch.Tensor, s_modes, keys) -> torch.Tensor:
    """``|Perm(U_{s,n})|^2 / prod_j n_j!`` -- the boson probabilities, for VERIFICATION only.

    Brute-force permanent over ``k!`` permutations, so this is for small ``k`` and for asserting
    that :func:`sandwich_unitary_at` reproduces the merlin layer.  The teacher never calls it.
    """
    from itertools import permutations

    rows = torch.as_tensor(np.asarray(s_modes), dtype=torch.long)
    Ur = U[:, rows, :]
    out = torch.zeros(U.shape[0], len(keys), dtype=torch.float32)
    for j, key in enumerate(keys):
        cols = torch.as_tensor(occupied_modes(key), dtype=torch.long)
        A = Ur[:, :, cols]                                        # (N, k, k)
        k = A.shape[1]
        tot = torch.zeros(A.shape[0], dtype=A.dtype)
        for p in permutations(range(k)):
            term = torch.ones(A.shape[0], dtype=A.dtype)
            for i in range(k):
                term = term * A[:, i, p[i]]
            tot = tot + term
        denom = math.prod(math.factorial(int(c)) for c in key)
        out[:, j] = tot.abs() ** 2 / denom
    return out


class FermionPhotonicTeacher(Teacher):
    """``X -> soft``: the sandwich circuit read out fermionically (``det``), scored by ``observable``.

    Matched to :class:`~model.photonic.PhotonicTeacher` at the same ``(m, k, n_features, seed,
    observable)`` -- same circuit, same input state, same Fock basis, same scoring code -- with
    ``|det|^2`` replacing ``|Perm|^2``.  Every registry observable applies.
    """

    name = "fermion_photonic"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0,
                 n_vertices: int | None = None, graph_seed: int | None = None,
                 angle_seed: int | None = None, graph_density: float | None = None):
        super().__init__(n_features)
        if not is_known_observable(observable):
            raise ValueError(f"unknown observable {observable!r}; expected one of: "
                             f"{observable_help()}")
        if k > m:
            raise ValueError(f"fermions need k <= m (Pauli exclusion); got m={m}, k={k}")
        self.m, self.k, self.observable, self.nsample = int(m), int(k), observable, int(nsample)
        self.seed = int(seed)
        self._noise_seed = self.seed + 13
        self._capture = False
        self._dist_probs: list = []

        from .photonic_circuit import default_input_state

        self._fock_keys = fock_keys(self.m, self.k)
        self.input_state = default_input_state(self.m, self.k)
        if max(self.input_state) > 1:
            raise ValueError("fermion_photonic needs a collision-free input state; "
                             f"default_input_state({m}, {k}) = {self.input_state} bunches")
        self.s_modes = occupied_modes(self.input_state)
        self.cf_positions, self.cf_columns = collision_free_index(self._fock_keys)
        self.W1, self.W2 = sandwich_unitaries(self.m, self.seed)

        self.obs = resolve_observable(observable, ObservableContext(
            m=self.m, k=self.k, keys=self._fock_keys, seed=self.seed, graph_seed=graph_seed,
            angle_seed=angle_seed, n_vertices=n_vertices, graph_density=graph_density,
            input_state=self.input_state, reference_probs=self.exact_probs_at_zero))
        # Peak intermediate is (batch, n_cf, k, k) complex64, so size the chunk on that.
        n_cf = max(len(self.cf_positions), 1)
        self.forward_batch = max(1, 16_777_216 // (n_cf * self.k * self.k))

    @property
    def n_collision_free(self) -> int:
        """``C(m, k)`` -- the support size; the rest of the Fock basis is identically 0."""
        return len(self.cf_positions)

    # --- the map ---------------------------------------------------------------------------- #

    @torch.no_grad()
    def unitary(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, m, m)`` sandwich unitary ``U(x)``, the same one the photonic teacher realises."""
        return sandwich_unitary_at(self.W1, self.W2, X, self.n_features)

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, n_fock)`` free-fermion distribution ``|det(U_{s,n})|^2``."""
        return determinant_probs(self.unitary(X), self.s_modes, self.cf_positions,
                                 self.cf_columns, len(self._fock_keys))

    @torch.no_grad()
    def boson_probs(self, X: torch.Tensor) -> torch.Tensor:
        """The ``|Perm|^2`` distribution of the same circuit -- for verification, ``k!`` cost."""
        return boson_probs_reference(self.unitary(X), self.s_modes, self._fock_keys)

    @torch.no_grad()
    def exact_probs_at_zero(self) -> torch.Tensor:
        """``q``: the ``(n_fock,)`` distribution at ``x = 0`` -- the reference ``xent`` needs."""
        return self.probs(torch.zeros(1, self.n_features))[0]

    @property
    def score_vec(self) -> torch.Tensor:
        """The observable's per-outcome score vector (the linear/diagonal families)."""
        return self.obs.score_vec

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bs = self.forward_batch
        if bs is None or bs <= 0 or X.shape[0] <= bs:
            return self._forward_chunk(X)
        chunks = [self._forward_chunk(X[i:i + bs]) for i in range(0, X.shape[0], bs)]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _forward_chunk(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.probs(X)
        if self.nsample > 0:
            probs = self._shot_sample(probs)
        if self._capture:
            self._dist_probs.append(probs.detach().cpu().numpy())
        return self.obs.score(probs).unsqueeze(-1)

    def _shot_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Replace each row by its ``nsample``-shot empirical distribution (finite-shot teacher)."""
        gen = torch.Generator().manual_seed(self._noise_seed)
        counts = torch.multinomial(probs.clamp(min=0), self.nsample, replacement=True,
                                   generator=gen)
        out = torch.zeros_like(probs)
        out.scatter_add_(1, counts, torch.ones_like(counts, dtype=probs.dtype))
        return out / self.nsample

    # --- distribution capture (same shape as PhotonicTeacher's) ----------------------------- #

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every forward's full distribution so it can be persisted and re-scored offline."""
        self._capture = bool(enable)
        self._dist_probs = []

    def captured_distributions(self) -> dict:
        """Recorded distributions, in the shape :func:`model.spoqc_magic.write_distributions` takes."""
        if not self._dist_probs:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        return {"keys": np.array(self._fock_keys, dtype=np.int16),
                "probs": np.vstack(self._dist_probs), "readout_modes": (),
                "m": self.m, "k": self.k, "observable": self.observable,
                "t_var": None, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        from .spoqc_magic import write_distributions
        return write_distributions(path, self.captured_distributions())

    # --- self-description -------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "FermionPhotonicTeacher":
        p = cfg.problem
        return cls(m=p.m, k=p.k, n_features=cfg.resolved_n_features,
                   observable=p.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample, n_vertices=p.n_vertices,
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed,
                   graph_density=p.graph_density)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        """Observable identity plus the readout, so ``det`` and ``Perm`` datasets never collide."""
        p = cfg.problem
        spec = {"observable": p.observable, "nsample": cfg.generation.nsample,
                "readout": "determinant_free_fermion"}
        spec.update(observable_hash_spec(p.observable, ObservableContext(
            m=p.m, k=p.k, seed=cfg.seeds.teacher_seed, graph_seed=p.graph_seed,
            angle_seed=p.angle_seed, n_vertices=p.n_vertices, graph_density=p.graph_density)))
        return spec
