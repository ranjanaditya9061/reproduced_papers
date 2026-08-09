"""Determinant readout of the SAME sandwich circuit that :mod:`v2.model.photonic` reads with ``Perm``.

The pipeline is one line long: the input state and an outcome ``n`` pick a ``k x k`` block out of the
circuit matrix, and the outcome's weight is ``|det|^2`` of that block, normalised over the basis.
Nothing else.

**Which index picks rows is fixed by the physics.**  On ``M = W2 D(x) W1`` the input's occupied modes
``S`` pick *columns* and the outcome's ``T`` pick *rows*, so the block is ``M[T, S]``.  Equivalently on
``U = M^T`` -- what :func:`~circuit.photonic_circuit.sandwich_unitary_at` returns and what the code
below indexes -- ``S`` picks rows and ``T`` picks columns, since ``U[S, T] = M[T, S]^T``.  Both
readings give the same number because ``det`` is transpose-invariant; the ``.T`` is the convention, and
dropping it would evaluate on the transposed unitary and silently change every probability
(``U[T, S]^T = U^T[S, T]``, a measured ``|det|^2`` of 0.0585 against 0.0011 on one draw).  What is free
is transposing the block once selected, which is why the gather below needs no permute.

**Why the swap is the interesting one.**  A permanent is ``#P``-hard (Valiant) while a determinant is
in ``P`` (an ``O(k^3)`` LU), and correspondingly boson sampling is hard to simulate while fermion
sampling is classically efficient (Valiant, Terhal-DiVincenzo).  Everything else -- the circuit, its
``2m^2 - 1`` parameters, the seed, the input state, the outcome basis, the observable scoring it -- is
held fixed, so a difference against the photonic model is attributable to the matrix function alone.

But note what that does and does not license.  The Fisher-spectrum comparison in
:mod:`v2.metrics.distribution` measures describability and estimability of the labelling function,
**not** computational hardness -- boson sampling is itself the counterexample, being indexed by
``O(m^2)`` numbers while each ``p(n)`` is ``#P``-hard.  A spectral difference here is a learnability
statement, not evidence about the ``Perm``/``det`` separation.

**The repeated-column problem, and the fix.**  ``T`` is the outcome's occupied modes with
multiplicity, so a bunched outcome repeats a column and the determinant vanishes identically -- Pauli
exclusion.  Strict free fermions therefore put *exactly zero* mass on the bunched sector, leaving
support at ``C(m, k)`` of the ``C(m+k-1, k)`` basis states (20 of 56 at ``m=6, k=3``), and a naive
comparison against the boson model then conflates *support size* with *matrix function*.

The ``p``-th copy of a mode's column is instead taken as::

    col_p(c) = (c / |c|)^p * |c|^(1 + s (1/p - 1))         s = k / m

-- the phase raised to the *integer* power ``p``, the modulus to a fractional one.  The two exponents
do different jobs and the split is not cosmetic:

* the **phase** power is what breaks the degeneracy.  Holding it at 1 and scaling only the modulus
  leaves both columns with identical entrywise phases, hence near-proportional (measured
  ``cond ~ 26`` against ``~5``), and the bunched mass *collapses* to 0.07;
* the phase exponent must be an **integer**, because ``arg`` is only defined mod ``2 pi``.  A
  fractional phase power (``c ** (1/p)``) is multivalued: as an entry of ``U`` crosses the principal
  branch cut, ``arg`` re-wraps by ``2 pi`` and the output phase jumps by ``2 pi / p``.  That flips
  *one entry* of one column, and ``|det|^2`` is **not** invariant under an entrywise phase (only
  under a whole-column one, ``A -> A diag(e^{i phi})``, which factors out of the determinant), so
  ``p(n)`` jumps -- measured ``0.318 -> 0.0038`` across a step of ``|dx| = 3.6e-9``, with the finite
  difference then diverging as ``1/h`` and silently poisoning any Fisher average that straddles it.
  Integer powers come around (``2 pi p == 0 mod 2 pi``) and are exactly continuous;
* the **modulus** power sets how much mass the bunched sector carries.  Since ``|U_ij| < 1``, a
  smaller exponent *enhances* bunching -- the same direction as boson sampling.  ``s = 0`` leaves
  moduli alone and undershoots; ``s = 1`` gives ``|c|^(1/p)`` and overshoots.

**Why ``s = k/m``.**  Boson bunched mass falls as modes get plentiful relative to photons, and the
filling fraction is the dimensionless quantity measuring that; fixed ``s`` is nearly flat in ``m`` and
so cannot track it.  Setting ``s = k/m`` matches the boson model's mean bunched mass to within 1.8%
on average (5.5% worst case) across ``m = 6..12`` at ``k = 3, 4`` and on the ``k = m/2`` diagonal up
to ``m=12, k=6``, with **no** free parameter:

====================  ======  ========  ========  ==========
outcome-mass error    s = 0   s = 0.5   s = 1     s = k/m
====================  ======  ========  ========  ==========
mean ``|err|``        18.0%   9.5%      31.4%     **1.8%**
worst case            -25.4%  +22.2%    +67.9%    **-5.5%**
====================  ======  ========  ========  ==========

``s = 1/2`` looks competitive only because the two configurations it was first tried at, ``(6,3)`` and
``(8,4)``, both have ``k/m = 1/2``.  Fitting ``s`` per configuration matches exactly but makes the
model partly defined by its comparator, and the objective is nearly flat near the optimum so the fit
is ill-conditioned anyway (at ``m=12, k=6``, ``s* = 0.477`` and ``s = 0.5`` differ by 0.4% in the
matched quantity).  ``s`` is exposed so the choice stays explicit, but the default is the rule.

**What this is and is not.**  Full ``C(m+k-1, k)`` support from a single ``O(k^3)`` determinant, no
flavour bookkeeping, differentiable, and boson-matched on both support and bunched mass -- so it is a
tightly controlled comparator for :func:`boson_probs_reference` on an identical circuit and seed.  But
no quantum state has this amplitude: ``det U[S, T]`` is a Slater determinant because ``T`` indexes
*distinct* output orbitals, and an elementwise power of an orbital is not an orbital.  So the bunched
sector does **not** carry the ``Perm``/``det`` hardness framing.

What does carry it is the collision-free sector, and it survives untouched: at a collision-free key
every mode has ``c = 1``, so only ``p = 1`` occurs and ``col_1(c) = (c/|c|) * |c|^(1 + s*0) = c``
*exactly*.  The construction therefore **contains** the free-fermion readout as its collision-free
restriction -- see :meth:`FermionModel.collision_free_probs`, which is ``|det U[S, T]|^2`` on the
shared ``C(m, k)`` support with no approximation.  Run the headline ``Perm``-vs-``det`` claim there.

No ``1 / prod_j n_j!`` factor is applied.  It is invisible on the collision-free sector (every
``n_j <= 1``), nothing physical here derives it for the bunched sector, and imposing it anyway breaks
the ``s = k/m`` rule badly (mean error 1.8% -> 32.9%, with no clean closed form replacing it).

Normalisation is genuine machinery, not float hygiene: Cauchy-Binet gives
``sum_{|S|=k} |det(U_{R,S})|^2 = 1`` only on the collision-free sector, so the ``x``-dependent
rescaling below enters every Jacobian taken through the full-support readout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from circuit.encoding import build_encoding
from circuit.photonic_circuit import default_input_state, sandwich_unitaries, sandwich_unitary_at
from .base import DistributionModel
from circuit.fock import fock_keys

if TYPE_CHECKING:
    from config import ExperimentConfig


def occupied_modes(occ) -> list[int]:
    """Mode indices of an occupation vector, each repeated by its count (length ``k``)."""
    out: list[int] = []
    for i, c in enumerate(occ):
        out += [i] * int(c)
    return out


def outcome_columns(keys, m: int, k: int) -> np.ndarray:
    """``(n_keys, k)`` -- the column recipe for every outcome, as one flat index.

    A column is identified by a *pair*: which output mode it is drawn from, and which copy of that
    mode it is (a mode occupied ``c`` times contributes copies ``1..c``, at powers ``1..c``).  Mode
    alone cannot say "second copy" and power alone cannot say which mode, so both are needed -- but
    they are packed into a single index ``(p - 1) * m + mode`` into the flattened ``(power, mode)``
    axis of the column stack, which keeps :func:`determinant_probs` to one gather.

    Depends only on ``keys``, so it is built once at construction.
    """
    rows = []
    for key in keys:
        row = [(p - 1) * m + j for j, c in enumerate(key) for p in range(1, int(c) + 1)]
        if len(row) != k:
            raise ValueError(f"key {key} has {len(row)} photons, expected {k}")
        rows.append(row)
    return np.asarray(rows, dtype=np.int64).reshape(len(keys), k)


def determinant_probs(U: torch.Tensor, rows_np, col_idx, s: float,
                      eps: float = 1e-12) -> torch.Tensor:
    """``(N, n_keys)`` -- one ``k x k`` ``|det|^2`` per outcome, normalised over the basis.

    ``rows_np`` are the input's occupied modes (``k`` of them); ``col_idx`` is the flat column recipe
    from :func:`outcome_columns`, applying ``col_p`` as described in the module docstring.  Integer
    phase powers are formed by repeated multiplication, never via ``exp(p log z)``, so no branch cut
    is crossed.

    Differentiable in ``U`` (hence in ``x``): ``torch.linalg.det`` has a gradient, which is what lets
    :mod:`v2.metrics` take exact input-Jacobians of this readout.
    """
    N, m = U.shape[0], U.shape[-1]
    n_keys, k = col_idx.shape

    rows = torch.as_tensor(rows_np, dtype=torch.long, device=U.device)
    V = U[:, rows, :]                                                 # (N, k, m)
    mag = V.abs().clamp(min=eps)
    phase = V / mag

    powers, acc = [], torch.ones_like(phase)
    for p in range(1, k + 1):
        acc = acc * phase                                             # integer power, cut-free
        powers.append(acc * mag ** (1.0 + s * (1.0 / p - 1.0)))
    cols = torch.stack(powers, dim=2).reshape(N, k, k * m)            # (N, row, (power, mode))

    idx = torch.as_tensor(col_idx.reshape(-1), dtype=torch.long, device=U.device)
    A = cols[:, :, idx].reshape(N, k, n_keys, k).permute(0, 2, 1, 3)  # (N, n_keys, k, k)

    p = torch.linalg.det(A).abs() ** 2
    return p / p.sum(dim=1, keepdim=True).clamp(min=1e-30)


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
    """``X -> probs``: the sandwich circuit read out with ``det`` in place of ``Perm``.

    Matched to :class:`~v2.model.photonic.PhotonicModel` at the same
    ``(m, k, n_features, seed, encoding)`` -- same circuit, same input state, same Fock basis, same
    scoring code.  Every registry observable applies.

    ``bunching_s`` is the ``s`` of the module docstring; ``None`` means the ``k/m`` rule.  Set it to
    ``0.0`` to leave moduli untouched, or to a fitted value if you want the bunched mass matched
    exactly at one configuration.
    """

    name = "fermion"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 bunching_s: float | None = None, encoding="phase"):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        self.bunching_s = float(k) / float(m) if bunching_s is None else float(bunching_s)
        self.encoding = build_encoding(encoding) if isinstance(encoding, str) else encoding

        self._keys = fock_keys(self.m, self.k)
        self._input_state = default_input_state(self.m, self.k)
        self._rows = np.asarray(occupied_modes(self._input_state), dtype=np.int64)
        self.col_idx = outcome_columns(self._keys, self.m, self.k)
        self._collision_free = np.asarray([max(key) <= 1 for key in self._keys])
        self.W1, self.W2 = sandwich_unitaries(self.m, self.seed)

        # Peak intermediate is the (chunk, n_keys, k, k) stack of blocks.
        self.forward_batch = max(1, 16_777_216 // max(len(self._keys) * self.k * self.k, 1))

    @property
    def n_reachable(self) -> int:
        """Outcomes with non-zero mass -- the whole basis, unlike the strict free-fermion readout."""
        return len(self._keys)

    @property
    def collision_free_mask(self) -> np.ndarray:
        """``(n_keys,)`` bool -- the ``C(m, k)`` sector where this readout *is* free fermions."""
        return self._collision_free

    def unitary(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, m, m)`` sandwich unitary ``U(x)`` -- the same one the photonic model realises."""
        return sandwich_unitary_at(self.W1, self.W2, X, self.n_features, self.encoding)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        return determinant_probs(self.unitary(X), self._rows, self.col_idx, self.bunching_s)

    def collision_free_probs(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, C(m,k))`` exact free-fermion distribution ``|det U[S, T]|^2``, renormalised.

        Not an approximation: ``col_1(c) = c`` identically, so restricting to collision-free keys
        recovers the strict Pauli-excluded readout.  This is the sector on which the
        ``Perm``-vs-``det`` claim is honest, and :meth:`boson_probs` restricted the same way is its
        comparator on shared support.
        """
        p = self._probs(X)[:, self._collision_free]
        return p / p.sum(dim=1, keepdim=True).clamp(min=1e-30)

    def boson_probs(self, X: torch.Tensor) -> torch.Tensor:
        """The ``|Perm|^2`` distribution of the *same* circuit -- ``k!`` cost, differentiable."""
        return boson_probs_reference(self.unitary(X), occupied_modes(self._input_state), self._keys)

    def shot_counts(self, X, *, shots: int, shot_seed: int = 0):
        """Not implemented: the fermion model is a probability-distribution model.
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
        # `readout` keeps det and Perm datasets from ever colliding even at a matched seed, and
        # `bunching_s` keeps two column conventions from colliding either.
        return {"model": self.name, "readout": "phase_power_determinant",
                "bunching_s": self.bunching_s, **self.encoding.spec()}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "FermionModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed,
                   bunching_s=getattr(cfg.model, "bunching_s", None), encoding=cfg.model.encoding)

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        flavours = getattr(cfg.model, "flavours", None)
        if flavours is not None and int(flavours) != 1:
            raise ValueError(
                f"model.flavours is no longer supported (got {flavours}). The flavoured-fermion "
                "readout degenerated to Perm(|U|^2) -- the classical distinguishable-particle "
                "distribution, with no determinant left -- at flavours = k, since each flavour block "
                "is then 1x1. Full support now comes from the phase-power columns instead; use "
                "model.bunching_s (default k/m) and see the module docstring."
            )
        s = getattr(cfg.model, "bunching_s", None)
        if s is not None and not (0.0 <= float(s) <= 3.0):
            raise ValueError(f"model.bunching_s should lie in [0, 3] (got {s})")
