"""Magic-state spin-photon teacher (``perceval_spoqc``, MBQC T-teleportation).

Every other spoqc teacher (:mod:`model.spoqc`, ``spoqc_low``, ``spoqc_prime``)
prepares the spins and then **traces them out** at detection.  Because dual-rail
emission is Z-basis correlated, the traced photonic input is always a *classical
mixture* over Fock states (only ``|amplitude|^2`` survives) -- so those teachers
reduce to ``soft(x) = sum_z p_z f_z(x)``, a convex combination of ordinary
photonic teachers.  No phase, no coherence, no "magic".

This teacher breaks that on purpose, using only spin gates + emission + a *photon*
measurement + classical feedforward (the spin is **never measured directly**):

    spin |+>
      emit ph1 (dual-rail into modes 0,1)
      H [; T]                     <- cluster gap, optionally a non-Clifford T (magic)
      emit ph2 ; H [; T] ...      (Lindner-Rudolph emitter train -> linear cluster)
      H                           <- last gap doubles as the readout rotation
      emit ph_readout (modes m, m+1)
      measure the readout photon in Z  -> outcome mu     (a PHOTON, not the spin)
      post-select mu = 0 (== feedforward-correct the mu=1 branch)

The number of ``T`` gates is the ``t_var`` knob (``problem.t_var``): each of the
``k`` data emissions is followed by a cluster-gap ``H``, and ``t_var`` of those
gaps also get a ``T``.  ``t_var=0`` is a pure cluster (stabilizer -> no magic, so
the teacher collapses back to a classical Fock mixture like the other spoqc ones);
larger ``t_var`` injects more magic.

Note: the last gap's ``H`` doubles as the readout rotation, so a ``T`` in that gap
(``t_var = k``) sits as a pure phase right before the Z-basis readout + post-select
and is **washed out** -- exactly the "phase before a Z-measurement is invisible"
effect.  Hence ``t_var = k`` is identical to ``t_var = k-1``; the genuinely distinct
levels are ``0 .. k-1``.  ``[0, k]`` is still accepted (the saturation is a feature,
not an error).

The readout measurement collapses the spin-photon entanglement and *teleports*
the spin's non-stabilizer phase onto the data photons.  The resulting k-photon
interferometer input is a **pure, non-stabilizer (magic) state** -- verified in
``model/tests``: purity 1.0 and several Pauli expectations off ``{0, 1}``.  It is
then sent through the SAME ``W1(Haar) -> PS(x) -> W2(Haar)`` sandwich and scored
by the SAME observables as :class:`model.photonic.PhotonicTeacher`.

Relationship to the ideal target
--------------------------------
The clean MBQC target (a *logical* T on the cluster qubit) is

    |phi_mu> = (1/2 sqrt2) sum_{abc} (-1)^{ab+bc} [1 + (-1)^mu e^{i pi b/4} (-1)^c] |abc>

but that exact stabilizer frame is **not realizable by a bare emitter train**: a
single emitter cannot confine the T-phase to the readout ``|1>`` branch without an
extra measurement, so the hardware realizes a Clifford-frame-shifted magic state
(``T`` acting on photon 2 of a linear cluster).  Same magic content (one non-Clifford
gate), different -- and physically honest -- stabilizer frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .photonic import (
    _bunching_score,
    _first_mode_score,
    _majority_score,
    _parity_score,
)
from .spoqc_utils import _sandwich_concrete

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

OBSERVABLES = ("parity", "majority", "bunching", "n_first")


def _build_magic_processor(x, *, m, k, n_features, seed, t_var):
    """Emitter-train magic processor for one input ``x`` (spin never measured).

    ``k`` data photons are emitted dual-rail into modes ``0..2k-1``; the readout
    photon uses the two extra modes ``m, m+1``.  The ``W1.PS(x).W2`` sandwich acts
    on the ``m`` data modes, exactly as in the photonic teacher.

    Each data emission is followed by an ``H`` (a cluster gap); ``t_var`` of those
    gaps also get a non-Clifford ``T`` -- so ``t_var=0`` is a pure cluster
    (stabilizer, no magic) and ``t_var=k`` puts a ``T`` after every emission.
    """
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    if 2 * k > m:
        raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
    r0, r1 = m, m + 1                                   # readout photon modes
    p = HybridProcessor(num_sources=1, num_modes=m + 2, num_records=m + 2,
                        allow_carry_over=True)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    p.with_initial_source_state(np.outer(plus, plus.conj()))   # spin |+>

    for j in range(k):
        p.emit(0, into=(2 * j, 2 * j + 1))             # data photon j+1
        p.gate.h(0)                                     # cluster link (last one = readout rotation)
        if j < t_var:
            p.gate.t(0)                                 # magic: non-Clifford T in this gap
    p.emit(0, into=(r0, r1))                            # readout photon
    p.add(r0, Detector(), record=r0)                   # measure readout in Z ...
    p.add(r1, Detector(), record=r1)                   # ... (r0,r1)=(1,0) -> mu=0

    p.add(0, _sandwich_concrete(m, n_features, seed, x))   # interferometer on data modes
    for md in range(m):
        p.add(md, Detector(), record=md)
    return p, (r0, r1)


def _score_magic(p, readout_modes, *, m, k, observable) -> float:
    """Post-select the readout photon on ``mu=0`` and score the data modes.

    Post-selecting ``mu=0`` yields exactly the feedforward-corrected deterministic
    input, so ``soft`` is a well-defined function of ``x`` (the discarded ``mu=1``
    mass is the other, Clifford-equivalent branch).
    """
    r0, r1 = readout_modes
    num, den = 0.0, 0.0
    for key, pr in p.probabilities().items():
        if int(key[r0]) != 1 or int(key[r1]) != 0:     # keep readout photon in mode r0
            continue
        den += pr
        if observable == "parity":
            num += pr * _parity_score(key, tuple(range((m + 1) // 2)))
        elif observable == "majority":
            num += pr * _majority_score(key, m, k)
        elif observable == "n_first":
            num += pr * _first_mode_score(key)
        else:  # bunching
            num += pr * _bunching_score(key)
    print(num, den)
    return float(num / den) if den > 1e-12 else 0.0


class SpoqcMagicPhotonicTeacher(Teacher):
    """T-teleported magic-state photonic teacher; ``soft`` in ``[-1, 1]``.

    Unlike the other spoqc teachers, the interferometer input is a *coherent,
    non-stabilizer* state (see module docstring); there is no ``cx_pairs`` /
    ``angle_levels`` knob -- the magic comes from the fixed ``T`` gate.
    """

    name = "spoqc_magic_photonic"

    #: number of non-Clifford T gates (magic) if ``t_var`` is not given in the config.
    T_VAR = 1

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, t_var: int | None = None):
        super().__init__(n_features)
        if observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
        if m % 2:
            raise ValueError("spoqc_photonic uses dual-rail photons -> requires even m")
        if k < 2:
            raise ValueError("magic teleportation needs k >= 2 data photons (T sits before emit-2)")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        self.m, self.k, self.observable, self.seed = m, k, observable, int(seed)
        self.t_var = self.T_VAR if t_var is None else int(t_var)
        if not 0 <= self.t_var <= k:
            raise ValueError(f"t_var must be in [0, k]={0, k} (got {self.t_var})")

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xn = X.detach().cpu().numpy()
        vals = []
        for row in Xn:
            p, ro = _build_magic_processor(row, m=self.m, k=self.k, t_var=self.t_var,
                                           n_features=self.n_features, seed=self.seed)
            vals.append(_score_magic(p, ro, m=self.m, k=self.k, observable=self.observable))
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

    def render(self, path: str, x=None) -> str:
        import matplotlib.pyplot as plt

        xv = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        p, _ = _build_magic_processor(xv, m=self.m, k=self.k, t_var=self.t_var,
                                      n_features=self.n_features, seed=self.seed)
        p.pdisplay_hybrid()
        plt.savefig(path)
        plt.close("all")
        return path

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "SpoqcMagicPhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed,
                   t_var=cfg.problem.t_var)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        # t_var (# of magic T gates) is a spoqc_magic-only knob -> only this teacher's
        # hash sees it; it identifies the dataset alongside the emitter-train prep.
        t_var = cls.T_VAR if cfg.problem.t_var is None else int(cfg.problem.t_var)
        return {"observable": cfg.problem.observable,
                "prep": f"magic_T{t_var}_emitter_train_postselect_mu0"}
