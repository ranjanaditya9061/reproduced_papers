"""spoqc_magic variant: each cluster gap injects a Haar-random single-qubit unitary.

Same emitter-train + readout + post-select-``mu=0`` gadget as
:class:`model.spoqc_magic.SpoqcMagicPhotonicTeacher`, but each injected gap gate
(gaps ``j < t_var``) is a random ``U in SU(2)`` applied as ``Rz(phi) Ry(theta) Rz(lam)``.
The per-gap Euler angles are drawn **once** from ``seed`` under the Haar measure and
kept fixed for the teacher (they are the "teacher weights"), so ``soft`` stays a
deterministic function of ``x``.  A generic ``SU(2)`` element is non-Clifford with
probability 1, so this is a random magic injection -- distinct datasets come from
distinct ``teacher_seed`` (already part of the artifact hash).

Heavy machinery (build/score/capture/save, per-row parallelism) is inherited; only the
gap gate and the hash prep tag change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .spoqc_magic import SpoqcMagicPhotonicTeacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


class SpoqcMagicRandXPhotonicTeacher(SpoqcMagicPhotonicTeacher):
    """spoqc_magic with a Haar-random ``SU(2)`` as the cluster-gap gate (fixed per seed)."""

    name = "spoqc_magic_rand_x_photonic"
    # Gate_kind = "u3_x"
    # gate_kinds = ["u3_x","u3_x_m","u3_x_s"]

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, t_var: int | None = None, gate_kind: str | None = None,
                 n_jobs: int = 1):
        print(gate_kind,"G")
        super().__init__(m, k, n_features, observable=observable, seed=seed,
                         t_var=t_var, gate_kind=gate_kind, n_jobs=n_jobs)
        # Haar SU(2) via ZYZ Euler angles: phi, lam ~ U(0, 2pi); theta with sin(theta)
        # weight (cos theta uniform in [-1, 1]) -> uniform on the Bloch sphere.
        rng = np.random.default_rng(self.seed)
        phi = rng.uniform(0.0, 2 * np.pi, size=k)
        lam = rng.uniform(0.0, 2 * np.pi, size=k)
        theta = rng.uniform(-np.pi/2, np.pi/2, size=k)
        self.gate_params = [(float(theta[j]), float(phi[j]), float(lam[j])) for j in range(k)]
