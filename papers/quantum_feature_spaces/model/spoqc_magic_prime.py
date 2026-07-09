"""spoqc_magic variant: the cluster-gap magic gate is ``Rz(prime_j)`` instead of ``T``.

Same emitter-train + readout + post-select-``mu=0`` gadget as
:class:`model.spoqc_magic.SpoqcMagicPhotonicTeacher`, but each injected gap gate
(gaps ``j < t_var``) is a ``Rz`` of the ``(j+1)``-th prime **in radians** (2, 3, 5,
7, ...).  A prime is never a multiple of ``pi/2``, so each is a non-Clifford phase --
like ``T`` it makes the teleported interferometer input non-stabilizer, but with a
distinct, deterministic magic angle per gap rather than the single fixed ``pi/4``.

All the heavy machinery (build/score/capture/save, per-row parallelism) is inherited;
only the gap gate and the hash prep tag change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .spoqc_magic import SpoqcMagicPhotonicTeacher
from .spoqc_prime import _first_primes

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


class SpoqcMagicPrimePhotonicTeacher(SpoqcMagicPhotonicTeacher):
    """spoqc_magic with ``Rz(prime_j)`` (increasing primes) as the cluster-gap gate."""

    name = "spoqc_magic_prime_photonic"
    Gate_kind = "rz"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, t_var: int | None = None,gate_kind: str | None = None,
                 n_jobs: int = 1):
        super().__init__(m, k, n_features, observable=observable, seed=seed,
                         t_var=t_var, gate_kind=gate_kind, n_jobs=n_jobs)
        self.gate_params = [float(p) for p in _first_primes(k)]   # Rz(prime_j) per gap

    @classmethod
    def _prep_tag(cls, cfg: "ExperimentConfig", t_var: int, gate_kind:str) -> str:
        """Prep string folded into the dataset hash; overridden by each gap-gate variant."""
        return f"magic_rzprime_T{t_var}_emitter_train_postselect_mu0"