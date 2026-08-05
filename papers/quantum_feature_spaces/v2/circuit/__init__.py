"""Circuit construction, shared by the frozen models and the parametric ones.

Three orthogonal axes, each its own registry, so a variant is a *parameter* rather than a class:

* :mod:`.encoding` -- how ``x`` enters (``phase``; deliberately open-ended).
* :mod:`.prep` -- how the photons are produced (``fock`` | ``spin`` | ``spin_magic``).  This is
  where the seven legacy ``spoqc*`` modules collapse into three preps and their knobs.
* :mod:`.photonic` -- the ``W1 -> encode(x) -> W2`` sandwich itself, available both as a merlin
  ``QuantumLayer`` and as explicit numpy/torch unitaries so the ``Perm`` and ``det`` readouts can
  share one circuit exactly.

:mod:`.spin` holds the numpy spin-state preparation and the per-row process pool.
"""

from __future__ import annotations

from .encoding import ENCODINGS, Encoding, PhaseEncoding, build_encoding
from .photonic_circuit import (build_quantum_layer, build_sandwich_circuit, default_input_state,
                       n_circuit_parameters, sandwich_unitaries, sandwich_unitary_at)
from .prep import PREPS, FockPrep, SpinMagicPrep, SpinPrep, StatePrep, build_prep

__all__ = [
    "ENCODINGS", "Encoding", "PhaseEncoding", "build_encoding",
    "build_quantum_layer", "build_sandwich_circuit", "default_input_state",
    "n_circuit_parameters", "sandwich_unitaries", "sandwich_unitary_at",
    "PREPS", "StatePrep", "FockPrep", "SpinPrep", "SpinMagicPrep", "build_prep",
]
