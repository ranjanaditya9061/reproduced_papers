"""Distribution models: ``X -> probs`` over a labelled outcome basis, and nothing else.

    from v2.model import build_model, sample_X

    model = build_model(cfg)                       # picks cfg.model.kind
    X     = sample_X(n, cfg.problem.n_features, cfg.seeds.sample_seed)
    probs = model.probs(X)                         # (N, n_outcomes)
    keys  = model.outcome_keys()                   # aligned to the columns

A model returns a **distribution**, not a score: the readout is applied afterwards by
:mod:`v2.observable` and cached separately, which is what makes the artifact readout-agnostic.

The roster, all comparable at one fixed ``n_features``:

===================  ===============================================  =====================  ==========================
kind                 ``probs``                                        outcome basis          parameters
===================  ===============================================  =====================  ==========================
``photonic``         merlin boson sampling, ``prep`` + ``encoding``   Fock ``C(m+k-1, k)``   ``2m^2 - 1``
``fermion``          flavoured ``|det|^2`` on the SAME circuit         Fock, occ. ``<= r``    ``2m^2 - 1`` (identical)
``quadratic_fock``   ``|f(x)^T W phi(n)|^2``                          Fock                   ``O(m^2)``, matchable
``mlp_fock``         dense ``2*n_fock`` head                          Fock                   exponential in ``m``
``qubit``            ``|IQP statevector|^2``                          computational ``2^n``  ``O(n k)``
``mlp``              softmax                                          2 class outcomes       ``O(n w k)``
``analytical``       ``p_1 = (1 + s)/2``                              2 class outcomes       none
===================  ===============================================  =====================  ==========================

``photonic`` alone covers eight legacy teacher classes, because the state preparation is a registry
choice (:mod:`v2.circuit.prep`): ``fock`` | ``spin`` | ``spin_magic`` replace ``model/photonic.py``
plus the seven ``spoqc*`` modules.  And the last two rows are what make the pipeline single-shaped:
on a 2-outcome basis ``parity`` returns the signed score exactly, so no model needs a scalar path.
"""

from __future__ import annotations

from .base import MODELS, DistributionModel, build_model
from circuit.fock import binary_keys, fock_keys, n_fock
from .features import TEACHER_FOURIER_VERSION, fourier_features
from .sampler import sample_X

# Concrete models, imported for their auto-registration side effect.  (photonic imports
# perceval/merlin lazily, on construction -- not here.)
from .classical import AnalyticalModel, MlpModel
from .fermion import FermionModel
from .mlp_fock import MlpFockModel
from .photonic import PhotonicModel
from .quadratic_fock import QuadraticFockModel
from .qubit import QubitFeatureMap, QubitModel

__all__ = [
    "DistributionModel", "MODELS", "build_model", "sample_X",
    "fock_keys", "binary_keys", "n_fock",
    "fourier_features", "TEACHER_FOURIER_VERSION",
    "PhotonicModel", "FermionModel", "QuadraticFockModel", "MlpFockModel",
    "QubitModel", "QubitFeatureMap", "MlpModel", "AnalyticalModel",
]
