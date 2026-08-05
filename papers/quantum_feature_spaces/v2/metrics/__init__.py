"""Two separate analyses, deliberately kept apart.

===========  =====================================  =====================================
             **A. distribution** (:mod:`.distribution`)  **B. observable** (:mod:`.observable`)
===========  =====================================  =====================================
input        ``p_x`` alone -- **no observable**      ``p_x`` **and** a readout ``O`` (or a set)
object       ``F = J^T J``, the ``n_f x n_f`` input FIM  ``eta_O``, and the ``R x n_f`` set Jacobian
asks         how much information about ``x`` does   how much of that information does this
             the distribution carry, and how spread  measurement actually capture
ceiling      ``F`` itself                            ``F_O <= F``, equality iff ``O`` is the score
compares     models, and across ``(m, k)``           the ~17 registry observable families
===========  =====================================  =====================================

They answer different questions, take different inputs, and **must not be reported as one number**.

Both rest on the same two exact objects, and neither samples: the score matrix
``S = d log p/dx`` (``(n_out, n_f)``) and the registry's influence functions ``psi``.  Since we
simulate the full distribution, the sampled score estimator -- whose variance
``(E[s^4] - I^2)/N`` is dominated by low-probability outcomes, and our distributions are
Porter-Thomas with ``log10 p`` down to ``-6.95`` at ``m=6, k=3`` -- is never needed.

The differentiation target is the **input** ``x``, not the circuit weights, for three reasons
spelled out in :mod:`.distribution`.  Consequently nothing here needs a trainable
re-implementation of any circuit: the metrics run directly on the frozen :mod:`v2.model`
distribution models.

**Derivatives are built outside the sampler.**  :mod:`.fd` wraps *any* ``x -> probs`` callable --
torch model, perceval backend, spin-photon simulator, or a shot draw -- in two forward evaluations
per direction, so no model has to be autograd-differentiable for analysis A or B to run.  Autograd
is preferred where it exists (exact, one backward pass instead of ``2 n_f`` forwards), and FD on the
exact distribution matches it to ``1.6e-4`` on the Fisher spectrum.  FD on a *shot* sampler is
noise-dominated (32% at 50k shots) -- shots are for labels, not derivatives.
"""

from __future__ import annotations

from .fd import FD_EPS, fd_jacobian, probs_and_fd_jacobian, sampler
from .distribution import (SUPPORT_TOL, conditional_sqrt_jacobian, description_cost,
                           effective_dimension, finite_difference_jacobian, fisher_spectrum,
                           input_fisher, phase_eigenvalue, probs_and_jacobian, project_physical,
                           r_eff, r_eff_curve, shared_support, spectrum_from_jacobian,
                           sqrt_jacobian, tau_N)

__all__ = [
    "FD_EPS", "fd_jacobian", "probs_and_fd_jacobian", "sampler",
    "SUPPORT_TOL", "conditional_sqrt_jacobian", "description_cost", "effective_dimension",
    "finite_difference_jacobian", "fisher_spectrum", "input_fisher", "phase_eigenvalue",
    "probs_and_jacobian", "project_physical", "r_eff", "r_eff_curve", "shared_support",
    "spectrum_from_jacobian", "sqrt_jacobian", "tau_N",
]
