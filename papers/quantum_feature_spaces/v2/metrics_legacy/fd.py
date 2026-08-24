"""Finite differences as a wrapper around *any* sampler -- exact or shot-based.

    fn = sampler(model)                       # x -> probs, exact
    fn = sampler(model, shots=50_000)         # x -> probs, empirical
    dp = fd_jacobian(fn, x)                   # (n_out, n_f), same call either way

**The derivative is built outside the sampler.**  Nothing in :mod:`v2.model` has to be
autograd-differentiable for the metrics to work: the Jacobian is two forward evaluations per input
direction, so a torch model, a perceval backend, a spin-photon simulator and a shot draw are all
differentiated by the same code.  That decouples the metrics from the simulator entirely, which is
what lets the exact branch move between backends without touching analysis A or B.

Autograd is still preferred where it exists (:func:`~v2.metrics.distribution.probs_and_jacobian`),
because it is exact and one backward pass rather than ``2 n_f`` forwards.  This is the fallback and
the general path.

**Accuracy, measured -- read before differentiating a shot sampler.**

=================================  ====================================
target                             max rel error vs the exact Jacobian
=================================  ====================================
exact distribution, ``eps=1e-2``   ``1.4e-4``
50k shots, common random numbers   ``0.32``
50k shots, independent draws       ``1.06``
=================================  ====================================

So FD on the *exact* distribution is fine -- it holds the Fisher spectrum to ``1.6e-4`` and matches
autograd to four significant figures at ``m=8, k=4``.  FD on a *shot* sampler is not: shot noise on
``p`` is ``~sqrt(p/S)`` and dividing by ``2 eps`` amplifies it ~50x at ``eps = 1e-2``.  Shots are for
**labels**, not for derivatives.  The interface is uniform; the accuracy is not.

:func:`sampler` reuses one ``shot_seed`` across both FD legs, so ``x`` and ``x + delta`` are drawn
from the *same* substreams -- common random numbers, which :mod:`pipeline.shots`'s seeding gives for
free (:func:`~pipeline.shots.offset_seed` keys the stream on ``(shot_seed, offset)``, never on
``x``).  Measured 3.4x better than independent draws, and free.  It does not rescue the derivative,
only its variance.
"""

from __future__ import annotations

from typing import Callable

import torch

#: Default step.  Large deliberately: FD error is ``O(eps^2)`` truncation plus
#: ``O(float_eps/eps)`` round-off, and these models are float32-rooted, so shrinking ``eps`` makes it
#: *worse* -- measured ``1.9e-4`` at ``1e-3`` against ``2.7e-2`` at ``1e-5``.
FD_EPS = 1e-2


def sampler(model, *, shots: int = 0, shot_seed: int = 0) -> Callable:
    """``x (n_f,) -> probs (n_out,)``: the callable :func:`fd_jacobian` differentiates.

    ``shots = 0`` gives the exact distribution.  ``shots > 0`` rounds up to whole blocks and returns
    the empirical distribution, drawn from substreams that do not depend on ``x`` -- so successive
    calls at nearby ``x`` share randomness.

    Raises if the model does not support shots, naming which ones do; see
    :meth:`v2.model.base.DistributionModel.shot_counts`.
    """
    if shots <= 0:
        return lambda x: model.probs(x.unsqueeze(0), grad=False)[0]

    # A shot draw reports only the outcomes it observed, and the observed set moves with x -- so for
    # FD to be well defined the two legs must share a column basis.  Align onto the model's declared
    # basis, which exists only where it is enumerable.  That is not a limitation in practice: FD on a
    # shot sampler is noise-dominated anyway (32% vs 1.4e-4 exact), so this path is a diagnostic.
    index = {tuple(int(c) for c in key): i for i, key in enumerate(model.outcome_keys())}

    def fn(x: torch.Tensor) -> torch.Tensor:
        row = model.shot_counts(x.unsqueeze(0), shots=shots, shot_seed=shot_seed)[0]
        dense = torch.zeros(len(index), dtype=torch.float64)
        for key in row:
            dense[index[key]] += 1.0
        return (dense / dense.sum().clamp(min=1)).to(torch.float32)

    return fn


def fd_jacobian(fn: Callable, x: torch.Tensor, *, eps: float = FD_EPS) -> torch.Tensor:
    """Central-difference Jacobian ``(n_out, n_f)`` of any ``x -> vector`` map.

    ``2 n_f`` evaluations, no gradient support required of ``fn``.
    """
    cols = []
    for i in range(int(x.shape[0])):
        h = torch.zeros_like(x)
        h[i] = float(eps)
        cols.append((fn(x + h) - fn(x - h)) / (2.0 * float(eps)))
    return torch.stack(cols, dim=1).detach()


def probs_and_fd_jacobian(model, x: torch.Tensor, *, shots: int = 0, shot_seed: int = 0,
                          eps: float = FD_EPS):
    """``(p, dp)`` for one input via finite differences -- the signature
    :func:`~v2.metrics.distribution.probs_and_jacobian` has, so the two are interchangeable in
    analysis A and B."""
    fn = sampler(model, shots=shots, shot_seed=shot_seed)
    return fn(x).detach(), fd_jacobian(fn, x, eps=eps)
