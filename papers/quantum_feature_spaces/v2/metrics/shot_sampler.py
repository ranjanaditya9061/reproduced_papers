"""Fallback shot sampler: draw multinomial shots from an already-computed exact ``p``.

    from metrics.shot_sampler import sample_shots_from_probs
    counts = sample_shots_from_probs(probs_row, shots=10000, seed=0)   # (n_out,) int counts

**Only for models/preps without a native, more efficient sampler.**  ``pipeline.shots`` deliberately
excludes a ``"multinomial"`` method (its own docstring: sampling from a stored ``p`` "requires the
full distribution, so it silently reinstates the dependency this branch exists to remove" -- the
whole point of ``pipeline.shots``'s Clifford/MH methods is to draw shots *without* ever forming
``p``, which is what lets them reach sizes where the outcome basis cannot be enumerated at all).

That constraint does not apply here.  This module is for **metrics** work, where the exact
distribution has *already* been generated and is sitting in memory/on disk (``dist.probs``) -- we
are not trying to avoid forming ``p``, we already paid for it.  Use this only for configs whose
model does not have its own ``shot_counts`` (``model.supports_shots == False`` -- every prep except
photonic ``fock`` and fermion), where no more efficient sampler exists to call instead.  For
photonic (``fock``) and fermion configs, use the model's own ``shot_counts``
(Clifford-and-Clifford / Metropolis-Hastings respectively) via :mod:`pipeline.shots` -- this module
is deliberately **not** a replacement for either, only a fallback where neither is implemented.

**What this draws.**  Straightforward multinomial sampling over the *declared* outcome basis (the
full ``keys``, not just the observed support) -- appropriate here because, unlike the Clifford/MH
methods, we already have every outcome's exact probability, so there is nothing to be saved by
tracking only observed outcomes the way :mod:`pipeline.shots`'s sequence format does.
"""

from __future__ import annotations

import torch


def sample_shots_from_probs(probs: torch.Tensor, *, shots: int, seed: int = 0) -> torch.Tensor:
    """``(n_out,)`` int64 counts from ``shots`` i.i.d. draws of the categorical distribution
    ``probs`` -- one row's exact distribution in, one row's shot-count vector out.

    ``torch.multinomial`` (single categorical, ``shots`` draws with replacement) rather than
    ``np.random.multinomial`` -- keeps everything on the same tensor/dtype/device convention as the
    rest of this repo and avoids a numpy round-trip for a ``(n_out,)`` vector that may already be a
    GPU tensor.
    """
    gen = torch.Generator(device=probs.device).manual_seed(int(seed))
    draws = torch.multinomial(probs.double().clamp(min=0.0), int(shots), replacement=True,
                              generator=gen)
    counts = torch.zeros(probs.shape[-1], dtype=torch.int64, device=probs.device)
    counts.scatter_add_(0, draws, torch.ones_like(draws))
    return counts


def sample_shots_from_probs_batch(probs: torch.Tensor, *, shots: int, seed: int = 0) -> torch.Tensor:
    """``(N, n_out)`` int64 counts, one row per input in a batched ``(N, n_out)`` ``probs`` --
    every row draws from the **same** seed offset by its own row index (``seed + i``), so extending
    the pool with more rows never changes an earlier row's draw, mirroring
    :func:`pipeline.shots.offset_seed`'s own stream-independence discipline (row order should not
    silently perturb an already-drawn row).
    """
    N = probs.shape[0]
    out = torch.empty((N, probs.shape[1]), dtype=torch.int64, device=probs.device)
    for i in range(N):
        out[i] = sample_shots_from_probs(probs[i], shots=shots, seed=int(seed) + i)
    return out


def empirical_probs_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """``counts -> p_hat`` -- the plug-in empirical distribution, ``counts / counts.sum()``."""
    total = counts.sum().clamp(min=1)
    return counts.double() / total
