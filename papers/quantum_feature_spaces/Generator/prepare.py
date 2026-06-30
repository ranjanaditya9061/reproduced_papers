"""Labelling + a *diagnostic* margin-filter / class-balance over a saved pool.

This is **not** part of the load/train path: :func:`Generator.load_split` returns
the full dataset (nothing discarded).  Margin filtering and balancing answer a
separate analysis question -- *"how much of this data is cleanly separable for
binary classification?"* -- so they live here and are invoked by the analyzer,
never silently between generation and learning.

:func:`prepare_indices` returns the surviving row indices (so the same selection
can be applied to a stored embedding matrix); :func:`prepare` is the convenience
that also slices ``X``/``soft``.

Labelling is inferred from the *output representation*, not from the model:

- ``soft`` shape ``(N, 1)``  -> a signed score: ``y = (soft >= 0)``, margin ``|soft|``.
- ``soft`` shape ``(N, c)``  -> class scores/probs: ``y = argmax``, margin = gap
  between the top two entries (``|p1 - p0|`` in the binary ``c = 2`` case).

Everything here is a pure function of its arguments, so every consumer of an
artifact sees an identical labelled/filtered/balanced view.
"""

from __future__ import annotations

import torch


def derive_labels(soft: torch.Tensor) -> torch.Tensor:
    """Labels inferred from the soft output shape (binary score or ``c``-class)."""
    if soft.shape[-1] == 1:
        return (soft[:, 0] >= 0).long()
    return soft.argmax(dim=-1).long()


def derive_confidence(soft: torch.Tensor) -> torch.Tensor:
    """Per-sample margin in ``[0, 1]`` inferred from the soft output shape."""
    if soft.shape[-1] == 1:
        return soft[:, 0].abs()
    top2 = soft.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).abs()


def prepare_indices(
    soft: torch.Tensor,
    *,
    min_margin: float = 0.0,
    balanced: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Indices (into ``soft``'s rows) surviving the margin filter + optional balance.

    Deterministic in ``(soft, min_margin, balanced, seed)``.  Use these to select
    the same rows from a stored feature/embedding matrix.
    """
    idx = torch.arange(soft.shape[0])
    if min_margin > 0.0:
        idx = idx[derive_confidence(soft) >= min_margin]
    if balanced:
        if idx.numel() == 0:
            raise ValueError(f"min_margin={min_margin} filtered out every sample; lower it.")
        y = derive_labels(soft)[idx]
        classes = torch.unique(y)
        if classes.numel() < 2:
            raise ValueError(
                f"min_margin={min_margin} leaves only one class; cannot balance — "
                f"lower min_margin (the teacher's soft output is concentrated near 0)."
            )
        n_each = int(min((y == c).sum() for c in classes))
        idx = torch.cat([idx[(y == c).nonzero(as_tuple=True)[0][:n_each]] for c in classes])
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))
    return idx[perm]


def prepare(
    X: torch.Tensor,
    soft: torch.Tensor,
    *,
    min_margin: float,
    balanced: bool,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnostic view ``(X', y', soft')`` for the given filter/balance settings.

    Deterministic in ``(X, soft, min_margin, balanced, split_seed)``.  Not used by
    the load/train path -- see the module docstring.
    """
    idx = prepare_indices(soft, min_margin=min_margin, balanced=balanced, seed=split_seed)
    return X[idx], derive_labels(soft)[idx], soft[idx]
