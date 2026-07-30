"""``max_prob``: the peak of the output distribution, ``max_n p(n)``.

A concentration probe rather than a diagonal expectation -- there is no per-Fock-state score
vector, so this is the one plain family that subclasses :class:`~.base.Observable` directly.
Clamped away from 0 because it is also used as a denominator elsewhere (see
:mod:`.single_output`) and to keep ``log``-scale plots finite.
"""

from __future__ import annotations

import torch

from .base import Observable, ObservableContext, ObservableFamily, register


class MaxProbObservable(Observable):
    """``max_n p(n)`` per row, clamped to ``>= 1e-10``."""

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return probs.max(dim=1).values.clamp(min=1e-10)


class MaxProbFamily(ObservableFamily):
    describe = "max_prob"

    def matches(self, name: str) -> bool:
        return name == "max_prob"

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        return MaxProbObservable()


register(MaxProbFamily())