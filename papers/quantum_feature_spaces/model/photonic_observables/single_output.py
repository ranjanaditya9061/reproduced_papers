"""``single_output``: does the outcome reproduce the input state (``+1``) or its mirror (``-1``)?

Scores ``(P(in) - P(reversed in)) / max_n p(n)``: the signed contrast between the two marked
outcomes, divided by the peak probability so the value is comparable across inputs (a nearly
flat distribution would otherwise give a vanishing contrast for every ``x``).

Unlike the other plain observables this needs the teacher's ``input_state``, which is *not*
persisted in a saved distribution -- so it is the one observable that cannot be re-scored
offline.  It is therefore not registered as a reusable base scorer either.
"""

from __future__ import annotations

import torch

from .base import Observable, ObservableFamily, ObservableContext, _as_vec, register


def single_output_score(key, input_state) -> int:
    """``+1`` on the input state, ``-1`` on its reverse, ``0`` on every other outcome."""
    kl = [int(key[i]) for i in range(len(input_state))]
    if kl == list(input_state):
        return 1
    if kl == list(reversed(input_state)):
        return -1
    return 0


class SingleOutputObservable(Observable):
    """``(probs @ score_vec) / max_n p(n)`` -- the marked-outcome contrast, peak-normalised."""

    def __init__(self, score_vec):
        super().__init__()
        self.register_buffer("score_vec", _as_vec(score_vec))

    def score(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs @ self.score_vec) / probs.max(dim=1).values.clamp(min=1e-10)


class SingleOutputFamily(ObservableFamily):
    describe = "single_output"

    def matches(self, name: str) -> bool:
        return name == "single_output"

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        if ctx.input_state is None:
            raise ValueError("observable 'single_output' needs the teacher's input_state, which "
                             "is not persisted in a saved distribution (cannot be re-scored "
                             "offline)")
        return SingleOutputObservable(
            [single_output_score(key, ctx.input_state) for key in ctx.keys])


register(SingleOutputFamily())