"""Teacher base class + registry.

A *teacher* is the data-generating model: it maps inputs ``X`` to a continuous
``soft`` output — no labelling, filtering or balancing (those happen at load
time in :mod:`Generator.prepare`).

A teacher's only job is ``forward(X) -> soft``.  It knows nothing about labels,
margins or thresholds — turning ``soft`` into a binary label is a separate
*labelling* concern handled at load time (:mod:`Generator.prepare`), inferred
from the output representation itself.

Adding a model is intentionally trivial: subclass :class:`Teacher`, give it a
``name``, and it auto-registers.  A teacher describes only *itself*:

- ``name``            : registry key (also the ``generation.generator`` value).
- ``from_config(cfg)``: build the module from an experiment config.
- ``hash_spec(cfg)``  : a dict of *model-specific* identifying knobs ("variances").
  These are folded into the artifact hash and stored in ``meta.json`` — so a new
  model that needs an extra parameter just returns it here; nothing else changes.

Teachers are :class:`torch.nn.Module` s, so their parameters round-trip through
``state_dict`` (saved as ``teacher.pt``); reproducibility is also guaranteed by
construction (same config + ``teacher_seed`` rebuilds an identical teacher).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # avoid a runtime import cycle (Generator -> model -> Generator)
    from Generator.config import ExperimentConfig

#: name -> Teacher subclass, populated automatically on subclassing.
TEACHERS: dict[str, type["Teacher"]] = {}


class Teacher(nn.Module):
    """Maps ``X (N, n_features)`` -> continuous ``soft`` ``(N, 1)`` or ``(N, c)``."""

    name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            TEACHERS[cls.name] = cls

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = int(n_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # --- self-description (override in subclasses) ------------------------- #

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "Teacher":  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        """Model-specific knobs that should affect the artifact identity.

        Default: none.  Override to declare extra "variances" — they are hashed
        and saved automatically.
        """
        return {}


def build_teacher(cfg: "ExperimentConfig") -> Teacher:
    """Instantiate the teacher named by ``cfg.generation.generator``."""
    name = cfg.generation.generator
    if name not in TEACHERS:
        raise ValueError(f"unknown teacher {name!r}; registered: {sorted(TEACHERS)}")
    return TEACHERS[name].from_config(cfg)
