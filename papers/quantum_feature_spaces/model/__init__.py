"""Teacher models for the generation pipeline.

A teacher maps ``X -> soft`` (continuous) and nothing else; labelling happens
downstream.  Importing this package registers every teacher in :data:`TEACHERS`
(keyed by ``name``), so adding a model is just adding a subclass module here.

    from model import build_teacher, sample_X, TEACHERS

    teacher = build_teacher(cfg)          # picks cfg.generation.generator
    X    = sample_X(n, cfg.resolved_n_features, cfg.seeds.sample_seed)
    soft = teacher(X)
"""

from __future__ import annotations

from .base import TEACHERS, Teacher, build_teacher
from .sampler import sample_X

# Import concrete teachers for their auto-registration side effect.
# (photonic imports perceval/merlin lazily, on construction — not here.)
from .analytical import AnalyticalTeacher
from .mlp import MLPTeacher
from .qubit import QubitTeacher
from .photonic import PhotonicTeacher
from .spoqc import SpoqcPhotonicTeacher
from .spoqc_low import SpoqcLowPhotonicTeacher
from .spoqc_prime import SpoqcPrimePhotonicTeacher
from .spoqc_magic import SpoqcMagicPhotonicTeacher
from .spoqc_magic_prime import SpoqcMagicPrimePhotonicTeacher
from .spoqc_magic_rand import SpoqcMagicRandPhotonicTeacher

__all__ = [
    "Teacher",
    "TEACHERS",
    "build_teacher",
    "sample_X",
    "AnalyticalTeacher",
    "MLPTeacher",
    "QubitTeacher",
    "PhotonicTeacher",
    "SpoqcPhotonicTeacher",
    "SpoqcLowPhotonicTeacher",
    "SpoqcPrimePhotonicTeacher",
    "SpoqcMagicPhotonicTeacher",
    "SpoqcMagicPrimePhotonicTeacher",
    "SpoqcMagicRandPhotonicTeacher",
]