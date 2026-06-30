"""Analysis stage: load saved datasets and inspect them.

Currently provides a loader/preview; complexity metrics will live here next.

    from analyzer import load_dataset, preview
    X, soft, meta = load_dataset("configs/example_photonic.yaml")
"""

from __future__ import annotations

from .compare import compare_from_embeddings, compare_kernels, kernel_grams, print_matrix
from .loader import load_dataset, preview
from .metrics import center_gram, kernel_alignment, kernel_kernel_matrix

__all__ = [
    "load_dataset",
    "preview",
    "center_gram",
    "kernel_alignment",
    "kernel_kernel_matrix",
    "kernel_grams",
    "compare_kernels",
    "compare_from_embeddings",
    "print_matrix",
]