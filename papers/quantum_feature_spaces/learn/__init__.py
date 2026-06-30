"""Learn stage: train a classifier on each embedding and score it on the split.

This is the downstream consumer the decoupling enables: the embedding stage stores
features over the **full** pool, and :func:`Generator.load_split` provides the one
shared train/test partition as indices -- so a learner just slices the stored
feature matrix and fits.  Nothing is re-embedded, and no margin filtering happens
on the train path (that stays a separate analyzer diagnostic).

    from learn import run_svm
    rows, meta = run_svm("configs/embed_example.yaml")

CLI::

    python -m learn --config configs/embed_example.yaml [--C 1.0] [--n-train 2000]
"""

from __future__ import annotations

from .svm import run_svm

# run_grid lives in learn.grid (imported lazily to avoid pulling matplotlib/sklearn
# at package import, and to avoid the runpy double-import warning for `python -m learn.grid`).
__all__ = ["run_svm", "run_grid"]


def __getattr__(name):
    if name == "run_grid":
        from .grid import run_grid
        return run_grid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
