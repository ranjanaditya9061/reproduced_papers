"""One distribution format, one reader.

    dist = load_dist(path)          # X, probs, keys, probs_at_zero
    dist.probs                      # (N, n_out) -- THE artifact

Replaces the legacy pair of mutually incompatible readers
(``model/photonic.score_from_distribution`` and ``model/spoqc_magic.score_from_distribution``,
which read the same ``.npz`` expecting different fields).  There is exactly one writer and one
reader here, and neither knows what an observable is.

``X`` is **not stored**.  ``sample_X`` is a single contiguous draw from a fixed seed, so it is
prefix-stable and ``X`` is reconstructed exactly from ``meta.json``'s ``n_features``,
``sample_seed`` and ``size``.  Storing it would add a second source of truth for the input pool
that could drift from the seed the hash is taken over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model.sampler import sample_X
from .artifact import DIST_FILENAME, load_meta


@dataclass
class Distribution:
    """A loaded artifact: the input pool and its distribution over a labelled basis."""

    X: torch.Tensor                 # (N, n_features)
    probs: torch.Tensor             # (N, n_out)
    keys: tuple                     # n_out occupation tuples, aligned to the columns
    probs_at_zero: torch.Tensor     # (n_out,)  the q that `xent` scores against
    meta: dict

    def __len__(self) -> int:
        return int(self.probs.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.probs.shape[1])

    def head(self, n: int) -> "Distribution":
        """The first ``n`` rows -- the metrics evaluate on a subsample of the pool."""
        return Distribution(self.X[:n], self.probs[:n], self.keys, self.probs_at_zero, self.meta)


def dist_bytes(size: int, n_out: int) -> int:
    """Bytes the ``(size, n_out)`` float32 ``probs`` matrix will occupy."""
    return int(size) * int(n_out) * 4


def check_size(size: int, n_out: int, max_bytes: int, *, m: int, k: int) -> None:
    """Refuse to generate an artifact whose ``probs`` exceeds the budget.

    Errors with the computed size and ``(m, k, N)`` rather than silently downcasting: at
    ``N = 10_000``, ``n_out = 2002`` is 80 MB, ``12376`` is 495 MB and ``77520`` is 3.1 GB.
    """
    need = dist_bytes(size, n_out)
    if need > int(max_bytes):
        raise ValueError(
            f"probs would be {need / 1024 ** 3:.2f} GiB at m={m}, k={k}, N={size} "
            f"({n_out} outcomes x float32), over generation.max_dist_bytes = "
            f"{int(max_bytes) / 1024 ** 3:.2f} GiB. Reduce generation.size or (m, k), or raise "
            "the budget deliberately -- it is not downcast automatically."
        )


def write_dist(path: str | Path, *, probs: torch.Tensor, keys, probs_at_zero: torch.Tensor) -> Path:
    """Write ``dist.npz``.  Note there is no ``observable`` field, by construction."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    out = path / DIST_FILENAME
    np.savez_compressed(
        out,
        keys=np.asarray([[int(c) for c in key] for key in keys], dtype=np.int16),
        probs=probs.detach().cpu().numpy().astype(np.float32),
        probs_at_zero=probs_at_zero.detach().cpu().numpy().astype(np.float32),
    )
    return out


def load_dist(path: str | Path, *, size: int | None = None) -> Distribution:
    """Load an artifact, reconstructing ``X`` from the seed recorded in ``meta.json``.

    ``size`` truncates to the first rows of the pool (the prefix is stable, so this is a genuine
    subsample of the same dataset rather than a different one).
    """
    path = Path(path)
    meta = load_meta(path)
    with np.load(path / DIST_FILENAME) as z:
        keys = tuple(tuple(int(c) for c in row) for row in z["keys"])
        probs = torch.from_numpy(z["probs"].astype(np.float32))
        q = torch.from_numpy(z["probs_at_zero"].astype(np.float32))

    n = probs.shape[0] if size is None else min(int(size), probs.shape[0])
    X = sample_X(int(meta["size"]), int(meta["n_features"]), int(meta["sample_seed"]))
    return Distribution(X=X[:n], probs=probs[:n], keys=keys, probs_at_zero=q, meta=meta)
