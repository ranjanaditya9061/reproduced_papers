"""The shots branch: one ``{outcome_key: count}`` dict per row, drawn in additive blocks.

    shots_v2/<circuit_hash>/<shot_hash>/counts.npz     ragged per-row counts
                                        meta.json      n_blocks  <- the EXTENDABLE number

**The representation is a list of dicts.**  ``S`` shots touch at most ``S`` outcomes while the basis
is ``C(m+k-1, k)`` (``20_030_010`` at ``m=20, k=10``), so a row is a mapping over what it actually
observed and nothing is ever sized by the full basis.  The ``(N, n_observed)`` matrix only appears at
the scoring boundary, in :func:`to_sparse`, where an observable needs aligned columns.

**Counts, not probs.**  Integers are what make blocks additive: extending a 20k draw to 30k is
``Counter(row) + Counter(new)``, exact and with no round-off.  :func:`to_sparse` normalises, so probs
exist everywhere they are wanted without being the stored form.

**A sibling of the exact distribution, not a readout of it.**  perceval implements the two with
disjoint backends -- ``CliffordClifford2017`` samples and exposes no ``all_prob``, ``SLOS``/``Naive``
expose ``all_prob`` and never sample -- and at large ``(m, k)`` the exact distribution cannot be
computed at all.  So shots cannot be *defined* as a downstream stage of the full distribution the way
an observable can.  What the branches share is the circuit and the input pool, which is exactly what
:func:`v2.pipeline.artifact.circuit_hash` covers.

**What is hashed.**  ``BLOCK``, ``shot_seed`` and ``method`` go into :func:`shot_hash`, because each
changes the draws.  ``n_blocks`` does **not**: it is the quantity you extend, so it lives in
``meta.json``, exactly as ``size`` does for the row pool.  Raising ``generation.shots`` therefore
adds blocks to the same store instead of landing in a new directory and redrawing from shot zero.

**One seed per block.**  ``exqalibur`` is seeded once per block, so the draws within a block depend
on how the rows are traversed; a block is reproducible as a whole, an individual row is not.  That is
a deliberate trade for ``backend.samples(BLOCK)`` in bulk over per-sample draws.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

#: Shots per block.  **Part of the shot identity**: changing it repartitions the draws, so it is
#: hashed rather than documented.  10k is small enough for fine budget steps and large enough that
#: the per-block overhead is negligible.
BLOCK = 10_000

COUNTS_FILENAME = "counts.npz"
SHOT_META_FILENAME = "meta.json"

#: How the shots were produced.  ``"clifford"`` samples the interferometer directly -- no
#: distribution is ever formed, which is what makes this branch a sibling of the exact one.  A
#: ``"multinomial"`` method (draw from a stored exact ``p``) is deliberately **absent**: it would be
#: faster but requires the full distribution, so it silently reinstates the dependency this branch
#: exists to remove and implies a scaling route that does not exist.
METHODS = ("clifford",)


def n_blocks_for(shots: int) -> int:
    """Blocks needed to cover ``shots``, rounding **up**: ``45_000 -> 5`` blocks (``50_000``)."""
    return int(math.ceil(max(int(shots), 0) / BLOCK))


def realised_shots(n_blocks: int) -> int:
    return int(n_blocks) * BLOCK


def block_seed(shot_seed: int, block: int) -> int:
    """32-bit ``exqalibur`` seed for one block.

    Hash-derived rather than ``base + block`` so the shot stream never aliases the circuit's weight
    stream -- with an additive offset, ``noise(model_seed=42)`` collided with ``weights(model_seed=55)``.
    Folded to 32 bits here because that is what ``exqalibur.set_seed`` takes.
    """
    blob = f"{int(shot_seed)}:shot:{int(block)}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


# --- identity -------------------------------------------------------------------------------- #


def shot_spec(cfg, *, method: str = "clifford") -> dict:
    """Identity fields of a shot *realisation*.  Note ``n_blocks`` is deliberately absent."""
    if method not in METHODS:
        raise ValueError(f"unknown shot method {method!r}; choose from {list(METHODS)}")
    return {"block": BLOCK, "shot_seed": int(cfg.seeds.shot_seed), "method": method}


def shot_hash(cfg, *, method: str = "clifford") -> str:
    blob = json.dumps(shot_spec(cfg, method=method), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def shots_path(cfg, model, *, method: str = "clifford",
               shots_root: str | Path = "shots_v2") -> Path:
    from .artifact import circuit_hash
    return Path(shots_root) / circuit_hash(cfg, model) / shot_hash(cfg, method=method)


# --- the rows --------------------------------------------------------------------------------- #
#
# A draw is `list[dict[tuple[int, ...], int]]`: one mapping per row, occupation tuple -> count.
# There is no wrapper class -- a list of dicts already supports everything the branch needs
# (`+` extends blocks, `+` on the lists extends rows, `len` is the row count).


def merge_shots(a: list[dict], b: list[dict]) -> list[dict]:
    """Add ``b``'s counts into ``a``'s, row by row -- how the shot budget extends.

    Rows past the end of either side are carried through, so this also tolerates a ragged pair.
    Counts stay integers, so a smaller budget stays a prefix of a larger one.
    """
    out = []
    for i in range(max(len(a), len(b))):
        lhs = Counter(a[i]) if i < len(a) else Counter()
        lhs.update(b[i] if i < len(b) else {})
        out.append(dict(lhs))
    return out


def observed_keys(rows: list[dict]) -> tuple:
    """Sorted union of the keys any row observed.  Sorted so the column order is deterministic."""
    seen: set = set()
    for row in rows:
        seen.update(row)
    return tuple(sorted(seen))


def total_shots(rows: list[dict]) -> int:
    """Shots in the first row -- the per-row budget (every row gets the same number)."""
    return int(sum(rows[0].values())) if rows else 0


def to_sparse(rows: list[dict], keys: tuple | None = None):
    """``(keys, probs)``: the per-row dicts as an aligned ``(N, len(keys))`` float32 matrix.

    The one place the matrix form is built, because an observable needs a score vector aligned to
    columns.  ``keys`` defaults to the observed union, so the width is the number of *observed*
    outcomes and never ``C(m+k-1, k)``; pass it explicitly to align onto a declared basis.
    """
    keys = observed_keys(rows) if keys is None else tuple(keys)
    col = {key: i for i, key in enumerate(keys)}
    counts = torch.zeros(len(rows), len(keys), dtype=torch.float64)
    for i, row in enumerate(rows):
        for key, n in row.items():
            counts[i, col[key]] = float(n)
    probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    return keys, probs.to(torch.float32)


def score_sparse(name: str, ctx, rows: list[dict]) -> torch.Tensor:
    """Score ``name`` from the observed outcomes alone -- the finite-sample readout.

    Equivalent to scoring the dense empirical distribution, to float round-off, but never builds a
    table over the unobserved outcomes.  Every implemented observable agrees with its full-basis
    value on such a subset, because an unobserved outcome contributes nothing -- see
    :func:`v2.observable.base.observable_on_keys` for the one shape that would not.
    """
    from observable import observable_on_keys

    keys, probs = to_sparse(rows)
    return observable_on_keys(name, ctx, keys).score(probs)


# --- store ------------------------------------------------------------------------------------ #
#
# Ragged dicts go to disk in the flat CSR-ish triple (indptr, keys, counts) -- so the file tracks the
# number of observed outcomes, never the basis size, and there is no padding to a common width.


def save_shots(path: str | Path, rows: list[dict], spec: dict, *, n_blocks: int) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    flat_keys = [key for row in rows for key in row]
    n_modes = len(flat_keys[0]) if flat_keys else 0
    np.savez_compressed(
        path / COUNTS_FILENAME,
        indptr=np.cumsum([0] + [len(row) for row in rows]).astype(np.int64),
        keys=np.asarray(flat_keys, dtype=np.int16).reshape(len(flat_keys), n_modes),
        counts=np.asarray([n for row in rows for n in row.values()], dtype=np.int32),
    )
    (path / SHOT_META_FILENAME).write_text(json.dumps(
        {**spec, "n_blocks": int(n_blocks), "shots": realised_shots(n_blocks),
         "size": len(rows), "n_observed": len(observed_keys(rows))}, indent=2))
    return path


def load_shots(path: str | Path):
    """``(rows, meta)`` -- one ``{key: count}`` per row, and ``n_blocks``."""
    path = Path(path)
    with np.load(path / COUNTS_FILENAME) as z:
        indptr, keys, counts = z["indptr"], z["keys"], z["counts"]
    rows = [{tuple(int(c) for c in keys[j]): int(counts[j]) for j in range(indptr[i], indptr[i + 1])}
            for i in range(len(indptr) - 1)]
    return rows, json.loads((path / SHOT_META_FILENAME).read_text())


def load_shot_probs(path: str | Path):
    """``(keys, probs, meta)`` -- the store as the aligned matrix scoring wants."""
    rows, meta = load_shots(path)
    keys, probs = to_sparse(rows)
    return keys, probs, meta
