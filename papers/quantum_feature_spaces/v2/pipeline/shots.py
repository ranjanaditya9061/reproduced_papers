"""The counts branch: ``<root>/<circuit_hash>/counts/<method>_s<shot_seed>/``.

    shots.npz     keys  (n_observed, n_modes) int16    sorted union of the outcomes seen
                  seq   (N, S)                uint16   one entry per shot, IN DRAW ORDER
    meta.json     the shared circuit identity + shot_seed / method / shots / size / n_observed

**The shot sequence, not aggregated counts.**  Storing ``{key: count}`` per row throws away the
order the shots came in, and with it the ability to ask for a *smaller* budget: from a 40k store the
only route back to 10k is a hypergeometric subsample, which is a different random draw from the one
a real 10k run produced.  Recording the sequence makes a smaller budget ``seq[:, :n]`` -- a true
prefix of the recorded draw, no sampling anywhere.  Counts are one :func:`to_counts` away at the
scoring boundary, so nothing is lost by not storing them.

``seq`` holds *indices* into ``keys`` rather than occupation tuples, so a shot costs 2 bytes and the
outcome basis is never materialised: the key table is the number of **observed** outcomes, against
``C(m+k-1, k)`` for the full basis -- ``20_030_010`` at ``m=20, k=10``.  The cost of the format is
that it is ``N x S``: 20 rows x 40k shots is 800k entries, but 2000 rows x 100k shots is 2e8, so the
budget is now a real storage decision rather than a free parameter.

**Seeded on the shot offset.**  :func:`offset_seed` keys ``exqalibur`` on *how many shots are already
saved* -- nothing stored means offset 0, 10k stored means offset 10_000.  Extending to 30k seeds at
10_000 and draws only the 20k new shots, which are appended; the stored prefix is never rewritten, so
the 10k answer stays bit-identical after you grow.  Both directions work and there is nothing to
quantise, which is why ``BLOCK`` and its four helper functions are gone and ``generation.shots`` is
now used literally instead of being rounded up.

**Rows share a stream within one draw call**, so which shots a given row gets depends on how many
rows precede it.  A draw is reproducible as a whole and a *shot* extension is exact; a **row**
extension is not bit-stable, which is the one thing this scheme does not fix.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .artifact import COUNTS_DIR, SHOTS_FILENAME, circuit_path, load_meta, save_meta

#: How the shots were produced.  ``"clifford"`` samples the interferometer directly -- no
#: distribution is ever formed, which is what makes this branch a sibling of the exact one.  A
#: ``"multinomial"`` method (draw from a stored exact ``p``) is deliberately **absent**: it would be
#: faster but requires the full distribution, so it silently reinstates the dependency this branch
#: exists to remove and implies a scaling route that does not exist.
METHODS = ("clifford",)


def offset_seed(shot_seed: int, offset: int) -> int:
    """32-bit ``exqalibur`` seed for the draw that starts at shot ``offset``.

    Hash-derived rather than ``base + offset`` so the shot stream never aliases the circuit's weight
    stream -- with an additive offset, ``noise(model_seed=42)`` collided with ``weights(model_seed=55)``.
    Folded to 32 bits because that is what ``exqalibur.set_seed`` takes.
    """
    blob = f"{int(shot_seed)}:shot:{int(offset)}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def shot_spec(cfg, *, method: str = "clifford") -> dict:
    """Identity fields of a shot *realisation*.  The budget is deliberately absent -- you extend it."""
    if method not in METHODS:
        raise ValueError(f"unknown shot method {method!r}; choose from {list(METHODS)}")
    return {"shot_seed": int(cfg.seeds.shot_seed), "method": method}


def shot_tag(cfg, *, method: str = "clifford") -> str:
    """``<method>_s<shot_seed>`` -- the realisation's directory name.

    Spelled out rather than hashed: the circuit needs a digest because its ``spec`` is an arbitrary
    dict, but a realisation is two scalars, so the tag just contains them.  ``shot_seed`` is the
    config seed, **not** what exqalibur is seeded with -- that is :func:`offset_seed`.
    """
    spec = shot_spec(cfg, method=method)
    return f"{spec['method']}_s{spec['shot_seed']}"


def shots_path(cfg, model, *, method: str = "clifford", root: str | Path = "datasets") -> Path:
    """``<root>/<circuit_hash>/counts/<shot_tag>`` -- beside the exact branch, same circuit."""
    return circuit_path(cfg, model, root) / COUNTS_DIR / shot_tag(cfg, method=method)


def shot_source_tag(shot_meta: dict) -> str:
    """``<method>_s<shot_seed>_n<shots>`` -- names the *labels* a draw produces, for the score cache.

    The realisation **and** the budget, because both change the labels: 10k-shot ``parity`` is not
    40k-shot ``parity``, so they cannot share a cache entry.  Takes ``meta`` rather than ``cfg``
    because the scorer has only the loaded store -- and lives here, beside :func:`shot_tag`, so the
    naming is not re-derived by hand in :mod:`v2.pipeline.score`.

    Uses the *returned* ``meta["shots"]``, which :func:`load_shots` sets to the cropped width -- so a
    crop to 10k caches under ``_n10000`` even when the store holds 40k.
    """
    return f"{shot_meta['method']}_s{int(shot_meta['shot_seed'])}_n{int(shot_meta['shots'])}"


# --- the in-memory form ------------------------------------------------------------------------- #
#
# `list[list[tuple[int, ...]]]`: one list of occupation tuples per row, in draw order.  Plain lists,
# so both extensions are concatenation -- `a + b` per row grows the budget, `seqs + new` grows the
# pool -- and no wrapper class is needed.


def to_index(seqs: list[list[tuple]]):
    """``(keys, seq)``: raw outcome tuples as a sorted key table plus an ``(N, S)`` index array.

    Every row must have the same number of shots; a ragged draw is a bug in the caller, not
    something to pad over.
    """
    widths = {len(row) for row in seqs}
    if len(widths) > 1:
        raise ValueError(f"rows have differing shot counts {sorted(widths)}; every row gets the same "
                         "budget, so this means a draw was assembled wrongly")
    keys = tuple(sorted({key for row in seqs for key in row}))
    col = {key: i for i, key in enumerate(keys)}
    dtype = np.uint16 if len(keys) < np.iinfo(np.uint16).max else np.int32
    seq = np.asarray([[col[key] for key in row] for row in seqs], dtype=dtype)
    return keys, seq.reshape(len(seqs), widths.pop() if widths else 0)


def to_seqs(keys: tuple, seq: np.ndarray) -> list[list[tuple]]:
    """Inverse of :func:`to_index` -- back to raw tuples, which is what extension concatenates."""
    return [[keys[j] for j in row] for row in seq.tolist()]


def to_counts(keys: tuple, seq: np.ndarray) -> torch.Tensor:
    """``(N, n_keys)`` int64 counts -- the aggregate form scoring wants, built at the boundary."""
    counts = np.zeros((seq.shape[0], len(keys)), dtype=np.int64)
    for i, row in enumerate(seq):
        counts[i] = np.bincount(row, minlength=len(keys))
    return torch.from_numpy(counts)


# --- store ---------------------------------------------------------------------------------------- #


def save_shots(path: str | Path, seqs: list[list[tuple]], cfg, model, *,
               method: str = "clifford") -> Path:
    """Write ``shots.npz`` and ``meta.json`` together, so the store is never half-described."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    keys, seq = to_index(seqs)
    np.savez_compressed(
        path / SHOTS_FILENAME,
        keys=(np.asarray([[int(c) for c in key] for key in keys], dtype=np.int16) if keys
              else np.zeros((0, 0), dtype=np.int16)),
        seq=seq,
    )
    save_meta(path, cfg, model, **shot_spec(cfg, method=method), shots=int(seq.shape[1]),
              size=int(seq.shape[0]), n_observed=len(keys))
    return path


def load_shot_probs(path: str | Path, num_shots: int | None = None):
    """``(keys, probs, meta)`` -- the store as the ``(N, n_keys)`` float32 matrix scoring wants.

    The empirical distribution over the first ``num_shots`` shots of each row.  Every row carries the
    same budget, so the normaliser is ``meta["shots"]`` rather than a per-row sum.
    """
    keys, seq, meta = load_shots(path, num_shots)
    shots = int(meta["shots"]) 
    if num_shots:
        shots = num_shots
    probs = to_counts(keys, seq).to(torch.float32) / shots
    return keys, probs, meta


def load_shots(path: str | Path, num_shots: int | None = None):
    """``(keys, seq, meta)``.  ``num_shots`` crops to that many shots per row; ``None`` is everything.

    The crop is a prefix of the recorded draw, so it *is* the first ``num_shots`` shots -- not a
    resample of the stored ones.  ``meta["shots"]`` is set to what was returned, so normalising with
    it stays correct after a crop.
    """
    path = Path(path)
    with np.load(path / SHOTS_FILENAME) as z:
        keys = tuple(tuple(int(c) for c in row) for row in z["keys"])
        seq = z["seq"]

    if num_shots is not None:
        if int(num_shots) > seq.shape[1]:
            raise ValueError(f"asked for {int(num_shots)} shots but {path} stores {seq.shape[1]}; "
                             "raise generation.shots and re-run generate to draw the rest")
        seq = seq[:, :int(num_shots)]

    meta = load_meta(path)
    meta["shots"] = int(seq.shape[1])
    return keys, seq, meta