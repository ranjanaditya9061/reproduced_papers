"""Artifacts and stores.  One circuit identity, two sibling branches, cached readouts.

    datasets/<circuit_hash>/exact/                dist.npz + circuit.pt + meta.json
    datasets/<circuit_hash>/counts/<shot_tag>/    shots.npz (keys, counts) + meta.json
    scores/<circuit_hash>/<source>/<obs_hash>.pt  soft (N,)

``circuit_hash`` covers the **input and the circuit only** -- no shot budget, no readout -- so one
simulation serves every observable and every shot budget.  The exact distribution and the shots are
*siblings* hanging off it, not parent and child: perceval implements them with disjoint backends, and
at large ``(m, k)`` the exact distribution cannot be computed at all.

Quantities you EXTEND stay out of the hash and live in ``meta.json``: ``size`` for rows, ``n_blocks``
for shots.  Quantities that CHANGE the data name the directory: ``shot_seed`` and ``BLOCK`` in
``shot_tag``, the observable in the score filename, the label source in ``<source>``.

The two CLI stages (:mod:`.generate`, :mod:`.score`) are deliberately **not** imported here, so
``python -m pipeline.generate`` does not import that module twice.  Import them by module.
"""

from __future__ import annotations

from .artifact import (circuit_hash, circuit_path, exact_path, hash_fields, load_meta,
                       save_circuit, save_meta)
from .distribution import Distribution, check_size, load_dist, save_dist
from .shots import (METHODS, load_shot_probs, load_shots, offset_seed, save_shots, shot_source_tag,
                    shot_spec, shot_tag, shots_path, to_counts, to_index, to_seqs)
from .split import split_indices

__all__ = [
    "circuit_hash", "circuit_path", "exact_path", "hash_fields", "load_meta",
    "save_circuit", "save_meta",
    "Distribution", "check_size", "load_dist", "save_dist", "split_indices",
    "METHODS", "load_shot_probs", "load_shots", "offset_seed", "save_shots", "shot_source_tag",
    "shot_spec", "shot_tag", "shots_path", "to_counts", "to_index", "to_seqs",
]
