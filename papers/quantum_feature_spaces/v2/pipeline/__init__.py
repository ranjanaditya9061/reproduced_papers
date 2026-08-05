"""Artifacts and stores.  One circuit identity, two sibling branches, cached readouts.

    datasets_v2/<model>_m<m>_k<k>_n<n_f>_<circuit_hash>/   dist.npz + circuit.pt + meta.json
    shots_v2/<circuit_hash>/<shot_hash>/                   counts.npz + meta.json (n_blocks)
    scores_v2/<circuit_hash>/<source>/<obs_hash>.pt        soft (N,)

``circuit_hash`` covers the **input and the circuit only** -- no shot budget, no readout -- so one
simulation serves every observable and every shot budget.  The exact distribution and the shots are
*siblings* hanging off it, not parent and child: perceval implements them with disjoint backends, and
at large ``(m, k)`` the exact distribution cannot be computed at all.

Quantities you EXTEND stay out of the hash and live in ``meta.json``: ``size`` for rows, ``n_blocks``
for shots.  Quantities that CHANGE the data are hashed: ``shot_seed`` and ``BLOCK`` into
``shot_hash``, the observable into the score filename, the label source into ``<source>``.

The two CLI stages (:mod:`.generate`, :mod:`.score`) are deliberately **not** imported here, so
``python -m v2.pipeline.generate`` does not import that module twice.  Import them by module.
"""

from __future__ import annotations

from .artifact import (artifact_dirname, artifact_path, circuit_hash, hash_fields, load_meta,
                       save_circuit, save_meta)
from .distribution import Distribution, check_size, dist_bytes, load_dist, write_dist
from .shots import (BLOCK, METHODS, block_seed, load_shot_probs, load_shots, merge_shots,
                    n_blocks_for, observed_keys, realised_shots, save_shots, score_sparse,
                    shot_hash, shot_spec, shots_path, to_sparse, total_shots)
from .split import split_indices

__all__ = [
    "artifact_dirname", "artifact_path", "circuit_hash", "hash_fields", "load_meta",
    "save_circuit", "save_meta",
    "Distribution", "check_size", "dist_bytes", "load_dist", "write_dist", "split_indices",
    "BLOCK", "METHODS", "block_seed", "load_shot_probs", "load_shots", "merge_shots",
    "n_blocks_for", "observed_keys", "realised_shots", "save_shots", "score_sparse",
    "shot_hash", "shot_spec", "shots_path", "to_sparse", "total_shots",
]
