"""Identity, layout, and the one writer of ``meta.json``.

One directory per circuit, one subdirectory per branch::

    datasets/<circuit_hash>/
    |-- exact/                    dist.npz  (keys, probs, probs_at_zero)
    |   |                         circuit.pt
    |   `-- meta.json
    `-- counts/<shot_tag>/        shots.npz (keys, counts)
        `-- meta.json

Both branches live under the **same** circuit directory because they are siblings of one generating
process: perceval implements them with disjoint backends (``CliffordClifford2017`` samples and
exposes no ``all_prob``; ``SLOS``/``Naive`` expose ``all_prob`` and never sample), and at large
``(m, k)`` the exact distribution cannot be computed at all -- so shots are not a readout of the
distribution, but they *are* the same circuit on the same input pool.  ``<shot_tag>`` sits one level
down because a circuit has one exact distribution but many shot realisations.

**One meta writer.**  :func:`save_meta` is the only function anywhere that writes ``meta.json``, and
both branches call it, so both carry the same circuit-identity block (hash, model, geometry, spec,
``input_state``, ``readout_modes``) and differ only in the ``**extra`` each passes.  The shots store
used to write its own ad-hoc meta with none of that, which meant a counts directory could not say
which circuit produced it.

**The observable appears nowhere.**  :func:`circuit_hash` covers the input and the circuit only, so
one simulation serves every readout; :func:`_check_spec` raises if a model tries to smuggle an
observable into its ``circuit_spec``, because that is the one mistake that silently reintroduces the
coupling (in the legacy pipeline ``parity`` and ``osc`` on an identical circuit produced two
directories and two independent boson-sampling runs).

Quantities you EXTEND stay out of the hash and live in ``meta.json`` -- ``size`` for rows,
``n_blocks`` for shots -- so growing either keeps the same directory instead of redrawing from zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

FORMAT_VERSION = 3

EXACT_DIR = "exact"
COUNTS_DIR = "counts"

DIST_FILENAME = "dist.npz"
SHOTS_FILENAME = "shots.npz"
CIRCUIT_FILENAME = "circuit.pt"
META_FILENAME = "meta.json"

#: Keys that must never appear in a ``circuit_spec``.  A readout is not part of a dataset's identity.
_FORBIDDEN_SPEC_KEYS = ("observable", "observables", "readout_observable", "score", "soft")


def _check_spec(spec: dict, model_name: str) -> dict:
    """Reject an observable smuggled into a model's ``circuit_spec``."""
    bad = sorted(set(spec) & set(_FORBIDDEN_SPEC_KEYS))
    if bad:
        raise ValueError(
            f"model {model_name!r} put {bad} in circuit_spec(). The artifact must depend on the "
            "INPUT and the CIRCUIT only -- an observable is a readout applied afterwards, and "
            "including it here would make the same circuit simulate once per readout.\n"
            "('readout' itself IS allowed: fermion uses it for det-vs-Perm, a property of the "
            "circuit's physics rather than a choice of measurement.)"
        )
    return spec


def hash_fields(cfg, model) -> dict:
    """The identity fields: input + circuit, and nothing else.

    Excludes ``size`` and the shot budget (both extendable, both recorded in ``meta.json``) and
    every readout.  So one hash names one *generating process*, and both branches hang off it.
    """
    return {
        "format_version": FORMAT_VERSION,
        "model": cfg.model.kind,
        "m": int(cfg.problem.m),
        "k": int(cfg.problem.k),
        "n_features": int(cfg.problem.n_features),
        "sample_seed": int(cfg.seeds.sample_seed),
        "model_seed": int(cfg.seeds.model_seed),
        "spec": _check_spec(model.circuit_spec(), cfg.model.kind),
    }


def circuit_hash(cfg, model) -> str:
    """8-char content hash over the input + circuit fields."""
    blob = json.dumps(hash_fields(cfg, model), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def circuit_path(cfg, model, root: str | Path = "datasets") -> Path:
    """``<root>/<circuit_hash>`` -- the directory both branches share."""
    return Path(root) / circuit_hash(cfg, model)


def exact_path(cfg, model, root: str | Path = "datasets") -> Path:
    """``<root>/<circuit_hash>/exact`` -- the exact-distribution branch."""
    return circuit_path(cfg, model, root) / EXACT_DIR


def save_meta(path: str | Path, cfg, model, **extra) -> Path:
    """Write ``meta.json``: the shared circuit identity plus this branch's ``extra`` fields.

    ``input_state`` and ``readout_modes`` are carried on both branches so every observable stays
    re-scorable offline -- both are functions of the circuit alone, and the legacy format stored
    neither, so ``single_output`` could not be re-scored at all.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model.input_state()
    meta = {
        **hash_fields(cfg, model),
        "hash": circuit_hash(cfg, model),
        "n_model_parameters": int(model.n_model_parameters()),
        "input_state": None if state is None else [int(v) for v in state],
        "readout_modes": [int(v) for v in model.readout_modes()],
        **extra,
    }
    (path / META_FILENAME).write_text(json.dumps(meta, indent=2))
    return path


def load_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / META_FILENAME).read_text())


def save_circuit(path: str | Path, model) -> Path | None:
    """Persist the model's fixed weights, when it has any."""
    state = model.state_dict()
    if not state:
        return None
    out = Path(path) / CIRCUIT_FILENAME
    torch.save(state, out)
    return out
