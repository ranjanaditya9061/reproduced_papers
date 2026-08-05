"""The on-disk artifact, keyed on the **input and the circuit only**.

Layout, one directory per logical dataset::

    datasets_v2/<model>_m<m>_k<k>_n<n_features>_<hash>/
    |-- dist.npz      keys (n_out, n_modes) int16
    |                 probs (N, n_out) float32          <- THE artifact
    |                 probs_at_zero (n_out,) float32     <- q, so xent is re-scorable
    |-- circuit.pt    the model's state_dict (W1/W2, theta, W_re/W_im, ...)
    `-- meta.json     provenance + input_state + circuit_spec

**The observable appears nowhere.**  That is the point of this module.  In the legacy pipeline
``Generator/artifact.py::_hash_fields`` folded ``TEACHERS[...].hash_spec(cfg)`` into the hash, and
every Fock teacher's ``hash_spec`` returned ``{"observable": ...}`` -- so ``parity`` and ``osc`` on
an *identical* circuit with *identical* inputs produced two separate directories and two
independent boson-sampling runs.  Here the distribution is the artifact and the readout is a cheap
downstream stage (:mod:`v2.pipeline.score`), so one simulation serves every observable.

:func:`circuit_hash` **enforces** that: if a model's ``circuit_spec`` contains an observable-ish key it
raises, because that is the one mistake that would silently reintroduce the coupling.

``size`` is deliberately excluded from the hash, so growing a pool keeps the same directory;
``meta.json["size"]`` records the current length.

Two fields exist purely so that every observable is re-scorable offline, which the legacy format
could not manage: ``probs_at_zero`` (the ``q`` that ``xent`` scores against) and ``input_state``
(which ``single_output`` marks).  Both are functions of the circuit alone, so they belong here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

FORMAT_VERSION = 2

DIST_FILENAME = "dist.npz"
CIRCUIT_FILENAME = "circuit.pt"
META_FILENAME = "meta.json"

#: Keys that must never appear in a ``circuit_spec``.  A readout is not part of a dataset's
#: identity; see the module docstring.
_FORBIDDEN_SPEC_KEYS = ("observable", "observables", "readout_observable", "score", "soft")


def _check_spec(spec: dict, model_name: str) -> dict:
    """Reject an observable smuggled into a model's ``circuit_spec``."""
    bad = sorted(set(spec) & set(_FORBIDDEN_SPEC_KEYS))
    if bad:
        raise ValueError(
            f"model {model_name!r} put {bad} in circuit_spec(). The distribution artifact must "
            "depend on the INPUT and the CIRCUIT only -- an observable is a readout applied "
            "afterwards and cached by v2.pipeline.score. Including it here would make the same "
            "circuit simulate once per readout, which is exactly what v2 exists to fix.\n"
            "(Note 'readout' itself IS allowed: fermion uses it for det-vs-Perm, which is a "
            "property of the circuit's physics, not a choice of measurement.)"
        )
    return spec


def hash_fields(cfg, model) -> dict:
    """The identity fields: input + circuit, and nothing else.

    * input -- ``n_features``, ``sample_seed``, and ``size`` only implicitly (excluded, see above)
    * circuit -- ``model`` kind, ``m``, ``k``, ``model_seed``, and the model's own
      ``circuit_spec()`` (prep, encoding, per-family knobs)

    **No shot fields.**  An earlier version hashed ``nsample``, on the argument that it changes the
    stored distribution so two shot budgets are two datasets.  That is true of what is *stored* but
    wrong about identity, and it cost three things: a 20k and a 30k budget landed in different
    directories, so the 30k run re-ran the exact simulation (measured 76% of the work, bit-identical
    output) *and* redrew from shot zero *and* produced a draw that was not even an extension of the
    20k one.

    The right rule is the one this module already applies to ``size``: a quantity you EXTEND belongs
    in ``meta.json``, not in the hash.  Shot blocks are nested and additive
    (:mod:`v2.pipeline.shots`), so growing the budget keeps the same store, exactly as growing the
    pool keeps the same directory.  Which *realisation* and how it is partitioned do change the
    data, so ``shot_seed`` and ``BLOCK`` are hashed -- into the SHOT identity
    (:func:`v2.pipeline.shots.shot_hash`), never into this one, which stays input + circuit only.
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
    """8-char content hash over input + circuit fields.

    Excludes ``size``, the shot budget, and every readout -- so one hash names one *generating
    process*, and both branches (exact distribution and shots) hang off it as siblings.
    """
    blob = json.dumps(hash_fields(cfg, model), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def artifact_dirname(cfg, model) -> str:
    p = cfg.problem
    return f"{cfg.model.kind}_m{p.m}_k{p.k}_n{p.n_features}_{circuit_hash(cfg, model)}"


def artifact_path(cfg, model, out_root: str | Path = "datasets_v2") -> Path:
    return Path(out_root) / artifact_dirname(cfg, model)


def save_meta(path: str | Path, cfg, model, *, size: int, n_out: int,
              exact_available: bool = True) -> Path:
    """Write ``meta.json``; returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model.input_state()
    meta = {
        **hash_fields(cfg, model),
        "hash": circuit_hash(cfg, model),
        "size": int(size),
        "n_outcomes": int(n_out),
        "n_model_parameters": int(model.n_model_parameters()),
        # Whether dist.npz carries an exact `probs`.  False in the regime where the outcome basis
        # is too large to enumerate, which is what gates analysis A and B -- both differentiate the
        # exact p, so neither can run from shots alone.
        "exact_available": bool(exact_available),
        # Circuit context that makes every observable offline-re-scorable.  The legacy .npz stored
        # neither, so `single_output` could not be re-scored at all and `xent` needed a
        # matched-seed teacher rebuilt by hand.
        "input_state": None if state is None else [int(v) for v in state],
        "readout_modes": [int(v) for v in model.readout_modes()],
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
