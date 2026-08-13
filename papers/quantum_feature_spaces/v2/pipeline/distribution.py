"""The exact branch: ``<root>/<circuit_hash>/exact/`` -- one writer, one reader.

    save_dist(path, cfg, model, probs)      # dist.npz + circuit.pt + meta.json, together
    dist = load_dist(path)                  # X, probs, keys, probs_at_zero, meta

:func:`save_dist` writes the payload, the weights and the meta in one call, so an artifact cannot
exist with a stale or missing ``meta.json``; the meta itself goes through
:func:`~pipeline.artifact.save_meta`, shared with the counts branch.

``X`` is **not stored**.  ``sample_X`` is a single contiguous draw from a fixed seed, so it is
prefix-stable and ``X`` is reconstructed exactly from ``meta.json``'s ``n_features``,
``sample_seed`` and ``size``.  Storing it would add a second source of truth for the input pool that
could drift from the seed the hash is taken over.

Neither function knows what an observable is: ``dist.npz`` has no ``observable`` field, by
construction.  ``probs_at_zero`` (the ``q`` that ``xent`` scores against) is a function of the
circuit alone, so it belongs here and makes ``xent`` re-scorable without rebuilding a teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model.sampler import sample_X

from .artifact import DIST_FILENAME, load_meta, save_circuit, save_meta


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


def check_size(size: int, n_out: int, max_bytes: int, *, m: int, k: int) -> None:
    """Refuse to generate a ``probs`` matrix over the budget.

    Errors with the computed size and ``(m, k, N)`` rather than silently downcasting: at
    ``N = 10_000``, ``n_out = 2002`` is 80 MB, ``12376`` is 495 MB and ``77520`` is 3.1 GB.
    """
    need = int(size) * int(n_out) * 4
    if need > int(max_bytes):
        raise ValueError(
            f"probs would be {need / 1024 ** 3:.2f} GiB at m={m}, k={k}, N={size} "
            f"({n_out} outcomes x float32), over generation.max_dist_bytes = "
            f"{int(max_bytes) / 1024 ** 3:.2f} GiB. Reduce generation.size or (m, k), or raise "
            "the budget deliberately -- it is not downcast automatically."
        )


def save_dist(path: str | Path, cfg, model, probs: torch.Tensor) -> Path:
    """Write the exact branch whole: ``dist.npz``, ``circuit.pt`` and ``meta.json``.

    ``keys`` and ``probs_at_zero`` come off the model rather than the caller -- they are properties
    of the circuit, so there is nothing for a caller to get wrong.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    keys = model.outcome_keys()
    np.savez_compressed(
        path / DIST_FILENAME,
        keys=np.asarray([[int(c) for c in key] for key in keys], dtype=np.int16),
        probs=probs.detach().cpu().numpy().astype(np.float32),
        probs_at_zero=model.probs_at_zero().detach().cpu().numpy().astype(np.float32),
    )
    save_circuit(path, model)
    # `exact_available` records that dist.npz really carries an exact p.  Analyses A and B
    # differentiate p, so neither can run from the counts branch alone.
    save_meta(path, cfg, model, size=int(probs.shape[0]), n_outcomes=len(keys),
              exact_available=True)
    return path


def readout_condition(tensors: tuple, keys: tuple, readout: tuple,
                      *, structure: str | None = None) -> tuple[tuple, tuple]:
    """Condition each of ``tensors`` (last axis aligned to ``keys``) on the readout modes, drop
    the readout columns from ``keys``, and return ``(conditioned_tensors, data_keys)``.

    Shared by :func:`~pipeline.distribution.load_dist` (a loaded artifact's stored ``probs``/
    ``probs_at_zero``) and :func:`eval.sweep_delta.readout_zero_probs` (a live model's ``probs``
    batch) -- both read the same two fields
    (:meth:`model.base.DistributionModel.readout_modes`/``outcome_keys``, persisted into
    ``meta.json`` by :func:`~pipeline.artifact.save_meta`), so one function suffices for both.  A
    no-op when ``readout`` is empty (every prep except ``spin_magic*``).

    Every structure but ``"ghz"`` post-selects on the readout reading dual-rail ``0`` (one photon
    in the first readout mode, none in the second) and renormalises, matching every other
    (data-mode-only) arm's basis. ``structure == "ghz"`` never re-superposes the spin after its
    first gap, so the final ``H`` ahead of the readout emission acts on a fixed Z-eigenstate rather
    than a genuine superposition: the readout branches deterministically, and masking to the
    dual-rail-``0`` branch alone would discard the outcome and, for some ``x``, leave zero mass on
    the kept branch. For ``ghz`` we instead marginalize -- sum both dual-rail readout outcomes
    together -- rather than post-select on either.
    """
    if not readout:
        return tensors, keys
    r0, r1 = readout

    # ``ghz`` keeps both readout branches (marginalize); every other structure keeps only the
    # dual-rail-0 branch (post-select). Either way, surviving rows are then grouped by their
    # data-only key: for the post-select mask that grouping is already 1:1 (a no-op sum), for the
    # marginalize mask it is what sums the two readout branches together.
    if structure == "ghz":
        mask = torch.ones(len(keys), dtype=torch.bool)
    else:
        mask = torch.tensor([int(key[r0]) == 1 and int(key[r1]) == 0 for key in keys])

    data_keys, index = [], {}
    for key, keep in zip(keys, mask.tolist()):
        if not keep:
            continue
        data_key = tuple(v for j, v in enumerate(key) if j not in readout)
        if data_key not in index:
            index[data_key] = len(data_keys)
            data_keys.append(data_key)
    data_keys = tuple(data_keys)

    def _condition(P: torch.Tensor) -> torch.Tensor:
        Pc = torch.zeros(*P.shape[:-1], len(data_keys), dtype=P.dtype)
        for j, (key, keep) in enumerate(zip(keys, mask.tolist())):
            if not keep:
                continue
            data_key = tuple(v for i, v in enumerate(key) if i not in readout)
            Pc[..., index[data_key]] += P[..., j]
        totals = Pc.sum(dim=-1, keepdim=True)
        if bool((totals <= 0).any()):
            raise ValueError("readout post-selection has zero mass at some x")
        return Pc / totals

    return tuple(_condition(P) for P in tensors), data_keys


def load_dist(path: str | Path, *, size: int | None = None, load_full: bool = False) -> Distribution:
    """Load the exact branch, reconstructing ``X`` from the seed recorded in ``meta.json``.

    ``size`` truncates to the first rows of the pool -- the prefix is stable, so this is a genuine
    subsample of the same dataset rather than a different one.

    **``load_full``.**  A prep never post-selects its own readout modes -- ``spin_magic`` saves the
    full, unselected distribution over its data modes *and* its two readout modes, so any later
    choice of post-selection is computed offline (see :mod:`circuit.prep`'s module docstring).
    ``load_full=False`` (the default) applies the ``mu = 0`` readout post-selection here, so a
    caller gets the *data-mode-only* distribution -- comparable to every other (non-``spin_magic``)
    arm's -- without knowing which preps carry readout modes at all.  ``load_full=True`` returns the
    raw stored distribution unchanged, readout modes and all.  A no-op either way when
    ``readout_modes`` is empty, which is every prep except ``spin_magic*``.
    """
    path = Path(path)
    meta = load_meta(path)
    with np.load(path / DIST_FILENAME) as z:
        keys = tuple(tuple(int(c) for c in row) for row in z["keys"])
        probs = torch.from_numpy(z["probs"].astype(np.float32))
        probs_at_zero = torch.from_numpy(z["probs_at_zero"].astype(np.float32))

    if not load_full:
        readout = tuple(int(v) for v in (meta.get("readout_modes") or ()))
        structure = (meta.get("spec") or {}).get("structure")
        (probs, probs_at_zero_2d), keys = readout_condition(
            (probs, probs_at_zero.unsqueeze(0)), keys, readout, structure=structure)
        probs_at_zero = probs_at_zero_2d[0]

    n = probs.shape[0] if size is None else min(int(size), probs.shape[0])
    X = sample_X(int(meta["size"]), int(meta["n_features"]), int(meta["sample_seed"]))
    return Distribution(X=X[:n], probs=probs[:n], keys=keys, probs_at_zero=probs_at_zero, meta=meta)
