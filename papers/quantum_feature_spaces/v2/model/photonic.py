"""Boson sampling: ``W1(Haar) -> encode(x) -> W2(Haar)``, read out in the Fock basis.

``p(n) = |Perm(U_{s,n})|^2 / prod_j n_j!`` -- genuine multiphoton interference, so this wraps a
merlin ``QuantumLayer`` (perceval/merlin imported lazily, only on construction).

**One class covers eight legacy ones.**  The state preparation is a registry choice
(:mod:`v2.circuit.prep`), so this model is:

* ``prep="fock"`` -- the plain boson sampler (legacy ``model/photonic.py``)
* ``prep="spin"`` -- spin-qubit dual-rail emission (legacy ``spoqc``, ``spoqc_low``,
  ``spoqc_prime``, which differed only in ``cx_pairs`` / ``angle_levels`` / ``rz_angles``)
* ``prep="spin_magic"`` -- emitter-train cluster state with an injected non-Clifford gate (legacy
  ``spoqc_magic`` and its ``_prime`` / ``_rand`` / ``_rand_x`` variants, which differed only in
  ``gate_kind``)

The spin preps also gain the **full** observable registry.  The legacy spin teachers scored with a
private if-chain over four observable names, so ``ent``, ``osc``, ``prod_parity`` and the graph
families were simply unavailable on them; because every prep now returns a distribution, all of
them apply.

The ``fock`` prep is batched and differentiable in ``X`` (merlin is a differentiable layer), so
:mod:`v2.metrics` can take exact input-Jacobians through it.  The spin preps build one perceval
processor per row in a process pool, so they are neither batched nor differentiable -- metrics fall
back to finite differences there, which the metrics module states explicitly rather than silently.
"""

from __future__ import annotations

from collections import Counter

import torch

from circuit.encoding import build_encoding
from circuit.prep import build_prep
from .base import DistributionModel


class PhotonicModel(DistributionModel):
    """``X -> probs``: the sandwich circuit's Fock output distribution.

    ``prep`` and ``encoding`` are registry names; everything else is geometry and the seed.
    """

    name = "photonic"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 prep=None, encoding="phase", n_jobs: int = 1):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        from circuit.prep import FockPrep

        self.prep = FockPrep() if prep is None else prep
        self.encoding = build_encoding(encoding) if isinstance(encoding, str) else encoding
        self.n_jobs = int(n_jobs)
        self._keys = None
        self._autosize_batch(self._resolve_keys_len())

    @property
    def supports_shots(self) -> bool:
        """True only for ``prep='fock'`` -- pure boson sampling.
        """
        return getattr(self.prep, "name", None) == "fock"

    def shot_counts(self, X, *, shots: int | None = None, blocks=None, rows=None,
                    shot_seed: int = 0) -> list[dict]:
        """One ``{occupation tuple: count}`` per requested row, from **direct** sampling.

        Goes through ``CliffordClifford2017``, which samples the interferometer *without forming a
        distribution* -- so this never calls :meth:`probs`, and never enumerates the outcome basis.
        Both matter: the first makes shots a genuine sibling of the exact branch rather than a readout
        of it, and the second is what keeps memory at the number of **observed** outcomes.  A draw of
        ``S`` shots touches at most ``S`` of them, against a basis of ``C(m+k-1, k)`` --
        ``20_030_010`` at ``m=20, k=10``, which no dense count vector can hold.

        ``X`` is the full pool; ``rows`` selects which row indices to draw and ``blocks`` which block
        indices, both defaulting to everything.  The return has one dict per *requested* row, in the
        order requested -- so a row extension is a list concatenation and a block extension is
        :func:`~v2.pipeline.shots.merge_shots`.

        ``exqalibur`` is seeded once per block, and the rows of a block are drawn from that one
        stream, so each block is sampled independently.
        """
        if not self.supports_shots:
            raise NotImplementedError(
                f"prep {getattr(self.prep, 'name', '?')!r} does not support finite-shot draws."
            )
        import exqalibur as xq
        import numpy as np
        import perceval as pcvl
        from perceval.backends import BACKEND_LIST

        from circuit.photonic_circuit import build_sandwich_circuit, default_input_state
        from pipeline.shots import BLOCK, block_seed, n_blocks_for

        if blocks is None:
            if shots is None:
                raise ValueError("pass either shots= or blocks=")
            blocks = range(n_blocks_for(shots))
        blocks = list(blocks)
        pool = X.detach().cpu().numpy()
        row_idx = list(range(pool.shape[0])) if rows is None else [int(r) for r in rows]
        state = pcvl.BasicState([int(v) for v in
                                (self.input_state() or default_input_state(self.m, self.k))])

        counters = [Counter() for _ in row_idx]
        backend = BACKEND_LIST["CliffordClifford2017"]()
        for b in blocks:
            xq.set_seed(block_seed(shot_seed, b))
            for slot, i in enumerate(row_idx):
                # The circuit is the only per-row state; the backend is reused across rows.
                backend.set_circuit(build_sandwich_circuit(
                    self.m, self.n_features, self.seed, self.encoding,
                    x=np.asarray(pool[i], dtype=float)))
                backend.set_input_state(state)
                for s, n in Counter(backend.samples(BLOCK)).items():
                    key = tuple(int(v) for v in s)
                    counters[slot][key] = counters[slot].get(key, 0) + n
                print(slot, counters[slot])
        return counters

    def _resolve_keys_len(self) -> int:
        """Outcome count, when the prep can state it up front (``fock`` can; the spin preps cannot)."""
        keys = self.prep.outcome_keys(m=self.m, k=self.k)
        if keys is None:
            # A spin prep discovers its basis from perceval per row, so size the chunk on the
            # nominal Fock dimension over the prep's mode count as an upper-bound proxy.
            from .fock import n_fock

            return n_fock(self.prep.outcome_modes(self.m), self.k + 1)
        self._keys = keys
        return len(keys)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.prep.probs(X, m=self.m, k=self.k, n_features=self.n_features,
                                seed=self.seed, encoding=self.encoding, n_jobs=self.n_jobs)
        keys = self.prep.outcome_keys(m=self.m, k=self.k)
        if keys is not None:
            self._keys = keys
        return probs

    def outcome_keys(self):
        if self._keys is None:
            keys = self.prep.outcome_keys(m=self.m, k=self.k)
            if keys is None:
                raise RuntimeError(
                    f"prep {self.prep.name!r} discovers its outcome basis from the backend, so "
                    "probs() must be called before outcome_keys(). This is why the generator "
                    "simulates before it writes the artifact."
                )
            self._keys = keys
        return self._keys

    def readout_modes(self) -> tuple:
        return tuple(self.prep.readout_modes(self.m))

    def input_state(self):
        return getattr(self.prep, "input_state", None)

    def n_model_parameters(self) -> int:
        """``2m^2 - 1`` -- the two Haar unitaries, less the unobservable global phase."""
        from circuit.photonic_circuit import n_circuit_parameters

        return n_circuit_parameters(self.m)

    def circuit_spec(self) -> dict:
        return {"model": self.name, **self.encoding.spec(), **self.prep.spec()}

    @classmethod
    def from_config(cls, cfg) -> "PhotonicModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed,
                   prep=build_prep(cfg), encoding=cfg.model.encoding,
                   n_jobs=cfg.generation.n_jobs)

    @classmethod
    def validate_config(cls, cfg) -> None:
        # Geometry checks live on the prep (dual-rail needs even m and 2k <= m) and on the
        # encoding (a phase encoding needs n_features <= m); both are exercised by build_prep /
        # Encoding.validate, which config.validate already calls.
        return None
