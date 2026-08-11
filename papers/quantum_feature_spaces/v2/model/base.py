"""``DistributionModel``: every model maps ``X`` to a distribution over a labelled outcome basis.

    probs = model.probs(X)          # (N, n_outcomes), rows sum to 1
    keys  = model.outcome_keys()    # n_outcomes occupation tuples, aligned to the columns

**A model returns a distribution, not a score.**  That is the change that makes the artifact
observable-independent: the readout is applied afterwards by :mod:`v2.observable` and cached
separately, so one simulation serves every observable.  In the legacy pipeline ``forward``
returned ``soft``, the observable was resolved *inside* the teacher, and it rode in the dataset
hash -- so ``parity`` and ``osc`` on an identical circuit cost two independent boson-sampling runs.

**This base class owns everything the four legacy Fock teachers each had a private copy of.**
``PhotonicTeacher``, ``FermionPhotonicTeacher``, ``MlpFockTeacher`` and ``EbmFockTeacher``
separately implemented forward batching, shot sampling, ``enable_distribution_capture``,
``captured_distributions``, ``save_distributions``, ``exact_probs_at_zero`` and observable
resolution -- ~120 near-identical lines, four times.  Here a subclass implements exactly two
things: :meth:`_probs` and :meth:`circuit_spec`.

Subclasses must **not** put an observable in :meth:`circuit_spec`.  That is enforced by
:mod:`v2.pipeline.artifact`, which raises if one appears, because it is the single mistake that
would reintroduce the coupling this rewrite removes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

#: Peak forward memory target, in fp32 elements (~128 MB), used to auto-size ``forward_batch``.
#: The per-call ``(N, n_out)`` matrix blows up for large ``(m, k)`` -- ``m=14, k=7`` is
#: ``n_out = 77520``, ~3 GB at ``N = 1e4`` -- and merlin's complex intermediates push peak higher,
#: so the row-chunking keeps peak at the chunk rather than the whole pool.
FORWARD_ELEMENTS = 33_554_432

#: name -> DistributionModel subclass, populated automatically on subclassing.
MODELS: dict[str, type["DistributionModel"]] = {}


class DistributionModel(nn.Module):
    """``X (N, n_features) -> probs (N, n_outcomes)`` over a fixed labelled outcome basis.

    An ``nn.Module`` so fixed weights round-trip through ``state_dict`` (saved as ``circuit.pt``);
    reproducibility is also guaranteed by construction, since the same config plus ``model_seed``
    rebuilds an identical model.

    Subclass contract -- implement these:

    * :meth:`_probs` -- the map, for one row-chunk.  **Do not decorate it with
      ``torch.no_grad``**: :mod:`v2.metrics` differentiates it w.r.t. ``X``.  The no-grad fast
      path for generation lives on :meth:`probs`, which callers use instead.
    * :meth:`outcome_keys` -- the basis the columns align to.
    * :meth:`circuit_spec` -- identity knobs, **never** including an observable.
    * :meth:`n_model_parameters` -- the parameter count, so the param-matching claims in the
      module docstrings are checkable rather than asserted.
    """

    name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            MODELS[cls.name] = cls

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42):
        super().__init__()
        self.m, self.k = int(m), int(k)
        self.n_features = int(n_features)
        self.seed = int(seed)
        self._capture = False
        self._captured: list = []
        #: Rows per chunk.  Auto-sized once the basis is known; ``<= 0`` disables chunking.
        self.forward_batch = 1

    # --- subclass hooks --------------------------------------------------------------------- #

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        """``(N_chunk, n_outcomes)`` distribution.  Differentiable in ``X`` where possible."""
        raise NotImplementedError

    def outcome_keys(self):
        """The outcome basis: a sequence of per-mode occupation tuples, one per column."""
        raise NotImplementedError

    def circuit_spec(self) -> dict:
        """Identity knobs for the artifact hash -- the circuit, and nothing about the readout."""
        raise NotImplementedError

    def n_model_parameters(self) -> int:
        """Parameters in the model proper, excluding scaffolding (feature tables, bases)."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg) -> "DistributionModel":
        raise NotImplementedError

    @classmethod
    def validate_config(cls, cfg) -> None:
        """Model-specific config checks, called by :meth:`v2.config.ExperimentConfig.validate`."""
        return None

    # --- shared machinery ------------------------------------------------------------------- #

    @property
    def n_outcomes(self) -> int:
        return len(self.outcome_keys())

    def readout_modes(self) -> tuple:
        """Modes reserved for a post-selection readout; ``()`` for most models."""
        return ()

    def input_state(self):
        """The photon injection, when the model has one (``None`` for the classical maps)."""
        return None

    def _autosize_batch(self, n_out: int) -> None:
        """Size the row-chunk so the per-call ``(chunk, n_out)`` matrix stays near 128 MB."""
        self.forward_batch = max(1, FORWARD_ELEMENTS // max(int(n_out), 1))

    def probs(self, X: torch.Tensor, *, grad: bool = False) -> torch.Tensor:
        """``(N, n_outcomes)`` distribution, row-chunked; the normal entry point.

        ``grad=False`` (the default) runs under ``no_grad`` -- the generation fast path.
        ``grad=True`` keeps the graph so :mod:`v2.metrics` can take input-Jacobians; note the
        chunking still applies, and a chunked call under ``grad=True`` concatenates chunk graphs,
        which is correct but holds them all live, so metrics pass small batches.

        **Always the EXACT distribution.**  Finite-shot sampling is not applied here: it is a
        parallel branch owned by :mod:`v2.pipeline.shots`, not a mode of the model.  Two reasons,
        both structural.  (1) Shots and the full distribution are *siblings*, not parent and child
        -- perceval implements them with disjoint backends (``CliffordClifford2017`` has
        ``sample``/``samples`` and no ``all_prob``; ``SLOS``/``Naive`` the reverse), and at large
        ``(m, k)`` the exact distribution cannot be computed at all, so shots cannot be defined as
        a readout of it.  (2) Sampling inside a row-chunked ``forward`` made the draw depend on the
        chunk boundary, which is a memory-tuning knob: the same config produced different data when
        ``FORWARD_ELEMENTS`` changed, with no change of hash.  Blocked substreams keyed on the
        absolute row offset fix that by construction.
        """
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            bs = self.forward_batch
            if bs is None or bs <= 0 or X.shape[0] <= bs:
                return self._forward_chunk(X)
            # Row order is preserved, so the result is identical to an unbatched call.
            chunks = [self._forward_chunk(X[i:i + bs]) for i in range(0, X.shape[0], bs)]
            return torch.cat(chunks, dim=0)

    def _forward_chunk(self, X: torch.Tensor) -> torch.Tensor:
        probs = self._probs(X)
        if self._capture:
            self._captured.append(probs.detach().cpu().numpy())
        return probs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`probs` -- a model's output *is* the distribution."""
        return self.probs(X)

    # --- the shots branch (opt-in, per model) ------------------------------------------------- #

    #: Whether this model can produce finite-shot draws.  **False by default**: shots are a
    #: capability a model has to earn, not something the base class can fake for it.  Drawing
    #: multinomially from a stored exact ``p`` is always *arithmetically* possible, but it is only
    #: the right shot model where sampling is the physically meaningful scaling path -- boson
    #: sampling.  For the other families the exact distribution IS the object of study, and a
    #: multinomial wrapper would suggest a scaling route that does not exist.
    supports_shots: bool = False

    def shot_counts(self, X: torch.Tensor, *, shots: int, offset: int = 0, rows=None,
                    shot_seed: int = 0) -> list[list[tuple]]:
        """One **shot sequence** (occupation tuples, in draw order) per requested row.

        **A sequence, not a vector over the declared basis.**  A shot draw observes at most
        ``shots`` outcomes however large the basis is, so returning a dense ``(N, n_outcomes)``
        array would reinstate the ``C(m+k-1, k)`` memory wall inside the one code path built to
        avoid it.  :meth:`outcome_keys` is not consulted -- the observed keys come from the draw
        itself.  Aggregate to counts with :func:`~pipeline.shots.to_counts` where scoring wants
        them; :func:`~pipeline.shots.to_index` gives the sorted key table plus an index array.

        **Order, so a budget is extended not redrawn.**  ``shots`` is how many *new* shots to draw
        and ``offset`` how many are already stored -- the stream is seeded on ``offset``
        (:func:`~pipeline.shots.offset_seed`), so ``draw(10k, offset=0) + draw(20k, offset=10k)``
        extends a 10k store to 30k without touching or reproducing the first 10k.

        ``X`` is always the **full pool**; ``rows`` selects which row indices to draw, defaulting
        to all. The result has one sequence per *requested* row, in the order requested -- so
        extending the pool is a list concatenation.

        Implemented only where sampling is the meaningful path; see :attr:`supports_shots`.
        """
        raise NotImplementedError(
            f"model {type(self).__name__} does not support finite-shot draws -- it is a "
            "probability-distribution model only. Shots are implemented for boson sampling "
            "(kind='photonic', prep='fock'), where sampling is the scaling path past an enumerable "
            "outcome basis. Use probs() for the exact distribution."
        )

    @torch.no_grad()
    def probs_at_zero(self) -> torch.Tensor:
        """``q``: the ``(n_outcomes,)`` distribution with every encoded feature set to zero.

        The circuit's *unencoded* pattern -- the fixed reference the ``xent`` family scores
        against.  A function of the circuit alone, so it is reproducible from the seed and carries
        no sampling noise.

        Persisted in the artifact (:mod:`v2.pipeline.artifact`), which is what makes ``xent``
        offline-re-scorable -- in the legacy pipeline it was not stored, so an offline re-score
        needed a matched-seed teacher rebuilt by hand.  Note this is also why ``xent`` is the one
        observable **unavailable** once the basis cannot be enumerated: it needs a full exact ``q``,
        where every other family only needs ``v`` evaluated on the outcomes actually observed.
        """
        capture, self._capture = self._capture, False
        try:
            return self._probs(torch.zeros(1, self.n_features))[0]
        finally:
            self._capture = capture

    # --- distribution capture ---------------------------------------------------------------- #

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every chunk's distribution so the pool can be persisted."""
        self._capture = bool(enable)
        self._captured = []

    def captured_distributions(self) -> dict:
        """The recorded distributions, in the shape :mod:`v2.pipeline.distribution` writes.

        NOTE there is no ``observable`` field, unlike the legacy ``.npz``: the whole point is that
        a saved distribution is readout-agnostic.  ``probs_at_zero`` and ``input_state`` travel
        with it so every observable -- including ``xent`` and ``single_output``, which the legacy
        format could not re-score -- can be applied offline.
        """
        if not self._captured:
            raise RuntimeError("no distributions captured; call enable_distribution_capture() "
                               "before probs()")
        keys = np.asarray([[int(c) for c in key] for key in self.outcome_keys()], dtype=np.int16)
        return {
            "keys": keys,
            "probs": np.vstack(self._captured).astype(np.float32),
            "probs_at_zero": self.probs_at_zero().detach().cpu().numpy().astype(np.float32),
            "m": self.m, "k": self.k,
            "readout_modes": tuple(int(v) for v in self.readout_modes()),
            "input_state": None if self.input_state() is None
            else [int(v) for v in self.input_state()],
            "seed": self.seed,
        }


def build_model(cfg) -> DistributionModel:
    """Instantiate the model named by ``cfg.model.kind``."""
    name = cfg.model.kind
    if name not in MODELS:
        raise ValueError(f"unknown model kind {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name].from_config(cfg)
