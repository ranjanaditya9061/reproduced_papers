"""Photonic sandwich teacher: W1(Haar) -> phase-encode(x) -> W2(Haar) -> measure.

Three pieces, split across modules:

* :mod:`model.photonic_circuit` -- the sandwich circuit, the Merlin ``QuantumLayer`` around it,
  and :class:`~model.photonic_circuit.PhotonicFeatureMap` (the amplitudes/probs embedding the
  photonic kernels use).
* :mod:`model.photonic_observables` -- one module per observable family, all built on a single
  framework: an observable is precomputed from the Fock basis, then maps a full distribution
  ``(N, n_fock)`` to ``(N,)`` scores.
* this module -- :class:`PhotonicTeacher`, which glues them together, plus
  :func:`score_from_distribution`, which re-scores a *saved* distribution through the identical
  observable objects (so the live and offline paths cannot drift).

``PhotonicTeacher.forward`` returns a continuous ``(N, 1)`` score chosen by ``observable``.
Every name the old single-file ``model.photonic`` exported is re-exported here, so existing
imports keep working; new code should import from the two modules above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .photonic_circuit import (PhotonicFeatureMap, build_quantum_layer,  # noqa: F401
                               build_sandwich_circuit, default_input_state)
from .photonic_observables import *  # noqa: F401,F403  (the whole observable surface)
from .photonic_observables import (ObservableContext, is_known_observable, observable_hash_spec,
                                   observable_help, resolve_observable)

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

#: Peak forward memory target, in fp32 elements (~128 MB), used to auto-size ``forward_batch``.
_FORWARD_ELEMENTS = 33_554_432


class PhotonicTeacher(Teacher):
    """``X -> soft``: run the sandwich circuit, score the Fock distribution with ``observable``.

    ``observable`` is any string the registry in :mod:`model.photonic_observables` recognises --
    a plain per-Fock-state scorer, a ``prod_parity`` monomial polynomial, a graph
    (``loop_path_``/``connected_``) reading, or a degree-2 (``sq_``/``pairprod``) form.  The
    graph / angle families need the extra knobs below; the seeds default to ``seed`` so a config
    that pins only ``seed`` still reproduces exactly.
    """

    name = "photonic_quantum"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0,
                 n_vertices: int | None = None, graph_seed: int | None = None,
                 angle_seed: int | None = None, graph_density: float | None = None):
        super().__init__(n_features)
        if not is_known_observable(observable):
            # Fail before building the (expensive) Merlin layer on a typo'd observable.
            raise ValueError(f"unknown observable {observable!r}; expected one of: "
                             f"{observable_help()}")
        self.m, self.k, self.observable, self.nsample = m, k, observable, int(nsample)
        self.seed = int(seed)
        self.n_vertices = None if n_vertices is None else int(n_vertices)
        self.graph_density = None if graph_density is None else float(graph_density)
        self.graph_seed = self.seed if graph_seed is None else int(graph_seed)
        self.angle_seed = self.seed if angle_seed is None else int(angle_seed)
        self._capture = False
        self._dist_probs: list = []       # per-forward (N, n_fock) prob matrices (capture)

        self.layer, self.input_state = build_quantum_layer(
            m, k, n_features, self.seed, measure="probs")
        self._fock_keys = list(self.layer.output_keys)
        self.obs = resolve_observable(observable, ObservableContext(
            m=m, k=k, keys=self._fock_keys, seed=self.seed, graph_seed=self.graph_seed,
            angle_seed=self.angle_seed, n_vertices=self.n_vertices,
            graph_density=self.graph_density, input_state=self.input_state))
        # Row-batch the forward so peak memory scales with the chunk, not the whole pool: the
        # per-forward (N, n_fock) prob matrix (n_fock = C(m+k-1, k)) blows up for large (m, k) --
        # e.g. m=14,k=7 -> n_fock=77520, ~3 GB at N=1e4 in fp32, and merlin's complex amplitudes
        # push peak higher -> the sampler gets OOM-killed.  Auto-size the chunk to ~128 MB per
        # call; override with ``teacher.forward_batch`` (larger if you have RAM, <= 0 to disable).
        self.forward_batch = max(1, _FORWARD_ELEMENTS // max(len(self._fock_keys), 1))

    # --- the observable's precomputed tables, kept reachable on the teacher ---------------- #

    @property
    def score_vec(self) -> torch.Tensor:
        """The observable's ``(n_fock,)`` per-Fock-state score vector (all but ``pairprod``/``max_prob``)."""
        return self.obs.score_vec

    @property
    def edges(self):
        """The observable's fixed seeded graph (``loop_path_``/``connected_`` only)."""
        return self.obs.edges

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bs = self.forward_batch
        if bs is None or bs <= 0 or X.shape[0] <= bs:
            return self._forward_chunk(X)
        # Process row-chunks and concatenate; peak memory stays at (bs, n_fock) instead of
        # (N, n_fock).  Row order is preserved, so the result is identical to an unbatched call.
        scores = [self._forward_chunk(X[i:i + bs]) for i in range(0, X.shape[0], bs)]
        return torch.cat(scores, dim=0)

    @torch.no_grad()
    def _forward_chunk(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.layer.forward(X, shots=self.nsample if self.nsample > 0 else None)
        if self._capture:
            self._dist_probs.append(probs.detach().cpu().numpy())
        return self.obs.score(probs).unsqueeze(-1)              # (N_chunk, 1)

    # --- optional full-distribution capture (parity with spoqc_magic) ---------------------- #

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every forward's full Fock distribution so it can be persisted.

        Off by default; the generator turns it on when ``generation.save_dist`` is set,
        letting the saved ``distributions.npz`` be re-scored offline under a different
        observable / ``loop_vars`` / ``path_vars`` (:func:`score_from_distribution`)
        without re-running the boson sampler.
        """
        self._capture = bool(enable)
        self._dist_probs = []

    def captured_distributions(self) -> dict:
        """Recorded distributions as a dict (same shape as :func:`spoqc_magic.load_distributions`)."""
        if not self._dist_probs:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        keys = np.array([[int(key[i]) for i in range(self.m)] for key in self._fock_keys],
                        dtype=np.int16)
        probs = np.vstack(self._dist_probs)
        return {"keys": keys, "probs": probs, "readout_modes": (),
                "m": self.m, "k": self.k, "observable": self.observable,
                "t_var": None, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        from .spoqc_magic import write_distributions
        return write_distributions(path, self.captured_distributions())

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "PhotonicTeacher":
        p = cfg.problem
        return cls(m=p.m, k=p.k, n_features=cfg.resolved_n_features,
                   observable=p.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample, n_vertices=p.n_vertices,
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed,
                   graph_density=p.graph_density)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        """Dataset identity: the observable string plus whatever its family canonicalises to.

        The family's contribution (:func:`~model.photonic_observables.base.observable_hash_spec`)
        resolves equivalent spellings to one dataset -- e.g. every ``prod_parity`` spec that
        expands to the same monomial set, or ``__L0-1__P2`` vs ``__P2__L0-1``.
        """
        p = cfg.problem
        spec = {"observable": p.observable, "nsample": cfg.generation.nsample}
        spec.update(observable_hash_spec(p.observable, ObservableContext(
            m=p.m, k=p.k, seed=cfg.seeds.teacher_seed, graph_seed=p.graph_seed,
            angle_seed=p.angle_seed, n_vertices=p.n_vertices, graph_density=p.graph_density)))
        return spec


def score_from_distribution(dist, observable: str | None = None, *,
                            n_vertices: int | None = None, loop_vars=None,
                            path_vars=None, graph_seed: int | None = None,
                            angle_seed: int | None = None,
                            graph_density: float | None = None):
    """Re-score a saved photonic distribution (dict from :func:`spoqc_magic.load_distributions`).

    Builds the *same* observable object the teacher would and applies it to the stored probs, so
    an offline re-score is exact by construction.  ``observable`` defaults to the stored one --
    pass another to re-label the same boson-sampling run under a different reading.

    The knobs that are not persisted in the ``.npz`` must be supplied for the families that need
    them: ``n_vertices`` (+ optional ``loop_vars`` / ``path_vars`` / ``graph_seed``) for
    ``loop_path_<base>``, ``graph_density`` (+ ``graph_seed``) for ``connected_<base>``.  Any
    ``__L``/``__P`` suffix in ``observable`` overrides the passed ``loop_vars`` / ``path_vars``, so
    a sweep can vary the selection purely through the observable string.  ``graph_seed`` and
    ``angle_seed`` default to the stored teacher ``seed``.  ``single_output`` is the one observable
    that cannot be re-scored: it needs the input state, which is not persisted.

    Returns ``(n_rows,)`` scores.
    """
    obs_name = dist["observable"] if observable is None else observable
    probs = torch.as_tensor(np.atleast_2d(np.asarray(dist["probs"])), dtype=torch.float32)
    ctx = ObservableContext(
        m=int(dist["m"]), k=int(dist["k"]),
        keys=[tuple(int(v) for v in row) for row in dist["keys"]],
        seed=int(dist["seed"]), graph_seed=graph_seed, angle_seed=angle_seed,
        n_vertices=n_vertices, graph_density=graph_density,
        loop_vars=loop_vars, path_vars=path_vars,
        input_state=None,                    # not persisted -> single_output is unavailable
    )
    return resolve_observable(obs_name, ctx).score(probs).numpy()


# --- back-compat aliases for the pre-split module-private names ---------------------------- #
#
# The old single-file model.photonic exposed these underscore helpers and they are imported
# across the repo (data.photonic_quantum, model.spoqc_utils, learn.*, Generator.config, the
# tests, the *_tmp.py scratch scripts).  They now live in model.photonic_observables under
# public names; these aliases keep the old spellings working.

_default_input_state = default_input_state
_parity_score = parity_score                                            # noqa: F405
_majority_score = majority_score                                        # noqa: F405
_bunching_score = bunching_score                                        # noqa: F405
_first_mode_score = first_mode_score                                    # noqa: F405
_single_output_score = single_output_score                               # noqa: F405
_prod_parity_score = prod_parity_score                                  # noqa: F405
_prod_parity_angle_score = prod_parity_angle_score                      # noqa: F405
_sq_base_vec = sq_base_vec                                              # noqa: F405
_occ_matrix = occ_matrix                                                # noqa: F405
_overlay_counts = overlay_counts                                        # noqa: F405
_graph_tables = graph_tables                                            # noqa: F405
_graph_selection = graph_selection                                      # noqa: F405
_clicked_max_component = clicked_max_component                          # noqa: F405
_connected_scores = connected_scores                                    # noqa: F405
_is_connected = is_connected_graph                                      # noqa: F405
_finish_kernel = finish_kernel                                          # noqa: F405
_parse_prod_segment = parse_prod_segment                                # noqa: F405
