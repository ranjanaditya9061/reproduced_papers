from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .photonic import (
    _bunching_score,
    _first_mode_score,
    _majority_score,
    _parity_score,
)
from .spoqc_utils import _sandwich_concrete, parallel_row_map

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

OBSERVABLES = ("parity", "majority", "bunching", "n_first",
               "max_prob", "max_state_sum", "max_prob_state_sum")


def _apply_gap_gate(p, gate_kind, gate_params, j) -> None:
    """Inject the non-Clifford "magic" gate into cluster gap ``j`` on the spin (mode 0).

    ``gate_kind`` selects the injected gate (a *picklable* string, so the whole task
    survives a process pool -- no closures):

    - ``"t"``  : the fixed ``T = P(pi/4)`` (base :class:`SpoqcMagicPhotonicTeacher`).
    - ``"rz"`` : ``Rz(gate_params[j])`` -- e.g. increasing primes (spoqc_magic_prime).
    - ``"u3"`` : a single-qubit unitary ``Rz(phi) Ry(theta) Rz(lam)`` with per-gap angles
                 ``gate_params[j] = (theta, phi, lam)`` (random, spoqc_magic_rand).
    """
    if gate_kind == "t":
        p.gate.t(0)
    elif gate_kind == "rz":
        p.gate.rz(0, float(gate_params[j]))
    elif gate_kind == "u3":
        theta, phi, lam = gate_params[j]
        p.gate.rz(0, float(lam))                        # ZYZ Euler: Rz(phi) Ry(theta) Rz(lam)
        p.gate.ry(0, float(theta))
        p.gate.rz(0, float(phi))
    elif gate_kind == "u3_x":
        theta, phi, lam = gate_params[j]
        p.gate.rz(0, float(lam))                        # ZYZ Euler: Rz(phi) Ry(theta) Rz(lam)
        p.gate.rx(0, float(theta))
        p.gate.rz(0, float(phi))
    elif gate_kind == "u3_x_m":
        theta, phi, lam = gate_params[j]
        p.gate.rz(0, float(lam/5))                        # ZYZ Euler: Rz(phi) Ry(theta) Rz(lam)
        p.gate.rx(0, float(theta/5))
        p.gate.rz(0, float(phi/5))
    elif gate_kind == "u3_x_s":
        theta, phi, lam = gate_params[j]
        p.gate.rz(0, float(lam/20))                        # ZYZ Euler: Rz(phi) Ry(theta) Rz(lam)
        p.gate.rx(0, float(theta/20))
        p.gate.rz(0, float(phi/20))
    else:
        raise ValueError(f"unknown gap gate_kind {gate_kind!r}")


def _build_magic_processor(x, *, m, k, n_features, seed, t_var, gate_kind="t", gate_params=None):

    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    if 2 * k > m:
        raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
    r0, r1 = m, m + 1                                   # readout photon modes
    p = HybridProcessor(num_sources=1, num_modes=m + 2, num_records=m + 2,
                        allow_carry_over=True)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    p.with_initial_source_state(np.outer(plus, plus.conj()))   # spin |+>

    for j in range(k):
        if j < t_var:
            _apply_gap_gate(p, gate_kind, gate_params, j)   # magic: non-Clifford gate in this gap
        p.emit(0, into=(2 * j, 2 * j + 1))             # data photon j+1
        p.gate.h(0)                                     # cluster link (last one = readout rotation)
        
    p.emit(0, into=(r0, r1))                            # readout photon
    p.add(r0, Detector(), record=r0)                   # measure readout in Z ...
    p.add(r1, Detector(), record=r1)                   # ... (r0,r1)=(1,0) -> mu=0

    p.add(0, _sandwich_concrete(m, n_features, seed, x))   # interferometer on data modes
    for md in range(m):
        p.add(md, Detector(), record=md)
    return p, (r0, r1)


def _full_distribution(p):
    """Full detection distribution of a built processor as plain arrays.

    Returns ``(keys, probs)`` where ``keys`` is an ``(n_states, n_modes)`` int
    array of per-mode photon counts (**all** modes, including the two readout
    modes) and ``probs`` the matching ``(n_states,)`` probabilities.  Nothing is
    post-selected or thresholded, so the ``mu=0`` selection and *any* observable
    can be recomputed offline from the saved arrays.
    """
    items = list(p.probabilities().items())
    if not items:
        return np.zeros((0, 0), dtype=np.int16), np.zeros(0, dtype=np.float64)
    n_modes = len(items[0][0])
    keys = np.array([[int(key[i]) for i in range(n_modes)] for key, _ in items],
                    dtype=np.int16)
    probs = np.array([pr for _, pr in items], dtype=np.float64)
    return keys, probs


def _score_distribution(keys, probs, readout_modes, *, m, k, observable) -> float:
    """Post-select ``mu=0`` on the readout modes and score the data modes.

    Post-selecting ``mu=0`` yields exactly the feedforward-corrected deterministic
    input, so ``soft`` is a well-defined function of ``x`` (the discarded ``mu=1``
    mass is the other, Clifford-equivalent branch).  Pure-array form of
    :func:`_score_magic`; also re-scores a saved distribution (see
    :func:`score_from_distribution`).  ``probs`` is the ``(n_states,)`` row aligned
    to the ``(n_states, n_modes)`` ``keys`` basis.
    """
    r0, r1 = readout_modes
    if keys.size == 0:
        return 0.0
    keep = (keys[:, r0] == 1) & (keys[:, r1] == 0)     # readout photon in mode r0
    den = float(probs[keep].sum())
    if den <= 1e-12:
        return 0.0
    ksel, psel = keys[keep], probs[keep]
    if observable in ("max_prob", "max_state_sum", "max_prob_state_sum"):
        # Modal-outcome observables: pick the single most-likely post-selected state.
        j = int(psel.argmax())
        max_prob = float(psel[j] / den)          # its probability, conditional on mu=0
        state_sum = float(np.arange(m) @ ksel[j][:m])   # index-weighted sum sum_i i*n_i over data modes
        if observable == "max_prob":
            return max_prob
        if observable == "max_state_sum":
            return state_sum
        return max_prob * state_sum              # max_prob_state_sum: product of the two
    
    if observable == "parity":
        modes = tuple([i*2 for i in range((m + 1) // 2)])
        sc = np.fromiter((_parity_score(row, modes) for row in ksel), float, len(ksel))
    elif observable == "majority":
        sc = np.fromiter((_majority_score(row, m, k) for row in ksel), float, len(ksel))
    elif observable == "n_first":
        sc = np.fromiter((_first_mode_score(row) for row in ksel), float, len(ksel))
    else:  # bunching
        sc = np.fromiter((_bunching_score(row) for row in ksel), float, len(ksel))
    # print(float((psel * sc).sum()), den)
    return float((psel * sc).sum() / den)


def _score_magic(p, readout_modes, *, m, k, observable) -> float:
    """Post-select the readout photon on ``mu=0`` and score the data modes."""
    keys, probs = _full_distribution(p)
    return _score_distribution(keys, probs, readout_modes, m=m, k=k, observable=observable)


# --- per-row parallelism (each row is an independent perceval simulation) ---- #

def _magic_row_worker(task):
    """Simulate one input row: ``(score, keys, probs)`` (arrays dropped if not captured).

    Module-level and pure (no shared state) so it is picklable for a process pool
    and safe to run concurrently -- each call reseeds its own circuit deterministically
    via ``_build_magic_processor``, so results are independent of worker/order.
    """
    row, m, k, t_var, n_features, seed, observable, capture, gate_kind, gate_params = task
    p, ro = _build_magic_processor(row, m=m, k=k, t_var=t_var, n_features=n_features, seed=seed,
                                   gate_kind=gate_kind, gate_params=gate_params)
    keys, probs = _full_distribution(p)
    score = _score_distribution(keys, probs, ro, m=m, k=k, observable=observable)
    return (score, keys, probs) if capture else (score, None, None)


# --- persisting the full distribution for offline re-scoring ---------------- #

def _merge_supports(basis, prob_rows, keys, probs):
    """Extend a shared key ``basis`` with any new Fock states, then align a new row.

    Fast path in :meth:`SpoqcMagicPhotonicTeacher._record` handles the usual case
    (every input shares the same support in the same order); this is the rare
    fallback when perceval prunes a different negligible-probability set per row.
    """
    index = {tuple(int(v) for v in row): i for i, row in enumerate(basis)}
    fresh = [tuple(int(v) for v in row) for row in keys]
    new = [t for t in dict.fromkeys(fresh) if t not in index]
    if new:
        basis = np.vstack([basis, np.array(new, dtype=np.int16)])
        pad = np.zeros(len(new))
        prob_rows = [np.concatenate([r, pad]) for r in prob_rows]
        for t in new:
            index[t] = len(index)
    vec = np.zeros(len(basis))
    for t, pr in zip(fresh, probs):
        vec[index[t]] = pr
    return basis, prob_rows + [vec]


def write_distributions(path, dist: dict):
    """Persist a distribution dict to ``path`` as a compressed ``.npz``; returns the path.

    ``dist`` is the shape produced by :meth:`SpoqcMagicPhotonicTeacher.captured_distributions`,
    :func:`load_distributions` or :func:`merge_distributions`: ``keys``
    ``(n_states, n_modes)`` shared basis, ``probs`` ``(n_rows, n_states)`` per-input
    matrix, plus the scalar metadata (``m``, ``k``, ``readout_modes``, ``observable``,
    ``t_var``, ``seed``) needed to re-score offline.  Reload with
    :func:`load_distributions`, re-score with :func:`score_from_distribution`.
    """
    import os

    path = os.fspath(path)
    np.savez_compressed(
        path,
        keys=np.asarray(dist["keys"], dtype=np.int16),
        probs=np.asarray(dist["probs"], dtype=np.float64),
        m=int(dist["m"]), k=int(dist["k"]),
        readout_modes=np.asarray(dist["readout_modes"], dtype=np.int64),
        observable=str(dist["observable"]),
        t_var=-1 if dist.get("t_var") is None else int(dist["t_var"]),
        seed=-1 if dist.get("seed") is None else int(dist["seed"]),
    )
    return path


def merge_distributions(old: dict, new: dict) -> dict:
    """Concatenate two distribution dicts row-wise onto a shared key basis.

    ``old`` rows come first, then ``new`` -- used to extend a saved pool by only
    simulating the added rows.  Takes the union of the two (possibly pruned) Fock
    supports and zero-fills, so files with slightly different supports still merge.
    Scalar metadata must match.
    """
    for f in ("m", "k", "observable", "t_var", "readout_modes"):
        if old[f] != new[f]:
            raise ValueError(f"cannot merge distributions: {f} differs "
                             f"({old[f]!r} != {new[f]!r})")
    basis = [tuple(int(v) for v in row) for row in old["keys"]]
    index = {t: i for i, t in enumerate(basis)}
    for row in new["keys"]:
        t = tuple(int(v) for v in row)
        if t not in index:
            index[t] = len(basis)
            basis.append(t)
    basis_arr = np.array(basis, dtype=np.int16)

    def _remap(d):
        p = np.atleast_2d(np.asarray(d["probs"], dtype=np.float64))
        out = np.zeros((p.shape[0], len(basis_arr)))
        if p.shape[0]:
            cols = [index[tuple(int(v) for v in row)] for row in d["keys"]]
            out[:, cols] = p
        return out

    return {**old, "keys": basis_arr, "probs": np.vstack([_remap(old), _remap(new)])}


def load_distributions(path) -> dict:
    """Load a distribution written by :func:`write_distributions`.

    Returns a dict with ``keys`` ``(n_states, n_modes)``, ``probs``
    ``(n_rows, n_states)``, ``readout_modes`` and the scalar metadata
    (``m``, ``k``, ``observable``, ``t_var``, ``seed``).
    """
    import os

    with np.load(os.fspath(path), allow_pickle=False) as z:
        return {
            "keys": z["keys"],
            "probs": z["probs"],
            "readout_modes": tuple(int(v) for v in z["readout_modes"]),
            "m": int(z["m"]), "k": int(z["k"]),
            "observable": str(z["observable"]),
            "t_var": int(z["t_var"]),
            "seed": int(z["seed"]),
        }


def score_from_distribution(dist, observable: str | None = None):
    """Re-score a loaded distribution (dict from :func:`load_distributions`).

    ``observable`` defaults to the stored one; pass another of :data:`OBSERVABLES`
    to re-score the *same* saved distribution under a different measurement -- the
    whole point of persisting the full distribution.  Returns ``(n_rows,)`` scores.
    """
    obs = dist["observable"] if observable is None else observable
    if obs not in OBSERVABLES:
        raise ValueError(f"observable must be one of {OBSERVABLES}, got {obs!r}")
    keys, probs, ro = dist["keys"], dist["probs"], dist["readout_modes"]
    m, k = dist["m"], dist["k"]
    return np.array(
        [_score_distribution(keys, probs[i], ro, m=m, k=k, observable=obs)
         for i in range(probs.shape[0])],
        dtype=np.float64,
    )


class SpoqcMagicPhotonicTeacher(Teacher):

    name = "spoqc_magic_photonic"

    #: number of non-Clifford T gates (magic) if ``t_var`` is not given in the config.
    T_VAR = 1

    #: gap-gate spec (picklable): base injects the fixed ``T``.  Subclasses override
    #: in ``__init__`` (see spoqc_magic_prime / spoqc_magic_rand).
    Gate_kind = "t"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, t_var: int | None = None, gate_kind: str | None = None,
                 n_jobs: int = 1):
        super().__init__(n_features)
        if observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
        if m % 2:
            raise ValueError("spoqc_photonic uses dual-rail photons -> requires even m")
        if k < 2:
            raise ValueError("magic teleportation needs k >= 2 data photons (T sits before emit-2)")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        self.m, self.k, self.observable, self.seed = m, k, observable, int(seed)
        self.t_var = self.T_VAR if t_var is None else int(t_var)
        print(gate_kind)
        self.gate_kind = self.Gate_kind if gate_kind is None else str(gate_kind)
        print(self.gate_kind)
        if not 0 <= self.t_var <= k:
            raise ValueError(f"t_var must be in [0, k]={0, k} (got {self.t_var})")
        self.n_jobs = int(n_jobs)         # 1=serial, -1=auto (CPUs-1), N=explicit workers
        self.gate_params = None           # per-gap angles for the injected gate (subclasses set)
        self._capture = False
        self._dist_keys = None            # shared (n_states, n_modes) key basis
        self._dist_probs: list = []       # per-row prob vectors, aligned to _dist_keys

    @property
    def readout_modes(self) -> tuple[int, int]:
        return (self.m, self.m + 1)

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every row's full Fock distribution during :meth:`forward`.

        Off by default (adds memory + a per-row alignment cost); the generator
        turns it on when ``generation.save_dist`` is set so the distributions can
        be persisted next to the dataset (:func:`save_distributions`) and
        re-scored offline under any observable (:func:`score_from_distribution`).
        """
        self._capture = bool(enable)
        self._dist_keys = None
        self._dist_probs = []

    def _record(self, keys, probs) -> None:
        if self._dist_keys is None:
            self._dist_keys, self._dist_probs = keys, [probs]
        elif keys.shape == self._dist_keys.shape and np.array_equal(keys, self._dist_keys):
            self._dist_probs.append(probs)             # same support & order (usual case)
        else:                                          # rare: perceval pruned a different set
            self._dist_keys, self._dist_probs = _merge_supports(
                self._dist_keys, self._dist_probs, keys, probs)

    def captured_distributions(self) -> dict:
        """The distributions recorded so far as a dict (same shape as :func:`load_distributions`)."""
        if self._dist_keys is None:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        probs = (np.vstack(self._dist_probs) if self._dist_probs
                 else np.zeros((0, self._dist_keys.shape[1])))
        return {"keys": self._dist_keys, "probs": probs,
                "readout_modes": self.readout_modes, "m": self.m, "k": self.k,
                "observable": self.observable, "t_var": self.t_var, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        return write_distributions(path, self.captured_distributions())

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xn = X.detach().cpu().numpy()
        cap = self._capture
        print(self.gate_kind)
        print(self.t_var)
        tasks = [(row, self.m, self.k, self.t_var, self.n_features, self.seed,
                  self.observable, cap, self.gate_kind, self.gate_params) for row in Xn]
        # parallel_row_map preserves input order -> _record below stays row-aligned
        # even though rows finish out of order across the pool.
        results = parallel_row_map(_magic_row_worker, tasks, self.n_jobs)

        vals = []
        for score, keys, probs in results:
            vals.append(score)
            if cap:
                self._record(keys, probs)                    # ordered append (row order)
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

    def render(self, path: str, x=None) -> str:
        import matplotlib.pyplot as plt

        xv = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        p, _ = _build_magic_processor(xv, m=self.m, k=self.k, t_var=self.t_var,
                                      n_features=self.n_features, seed=self.seed,
                                      gate_kind=self.gate_kind, gate_params=self.gate_params)
        p.pdisplay_hybrid(compact=False)
        plt.savefig(path)
        plt.close("all")
        return path

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "SpoqcMagicPhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed,
                   t_var=cfg.problem.t_var, gate_kind = cfg.problem.gate_kind, n_jobs=cfg.generation.n_jobs)

    @classmethod
    def _prep_tag(cls, cfg: "ExperimentConfig", t_var: int, gate_kind:str) -> str:
        """Prep string folded into the dataset hash; overridden by each gap-gate variant."""
        return f"magic_{gate_kind}_{t_var}_emitter_train_postselect_mu0"

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        # t_var (# of injected magic gates) is a spoqc_magic-only knob -> only this
        # teacher's hash sees it; it identifies the dataset alongside the prep tag.
        t_var = cls.T_VAR if cfg.problem.t_var is None else int(cfg.problem.t_var)
        gate_kind = cls.Gate_kind if cfg.problem.gate_kind is None else str(cfg.problem.gate_kind)
        return {"observable": cfg.problem.observable, "prep": cls._prep_tag(cfg, t_var, gate_kind)}
