"""Embedding feature maps: turn a dataset ``X`` into a real feature matrix.

An *embedding* is a feature map ``features(X) -> (N, d)`` plus the params that
identify it for caching.  The kernel applied on top is always a Gaussian RBF
(:mod:`kernel`), so an embedding is fully described by *how the features are
produced*:

- ``rbf``              -- the raw angles (Huang's universal baseline).
- ``fourier_rbf``      -- Fourier features ``[sin(jx), cos(jx)]`` (periodicity-aware).
- ``qubit_projected``  -- Huang projected kernel: 1-qubit reduced-state Pauli
  expectations (``<X_q>, <Y_q>, <Z_q>`` -> ``3n``), from :class:`model.qubit.QubitFeatureMap`.
- ``photonic_projected`` -- per-mode occupation moments up to 2nd order
  (``<n_i>`` + ``<n_i n_j>`` -> ``m(m+1)/2``), from :class:`model.photonic.PhotonicFeatureMap`.

The fidelity kernels (``|<phi|phi'>|^2``) were removed: they are not realizable on
real hardware and were the only thing that needed an N x N Gram stored on disk.

Subclass with a ``name`` to auto-register in :data:`EMBEDDINGS`.
"""

from __future__ import annotations

import math

import torch

from model.mlp import fourier_features
from model.photonic import PhotonicFeatureMap
from model.qubit import QubitFeatureMap

#: name -> Embedding subclass, populated on subclassing.
EMBEDDINGS: dict[str, type["Embedding"]] = {}


def projected_features(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """1-qubit reduced-state Pauli expectations: ``(N, 3n)`` real features.

    Columns are ``[<X_0..X_{n-1}>, <Y_0..>, <Z_0..>]``.
    """
    n = n_qubits
    dev = states.device
    z = torch.arange(2 ** n, device=dev)
    bits = (z.unsqueeze(-1) >> torch.arange(n, device=dev)) & 1   # (2^n, n)
    signs = (1 - 2 * bits).float()                                # Z eigenvalues

    probs = (states.conj() * states).real                         # (N, 2^n)
    Z = probs @ signs                                             # (N, n)  <Z_q>

    X_cols, Y_cols = [], []
    for q in range(n):
        partner = z ^ (1 << q)
        cross = states.conj() * states[:, partner]                # (N, 2^n)
        X_cols.append(cross.real.sum(dim=1))                      # <X_q>
        Y_cols.append((-1j * signs[:, q] * cross).real.sum(dim=1))  # <Y_q>
    Xf = torch.stack(X_cols, dim=1)
    Yf = torch.stack(Y_cols, dim=1)
    return torch.cat([Xf, Yf, Z], dim=1)                          # (N, 3n)


class Embedding:
    name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            EMBEDDINGS[cls.name] = cls

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """The real feature matrix ``(N, d)`` for ``X``."""
        raise NotImplementedError

    def check_input(self, X: torch.Tensor) -> None:
        """Validate ``X``'s width against this map's expected input dim (no-op by default)."""
        return None

    def spec(self) -> dict:
        """Identifying params for the cache key (override to add knobs)."""
        return {"name": self.name}

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "Embedding":
        """Build from an embedding-config entry.

        Geometry (``m``/``k``/``n_features``) is read from ``spec`` if present, else
        inherited from the dataset's ``cfg`` -- so a fully-specified spec needs no
        dataset config, while an omitted one adapts to whatever dataset it's run on.
        """
        raise NotImplementedError


def _geom(spec: dict, cfg, key: str, cfg_value):
    """spec[key] if present, else the dataset cfg's value; error if neither is available."""
    if key in spec and spec[key] is not None:
        return spec[key]
    if cfg is not None:
        return cfg_value()
    raise ValueError(
        f"embedding {spec.get('type')!r} needs {key!r} in its spec when built without a dataset config"
    )


def build_embedding(spec: dict, cfg=None) -> Embedding:
    """Instantiate the embedding named by ``spec['type']``.

    ``cfg`` (a dataset config) is optional: it supplies geometry defaults for any
    of ``m``/``k``/``n_features`` not given directly in ``spec``.
    """
    t = spec.get("type")
    if t not in EMBEDDINGS:
        raise ValueError(f"unknown embedding type {t!r}; registered: {sorted(EMBEDDINGS)}")
    return EMBEDDINGS[t].from_spec(spec, cfg)


class RawEmbedding(Embedding):
    """The raw angle features (Huang's universal RBF baseline)."""

    name = "rbf"

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return X

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "RawEmbedding":
        return cls()


class FourierEmbedding(Embedding):
    """Fourier features ``[sin(jx), cos(jx)]_{j=1..order}`` -- periodicity-aware."""

    name = "fourier_rbf"

    def __init__(self, fourier_order: int = 3):
        self.fourier_order = int(fourier_order)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return fourier_features(X, self.fourier_order)

    def spec(self) -> dict:
        return {"name": self.name, "fourier_order": self.fourier_order}

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "FourierEmbedding":
        return cls(fourier_order=spec.get("fourier_order", 3))


def gaussian_bump_features(X: torch.Tensor, n_bumps: int,
                           x_min: float = 0.0, x_max: float = 2 * math.pi,
                           width: float | None = None) -> torch.Tensor:
    """Localized RBF bumps ``exp(-(x-c_k)^2 / 2s^2)`` at ``n_bumps`` centers spanning
    ``[x_min, x_max]``: angles ``(N, d)`` -> ``(N, d*n_bumps)``.

    The *local* complement to the (global) Fourier basis; ``width`` defaults to the
    center spacing so adjacent bumps overlap.  Centers are fixed (not data-dependent)
    so train and test share the same map.
    """
    centers = torch.linspace(x_min, x_max, n_bumps, device=X.device, dtype=X.dtype)
    s = width if width is not None else (x_max - x_min) / max(n_bumps - 1, 1)
    diff = X.unsqueeze(-1) - centers                              # (N, d, n_bumps)
    return torch.exp(-(diff ** 2) / (2 * s * s)).reshape(X.shape[0], -1)


class ComboEmbedding(Embedding):
    """``[x | Fourier(x) | Gaussian bumps]`` concatenated -- a broad, target-agnostic basis.

    Stacks three complementary views of the angles: the raw ``x`` (linear/identity
    structure), the periodic Fourier features ``[sin(jx), cos(jx)]_{j=1..order}``
    (global oscillations), and localized Gaussian bumps (local structure).  Fed through
    the Nystrom-RBF rank sweep (:func:`learn.svm._fit_score_rank`) this gives the kernel
    both periodic and local resolution, so its RKHS covers a wide range of targets
    without hand-picking a basis per dataset.  Width = ``2*order*d + d + n_bumps*d``.
    """

    name = "combo"

    def __init__(self, fourier_order: int = 4, n_bumps: int = 8,
                 x_min: float = 0.0, x_max: float = 2 * math.pi):
        self.fourier_order = int(fourier_order)
        self.n_bumps = int(n_bumps)
        self.x_min, self.x_max = float(x_min), float(x_max)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            X,                                                   # x itself
            fourier_features(X, self.fourier_order),             # Fourier of x
            gaussian_bump_features(X, self.n_bumps, self.x_min, self.x_max),  # "anti-Fourier"
        ], dim=1)

    def spec(self) -> dict:
        return {"name": self.name, "fourier_order": self.fourier_order,
                "n_bumps": self.n_bumps, "x_min": self.x_min, "x_max": self.x_max}

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "ComboEmbedding":
        return cls(fourier_order=spec.get("fourier_order", 4),
                   n_bumps=spec.get("n_bumps", 8),
                   x_min=spec.get("x_min", 0.0),
                   x_max=spec.get("x_max", 2 * math.pi))


class QubitProjectedEmbedding(Embedding):
    """1-qubit reduced-state features (``3n``); Huang projected kernel."""

    name = "qubit_projected"

    def __init__(self, n_features: int, depth: int, seed: int, lead: bool = True):
        self.n = n_features
        self.fm = QubitFeatureMap(n_features, depth=depth, seed=seed, lead=lead)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return projected_features(self.fm(X), self.n)

    def check_input(self, X: torch.Tensor) -> None:
        if X.shape[1] != self.n:
            raise ValueError(
                f"qubit_projected expects n_features={self.n} but X has width {X.shape[1]}"
            )

    def spec(self) -> dict:
        return {"name": self.name, "n_features": self.n, "depth": self.fm.depth,
                "seed": self.fm.seed, "lead": self.fm.lead}

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "QubitProjectedEmbedding":
        return cls(
            n_features=_geom(spec, cfg, "n_features", lambda: cfg.resolved_n_features),
            depth=_geom(spec, cfg, "depth", lambda: cfg.problem.k),
            seed=spec["seed"], lead=spec.get("lead", True),
        )


class PhotonicProjectedEmbedding(Embedding):
    """Occupation moments up to 2nd order (``m(m+1)/2`` features)."""

    name = "photonic_projected"

    def __init__(self, m: int, k: int, n_features: int, seed: int):
        self.m, self.k, self.n_features, self.seed = m, k, n_features, int(seed)
        self.fm = PhotonicFeatureMap(m, k, n_features, seed)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.fm.probs(X)                                  # (N, n_fock)
        occ = self.fm.occ                                         # (n_fock, m)
        n1 = probs @ occ                                          # <n_i>      (N, m)
        rows, cols = torch.triu_indices(self.m, self.m, offset=1)
        n2 = probs @ (occ[:, rows] * occ[:, cols])               # <n_i n_j>  (N, m(m-1)/2)
        return torch.cat([n1, n2], dim=1)                         # (N, m(m+1)/2)

    def check_input(self, X: torch.Tensor) -> None:
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"photonic_projected expects n_features={self.n_features} but X has width {X.shape[1]}"
            )

    def spec(self) -> dict:
        return {"name": self.name, "m": self.m, "k": self.k,
                "n_features": self.n_features, "seed": self.seed}

    @classmethod
    def from_spec(cls, spec: dict, cfg=None) -> "PhotonicProjectedEmbedding":
        return cls(
            m=_geom(spec, cfg, "m", lambda: cfg.problem.m),
            k=_geom(spec, cfg, "k", lambda: cfg.problem.k),
            n_features=_geom(spec, cfg, "n_features", lambda: cfg.resolved_n_features),
            seed=spec["seed"],
        )