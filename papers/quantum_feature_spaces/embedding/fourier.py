"""Learner-side Fourier featurisation -- deliberately separate from any teacher's encoding.

This is the basis the *learners* use (``fourier_rbf`` / ``combo`` embeddings,
:mod:`learn.linreg`, :mod:`learn.mkl`).  It is a standalone copy on purpose: the near-identical
:func:`model.mlp.fourier_features` is part of what three **teachers** compute
(:class:`~model.mlp.MLPTeacher`, :class:`~model.mlp_fock.MlpFockTeacher`,
:class:`~model.ebm_fock.EbmFockTeacher` all call it inside ``forward``), so editing that one
redefines the labels.  Editing this one cannot.

**Why the split matters, concretely.**  When the two were the same function, changing it moved the
learner's features *and* the classical teachers' encoding together -- and since
``mlp_fock``/``ebm_fock`` labels are functions of ``f(x)`` alone, handing the learner that same
``f(x)`` removes the whole decode-the-encoding half of their task.  Measured at ``m=6, k=3`` on
``parity``, test R^2 under three learner bases:

===============  ==========  =============
labels           raw x       matched f(x)
===============  ==========  =============
photonic            0.785        0.606
ebm_fock            0.689        0.957
===============  ==========  =============

The classical control gains ~0.27 and the quantum map *loses* 0.18 from the same change, so a
matched basis measures representation overlap rather than learnability.  Keep the two
implementations apart, and prefer ``rbf`` (raw angles) when comparing a teacher against its
classical control.

**Cache safety.**  :data:`FOURIER_FORMULA_VERSION` rides in every embedding ``spec()``, hence in
the on-disk cache key (``embeddings/<dataset_hash>/<name>_<spec_hash>.pt``).  Previously the key
recorded only ``fourier_order``, so editing the formula silently reused features from a different
vintage.  Bump the version whenever the formula changes and stale entries are recomputed instead.
"""

from __future__ import annotations

import torch

#: Bump on ANY change to :func:`fourier_features` semantics.  Rides in the embedding spec, so a
#: bump invalidates cached features rather than silently mixing vintages.
FOURIER_FORMULA_VERSION = 2

#: Harmonic argument modes.  ``"mul"`` is the standard trigonometric basis.  ``"div"`` uses the
#: reciprocal ``j / x`` -- see the caveat on :func:`fourier_features`.
FOURIER_MODES = ("mul", "div")


def harmonics(order: int, j0: int = 1, step: int = 1) -> list[int]:
    """The harmonic indices used: ``j0, j0+step, ..`` (``order`` of them)."""
    if order < 1 or step < 1:
        raise ValueError(f"need order >= 1 and step >= 1 (got order={order}, step={step})")
    return [int(j0) + i * int(step) for i in range(int(order))]


def fourier_features(X: torch.Tensor, order: int, *, mode: str = "mul", j0: int = 1,
                     step: int = 1, eps: float = 1e-6,
                     include_raw: bool = False) -> torch.Tensor:
    """Expand angles ``(N, d)`` into ``[sin(a_j), cos(a_j)]`` for ``order`` harmonics.

    ``mode="mul"``  -> ``a_j = j * x``, the standard periodicity-aware basis.
    ``mode="div"``  -> ``a_j = j / x``, the reciprocal basis.

    ``j0`` / ``step`` choose which harmonics (``j0, j0+step, ...``), so ``order=3, j0=1000``
    gives ``j = 1000, 1001, 1002``.  ``include_raw`` prepends ``x`` itself.  Output width is
    ``2 * order * d`` (``+ d`` with ``include_raw``).

    CAVEAT on ``mode="div"``: ``j / x`` blows up as ``x -> 0`` (``eps`` floors ``|x|`` to keep it
    finite), and with a large ``j0`` the features become effectively a *hash* of ``x`` -- the
    period in ``x`` is ``2*pi*x^2/j``, so at ``j0=1000`` the basis oscillates ~160 times across
    ``[0, 2*pi]`` and a point's nearest neighbour in feature space is unrelated to it in ``x``.
    That destroys any smooth signal: on photonic ``parity`` it took test R^2 from 0.606 to
    -0.001, while leaving a teacher whose own encoding uses the same basis at ~0.96.  Use it for
    deliberate experiments, not as a default.
    """
    if mode not in FOURIER_MODES:
        raise ValueError(f"mode must be one of {FOURIER_MODES}, got {mode!r}")
    if mode == "div":
        # Floor |x| away from 0 while keeping its sign, so j/x stays finite (sin(inf) is NaN).
        base = torch.where(X < 0, -1.0, 1.0) / X.abs().clamp(min=float(eps))
    parts = [X] if include_raw else []
    for j in harmonics(order, j0, step):
        arg = j * X if mode == "mul" else j * base
        parts.append(torch.sin(arg))
        parts.append(torch.cos(arg))
    return torch.cat(parts, dim=1)


def fourier_spec(order: int, *, mode: str = "mul", j0: int = 1, step: int = 1,
                 eps: float = 1e-6, include_raw: bool = False) -> dict:
    """The identifying fields for a Fourier featurisation, for an embedding ``spec()``.

    Every argument that changes the features appears here, plus
    :data:`FOURIER_FORMULA_VERSION` -- so the cache key moves whenever the features do.
    """
    return {"fourier_order": int(order), "fourier_mode": mode, "fourier_j0": int(j0),
            "fourier_step": int(step), "fourier_eps": float(eps),
            "fourier_include_raw": bool(include_raw),
            "fourier_formula": FOURIER_FORMULA_VERSION}


def fourier_embedding_spec(order: int = 3, **kwargs) -> dict:
    """A ready-to-use ``fourier_rbf`` embedding spec (what ``build_embeddings_for`` consumes).

    Takes the short knob names -- ``order``, ``mode``, ``j0``, ``step``, ``eps``, ``include_raw``
    -- and returns the ``{"type": ..., "fourier_*": ...}`` dict, so callers never hand-write the
    prefixed keys.  Every knob lands in the spec, hence in the cache key.
    """
    return {"type": "fourier_rbf", **fourier_spec(order, **kwargs)}


def fourier_kwargs_from_spec(spec: dict, *, default_order: int = 3) -> dict:
    """Read :func:`fourier_features` kwargs back out of an embedding ``spec`` dict."""
    return {"order": int(spec.get("fourier_order", default_order)),
            "mode": spec.get("fourier_mode", "mul"),
            "j0": int(spec.get("fourier_j0", 1)),
            "step": int(spec.get("fourier_step", 1)),
            "eps": float(spec.get("fourier_eps", 1e-6)),
            "include_raw": bool(spec.get("fourier_include_raw", False))}
