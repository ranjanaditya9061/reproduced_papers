"""Merlin photonic sandwich dataset generator.

Teacher circuit::

    W1 (Haar) → phase encode(x) → W2 (Haar) → photon-number measurement

The number of encoded features ``n_features`` is independent of ``m``:
only the first ``n_features`` modes receive a data-dependent phase shift
(the remaining modes are unencoded).  This lets you use a larger Fock space
(more modes/photons for expressivity) while keeping the input space low-
dimensional for visualisation.  Default: ``n_features = m − 1``.

Four observables are supported (``observable`` argument):

``"parity"`` (default)
    soft = Σ_s P(s) · (−1)^{n_parity(s)},  label = sign.
    n_parity = photon count on first ⌈m/2⌉ modes.

``"majority"``
    soft = E[(n_left − n_right) / k],  label = sign.
    n_left = photons in first m//2 modes, n_right = photons in last m//2 modes.
    Requires even m.  Ties (soft = 0) filtered by ``min_margin > 0``.

``"bunching"``
    soft = P(anti-bunched) − P(bunched),  label = sign.
    Anti-bunched = all photons in distinct modes (max count ≤ 1).
    Related to Hong-Ou-Mandel interference and the boson-sampling permanent.

``"single_output"``
    soft = (P(output = input_state) − P(output = reversed_input_state)) / max_s P(s),
    label = sign.
    Probes the probability of two specific Fock states: the injected input state
    and its mode-reversal.  Each probability is |perm(submatrix of U(x))|² —
    an interference term that can oscillate at high spatial frequency with x.
    The difference is normalised by the highest output probability so that the
    score stays O(1) as m and k grow (raw permanents shrink as 1/dim(Fock)).
    The difference is naturally centred near 0, giving balanced classes.

This module provides:

- :func:`generate_photonic_quantum`  : public entry point registered in
  :data:`data.GENERATORS`.
- circuit builders :func:`build_sandwich_experiment`, :func:`make_quantum_layer`.
- observable utilities :func:`parity_of_key`, :func:`majority_score_of_key`,
  :func:`bunching_score_of_key`, :func:`single_output_score_of_key`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import perceval as pcvl
import merlin as ML

from .base import Dataset
from ._resample import filter_resample


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_sandwich_experiment(
    m: int,
    n_features: int | None = None,
    phase_prefix: str = "x",
) -> tuple[pcvl.Experiment, "np.ndarray", "np.ndarray"]:
    """Build the W1 → P(x) → W2 circuit.

    Only the first ``n_features`` modes receive a data-dependent phase shift.
    Default ``n_features = m − 1`` (all modes except the last, which removes
    the global-phase redundancy).

    Returns
    -------
    experiment : pcvl.Experiment
    W1 : np.ndarray  shape (m, m) complex
    W2 : np.ndarray  shape (m, m) complex
    """
    if n_features is None:
        n_features = m - 1
    if not (1 <= n_features <= m - 1):
        raise ValueError(f"n_features must be in [1, m−1], got {n_features} for m={m}.")

    circuit = pcvl.Circuit(m, name="haar_phase_haar")

    _W1 = pcvl.Matrix.random_unitary(m)
    circuit.add(0, pcvl.Unitary(_W1), merge=True)

    for i in range(n_features):
        circuit.add(i, pcvl.PS(pcvl.P(f"{phase_prefix}{i}")))

    _W2 = pcvl.Matrix.random_unitary(m)
    circuit.add(0, pcvl.Unitary(_W2), merge=True)

    return pcvl.Experiment(circuit), np.asarray(_W1), np.asarray(_W2)


def default_input_state(m: int, k: int) -> list[int]:
    """Inject k photons evenly spaced across m modes.

    Photon i lands on mode round(i * m / k), spreading input across the full
    interferometer so no region is out of the light cone.

    Examples
    --------
    m=4, k=2  →  [1, 0, 1, 0]
    m=6, k=2  →  [1, 0, 0, 1, 0, 0]
    m=6, k=3  →  [1, 0, 1, 0, 1, 0]
    """
    state = [0] * m
    for i in range(k):
        state[round(i * m / k)] = 1
    return state


def parity_of_key(key, parity_modes: Iterable[int]) -> int:
    """Return +1 for even photon number on ``parity_modes``, -1 otherwise."""
    n = sum(int(key[i]) for i in parity_modes)
    return +1 if n % 2 == 0 else -1


def majority_score_of_key(key, m: int, k: int) -> float:
    """Return (n_left − n_right) / k for the majority observable.

    n_left  = photons in modes 0 .. m//2 − 1
    n_right = photons in modes m//2 .. m − 1
    Result ∈ [−1, +1]; 0 means tie (only possible when k is even).
    Requires even m so the two halves are the same size.
    """
    split = m // 2
    n_left  = sum(int(key[i]) for i in range(split))
    n_right = sum(int(key[i]) for i in range(split, m))
    return (n_left - n_right) / k


def bunching_score_of_key(key) -> int:
    """Return +1 if anti-bunched (all photons in distinct modes), −1 if bunched.

    Anti-bunched ⟺ no mode has more than one photon (max count ≤ 1).
    This is the observable related to Hong-Ou-Mandel interference: the
    probability of anti-bunching is given by the permanent of a submatrix
    of the unitary, making it classically hard to compute for large circuits.
    """
    return +1 if max(int(n) for n in key) <= 1 else -1


def single_output_score_of_key(key, input_state: list[int]) -> int:
    """Return +1 if key matches input_state, −1 if it matches reversed input_state, else 0.

    Defines the unnormalised score_vec for the ``"single_output"`` observable.
    Only two Fock states carry non-zero weight; all others contribute 0.
    The raw dot product probs @ score_vec = P(A) − P(B) is then divided by
    max_s P(s) inside ``generate_photonic_quantum`` to keep the score O(1)
    as the Fock-space dimension grows with m and k.
    """
    key_list = [int(key[i]) for i in range(len(input_state))]
    if key_list == list(input_state):
        return +1
    if key_list == list(reversed(input_state)):
        return -1
    return 0


def make_quantum_layer(
    m: int,
    k: int,
    n_features: int | None = None,
    seed: int = 1234,
    input_state: list[int] | None = None,
    phase_prefix: str = "x",
    return_object: bool = False,
) -> tuple[ML.QuantumLayer, dict]:
    """Build the merlin QuantumLayer + a metadata dict (also stores W1, W2)."""
    if n_features is None:
        n_features = m - 1

    experiment, W1, W2 = build_sandwich_experiment(
        m=m, n_features=n_features, phase_prefix=phase_prefix,
    )

    if input_state is None:
        input_state = default_input_state(m, k)

    layer = ML.QuantumLayer(
        input_size=n_features,
        experiment=experiment,
        input_state=input_state,
        input_parameters=[phase_prefix],
        measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        return_object=return_object,
    )

    return layer, {
        "m": m,
        "k": k,
        "n_features": n_features,
        "seed": seed,
        "input_state": input_state,
        "phase_prefix": phase_prefix,
        "teacher_W1": W1,
        "teacher_W2": W2,
    }


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_photonic_quantum(
    size: int,
    m: int,
    k: int,
    n_features: int | None = None,
    seed: int = 1234,
    nsample: int = 0,
    observable: str = "parity",
    parity_modes: Iterable[int] | None = None,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_iter: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    **_,
) -> Dataset:
    """Photonic sandwich dataset from a Merlin circuit.

    Parameters
    ----------
    size : int
        Final number of examples (post-filter, post-balance).
    m : int
        Number of optical modes.
    k : int
        Number of injected photons (1 ≤ k ≤ m).
    n_features : int, optional
        Number of data-encoded phases (input dimension).
        Default: m − 1 (all modes except the last).
        Setting n_features < m − 1 decouples the input dimension from the
        circuit size, e.g. n_features=2 with m=4 gives a 2D feature space
        backed by a richer 4-mode Fock space.
    seed : int
        RNG seed for the Haar matrices and dataset draws.
    nsample : int
        0 ⇒ exact probabilities (default). >0 ⇒ empirical shot probabilities.
    observable : {"parity", "majority", "bunching"}
        ``"parity"``   soft = Σ_s P(s)·(−1)^{n_parity(s)}.
        ``"majority"`` soft = E[(n_left − n_right)/k].  Requires even m.
        ``"bunching"`` soft = P(anti-bunched) − P(bunched).
    parity_modes : iterable of int, optional
        Modes counted for ``observable="parity"``.
        Default: first ⌈m/2⌉ modes.
    balanced : bool
        If True, return size//2 rows per class.
    min_margin : float
        Drop samples with |soft| < min_margin.
    bail_threshold : float
        Abort resampling if first-batch yield is below this fraction.
    max_iter : int
        Maximum resample iterations.
    device, dtype
        Tensor placement / precision.

    Returns
    -------
    Dataset
        X (N, n_features) in [0, 2π], y (N,) binary labels,
        soft_targets (N, 1) in [−1, +1], metadata with W1/W2.
    """
    if not (1 <= k <= m):
        raise ValueError("Require 1 <= k <= m.")
    if nsample < 0:
        raise ValueError("nsample must be >= 0.")
    is_prod = observable.startswith("prod_parity")
    is_sq = observable.startswith("sq_")
    is_pairprod = observable == "pairprod"
    if (observable not in ("parity", "majority", "bunching", "single_output")
            and not is_prod and not is_sq and not is_pairprod):
        raise ValueError(
            f"observable must be 'parity', 'majority', 'bunching', 'single_output', "
            f"'prod_parity[__...]', 'sq_<base>', or 'pairprod', got {observable!r}."
        )
    if observable == "majority" and m % 2 != 0:
        raise ValueError("observable='majority' requires even m.")
    if observable == "sq_majority" and m % 2 != 0:
        raise ValueError("observable='sq_majority' requires even m.")

    if n_features is None:
        n_features = m - 1

    torch.manual_seed(seed)
    pcvl.random_seed(seed)

    layer, metadata = make_quantum_layer(m=m, k=k, n_features=n_features, seed=seed)
    layer = layer.to(device)
    output_keys = list(layer.output_keys)

    pair_kernel = None
    if is_prod:
        # (-1)^P(n): P a sum of square-free monomials in the counts. Reuse the canonical
        # monomial builder/scorer from the teacher so both paths agree exactly (the
        # consecutive variant derives its monomials from (m, k)).
        from model.photonic import _prod_parity_score, prod_family_monomials

        parity_modes = None
        monomials = prod_family_monomials(observable, m, k)
        score_vec = torch.tensor(
            [_prod_parity_score(key, monomials) for key in output_keys],
            dtype=dtype, device=device,
        )
    elif is_sq:
        # Diagonal degree-2 observable Σ_n <base>(n) p(n)^2 (nonlinear). Reuse the teacher's
        # per-Fock-state base scorer so both paths agree exactly.
        from model.photonic import _sq_base_vec, parse_sq_observable

        parity_modes = None
        base = parse_sq_observable(observable)[1]
        score_vec = torch.tensor(
            _sq_base_vec(base, output_keys, m=m, k=k), dtype=dtype, device=device,
        )
    elif is_pairprod:
        # Quadratic form p^T K p with a non-separable ±1 kernel (nonlinear). score_vec unused.
        from model.photonic import PAIRPROD_MAX_FOCK, pairprod_kernel

        parity_modes = None
        n_fock = len(output_keys)
        if n_fock > PAIRPROD_MAX_FOCK:
            raise ValueError(
                f"pairprod builds a dense {n_fock}x{n_fock} kernel (n_fock={n_fock} > "
                f"{PAIRPROD_MAX_FOCK}); lower (m, k) or use a sq_<base> observable"
            )
        pair_kernel = torch.tensor(pairprod_kernel(output_keys, m), dtype=dtype, device=device)
        score_vec = torch.zeros(n_fock, dtype=dtype, device=device)   # unused
    elif observable == "parity":
        if parity_modes is None:
            parity_modes = tuple(range((m + 1) // 2))
        else:
            parity_modes = tuple(parity_modes)
        score_vec = torch.tensor(
            [parity_of_key(key, parity_modes) for key in output_keys],
            dtype=dtype, device=device,
        )
    elif observable == "majority":
        parity_modes = None
        score_vec = torch.tensor(
            [majority_score_of_key(key, m=m, k=k) for key in output_keys],
            dtype=dtype, device=device,
        )
    elif observable == "bunching":
        parity_modes = None
        score_vec = torch.tensor(
            [bunching_score_of_key(key) for key in output_keys],
            dtype=dtype, device=device,
        )
    else:  # single_output
        parity_modes = None
        in_state = metadata["input_state"]
        score_vec = torch.tensor(
            [single_output_score_of_key(key, in_state) for key in output_keys],
            dtype=dtype, device=device,
        )

    def _draw(n: int):
        Xb = 2.0 * np.pi * torch.rand(n, n_features, dtype=dtype, device=device)
        probs = layer.forward(Xb, shots=nsample if nsample > 0 else None)
        if is_pairprod:
            score = (probs @ pair_kernel * probs).sum(dim=1)   # p^T K p (K symmetric ±1)
        elif is_sq:
            score = (probs * probs) @ score_vec                # Σ_n <base>(n) p(n)^2
        else:
            score = probs @ score_vec
        if observable == "single_output":
            # P(A) and P(B) shrink as 1/dim(Fock) with m and k, making the raw
            # difference exponentially small and min_margin meaningless.
            # Normalising by max_s P(s) gives a scale-invariant score in [-1,+1]
            # that stays O(1) regardless of circuit size.
            score = score / probs.max(dim=1).values.clamp(min=1e-10)
        yb = (score >= 0).to(torch.long)
        return Xb, yb, score, score.abs()

    (X, y, soft), info = filter_resample(
        target_size=size,
        balanced=balanced,
        min_margin=min_margin,
        draw_fn=_draw,
        perm_seed=seed + 7,
        low_survival_threshold=bail_threshold,
        max_iter=max_iter,
    )

    if info["n_iters"] > 0:
        print(
            f"  Drew {info['total_raw_drawn']} raw samples to reach {len(X)} "
            f"after filter+balance "
            f"(margin_survival={info['margin_survival']:.1%}, "
            f"balance_yield_initial={info['balance_yield']:.1%}, "
            f"iters={info['n_iters']})."
        )

    metadata.update({
        "generator": "photonic_quantum",
        "observable": observable,
        "n_features": n_features,
        "nsample": nsample,
        "parity_modes": parity_modes,
        "single_output_state_a": metadata["input_state"] if observable == "single_output" else None,
        "single_output_state_b": list(reversed(metadata["input_state"])) if observable == "single_output" else None,
        "balanced": balanced,
        "min_margin": min_margin,
        "total_raw_drawn": info["total_raw_drawn"],
        "class_1_fraction": (
            round(y.float().mean().item(), 4) if len(y) > 0 else float("nan")
        ),
    })

    return Dataset(
        X=X.cpu(),
        y=y.cpu(),
        soft_targets=soft.cpu().unsqueeze(-1),
        metadata=metadata,
    )


if __name__ == "__main__":
    ds = generate_photonic_quantum(size=10000, m=6, k=3, nsample=0, seed=69)
    print("X:", ds.X.shape)
    print("labels:", ds.y[:10])
    print("0:", (ds.y == 0).sum().item(), " 1:", (ds.y == 1).sum().item())
