"""Generate the full size-sweep config matrix into ``configs/size_sweep_full/``.

    python configs/generate_size_sweep.py

Sizes ``(m, k)`` run ``k = m // 2`` from ``(6, 3)`` to ``(12, 6)`` for every model kind, and
``(14, 7)`` to ``(18, 9)`` for ``photonic``/``fock`` only, on the shots branch.

**Why the split.**  The exact ``probs`` matrix is ``(size, C(m+k-1,k))`` float32: at ``(14, 7)`` that
is already ~2.9 GiB for 10k rows, and ``(18, 9)`` is ~116 GiB -- not a budget question, a hard wall.
``check_size`` (:mod:`pipeline.distribution`) refuses generation past ``generation.max_dist_bytes``
(default 2 GiB) for exactly this reason.  Only ``prep="fock"`` supports the shots branch
(:attr:`model.photonic.PhotonicModel.supports_shots`) -- every other kind (``fermion``, ``qubit``,
``spin``, ``spin_magic``, ``quadratic_fock``, ``mlp_fock``, ``mlp``, ``analytical``) has no path past
the exact-``probs`` wall, so they stop at ``(12, 6)``.

**``n_features`` is fixed at 5 across every size**, per the study-invariant design in
:mod:`config`: Fisher spectra are ``n_features x n_features``, so varying the input dimension across
a comparison would make cross-size spectra incomparable.

**Encoding axis (``photonic``/``fock`` only):** ``phase`` (the default, used everywhere else in
this matrix) plus ``bs`` and ``bs_phase`` -- two extra single configs, not crossed with any other
kind.  Both need ``n_features <= m - 1`` (one more mode than feature, vs. ``phase``'s
``n_features <= m``), which holds throughout at ``n_features=5, m>=6``.

**The spin/spin_magic sub-matrix, per size:**

* ``spin`` -- ``cx_pairs: chain``, which :func:`circuit.spin.normalize_cx_pairs` expands to the
  linear ladder ``[(0,1), (1,2), ..., (k-2,k-1)]`` at whatever ``k`` the config has -- the ladder
  itself lives in ``circuit/``, not computed here, so this script only ever writes the literal
  string.  The 3 encode cases (``encode_circuit`` only, ``encode_on_spin`` only, both -- see
  :class:`circuit.prep.SpinPrep`, whose ``encode_circuit`` flag this sweep is what motivated adding)
  are each run at the default ``layers=1``; ``layers=3`` is a separate, single config at the
  default encode case (``encode_circuit`` only) -- the two axes are not crossed, so this is 4
  configs, not the 3x2 product.
* ``spin_magic`` -- the 3 encode cases are each run at the default ``structure="linear"``;
  ``ghz``/``linear_u3`` are two separate, single configs at the default encode case
  (``encode_circuit`` only) -- the ``structure`` and encode-case axes are not crossed, so this is
  5 configs, not the 3x3 product.  ``t_var=0`` throughout (no extra sparse magic-gate injection).
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path(__file__).parent / "size_sweep_full"
N_FEATURES = 5
SIZE = 10_000

#: (m, k) with exact probs -- every model kind runs here.  Start at just (6, 3) to verify the
#: matrix is right before widening; add (8, 4), (10, 5), (12, 6) once confirmed.
EXACT_SIZES = [(6, 3)]
#: (m, k) past the exact-probs memory wall -- photonic/fock only, via the shots branch.  Empty
#: until the EXACT_SIZES tier is verified.
SHOTS_SIZES: list[tuple[int, int]] = []

BASE = {"split": {"test_fraction": 0.2, "split_seed": 0},
       "seeds": {"sample_seed": 42, "model_seed": 42}}

#: The 3 encode cases shared by spin and spin_magic: (suffix, {model kwargs}).
ENCODE_CASES = [
    ("enccirc", {"encode_circuit": True, "encode_on_spin": False}),
    ("encspin", {"encode_circuit": False, "encode_on_spin": True}),
    ("encboth", {"encode_circuit": True, "encode_on_spin": True}),
]


def _write(name: str, problem: dict, model: dict, generation: dict) -> Path:
    cfg = {"problem": problem, "model": model, "generation": generation, **BASE}
    path = OUT_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    return path


def configs_for_size(m: int, k: int, *, shots: int = 0) -> list[Path]:
    """Every config at one (m, k): the 7 single-variant kinds, plus the spin/spin_magic matrix."""
    problem = {"n_features": N_FEATURES, "m": m, "k": k}
    generation = {"size": SIZE} if not shots else {"size": SIZE, "shots": shots}
    tag = f"m{m}k{k}"
    paths = []

    single = [
        ("photonic_fock", {"kind": "photonic", "prep": "fock", "encoding": "phase"}),
        ("fermion", {"kind": "fermion", "encoding": "phase"}),
        ("qubit", {"kind": "qubit", "encoding": "phase"}),
        ("quadratic_fock", {"kind": "quadratic_fock", "encoding": "phase", "param_matched": True}),
        ("mlp_fock", {"kind": "mlp_fock", "encoding": "phase"}),
        ("mlp", {"kind": "mlp", "encoding": "phase"}),
        ("analytical", {"kind": "analytical", "encoding": "phase"}),
    ]
    for name, model in single:
        if shots and name != "photonic_fock":
            continue                            # only fock supports the shots branch
        paths.append(_write(f"{name}_{tag}", problem, model, generation))

    # Encoding axis, photonic/fock only: bs and bs_phase, not crossed with any other kind.
    for enc_name in ("bs", "bs_phase"):
        model = {"kind": "photonic", "prep": "fock", "encoding": enc_name}
        paths.append(_write(f"photonic_fock_{enc_name}_{tag}", problem, model, generation))

    if shots:
        return paths                            # spin/spin_magic have no shots path -- stop here

    # Encode-case axis: 3 configs, all at the default layers=1 -- not crossed with the layers
    # axis below, so this is not a 3x2 product.
    for enc_tag, enc_kwargs in ENCODE_CASES:
        model = {"kind": "photonic", "prep": "spin", "encoding": "phase",
                 "cx_pairs": "chain", **enc_kwargs}
        paths.append(_write(f"spin_{tag}_{enc_tag}_l1", problem, model, generation))

    # Layers axis: one extra config at layers=3, held at the default encode case.
    model = {"kind": "photonic", "prep": "spin", "encoding": "phase",
             "cx_pairs": "chain", "layers": 3, **dict(ENCODE_CASES[0][1])}
    paths.append(_write(f"spin_{tag}_{ENCODE_CASES[0][0]}_l3", problem, model, generation))

    # Encode-case axis: 3 configs, all at the default structure="linear" -- not crossed with the
    # structure axis below, so this is not a 3x3 product.
    for enc_tag, enc_kwargs in ENCODE_CASES:
        model = {"kind": "photonic", "prep": "spin_magic", "encoding": "phase",
                 "structure": "linear", "t_var": 0, **enc_kwargs}
        paths.append(_write(f"spin_magic_{tag}_linear_{enc_tag}", problem, model, generation))

    # Structure axis: two extra configs (ghz, linear_u3), held at the default encode case.
    for structure in ("ghz", "linear_u3"):
        model = {"kind": "photonic", "prep": "spin_magic", "encoding": "phase",
                 "structure": structure, "t_var": 0, **dict(ENCODE_CASES[0][1])}
        paths.append(_write(f"spin_magic_{tag}_{structure}_{ENCODE_CASES[0][0]}", problem, model,
                            generation))

    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for m, k in EXACT_SIZES:
        written += configs_for_size(m, k)
    for m, k in SHOTS_SIZES:
        written += configs_for_size(m, k, shots=SIZE)
    print(f"wrote {len(written)} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
