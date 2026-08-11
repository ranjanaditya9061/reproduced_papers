# Mathematical model of every circuit/model in `v2/model/`

Every model implements one map `X (N, n_features) -> probs (N, n_outcomes)`, a genuine
distribution (`probs.sum(dim=1) == 1`). This document gives the exact formula each one computes,
not just its role. Shared notation:

- `x ∈ ℝ^{n_features}` — one input row, angles.
- `m` — number of modes (photonic models) or the geometry parameter that stands in for network
  depth/qubit count elsewhere.
- `k` — photon number (photonic models) or a model-specific meaning noted per model.
- `n_fock(m,k) = C(m+k-1, k)` — size of the Fock basis (occupation tuples of `k` indistinguishable
  photons over `m` modes, bunching allowed).

---

## 1. Shared circuit: the sandwich `W1(Haar) → encode(x) → W2(Haar)`

`circuit/photonic_circuit.py`, used by `photonic` (`prep="fock"`) and `fermion`.

Two Haar-random unitaries are drawn once from `model_seed`, in this fixed order:

```
W1 = random_unitary(m)          (drawn first)
W2 = random_unitary(m)          (drawn second, same seed stream)
```

An **encoding** `Encoding` supplies the `x`-dependent middle piece, either as a diagonal
`D(x) ∈ U(m)` (`phase`) or a general unitary `E(x) ∈ U(m)` (`bs`, `bs_phase`):

```
phase:      D(x) = diag(e^{i x_0}, ..., e^{i x_{n_f-1}}, 1, ..., 1)          n_features ≤ m
bs:         E(x) = ∏_{i=n_f-1}^{0}  BS_i(θ=x_i)                              n_features ≤ m-1
                    BS_i(θ) acting on modes (i,i+1):  [[cos(θ/2), i sin(θ/2)],
                                                        [i sin(θ/2), cos(θ/2)]]
bs_phase:   E(x) = ∏_{i=n_f-1}^{0}  PS_i(x_i) · BS_i(θ=x_i)                   n_features ≤ m-1
                    (per feature i: BS_i(x_i) first, then PS_i(x_i) on mode i, in add-order)
```

The realised circuit unitary, in the convention `sandwich_unitary_at` returns (`U = M^T`, see §6
for why the transpose matters):

```
M(x) = W2 · D(x) · W1              (perceval's own circuit unitary)
U(x) = M(x)^T = W1^T · D(x)^T · W2^T
```

`n_circuit_parameters(m) = 2m² − 1`: `dim U(m) = m²` per Haar factor, two factors, minus one
unobservable global phase.

---

## 2. `photonic` (`model/photonic.py`, `prep="fock"`) — boson sampling

```
p(n | x) = |Perm(U(x)_{S,T})|² / ∏_j n_j!          n ∈ Fock basis, |n| = k
```

- `S` = input's occupied modes (`k` of them, from `default_input_state(m,k)`, evenly spaced).
- `T` = outcome `n`'s occupied modes, with multiplicity — a bunched `n` repeats a column of the
  `k×k` submatrix `U(x)_{S,T}`.
- `Perm` is the matrix permanent, `O(k!)` naively, `#P`-hard in general (Valiant).
- Computed via a merlin `QuantumLayer` (differentiable), not by brute force, except in the pure
  torch reference `boson_probs_reference` used for cross-checking.

`n_outcomes = n_fock(m,k) = C(m+k-1,k)`. `n_model_parameters = 2m²−1`.

---

## 3. `fermion` (`model/fermion.py`) — same circuit, `det` in place of `Perm`

Same `U(x)`, same `S`, same seed, same outcome basis as `photonic`. The naive determinant
readout would vanish on any bunched outcome (repeated column ⇒ `det = 0`, Pauli exclusion), so the
implemented model modifies repeated columns by a **phase-power correction** before taking the
determinant:

```
col_p(c) = (c / |c|)^p · |c|^{1 + s(1/p − 1)}          p = 1..(multiplicity of that mode in n)
                                                        s = bunching_s  (default k/m)
p(n | x) = |det(A(x,n))|² / Σ_{n'} |det(A(x,n'))|²
```

where `A(x,n)` is the `k×k` matrix built by taking, for each of the `k` rows `U(x)_{S_i, ·}`, the
columns corresponding to `n`'s occupied output modes — with the `p`-th copy of a repeated mode's
column entry raised to `col_p` (phase kept as an **integer** power to stay branch-cut-free;
modulus raised to a **fractional** power, exponent set by `s`). On the **collision-free** sector
(`max(n_i) ≤ 1`, support `C(m,k)` of the full `C(m+k-1,k)` basis) every `p=1` and `col_1(c) = c`
exactly, so there:

```
p(n | x) = |det(U(x)_{S,T})|²                          T = n's occupied modes, no correction
```

which is the honest free-fermion / Slater-determinant distribution and the sector on which the
`Perm` vs `det` comparison is physically meaningful. Off that sector the modified amplitude is not
a physical quantum state's — it is a matched *classical control readout* on the same circuit.

`n_outcomes = n_fock(m,k)` (same basis as `photonic`, full support by construction).
`n_model_parameters = 2m² − 1` (identical to `photonic`).

---

## 4. `qubit` (`model/qubit.py`) — IQP sandwich (Havlicek/Huang-style)

A qubit analogue of the same "sandwich" idea, `n_qubits = m`, acting on `n_qubits` qubits with
`n_features ≤ n_qubits` (unencoded qubits held at `x_i = π`):

```
|ψ(x)⟩ = W_trail(θ_trail) · H^{⊗n} · U_φ(x) · H^{⊗n} · U_φ(x) · V_lead(θ_lead) |0⟩^{⊗n}

U_φ(x) = diag( exp( i Σ_i x̃_i Z_i  +  i Σ_{i<j} (π−x̃_i)(π−x̃_j) Z_i Z_j ) )
         x̃ = x padded to length n with π at the unencoded positions

p(n | x) = |⟨n|ψ(x)⟩|²                     n ∈ {0,1}^{n_qubits}, n_outcomes = 2^{n_qubits}
```

`V_lead`, `W_trail` are fixed alternating `RY`/`RZ` layers with a `CZ`-chain entangler between
layers, both seeded from `model_seed` (lead drawn first, trail second — the `W1`/`W2` analogue;
`lead=False` drops `V_lead` but keeps the same trail draw). `n_model_parameters = 2·(depth+1)·n·2`
(both variational blocks' rotation angles).

---

## 5. `quadratic_fock` (`model/quadratic_fock.py`) — poly-size bilinear classical control

Fixed outcome feature map (no `x`-dependence), one row per Fock key:

```
φ(n) = [n_i]_{i=0}^{m-1}  ⊕  [n_i · n_j]_{i<j}                 d_φ = m(m+1)/2
```

standardised per-column at construction (zero mean, unit std across the `n_fock` rows). Input
featurisation (teacher-side Fourier map, `model/features.py`):

```
f(x) = [sin(x), cos(x), sin(2x), cos(2x), ..., sin(order·x), cos(order·x)]      d_x = 2·order·n_features
```

Two independent random (or low-rank) weight matrices `W_re, W_im ∈ ℝ^{d_x × d_φ}`:

```
full rank:   W_re, W_im ~ 𝒩(0, 1/d_x)                     count = 2 d_x d_φ
low rank r:  W_re = A_re B_re^T,  W_im = A_im B_im^T        count = 2r(d_x + d_φ)
             A ∈ ℝ^{d_x×r} ~ 𝒩(0, 1/d_x),  B ∈ ℝ^{d_φ×r} ~ 𝒩(0, 1/r)
```

Amplitude and distribution:

```
a_re(x,n) = f(x)^T W_re φ(n)          a_im(x,n) = f(x)^T W_im φ(n)
p(n | x)  = (a_re(x,n)² + a_im(x,n)²) / Σ_{n'} (...)
```

Expanding, `p_x(n) ~ φ(n)^T M(x) φ(n)` with `M(x) = W_re^T f f^T W_re + W_im^T f f^T W_im`, a PSD
quadratic form of rank ≤ 2 in `φ(n)` — hence the name. `matched_rank(m,d_x,d_φ)` solves for the `r`
minimising `|2r(d_x+d_φ) − (2m²−1)|`, so `param_matched=True` pins this model's count to the
circuit's at any `m`. `n_outcomes = n_fock(m,k)` (same Fock basis as `photonic`/`fermion`).

---

## 6. `mlp_fock` (`model/mlp_fock.py`) — capacity-unbounded classical reference

Same outcome basis and same `f(x)` Fourier featurisation as `quadratic_fock`, but the map from
`f(x)` to per-outcome amplitudes is a dense tanh network instead of a bilinear form:

```
h_0 = f(x)                                                            f(x) ∈ ℝ^{d_x}
h_l = tanh(W_l h_{l-1})           l = 1..depth,   W_l ∈ ℝ^{hidden × ·}
[a_re ; a_im] = W_out h_depth                     W_out ∈ ℝ^{2·n_fock × hidden}

p(n | x) = (a_re(x,n)² + a_im(x,n)²) / Σ_{n'} (...)
```

All weights bias-free, Xavier-initialised (gain = `weight_gain`, tanh-scaled on hidden layers),
fixed at construction (no training). `n_model_parameters` is dominated by `W_out`, hence
`Θ(n_fock) = Θ(C(m+k-1,k))` — **exponential in `m`**, unlike every other model here, which is the
model's entire point: it upper-bounds "is the map learnable by *any* classical function," at the
cost of not being a fair-capacity comparison.

---

## 7. `mlp` (`model/classical.py`) — 2-outcome softmax baseline

```
h_0 = f(x)                    f = teacher Fourier features, order = MLP_FOURIER_ORDER
h_l = tanh(W_l h_{l-1})        l = 1..k   (k = n_layers, bias-free, Xavier tanh-gain init)
logits = W_out h_k             W_out ∈ ℝ^{2 × hidden}
p(n | x) = softmax(logits)     n ∈ {(1,0), (0,1)}
```

`m` is unused; `k` sets the depth. Outcome basis is the 2-element `binary_keys(2)`, not Fock.

---

## 8. `analytical` (`model/classical.py`) — closed-form, zero-parameter baseline

```
s(x) = (1 / n_pairs) Σ_{i<j} sin(k · (x_i − x_j))        ∈ [−1, 1]
       (n_features == 1:  s(x) = sin(k x_0))
p(n | x) = [(1 − s(x))/2, (1 + s(x))/2]                   n ∈ {(1,0), (0,1)}
```

`n_model_parameters = 0`. With `keys = [(1,0),(0,1)]`, `parity` over 1 mode gives score vector
`v=(−1,+1)`, so `probs·v = p_1 − p_0 = s(x)` **exactly** — the exact-recovery property both
2-outcome models are built around, verified in tests to `2.6e-8` (float32 round-off).

---

## 9. State-preparation variants of `photonic` (`circuit/prep.py`)

All three preps below plug into `PhotonicModel`, sharing its `W1(Haar) → encode(x) → W2(Haar)`
sandwich (§1) for the interferometer stage; they differ only in the **input state** the photons
arrive in.

### 9.1 `fock` — `p(n|x) = |Perm(U(x)_{S,T})|²/∏ n_j!` exactly as in §2. `S` is the fixed
`default_input_state(m,k)`.

### 9.2 `spin` — dual-rail emission from `k` independently-prepared spin qubits

Numpy-built joint pure state, `layers` rounds of rotate-then-entangle (default `layers=1`):

```
|ψ_0⟩ = |0⟩^{⊗k}
[optional, once, before any layer]  per qubit q:  Rx(x_data[2q]) · Ry(x_data[2q+1])
for layer = 1..layers:
    per qubit q:  H → Rx(rx[layer,q]) → [Rz(rz[layer,q])] → Ry(ry[layer,q])
    for (c,t) in cx_pairs:  CX(c,t)
```

`rx`, `ry` seeded uniform on `[0,2π)` per layer (or discretised to `±0.1·{1..angle_levels}` when
`angle_levels` is set); `rz` = fixed increasing primes when `rz_angles="prime"`, replayed
identically at every layer (not re-seeded). Qubit `q`'s dual-rail photon emits into modes
`(2q, 2q+1)`; the resulting `k`-photon Fock state (built by tracing the spins, since photon
detection projects each spin's Z-basis onto its emitted rail) is Porter-Thomas-mixed **not** by a
pure amplitude but a genuine density matrix over Fock outcomes at that point, then propagated
through the shared interferometer `U(x)` and measured:

```
p(n | x) = ⟨n| U(x) ρ_spin U(x)^† |n⟩          ρ_spin = |ψ⟩⟨ψ| traced/projected onto photon number
```

`n_outcomes = m` (no readout pair). Basis discovered per-row from the perceval backend (perceval
prunes negligible outcomes), so it is not `n_fock(m,k)` in general.

### 9.3 `spin_magic` — single reused emitter, `m+2` modes with a dedicated readout pair

One spin, reused sequentially for all `k` emissions (an emitter-train cluster-state construction),
plus a fixed non-Clifford "magic" gate optionally injected into the first `t_var` gaps:

```
|ψ_0⟩ = |0⟩
for j = 0..k-1:
    structural gate(s), by `structure`:
        linear:      H
        ghz:         H   (j == 0 only; nothing at j > 0)
        linear_u3:   H,  then Rz(λ_j)·Ry(θ_j)·Rz(φ_j)   (Haar SU(2), every gap)
    [optional] Rx(x[2j mod n_f]) · Ry(x[2j+1 mod n_f])            -- encode_on_spin
    [optional, j < t_var] gap gate, by `gate_kind`:
        "t":    T = P(π/4)
        "rz":   Rz(prime_j / scale)
        "u3":   Rz(λ_j/scale) · Ry(θ_j/scale) · Rz(φ_j/scale)      (Haar SU(2))
        "u3_x": same with Rx in place of Ry
    emit spin's state into data modes (2j, 2j+1)
H                                                                  -- final re-superposition
emit into the 2 readout modes (m, m+1)

x_iface = x  if encode_circuit  else 0                              -- interferometer's input
p(n | x) = ⟨n| U(x_iface) ρ_train U(x_iface)^† |n⟩                  n over m+2 modes, unselected
```

`structure` and `gate_kind`/`t_var` are independent, composable axes (both may fire in the same
gap); `encode_on_spin` puts `x` on the spin every gap, `encode_circuit` puts `x` in the
interferometer — independently togglable, `False → x=0` (identity encoding) for whichever is off.
No `mu=0` post-selection is applied here: the full, unselected `m+2`-mode distribution is what
every prep persists, so any later choice of post-selection or observable is computed offline.
`n_outcomes = n_fock(m,k) · 2` empirically (the `m`-mode Fock support times the 2-way readout
photon placement), discovered per-row from the backend, same as `spin`.

---

## Summary table

| model | outcome basis | `n_outcomes` | params | poly in `m`? |
|---|---|---|---|---|
| `photonic` (`fock`) | Fock, `k` photons / `m` modes | `C(m+k-1,k)` | `2m²−1` | — (quantum) |
| `photonic` (`spin`) | `m` modes, backend-discovered | ≤ `C(m+k-1,k)` | `2m²−1` | — (quantum) |
| `photonic` (`spin_magic`) | `m+2` modes, backend-discovered | ≈ `2·C(m+k-1,k)` | `2m²−1` | — (quantum) |
| `fermion` | Fock, same as `photonic` | `C(m+k-1,k)` | `2m²−1` | poly (`det`, `O(k³)`) |
| `qubit` | computational, `n=m` qubits | `2^m` | `2(depth+1)·2m` | — (quantum) |
| `quadratic_fock` | Fock | `C(m+k-1,k)` | `2 d_x d_φ` or `2r(d_x+d_φ)` | yes, `O(m²)` |
| `mlp_fock` | Fock | `C(m+k-1,k)` | `Θ(n_fock)` | **no** — exponential |
| `mlp` | 2-outcome | 2 | fixed (Fourier width × depth × 2) | n/a (`m` unused) |
| `analytical` | 2-outcome | 2 | 0 | n/a (`m` unused) |
