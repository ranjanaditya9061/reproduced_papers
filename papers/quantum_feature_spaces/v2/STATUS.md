# QuEval v2 — detailed status report

Branch `ranjan-ad`, HEAD `cd279e2` ("Added hellinger distance metric snd fixed qubit feature
scaling"). Companion to `PIPELINE.md` (the math/rationale doc, 680 lines, checked against every
claim below). `PIPELINE.md:3` references a sibling `IMPLEMENTATION.md` — **that file does not
exist in this checkout**; this document fills that gap from the actual source.

Pipeline shape (`PIPELINE.md:10-16`):

```
X ──sample_X──► model ──circuit──► probs (N, n_out) ──observable──► score
                                       │
                                       ├──► learner   (embedding-based / nn-based)
                                       └──► metrics   (A: distribution   B: observable)
```

---

## 1. Models

### 1.1 Base contract (`model/base.py`)

`DistributionModel(nn.Module)` (`model/base.py:40`). A subclass implements exactly four things:

- `_probs(self, X) -> (N_chunk, n_outcomes)` (`model/base.py:77-79`) — **must not** be
  `@torch.no_grad`, since `metrics/` differentiates it w.r.t. `X`; the no-grad fast path lives on
  the wrapping `probs()` method instead (`model/base.py:120-146`).
- `outcome_keys(self)` — the basis the output columns align to (`model/base.py:81-83`).
- `circuit_spec(self) -> dict` — identity knobs for the artifact hash, **never** an observable key;
  `pipeline/artifact.py` raises if one appears (`model/base.py:85-87`, enforced at
  `pipeline/artifact.py:57-68`).
- `n_model_parameters(self)` — so parameter-matching claims are checkable, not asserted
  (`model/base.py:89-91`).

Plus classmethods `from_config`/`validate_config` (`model/base.py:93-100`). A registry
`MODELS: dict[str, type[DistributionModel]]` auto-populates via `__init_subclass__`
(`model/base.py:36-37, 60-63`); `build_model(cfg)` dispatches on `cfg.model.kind`
(`model/base.py:247-252`).

This one base class replaces four legacy per-model copies (`PhotonicTeacher`,
`FermionPhotonicTeacher`, `MlpFockTeacher`, `EbmFockTeacher`) that each separately implemented
~120 near-identical lines of batching, shot handling, and distribution capture
(`model/base.py:12-17`).

**Shared machinery, all on the base class:**
- **Auto-sized row-chunking** (`_autosize_batch`, `model/base.py:116-118`, target
  `FORWARD_ELEMENTS = 33_554_432` fp32 elements ≈ 128MB, `model/base.py:30-34`) — needed because
  `(N, n_out)` blows up at large `(m,k)`: `m=14,k=7` is `n_out=77520`, ~3GB at `N=1e4`.
- **Shots are opt-in per model**, `supports_shots: bool = False` by default
  (`model/base.py:160-166`) — deliberately not faked generically: multinomial resampling from a
  stored exact `p` is always arithmetically possible but is only physically meaningful where
  sampling is the actual scaling path past an unenumerable outcome basis (boson sampling); for
  every other family the exact distribution *is* the object of study.
- `probs_at_zero()` (`model/base.py:196-214`) — the circuit's unencoded output distribution `q`,
  used by the `xent` observable family; persisted in the artifact so `xent` is offline-re-scorable,
  which the legacy format could not do (it needed a matched-seed teacher rebuilt by hand).
- `enable_distribution_capture`/`captured_distributions` (`model/base.py:218-244`) — records every
  chunk so the pool can be persisted; explicitly carries **no** `observable` field, since a saved
  distribution must be readout-agnostic (`model/base.py:226-229`).

### 1.2 The seven model classes

| model | file | readout | outcome basis | params @ m=6,k=3,n_f=6 | in active sweeps? |
|---|---|---|---|---|---|
| `photonic` | `model/photonic.py` | merlin boson sampling, `\|Perm\|²` | Fock `C(m+k-1,k)`=56 | 71 = `2m²-1` | yes: both `eval/sweep_*.py` |
| `fermion` | `model/fermion.py` | same circuit, `\|det\|²` (phase-power columns) | Fock, same support | 71 (identical) | yes: both `eval/sweep_*.py` |
| `qubit` | `model/qubit.py` | IQP statevector | computational `2ⁿ`=64 | 96 | only `sweep_delta.py` |
| `quadratic_fock` | `model/quadratic_fock.py` | `\|f(x)ᵀWφ(n)\|²`, optionally low-rank | Fock | 1008 full-rank; matched to 71 | unit-tested only |
| `mlp_fock` | `model/mlp_fock.py` | dense `2·n_fock` head | Fock | 35,328 (exponential in m) | unit-tested only |
| `mlp` | `model/classical.py` | fixed random tanh MLP + softmax | 2 outcomes | 744 | baseline only |
| `analytical` | `model/classical.py` | closed-form `p₁=(1+s)/2` | 2 outcomes | 0 | baseline only |

(Table reproduced/cross-checked against `PIPELINE.md:73-81`, which independently states the same
seven rows and the same parameter counts.)

**`photonic`** (`model/photonic.py:36,44-45`) wraps a merlin `QuantumLayer` boson sampler:
`W1(Haar) → encode(x) → W2(Haar)`, Fock-basis readout (`circuit/photonic_circuit.py:35-56`).
Constructor: `PhotonicModel(*, m, k, n_features, seed=42, prep=None, encoding="phase", n_jobs=1)`.
Directory names like `photonic_m12_k3_n6` map directly onto `m` (optical modes), `k` (injected
photons), `n` (`n_features`, encoded input dimension). `n_model_parameters() = 2m²-1`
(`model/photonic.py:153-157`; derivation: `dim U(m) = m²` per Haar unitary, less one unobservable
global phase, `PIPELINE.md:528`). One `PhotonicModel` class replaces **eight** legacy classes,
since state preparation is now a registry choice — `fock`/`spin`/`spin_magic` cover
`model/photonic.py` plus all seven legacy `spoqc*` modules (`model/photonic.py:6-24`,
`PIPELINE.md:83-86`). Bonus: the spin preps now *gain* the full observable registry — the legacy
spin teachers scored via a private if-chain over four names, so `ent`/`osc`/`prod_parity`/graph
families were previously unavailable to them (`PIPELINE.md:85-86`).

**`fermion`** (`model/fermion.py:207-316`) — full docstring is ~95 lines (`model/fermion.py:1-95`)
and is one of the most load-bearing pieces of design reasoning in the repo:

- **Index convention is physics-fixed, not a choice.** On `M = W₂D(x)W₁`, the input's occupied
  modes `S` pick columns and the outcome's `T` pick rows, so the block is `M[T,S]`. The code
  indexes `U = Mᵀ` (what `sandwich_unitary_at` returns), so `S` picks rows and `T` picks columns —
  dropping the transpose would silently evaluate the wrong distribution: measured `|det|²` of
  `0.0585` vs `0.0011` on one draw (`model/fermion.py:7-14`).
- **The repeated-column / bunching problem.** A bunched outcome (any mode occupied >1×) repeats a
  column in a naive `det`, which vanishes identically (Pauli exclusion) — so strict free fermions
  put exactly zero mass on the bunched sector, support collapses to `C(m,k)` = 20 of 56 outcomes at
  `m=6,k=3` (`model/fermion.py:28-32`).
- **The fix — phase-power columns:** `col_p(c) = (c/|c|)^p · |c|^(1+s(1/p-1))`, `s=k/m`
  (`model/fermion.py:34-36`, code at `model/fermion.py:142-172`). Two exponents, two jobs:
  - the **integer** phase power breaks the degeneracy between repeated copies of a column — holding
    it at 1 leaves near-proportional columns (`cond≈26` vs `≈5` when broken) and bunched mass
    collapses to 0.07 (`model/fermion.py:41-43`). It must be an integer because `arg` is defined
    only mod `2π`: a fractional phase power is multivalued and jumps discontinuously across a
    branch cut — measured `0.318 → 0.0038` across `|Δx|=3.6e-9`, with the finite difference then
    diverging as `1/h` (`model/fermion.py:44-51`).
  - the **fractional modulus** power sets how much mass the bunched sector carries: smaller
    exponent enhances bunching (the boson direction), `s=0` undershoots, `s=1` overshoots
    (`model/fermion.py:52-54`).
- **Why `s=k/m` specifically** (the filling fraction, not a fitted constant): matches the boson
  model's mean bunched mass to within 1.8% average / 5.5% worst case across `m=6..12` at `k=3,4`,
  with zero free parameters (`model/fermion.py:56-73`). Measured comparison table
  (`model/fermion.py:62-67`):

  | | s=0 | s=0.5 | s=1 | **s=k/m** |
  |---|---|---|---|---|
  | mean \|err\| | 18.0% | 9.5% | 31.4% | **1.8%** |
  | worst case | -25.4% | +22.2% | +67.9% | **-5.5%** |

  `s=1/2` only looks competitive because the two configs it was first tried on (`(6,3)`, `(8,4)`)
  both happen to have `k/m=1/2` (`model/fermion.py:69-73`).
- **What this buys and costs, explicitly stated as a limitation:** full `C(m+k-1,k)` support from
  one `O(k³)` determinant, boson-matched on support *and* bunched mass, differentiable, no flavour
  bookkeeping — but no quantum state has this amplitude (an elementwise power of an orbital is not
  an orbital), so **the bunched sector does not carry the `Perm`/`det` hardness framing**
  (`model/fermion.py:75-80`). Only the **collision-free sector** does — `FermionModel.
  collision_free_probs` (`model/fermion.py:254-263`) is `|det U[S,T]|²` on the shared `C(m,k)`
  support with **no approximation**; this is where "run the headline `Perm`-vs-`det` claim" belongs
  (`model/fermion.py:82-86`).
- **A retired alternative, kept in the docstring as a documented dead end:** giving each mode `r`
  internal "flavour" states was tried first to open the bunched sector, but the interferometer is
  flavour-blind, so the lifted unitary block-factorises and at `r=k` every block is `1×1` — no
  determinant survives, and the model degenerates to the classical distinguishable-particle
  permanent distribution. `FermionModel.validate_config` now explicitly rejects `model.flavours != 1`
  with an error message explaining why (`model/fermion.py:302-312`).
- `|Perm|²` reference is computed independently in pure torch (`boson_probs_reference`,
  `model/fermion.py:175-204`), not through merlin, from the *same* `sandwich_unitary_at` the
  determinant path uses — verified against merlin to `6.7e-8` max difference, identical key
  ordering. Brute-force over `k!` permutations, so capped at small `k` (`k=3`→6 perms, `k=7`→5040).
- `bunching_s` is exposed as a constructor/config knob (`None` → `k/m` rule) so the choice stays
  explicit rather than hardcoded (`model/fermion.py:214-216, 221-224`).

**`qubit`** (`model/qubit.py`) — pure-torch statevector sandwich mirroring the photonic
`W1→encode→W2` structure but on qubits (Havlicek/Huang-style IQP feature map). `n_qubits` set by
`m`, decoupled from `n_features`. Used as a third arm in `eval/sweep_delta.py`'s `ARMS =
("photonic","fermion","qubit")` (`eval/sweep_delta.py:41`) but **absent** from `eval/sweep_size.py`'s
`ARMS = ("photonic","fermion")` (`eval/sweep_size.py:39`).

**`quadratic_fock`** — the actual **"classical, parameter-matched"** control (see §1.3).

**`mlp_fock`** — deliberately capacity-**unbounded**: `model/mlp_fock.py:1-16` states its
description grows exponentially in `m` (35.3k params at `m=6`, 19.9M at `m=14`, vs the circuit's
71/391) and that it can therefore only answer the weaker question "is this learnable by *some*
unconstrained classical map," never a fair-capacity comparison. Registered, has
`configs/mlp_fock.yaml`, exercised by the parameter-count unit test
(`tests/test_v2.py:248-263`) but absent from every `eval/sweep_*.py` script and from
`learner/compare.py`'s default arms.

**`mlp` / `analytical`** (`model/classical.py`) — unrelated 2-outcome baselines justified by an
exact-recovery argument: with `keys=[(1,0),(0,1)]`, `parity` over the first `⌈2/2⌉=1` mode gives
`v=(-1,+1)`, so `probs@v = p₁-p₀`; a model setting `p₁=(1+s)/2` recovers `s` **exactly** to
`2.6e-8` float32 round-off (`PIPELINE.md:68-71`). Not classical analogues of the photonic circuit —
they exist so every model (Fock and non-Fock alike) shares one pipeline shape with no
special-cased scalar path.

### 1.3 "Classical matched" — precisely what it means and where it lives

**The "classical model matched to the photonic one" is `quadratic_fock` with
`param_matched=True`** (`model/quadratic_fock.py:119-155`), **not** `model/classical.py`.

- Amplitude: `a(x,n) = f(x)ᵀ W φ(n)` for two independent random matrices `W_re, W_im`;
  `p_x(n) = (a_re²+a_im²)/Σ(...)` (`model/quadratic_fock.py:1-8`). `φ(n)` is a fixed feature map of
  the *outcome* — occupation moments `[nᵢ] + [nᵢnⱼ]_{i<j}`, `m(m+1)/2` numbers
  (`model/quadratic_fock.py:94-103`); `f(x)` is a Fourier expansion of the input angles at
  `FOURIER_ORDER=2` by default (`model/quadratic_fock.py:91`).
- **Why "quadratic":** expanding gives `p_x(n) ~ φ(n)ᵀM(x)φ(n)`, a PSD quadratic form of rank ≤2 in
  both `φ(n)` and `f(x)` — degree 2, contrasted explicitly with `mlp_fock`'s deep tanh stack
  (`model/quadratic_fock.py:10-15`).
- **Why this shape at all:** the photonic map is `2m²-1` numbers generating a distribution over
  `C(m+k-1,k)` outcomes — *poly parameters, exponential support*. A classical control meant to
  answer "is a small classical model enough" needs that same asymmetry, and `quadratic_fock` has it
  (unlike `mlp_fock`, whose *description* also grows exponentially, `model/quadratic_fock.py:17-23`).
- **Two corrected mistakes from the legacy version**, both documented (`model/quadratic_fock.py:29-51`):
  1. The legacy docstring claimed `O(m³)` scaling — an artifact of the old `n_features=m-1`
     coupling; with `n_features` held fixed per study, `d_x` is constant and the true scaling is
     `O(m²)`, matching the circuit.
  2. Same *scaling* isn't the same *count*: full-rank at `n_features=6,order=2` is `2·24·21=1008`
     against the circuit's `71`. So `W=ABᵀ` is optionally **low-rank**, giving
     `2r(d_x+d_φ)` parameters with `r` as the single matching dial
     (`matched_rank(m,d_x,d_φ)`, `model/quadratic_fock.py:106-116`, floored at 1). Measured table
     (`model/quadratic_fock.py:42-48`, also `PIPELINE.md:542-546`):

     | m | d_φ | circuit `2m²-1` | rank-1,order=1 | rank-2,order=1 |
     |---|---|---|---|---|
     | 6 | 21 | 71 | 66 | 132 |
     | 10 | 55 | 199 | 134 | 268 |
     | 14 | 105 | 391 | 234 | 468 |

  `param_matched=True` solves for the closest `r` and **records the resolved rank, not the flag, in
  `circuit_spec`** (`model/quadratic_fock.py:204-212`) — so a param-matched dataset is identified by
  the rank it landed on, deliberately colliding with an explicit `rank=` that happens to match.
- **A subtle correctness point about the amplitude construction:** using *two* independent random
  amplitude sets and squaring (rather than one real amplitude, or a Gibbs link `p~exp(θ·φ)`) is
  what makes the tail statistics land on Porter-Thomas — the law a Haar-random photonic circuit
  actually obeys — because `a_re,a_im` are near-Gaussian so `a_re²+a_im²` is exponential. A single
  real amplitude gives too many near-zeros; the Gibbs link gets tail *width* right but *shape*
  wrong (Gaussian `log p` vs Porter-Thomas's Gumbel-min: measured `log₁₀p` at the 0.1st percentile
  is `-3.87` vs the photonic model's `-4.95`) (`model/quadratic_fock.py:53-60`).
- **How to read a result using this model:** `parity` is a *calibration control*, not just a
  baseline — a map with too much high-frequency content is hard to learn whatever observable sits
  on top, so `parity` must come out easy; if it doesn't, lower `FOURIER_ORDER` until it does, and
  only then trust the nonlinear rows (`model/quadratic_fock.py:62-65`).

`quadratic_fock`/`mlp_fock` are both registered in `MODELS`, both have their own config files
(`configs/quadratic_fock.yaml`, `configs/mlp_fock.yaml`), and both are exercised by
`tests/test_v2.py:248-263` (parameter-count scaling assertions) — but **neither appears in any
`eval/sweep_*.py` script**. So the classical-matched control is implemented and unit-verified but
not yet run through the same size/delta sweeps as the quantum arms. Running the three-way
photonic/fermion/classical-matched comparison end-to-end would require adding `quadratic_fock` to
`ARMS` in `eval/sweep_size.py`/`eval/sweep_delta.py`, or passing `configs/quadratic_fock.yaml` as an
arm to `learner/compare.py` (see §4).

### 1.4 Circuit / encoding (`circuit/`)

Two orthogonal registry axes, not three separate encodings:
- **`prep`** (`circuit/prep.py`) — how photons are injected: `FockPrep` (`"fock"`, batched,
  differentiable via merlin), `SpinPrep` (`"spin"`, dual-rail spin-qubit emission, non-batched,
  process-pool per row), `SpinMagicPrep` (`"spin_magic"`, emitter-train cluster state with an
  injected non-Clifford gate, adds two readout modes) (`circuit/prep.py:40-300`).
- **`encoding`** (`circuit/encoding.py`) — how classical `x` enters the circuit. Only one ships:
  `PhaseEncoding` (`"phase"`), one phase shifter `PS(xᵢ)` per mode `i<n_features`
  (`circuit/encoding.py:69-105`). Deliberately left open for future encodings (data re-uploading,
  MZI-angle, dense mixing) — none exist yet (`circuit/encoding.py:1-15`).

"Photonic" in the `photonic_m12_k3_n6`-style directory-name sense is always `PhotonicModel` with
`prep="fock"` (the default) plus `PhaseEncoding`. `circuit/fock.py` is pure combinatorics
(`n_fock(m,k)=C(m+k-1,k)`, `fock_keys`, `binary_keys` for the non-Fock classical models),
`circuit/photonic_circuit.py` is the actual sandwich-circuit builder shared by both the merlin
path and the pure-torch fermion/boson-reference paths, `circuit/spin.py` holds the numpy
spin-state-preparation primitives.

### 1.5 Shared infra

- `model/sampler.py`: `sample_X(n_samples, n_features, seed)` — the one seeded, prefix-stable input
  sampler every model shares, so all models see identical `X` at a given `(n_features, seed)`.
- `model/features.py`: teacher-side Fourier featurization (`fourier_features`, `fourier_dim`) used
  by the classical/quadratic/mlp_fock label functions — **deliberately a separate implementation**
  from the learner-side Fourier map in `learner/embedding.py`. Sharing one implementation
  previously let the learner see the classical models' own encoding, inflating their scores:
  measured on `parity`, `0.689→0.957` for the bilinear model against `0.785→0.606` for photonic
  (`learner/embedding.py:6-12`). Unit-tested in `tests/test_v2.py:793-797`.

---

## 2. Probability distribution metrics — Analysis A (`metrics/distribution.py`)

Properties of `pₓ` alone, **no observable involved**. Differentiation target is the input `x`
(the encoded phase vector), not the circuit's weights (`PIPELINE.md:152-153`).

### 2.1 The input Fisher matrix

```
Fᵢⱼ(x) = Σₙ (1/pₙ)(∂pₙ/∂xᵢ)(∂pₙ/∂xⱼ) = (JᵀJ)ᵢⱼ,   Jₙᵢ = ∂(2√pₙ)/∂xᵢ
```

**Compute as a Gram factorization — SVD of `J`, never `eigvalsh(JᵀJ)`.** Forming the Gram squares
the condition number: a resolvable singular-value ratio of `1e-7` becomes an eigenvalue ratio of
`1e-14`, at the edge of double-precision round-off (`PIPELINE.md:162-165`). Function inventory
(`metrics/distribution.py`): `probs_and_jacobian` (autograd via `jacrev`, line 71), `sqrt_jacobian`
(110), `conditional_sqrt_jacobian` for shared-support comparisons (124),
`finite_difference_jacobian` as a non-autograd fallback (83), `spectrum_from_jacobian` (SVD, 150),
`input_fisher` (163), `fisher_spectrum` — trace, rank, spectral entropy, participation ratio,
`r_eff`, description cost (216), `r_eff_curve` (260), pool-level `analyse()` (282). Sum over the
**support only**; masking (not clamping) is required where `pₙ≡0` identically — the motivating case
being strict free fermions on the bunched sector.

**Three consequences of choosing `x` (input) over the circuit weights, all load-bearing**
(`PIPELINE.md:172-180`):
1. `F` is `n_f × n_f` for **every** model and every `(m,k)` — weight-space would be `72×72` at
   `m=6` and `392×392` at `m=14`: different spaces, no comparison possible.
2. No parametric re-implementation needed — `x` is already every model's input, and both readouts
   (`fermion` via `einsum` on `exp(1j·x)`, merlin as a differentiable layer) are differentiable in
   it directly.
3. It measures the *labelling function*, which is what learnability is actually about — the dataset
   varies `x`, not the weights.

**Gauge caveat:** only *rank* is reparametrization-invariant (`F → JᵨᵀFJᵨ` is a congruence, not a
similarity), so cross-model eigenvalue comparison is legitimate only in a fixed gauge — which this
setup has by construction, since `x` is the shared physical input in identical radian units for
every model (`PIPELINE.md:182-186`).

### 2.2 Conditioning on shared support requires centering

Comparing boson vs. determinant on the common collision-free support `C` means **conditioning**,
`q=p/Z_C`, whose score is the *centered* score `∂ᵢlog q = ∂ᵢlog p - E_q[∂ᵢlog p]`. Omitting the
centering overestimates by the rank-1 outer product `(∂log Z_C)(∂log Z_C)ᵀ` — not a small effect
here, since at `m=6,k=3` the collision-free sector is 20 of 56 outcomes and `Z_C` is strongly
`x`-dependent (`PIPELINE.md:188-200`). Implemented as `conditional_sqrt_jacobian`
(`metrics/distribution.py:124`), unit-tested at `tests/test_v2.py:335-357`
(`test_conditioning_centering_is_exactly_the_rank_one_correction`).

### 2.3 The global-phase mode and the projection invariant

When `n_features==m`, adding a constant to every `xᵢ` is exactly unobservable (a global phase), so
`rank F = n_f - 1` at that one grid point but `n_f` everywhere else — which would make the headline
`(m,k)` sweep at `(6,3),(8,4),(10,5)` inconsistent (rank 5 then 6 then 6). **Fix:** always evaluate
`F` on the `(n_f-1)`-dimensional orthogonal complement of `1/√n_f`, i.e. project
`J → J(I-11ᵀ/n_f)` before the SVD, at every `(m,k)` — giving a consistently 5-dimensional spectrum,
with the discarded direction's eigenvalue reported separately when it's informative (`n_f<m`)
(`PIPELINE.md:202-216`). This does **not** settle the `2m²-1` weight count, a different gauge
question (`PIPELINE.md:218-219`). Unit-tested: `test_global_phase_mode_and_the_projection_invariant`
(`tests/test_v2.py:316-335`).

### 2.4 Description cost: reverse water-filling, proved not asserted

`log 𝒩(ε) = ½Σᵢlog(1+λᵢ/ε²) ≈ ½Σ_{λᵢ>ε²}log(λᵢ/ε²)` (`metrics/distribution.py:193` /
`PIPELINE.md:223-227`). At fixed trace, spread is **proved** strictly more expensive than
concentration via `f'(r)>0` (Jensen's inequality argument, `PIPELINE.md:229-238`), giving the
ordering `r_eff=1 (cheapest) → gapped → log-uniform → flat (most expensive)`. Magnitude enters only
logarithmically, count enters linearly — "big eigenvalues cost more" is right per-direction and
misleading overall (at `k=40, T/ε²=10⁶`: flat ≈202 nats, `r_eff=1`≈6.9 nats,
`PIPELINE.md:240-242`).

**Permanent limitation, not deferred:** at `n_features=6` the `r_eff(τ)` curve has ≤6 steps and
**cannot distinguish** gapped/log-uniform/flat spectral shapes — `metrics/distribution.py`'s
`r_eff_curve` docstring (line 260) instructs printing **no shape label** and not fitting a slope to
6 points (`PIPELINE.md:259-263`). What *is* valid at `n_f=6`: cross-model and cross-`(m,k)`
comparison of `tr F`, `r_eff(τ_N)`, `rank`, `exp(H)`.

### 2.5 What the spectrum measures, and the central caveat of the whole framework

**Measures:** incompressibility of the parametrized family at the sample resolution, and
directional estimability. **Does not measure:** computational hardness of evaluating or sampling
`p` (`PIPELINE.md:285-289`). Boson sampling is the framework's own decisive counterexample: the
family is indexed by `O(m²)` numbers (`U`) — polynomially describable, maximally compressible —
while each `p(n)=|Perm(U_{S,T})|²` is `#P`-hard. **Three axes move in opposite directions as
`(m,k)` grow:** description complexity ↓ (bounded, `≤n_f`), sample complexity ↑ (`1/λ_min`),
computational complexity ↑ (cost of evaluating one `p(n)`). Small eigenvalues mean the family is
trivial to summarize *and* its parameters unlearnable — the same statement, via Pinsker
(`TV ≤ (D/2)√λ_max`, so no test on `N≲1/λ_max` samples can tell which `x` was used)
(`PIPELINE.md:300-310`). **Conclusion this framework already has:** the boson-vs-determinant
spectrum comparison is a learnability / effective-input-dimension statement and must never be
presented as evidence about the `Perm`/`det` complexity separation — this is stated in both
`PIPELINE.md:296-298` and mirrored verbatim in `model/fermion.py:22-26`.

**Do not use rank to hunt for barren plateaus** — a plateau is typically *full rank* with uniformly
tiny eigenvalues; only magnitudes detect it. An exact null direction instead means the
parametrization is redundant (the manifold is lower-dimensional), which makes learning *easier*
there, not harder (`PIPELINE.md:278-283`). The FIM is local and blind to discrete symmetries, so
full rank never certifies global identifiability (cites Rothenberg 1971 for the local-only claim).

### 2.6 Report absolute and shape, always both

`tr F = Σλᵢ` → sample complexity (absolute, matched physical units); `λᵢ/Σλⱼ` → compressibility,
cross-model comparison (shape). **Normalizing away the trace destroys the plateau signal**: if
`λᵢ(S)=2⁻ˢμᵢ`, the normalized spectrum is `S`-independent and cannot see uniform collapse
(`PIPELINE.md:265-273`). Secondary summaries: spectral entropy `H=-Σλ̂ᵢlogλ̂ᵢ`
(`exp(H)∈[1,n_f]`), participation ratio `(Σλ)²/Σλ²`, numerical rank at a stated tolerance.

### 2.7 Metric-naming trap and the shot-noise findings

**`metrics/fd.py` means "finite differences," not "Fisher discriminant."** It is a
finite-difference Jacobian wrapper usable on *any* `x→probs` callable, exact or shot-based, so no
model needs to be autograd-differentiable. Documented accuracy: FD on the exact distribution tracks
autograd to `1.6e-4`; FD on a 50k-shot sampler is noise-dominated (`0.32` relative error)
(`metrics/fd.py:17-36`). Unit-tested: `test_fd_wrapper_differentiates_any_sampler`
(`tests/test_v2.py:571-598`), `test_fd_on_a_shot_sampler_is_noise_dominated` (598-620).

**`metrics/fisher.py`** — single-input finite-difference input Fisher matrix, CLI
`python -m metrics.fisher --config ... --index 0 [--shots N]`. Documents a hard finding: a
**shot-based plug-in Fisher matrix is not usable as a measurement of F at any reachable budget** —
measured trace `2.2×` the true value even at `1e5` shots (`metrics/fisher.py:37-54`).
**Conclusion: shots are for labels, not derivatives.**

**Hellinger distance** was added recently (per the HEAD commit) but **lives in
`eval/sweep_delta.py`, not `metrics/`** — `hellinger_distance(p,q)=√(½Σ(√p-√q)²)`
(`eval/sweep_delta.py:49-59`). Justified there specifically for being bounded, proper, and needing
no ground metric on the outcome space, and framed as the natural bridge to the Fisher metric via
its local quadratic expansion (`eval/sweep_delta.py:15-19`).

---

## 3. Observables — Analysis B (`observable/`, `metrics/observable.py`)

### 3.1 Interface and the three functional shapes (`observable/base.py`)

`Observable(nn.Module)` (`observable/base.py:154`):
- `score(probs) -> (N,)` — must override (`observable/base.py:178-179`).
- `influence(probs) -> (N, n_out)` — `ψₙ=∂T/∂pₙ`, the object every efficiency quantity is built
  from; base raises `NotImplementedError` with explicit guidance: implement it if `T` is smooth in
  `p`, else set `is_differentiable=False` (`observable/base.py:184-213`).
- `effective_variance(probs)` — `V_eff`, generic from `influence` (`observable/base.py:215-228`).
- Class flags: `is_expectation` (only `Expectation` — exact single-shot variance), `is_differentiable`
  (False only for `max_prob`), `partial_basis_safe` (True when unobserved outcomes contribute 0).

Three concrete shapes (`observable/base.py:16-21`, matching `PIPELINE.md:94-98`):

| class | `T(p)` | influence `ψₙ=∂T/∂pₙ` | `V_eff` |
|---|---|---|---|
| `Expectation` | `p·v` | `vₙ` | `Var(v)` — **exact** single-shot |
| `Quadratic`/`DiagonalQuadratic` | `pᵀKp` | `2(Kp)ₙ` | `4Var(Kp)` — asymptotic |
| `ProbFunction` | `Σvₙφ(pₙ)` | `vₙφ'(pₙ)` | `Var(vφ'(p))` — asymptotic |

**All three are in scope for the efficiency metrics** — a documented correction from an earlier
design that gated efficiency metrics on `isinstance(obs, Expectation)`, wrongly claiming the others
"have no variance." That conflated two separate claims: no single-shot unbiased estimator with
variance `p·v² - (p·v)²` (**true**) vs. no finite asymptotic variance (**false**)
(`observable/base.py:23-26`, `PIPELINE.md:100-103`). Verified numerically:
`∂T/∂x` matches autograd to 8 decimals; `Var(T̂)·S` vs `Var_p(ψ)` gives `0.0693` vs `0.0685` (dense
quadratic) and `0.197` vs `0.187` (entropy) — the residual is exactly the `O(1/S)` bias correction
(`observable/base.py:41-43`, `PIPELINE.md:113-115`). Every implemented `ψ` matches autograd to
`≤2.8e-14` across 11 observables (unit test `test_influence_matches_autograd_on_p`,
`tests/test_v2.py:407-421`).

**For `Quadratic`:** `ψ=2Kp` is the Hájek projection of the order-2 U-statistic;
`Var(T̂) = 4(S-2)/(S(S-1))·ζ₁ + 2/(S(S-1))·ζ₂`, and the factors of 4 cancel in the efficiency
ratio, so a quadratic's `η` equals that of a *linear* observable with score vector `w=Kpₓ` — no new
machinery needed (`observable/base.py:259-274`, `PIPELINE.md:117-120`). `Quadratic.
u_statistic_degeneracy` (`observable/base.py:289-306`) computes `ζ₁/ζ₂` as a **flag, not an
exclusion**: when `≈0` the U-statistic is degenerate (convergence becomes `O(1/S)` with a
weighted-χ² limit, not Gaussian), so report it alongside rather than dropping the cell. Measured:
`sq_parity≈0.027`, `pairprod≈0.050` — both comfortably non-degenerate (`PIPELINE.md:458-460`).
Unit-tested: `test_quadratics_are_not_degenerate` (`tests/test_v2.py:473-482`).

For entropy (`ProbFunction`, `φ(u)=u·log u`), `ψ=log p+1`, so `V_eff=Var(log p)` — finite and
exact (`PIPELINE.md:122`).

### 3.2 Score-vector builders (Axis B, `observable/scorers/`)

All produce `Expectation`s — the axis is *how `v(n)` is constructed*, not a different measurement
class (`observable/base.py:45-49`, `PIPELINE.md:126-127`):
- `counting.py` — `parity`, `majority` (needs even `m`), `bunching`, `n_first`
  (`observable/scorers/counting.py:1-84`); these also populate `BASE_SCORERS`, letting composite
  families (`sq_<base>`, `ent_<base>`, `osc_<base>`, `xent_<base>`) reuse the same per-outcome
  scorer without duplicating logic (`observable/base.py:534-558`).
- `graph.py` — `connected_<base>`, `maxcc`: reads an outcome as a vertex set of a fixed seeded
  bounded-degree graph, `maxcc`=largest connected component (`observable/scorers/graph.py:1-190`).
- `exp_poly.py` — `prod_parity[__lo{N}]`, `prod_parity_consecutive[_second]`, and `_pi`/`_random`
  angle variants (`(-1)^P(n)`/`cos(P(n))` for a count polynomial `P`) (`observable/scorers/exp_poly.py:1-268`).
- `marked.py` — `single_output` (marked-outcome contrast, peak-normalized, needs `input_state`) and
  `xent`/`xent_<base>` (weighted `log q` against `probs_at_zero`, needs `reference_probs`)
  (`observable/scorers/marked.py:1-177`).
- `prob_fn.py` — `ent`/`ent_<base>` (`φ=p log p`), `osc`/`osc_<base>` (`φ=p sin(1/(p+ε))`,
  `OSC_EPS=1e-3`), `max_prob` (the sole `is_differentiable=False` observable)
  (`observable/prob_fn.py:1-257`).
- `quadratic.py` — `sq_<base>` (`DiagonalQuadratic`) and `pairprod` (dense `Quadratic`,
  `K[n1,n2]=(-1)^⟨n1,n2⟩`, capped at `PAIRPROD_MAX_OUT=4096` outcomes) (`observable/quadratic.py:1-120`).

**Dropped from the legacy package, deliberately:** `SelectiveObservable`, `loop_path`'s `__L`/`__P`
pre-selection, spoqc's `match{N}_` prefix — a post-selected mean is a score vector on a smaller
support and earns no class of its own (`observable/base.py:51-53`, `PIPELINE.md:134-135`).

**Three identities and a trap, measured at `m=6,k=3`** (`PIPELINE.md:139-146`):
- `prod_parity_consecutive_pi ≡ prod_parity_consecutive` — the `cos(πN)=(-1)^N` identity, a free
  cross-check on the angle path.
- `prod_parity_second ≡ bunching` — not coincidence: `Σ_{i≤j}nᵢnⱼ=(k²+Σnᵢ²)/2`, even exactly on
  collision-free outcomes at `k=3`. Reporting both double-counts one measurement.
- **Bare `prod_parity` is degenerate whenever `k<m`**: defaults to the `full` monomial
  `n₀···n_{m-1}` needing every mode occupied, so with 3 photons in 6 modes it is identically 0,
  `v≡+1` — constant score, zero variance, zero information. In-repo failure case for the
  gradient-free screen `G_O` (§3.4). Use `prod_parity__lo3` or `prod_parity_consecutive` (capped at
  `min(k,m)` precisely to avoid this) — also documented as `observable/scorers/exp_poly.py:30-35`.

### 3.3 Efficiency — the framework's central quantitative result

Two readings, both from the shared score matrix `S=∂log p/∂x` (`metrics/observable.py`,
`PIPELINE.md:319-347`):

```
local  (per direction i):   ρ²(O,sᵢ) = Cov(O,sᵢ)²/(Var(O)Var(sᵢ)) = F_{O,ii}/F_ii
joint  (all directions):    η_O      = gᵀF⁺g/V_eff,     gᵢ = Cov(ψ,sᵢ)
```

Implemented as `fisher_at` (`metrics/observable.py:80`), `influence_terms` (93),
`observable_fisher` (107), `efficiency` (112), `eta` (117), `rho2_per_direction` (127). `η∈[0,1]`
by Cauchy-Schwarz, **exact, not asymptotic** — a violation is unambiguously a code bug (Jacobian or
pseudo-inverse projection error), never a sampling artifact, which is what makes the boundary test
worth having (`PIPELINE.md:332-341`). Unit-tested: `test_eta_in_unit_interval_and_above_the_local_
reading` (`tests/test_v2.py:435-449`), `test_eta_is_one_when_the_observable_is_the_score`
(449-463, the tightness case), `test_mispaired_averages_can_exceed_the_bound` (501-514, a
deliberately constructed *negative* test confirming the bound only holds when both sides of a
sandwich come from the same pool).

**Equality iff `ψ` is the score — so the information-optimal readout is the score itself**, and
for a boson sampler evaluating the score's cost *is* the `#P`-hardness. **The optimal readout is
precisely the unusable one; every implementable observable is inefficient by construction**
(`PIPELINE.md:343-347`). This is the framework's reframing of "useful observable": not a search for
a perfect one, but a cost/information tradeoff.

**Measured `V_eff` ranking, reproducing a known result** (`PIPELINE.md:351-365`):

| observable | `V_eff` | note |
|---|---|---|
| `parity` | 1.00 | `v=±1` |
| `n_first` | 0.197 | |
| `ent` | 0.266 | **cheaper than parity** |
| `sq_parity` | 0.0025 | (η is scale-invariant, so this is fine) |
| `pairprod` | 0.161 | |
| `xent_parity` | 16.2 | |
| `single_output` | 35.3 | peak-normalized ratio |
| **`osc`** | **1763** | four orders above parity — hardest family to estimate, now *quantified* |

`ent` being cheaper than `parity` independently reproduces a shot-correlation table already
recorded elsewhere in the repo (`ent` 0.82 vs `parity` 0.79 at 100 shots, `PIPELINE.md:364-365`).

### 3.4 Rank-1 caveat, the observable set, and the A/D/E summaries

`F_O(x)=ggᵀ/V_eff` is rank 1 at any single point `x` — one scalar readout informs one direction at
one point. **Explicitly scoped as a caveat, not a headline**: this study never poses the inverse
problem of inferring `x` from `O` (the learner receives `x` directly), so there's no nuisance
parameter and the "effective Fisher is zero" fact is true but about a problem the pipeline doesn't
pose (`PIPELINE.md:367-386`). The prescribed fix is to compute `A/D/E` summaries on the
`x`-**averaged** `F̄_O=E_x[ggᵀ/V_eff]`, which is generically full rank.

For a vector of observables `O=(O₁..O_R)`: `F_O = Cov(O,s)ᵀΣ⁻¹Cov(O,s)`, and the optimal linear
combination `O*=βᵀO`, `β=Σ⁻¹Cov(O,s)` achieves `η_{O*}≥max_r η_{O_r}`, with `Σ⁻¹` automatically
discounting redundant observables (`PIPELINE.md:390-402`). Implemented as `set_fisher` (line 181),
`combination_weights` (190), `greedy_selection` (203) in `metrics/observable.py`.

**Scalar summaries** `ade_summaries` (`metrics/observable.py:152`), all `∈[0,1]`:

| summary | formula | reads as |
|---|---|---|
| A-type | `(1/k)tr(I⁻¹ᐟ²F_O I⁻¹ᐟ²)` | mean fraction captured |
| D-type | `(det F_O/det I)^{1/k}` | confidence-ellipsoid volume ratio |
| E-type | `λ_min(I⁻¹ᐟ²F_O I⁻¹ᐟ²)` | worst-direction fraction |

**Lead with E-type** — a single blind direction is exactly what the rank-1 caveat is about; D-type
is what the distinguishability literature optimizes instead (`PIPELINE.md:413-414`).

**Two implementation traps flagged in the doc, worth re-checking on any new use:**
1. All three are ill-defined on the raw `F` at rank-deficient configs (`rank F=5<6` at the base
   `n_f=6` config) — must evaluate on `range(F)` with `k→rank F` using the §2.3 projection, else
   D-type is `0/0` and E-type is identically `λ_min=0` for every observable set.
2. Averaged vs. pointwise efficiency have **different valid ranges** that must never be
   blanket-asserted: pointwise rank-1 is `tr(F(x)⁺F_O(x))∈[0,1]`, averaged/multi-observable is
   `tr(F̄⁺F̄_O)∈[0,k]` — exactly why A-type carries the `1/k` divisor
   (`PIPELINE.md:426-436`). Unit-tested: `test_ade_summaries_are_in_unit_interval_and_carry_the_1_
   over_k` (`tests/test_v2.py:482-501`).

### 3.5 Shot budget, bias, and the gradient-free screen

```
Var(O|x) = p·v² - (p·v)²                    exact, for an Expectation
shots_required_i(x) = V_eff / (∂⟨O⟩/∂xᵢ·δ)²
```

An observable whose `shots_required` exceeds the experiment's actual shot budget `S` cannot support
learning *regardless of the learner*, since the labels are noise-dominated at that budget.

**Bias, not variance, is what breaks nonlinear functionals at finite shots** — the plug-in `T(p̂)`
is biased at `O(1/S)`; for entropy-like functionals `~(support-1)/(2S)`, i.e. `-0.275` nats at
`n_out=56,S=100`, ~8% of bare `ent≈-3.5` but larger than `ent_parity≈-0.18` outright
(`observable/base.py:193-200`, `PIPELINE.md:452-456`). This gates **only** the shot-budget and
`R²`-ceiling readings (which assume zero-mean label noise); the exact `η_T` is unaffected since it
never samples.

**`G_O`, the gradient-free screen** (`metrics/observable.py:227`, `g_ratio`): exact law-of-total-variance
ratio `G_O = Var_x(μ_x)/E_x[σ²_x]`, needing only forward evaluations, no gradients
(`PIPELINE.md:465-478`). Related to `η` by a **one-directional** Poincaré inequality —
`G_O` large ⟹ `F_O` large somewhere (sound), but `F_O` large ⇏ `G_O` large (an oscillating `μ_x`
can have large local sensitivity but small global spread from cancellation). **So `G_O` prioritizes
work; it must never exclude a cell** — the in-repo failure case is `exp_poly`'s alternating-sign
score vectors, whose means are the likeliest in the library to oscillate and screen out while
remaining locally informative: **always run the gradient path on `exp_poly` regardless of `G_O`**
(`PIPELINE.md:491-494`). Two further blind spots flagged: dead zones (sharp jump in one sliver plus
flatness elsewhere gives healthy `G_O` while local estimation is hopeless) and non-injectivity
(`μ_{x1}=μ_{x2}` for well-separated inputs breaks identifiability while `F_O` looks fine) — check
monotonicity of `μ_x` (`monotonicity`, `metrics/observable.py:263`) as an additional guard. `G_O` is
**grid-dependent** — changing spacing/range changes the number without changing `μ_x`, so state the
grid; `η_O(x)` has no such dependence (`PIPELINE.md:499-500`).

**SNR ceiling** (`snr_r2_ceiling`, `metrics/observable.py:249`): at the dataset's actual global
scale, `S·G_O` is the intrinsic SNR and bounds `R² ≲ SNR/(1+SNR)`. **Applies only to `R²` measured
against shot-noisy test labels**, and the default generation config uses `shots=0` (verified:
`GenerationConfig.nsample` defaults to `0`; models apply `_shot_sample` only when `nsample>0`; the
learner regresses on the stored exact `soft`) — so with default configs the ceiling is 1 and any
comparison against it is **vacuous** (`PIPELINE.md:511-516`). This supersedes an earlier
`varsweep_tmp.py`, which computed only the numerator without the denominator that makes it an SNR
(`PIPELINE.md:520-521`). Also excluded from the SNR-ceiling reading: `ProbFunction` labels, per the
bias gate above.

---

## 4. Learners — no separate "generic" vs "Fourier-degree" learner class

### 4.1 What exists (`learner/base.py`, `learner/embedding.py`, `learner/nn.py`)

Three registered learners, all under one `Learner` ABC (`fit`, `predict`, `spec`,
`learner/base.py:43-67`):

- **`RidgeLearner`** (`name="ridge"`, `learner/embedding.py:69-103`) — closed-form ridge on a
  chosen feature basis (`raw`/`fourier`/`rff`/`combo`, `build_features`,
  `learner/embedding.py:56-66`). Convex and solved exactly, so a failure of this arm is *provably* a
  statement about the feature map's expressivity, never about an optimizer — which is exactly what
  makes it the right control for the paired protocol below: the "learner inadequate" verdict row
  can't be blamed on training dynamics here (`learner/embedding.py:73-76`).
- **`SvrLearner`** (`name="svr"`, `learner/embedding.py:105-132`) — RBF-SVR on the same feature
  bases, for contrast with the convex ridge readout.
- **`MlpLearner`** (`name="mlp"`, `learner/nn.py:23-97`) — trainable torch MLP with early stopping.

**"Fourier degree" is the `order` parameter of `fourier_features`**
(`learner/embedding.py:28-44`), not a distinct learner class. There is one embedding-based learner
family (ridge/SVR, parametrized by basis-choice and degree) and one nn-based learner (MLP),
matching the pipeline diagram's stated split ("learner: embedding-based / nn-based",
`PIPELINE.md:14`).

**Documented failure mode, explicitly framed as a feature not a bug of the ridge learner:** the
Fourier feature map is additive across input coordinates with **no interaction terms**
(`learner/embedding.py:28-39`) — so a ridge readout on it is a sum of univariate functions and
structurally cannot represent the multilinear cross-term structure of a Fock-basis permanent.
Measured on `fermion`/`parity`: `R²=-0.03` at order 3, `-0.06` at order 8, vs `+0.79` for the MLP on
the same data. The docstring explicitly warns: "this map is a genuine baseline, not a broken one —
but do not read its failure as evidence about the labels." A tensor-product basis would need
`order^{n_f}` width, so the prescribed alternatives are `rff` (implicit interactions, needs `gamma`
tuning) or the `mlp` learner.

**Fourier map isolation is deliberate and separately implemented from the teacher-side one** in
`model/features.py` — see §1.5; cross-contamination between the two previously inflated classical
model scores and depressed photonic ones.

### 4.2 Adjudication metric — a known, explicitly flagged open gap

`learner/base.py:1-31` (module docstring) states the design intent plainly: adjudicate on held-out
**log-likelihood**, since with exact `probs` available the mean held-out log-likelihood against the
ideal model is an unbiased KL estimate, stronger and less scale-sensitive than `R²`. This holds only
when labels carry genuine noise (`generation.shots>0`). **At the default `shots=0` (noiseless
labels)** the ideal model's log-likelihood is `+∞` and no finite KL exists — what
`gaussian_log_likelihood` (`learner/base.py:88-97`) actually reports is the learner's own Gaussian
predictive score, dominated by `σ²_train`, i.e. by label scale. Measured: paired-arm label
variances differ by up to `4.5e12` in one historical configuration, and the paired
`log_likelihood` difference there was `-12.8` while the paired `R²` difference was `+0.24` — opposite
signs, because one statistic renormalizes by label variance and the other doesn't
(`learner/base.py:19-25`). **Resolution used today: `R²` is the adjudicator for the decision table**
(§4.3), log-likelihood is reported as secondary. The code comments this explicitly as something to
revisit and promote once sweeps run at `shots>0` (`learner/base.py:29`) — the one clearly-flagged
"not yet done" item in `learner/`.

### 4.3 Cross-model comparison — what's wanted vs. what `learner/compare.py` implements

**What's wanted** (`learner/compare.py:1-35`, mirrors `PIPELINE.md §7`): a comparison where a
single-arm failure can't be blamed on architecture/optimizer choice — the standard objection that
kills training-based evidence. Solution: the **paired protocol** — run two model arms with
*everything* else identical (architecture, hyperparameters, `n_train`, seeds, split indices), and
report the paired difference. The fermion arm's role is explicitly framed as **calibrating the
learner**, not as an independent second result: "if the learner cannot fit the easy arm, it cannot
be used to make a claim about the hard one" (`learner/compare.py:6-10`).

**What's implemented:** `run_arm()` (`learner/compare.py:65-87`) fits one arm; `compare()`
(90-104) runs both arms over a list of observables; `verdict()` (107-116) is the fixed four-row
decision table at `SUCCESS_R2=0.5` (chosen and fixed *before* looking at numbers, so the table means
something — `learner/compare.py:59-62`):

| `det` | `perm` | verdict string |
|---|---|---|
| succeeds | fails | `"INFORMATIVE"` |
| succeeds | succeeds | `"no separation at this size"` |
| fails | fails | `"VOID: learner inadequate"` — **emitted loudly** (`main()`, lines 167-172, prints an explicit warning block: this is not a hardness signal, raise `n_train`/capacity or widen the feature map before reading anything into the perm arm) |
| fails | succeeds | `"INVESTIGATE: likely a bug or mismatched control"` |

Split indices come from `pipeline.split.split_indices`, deterministic by pool size alone
(`learner/compare.py:78-79`), so both arms see identical rows by construction — this is what makes
the comparison genuinely paired.

**`--confirm-split` mode** (`learner/compare.py:27-31, 140-141, 149-157`) guards against
multiple-comparisons cherry-picking: with `R` observables × `M` grid points you're implicitly
selecting a maximum, and bootstrap bands don't cover that selection — the pool is split into
"selection" and "confirmation" halves, and only the confirmation half's numbers may be quoted as a
result (enforced only by the printed label, not by any code-level restriction on what gets
reported downstream).

**Two-arm only, not N-way.** `compare.py`'s CLI (`main()`, lines 119-132) takes exactly `--perm`
and `--det` config paths, defaulting to `configs/photonic.yaml` vs `configs/fermion.yaml`. There is
no three-way (or N-way) comparator across photonic/fermion/classical-matched in one invocation —
running photonic-vs-classical-matched means manually passing `configs/quadratic_fock.yaml` as one
of the two arms; no script or config currently does this by default.

**What's not built, and explicitly why:** a trainable **quantum** student (parametric photonic/qubit
learner) — `learner/nn.py` notes to add it only "if the question becomes 'can a quantum student do
better'" (i.e. it's a deliberate scope decision, not an oversight).

### 4.4 What correlation between learners is actually wanted here

Per `PIPELINE.md §7` and the `verdict()` table: the wanted correlation is a **paired one** — same
learner, same hyperparameters, same split, evaluated on two (or more) models' labels, with the
*difference* (not either absolute score) being the reportable quantity. This is deliberately not a
correlation *across different learner types* (ridge vs SVR vs MLP) — those exist as alternative
readouts to cross-check a single model's labels (e.g. showing ridge fails on `fermion`/`parity`
while MLP succeeds, §4.1), not as a formal "agreement between learners" statistic. No code
currently computes an explicit learner-vs-learner agreement metric (e.g. rank correlation of
verdicts across ridge/SVR/MLP) — if that's wanted, it would be a new aggregation on top of
`compare()`'s existing per-arm `evaluate()` outputs.

---

## 5. What's actually been run (`eval/`)

- **Size sweep** (`eval/sweep_size.py`, `eval/plot_size.py`): bunched mass + input-Fisher spectrum,
  `photonic` vs `fermion` only (`ARMS=("photonic","fermion")`, line 39), via finite differences on
  both arms uniformly (deliberate — `eval/sweep_size.py:1-9`). Default CLI sweeps
  `m∈{6,8,10,12,14,16}`, `n_features=5`, `n_x=100` (lines 107-113), but the **committed result**
  (`eval/sweep_size.json`) stops at `m∈{6,8,10,12}` — a stated memory ceiling, not a missed run:
  at `m=14` the outcome count is 77,520 and the Fisher Jacobian alone is ~2GB
  (`eval/sweep_size.py:16-19`). Figures: `bunched_mass.png`, `mass_split.png`, `fisher_eigs.png`
  (5 panels, one per eigenvalue, vs `m`).
- **Delta sweep** (`eval/sweep_delta.py`, `eval/plot_delta.py`): Hellinger distance from a fixed
  base point `x0` over a local perturbation ball of radius `delta`, all three arms
  (`ARMS=("photonic","fermion","qubit")`, line 41). "Delta" = the ball radius in radians — a local
  sensitivity/smoothness axis, distinct from "size." Run at **six radii**
  (`1e-01…1e-06`) × **four sizes** (`m∈{6,8,10,12}`) × three arms — confirmed via direct inspection
  of `eval/results/sweep_delta_{1e-01…1e-06}.json`. Figures (`eval/plot_delta.py`) exist for
  `1e-01…1e-04` only; the `1e-05`/`1e-06` runs have JSON results but **no rendered overview PNG** —
  either the plotting script wasn't re-run for the two smallest deltas, or the PNGs weren't
  committed.
- Two loose duplicate files at `eval/` root (`sweep_delta.json`, `sweep_size.json`) rather than in
  `eval/results/`: `sweep_delta.json` matches the `1e-06` result exactly and looks like an
  earlier/default-args run predating the `results/` convention; `sweep_size.json` is the only
  size-sweep result and has no `results/` analog (the size sweep has no subfolder equivalent).
- `configs/size_sweep/` (`photonic_m6k3.yaml`, `photonic_m8k4.yaml`, `photonic_m10k5.yaml`, all
  `n_features=5`, `generation.size=10000`, `k=m/2`) and `configs/size_sweep_shots/` (same three
  points + `generation.shots=1000`) — the second set exists specifically for the exact-vs-shot-noise
  comparison the SNR ceiling (§3.5) needs, but no `eval/` script currently consumes it directly;
  it appears to be prepared infrastructure rather than a run-and-plotted result.
- Root-level `configs/photonic_m12k6.yaml`, `photonic_m14k7.yaml`, `photonic_m16k8.yaml` extend the
  size axis further but sit outside `size_sweep/`, and the fermion side's ad hoc configs
  (`configs/fermion_m6k3.yaml` … `fermion_m10k5.yaml`) stop at `m=10` — asymmetric coverage between
  the two arms' standalone config files (though both arms are covered together inside
  `eval/sweep_size.py`'s own sweep up to `m=12`).

---

## 6. Testing status

`tests/test_v2.py` — 33 test functions (verified count via `grep -n "^def test_"`), organized by
section, with line numbers:

**Artifact/cache identity (9 tests, lines 55-233):** `test_artifact_is_observable_independent`,
`test_shot_budget_is_not_part_of_the_circuit_identity`, `test_shot_draw_never_builds_the_
distribution`, `test_shot_draw_returns_one_dict_per_requested_row`, `test_shot_blocks_are_additive`,
`test_shot_seed_gives_independent_realisations_at_a_fixed_circuit`, `test_clifford_sampling_
converges_to_the_exact_distribution`, `test_noisy_scores_do_not_share_a_cache_key_with_exact`,
`test_circuit_spec_rejects_an_observable`.

**Model contract (3 tests, lines 233-274):** `test_parity_on_a_two_outcome_basis_is_the_signed_
score`, `test_parameter_counts_scale_as_2m2_minus_1` (checks `photonic`/`fermion`=`2m²-1` exactly,
`mlp_fock` super-linear growth dwarfing the circuit, `quadratic_fock(param_matched)` within
`[0.5×,2×]` of 71), `test_n_features_is_a_study_invariant` (config validation rejects mismatched
`n_features`).

**Analysis A / Fisher (8 tests, lines 274-407):** `test_fisher_is_symmetric_and_psd`, `test_
jacobian_matches_finite_differences`, `test_probability_derivatives_sum_to_zero`, `test_svdvals_
beats_gram_on_known_ground_truth` (constructed ground-truth singular values, numerically proves the
SVD-vs-Gram claim), `test_global_phase_mode_and_the_projection_invariant`, `test_conditioning_
centering_is_exactly_the_rank_one_correction`, `test_fermion_collision_free_sector_is_exactly_free_
fermions`, `test_fermion_bunched_mass_tracks_the_boson_model`.

**Analysis B / observable efficiency (8 tests, lines 407-514):** `test_influence_matches_autograd_
on_p` (11+ observables vs autograd to `1e-12`), `test_g_equals_the_gradient_of_the_label`, `test_
eta_in_unit_interval_and_above_the_local_reading`, `test_eta_is_one_when_the_observable_is_the_score`,
`test_max_prob_is_the_only_exclusion`, `test_quadratics_are_not_degenerate`, `test_ade_summaries_
are_in_unit_interval_and_carry_the_1_over_k`, `test_mispaired_averages_can_exceed_the_bound`
(deliberate negative test).

**Learner (2 tests, lines 514-534):** `test_verdict_table`, `test_split_is_deterministic_and_
partitions_the_pool`.

**Shots capability (4 tests, lines 534-620):** `test_only_pure_boson_sampling_supports_shots`,
`test_generate_refuses_shots_for_distribution_only_models`, `test_fd_wrapper_differentiates_any_
sampler`, `test_fd_on_a_shot_sampler_is_noise_dominated`.

**Partial-basis / finite-sample path (7 tests, lines 620-793):** `test_key_scorers_are_the_single_
source_of_the_dense_tables`, `test_score_on_a_partial_basis_equals_the_full_basis`, `test_partial_
basis_guard_fires_when_phi_does_not_vanish_at_zero`, `test_max_prob_is_partial_basis_safe_but_
still_excluded_from_analysis_B`, `test_merlin_perm_matches_the_analytic_permanent_on_the_FISHER_
spectrum`, `test_combinatorial_fock_basis_matches_merlin_exactly`, `test_learner_fourier_map_is_
independent_of_the_teacher_map`.

### 6.1 The test suite currently cannot be collected — reproduced directly, not just reported

Running `python -m pytest tests/test_v2.py --collect-only -q` in this checkout fails immediately:

```
ImportError while importing test module '...\tests\test_v2.py'.
tests\test_v2.py:26: in <module>
    from model import MODELS, build_model, sample_X
E   ImportError: cannot import name 'MODELS' from 'model' (...\quantum_feature_spaces\model\__init__.py)
```

Root cause is a **package-shadowing bug**: both `v2/__init__.py` and `v2/tests/__init__.py` exist,
so under pytest's default "prepend" import mode, `tests` resolves as a subpackage of `v2` (also a
package), and pytest climbs to the first ancestor directory *without* an `__init__.py` —
`papers/quantum_feature_spaces/` — inserting *that* onto `sys.path[0]`. That directory contains a
**different, older pre-v2 `model/` package** which shadows `v2/model/` for every absolute import
(`from model import ...`) in the test file. Reproduced with `--noconftest` and
`--import-mode=importlib` too, so it's independent of any parent `conftest.py`.

**A second, independent problem exists underneath this one** (found by direct `grep`, not just
collection failure): the test file references pipeline APIs that the current `pipeline/shots.py`
no longer has —

```
tests/test_v2.py:78-93    from pipeline.artifact import artifact_path
                            from pipeline.shots import BLOCK, load_shots
                            generate_shots(cfg, shots_root=shots_root)
tests/test_v2.py:133       from pipeline.shots import BLOCK
tests/test_v2.py:155-161   from pipeline.shots import BLOCK, merge_shots
                            merged = merge_shots(b0, b1)
tests/test_v2.py:646-688   from pipeline.shots import score_sparse, to_sparse
tests/test_v2.py:737-740   from circuit.photonic import (...)      # current module: circuit.photonic_circuit
                            from model.fock import fock_keys        # current module: circuit.fock
tests/test_v2.py:783-784   from circuit.photonic import build_quantum_layer
                            from model.fock import fock_keys
```

None of `BLOCK`, `merge_shots`, `to_sparse`, `score_sparse`, or `artifact_path` exist anywhere in
the current `pipeline/` (confirmed by grep across the module), and `pipeline/shots.py`'s own
docstring states outright that "`BLOCK` and its four helper functions are gone" — a deliberate,
intentional API removal that the test file predates. `generate_exact`/`generate_shots` also take a
single `root=` kwarg today, not the `out_root=`/`shots_root=` the tests pass.

**Net: `pytest tests/test_v2.py` cannot currently run at all** — it fails at import/collection
time, before any assertion executes, for two independent reasons stacked on top of each other. The
33 test bodies read as a thorough, carefully-reasoned regression suite with numeric tolerances tied
to specific measured numbers in the module docstrings (e.g. the `1e-12` influence-function
tolerance, the `2m²-1` parameter-count check), but **none of them can currently be verified to pass
in this checkout** without first fixing (a) the `__init__.py` package-shadowing issue and (b) the
stale `pipeline.shots`/`circuit.photonic`/`model.fock` references.

---

## 7. Config (`config.py`, `configs/`)

`ExperimentConfig` (`config.py:52-147`) has `problem`/`model`/`generation`/`split`/`seeds`
sections. Key invariant: `problem.n_features` is **required with no `m-1` fallback**, and
`check_commensurable(cfgs)` (`config.py:237-251`) rejects any sweep/grid mixing `n_features`
values, since every Fisher-based metric in §2 is denominated in it. Deliberately **no `observable`
field anywhere** in the config schema — `load_config` raises if one is found in any section
(`config.py:214-222`), enforcing the artifact/observable independence from §1.1.
`ModelConfig` documents per-family knobs (`prep`, `encoding`, `cx_pairs`, `bunching_s`,
`rank`/`param_matched`, etc., `config.py:61-101`), each hashed into the artifact identity only when
actually set.

---

## 8. Dataset hygiene note

Four of six on-disk fermion dataset directories (`07401951`, `38e56e7b`, `f5a4b680`, `b69147a7`)
carry the retired `flavours`-based spec (`{"readout":"determinant_free_fermion","flavours":1|2|3}`)
that `FermionModel.validate_config` now explicitly rejects (`model/fermion.py:302-312`) — orphaned
from before the phase-power-column rewrite, and cannot be regenerated by current code (any config
with `model.flavours != 1` fails validation immediately). The `datasets_v2`/`shots_v2` trees
(currently being deleted per `git status`, matching the "Removed data from git" commit pattern) are
an even older artifact-naming scheme — directory names embed `m/k/n` directly
(`photonic_m12_k3_n6`) — superseded by the current opaque content-hash `datasets/<hash>/{exact,
counts}/` layout defined in `pipeline/artifact.py`.

`load_npz.py` is a tiny ad hoc inspection script, not part of the pipeline proper — it just
`numpy.load`s one `dist.npz` and prints shapes; currently points at `datasets/b69147a7/exact/
dist.npz` (one of the stale flavoured-fermion artifacts).
