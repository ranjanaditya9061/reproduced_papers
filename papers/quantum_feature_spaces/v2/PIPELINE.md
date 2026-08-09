# v2 pipeline — concepts and mathematics

Companion to [`IMPLEMENTATION.md`](IMPLEMENTATION.md), which maps this onto files and status.
This document is the *why*: the pipeline's shape, and the mathematics each stage rests on.

---

## 1. The pipeline

```
    X  ─────►  model  ─────►  probs (N, n_out)  ─────►  observable  ─────►  soft (N,)
 sample_X    circuit       THE ARTIFACT                 readout          score cache
                              │
                              ├──►  learner    (embedding-based / nn-based)
                              └──►  metrics    (A: distribution   B: observable)
```

Four stages, each with one responsibility:

| stage | input | output | depends on |
|---|---|---|---|
| **generate** | config | `probs` over a labelled outcome basis | input + circuit **only** |
| **score** | `probs`, observable name | `soft` | + the observable |
| **learn** | `X`, `soft` | fitted model, test score | + the learner |
| **metrics** | `probs` (+ observable for B) | Fisher / efficiency numbers | — |

### 1.1 The one invariant that shapes everything: `n_features` is fixed

`problem.n_features` is required per config (no `m - 1` fallback), and `v2.config.check_commensurable`
rejects any grid or sweep that mixes values. **Not a knob within a comparison.** Every complexity
measure in §3–4 is denominated in it — the input Fisher matrix is `n_features × n_features`,
`r_eff ∈ [0, n_f]`, `exp(H) ∈ [1, n_f]`, and the description cost `½ Σᵢ log(1 + λᵢ/ε²)` has `n_f`
terms. Vary it and you conflate "more input dimensions" with "harder map", which is the same
objection that rules out a weight-space Fisher matrix. Results at two values are two non-comparable
studies — which is why the check is on the grid rather than on a global constant.

What it buys: **the `(m, k)` sweep becomes well-posed.** At fixed `n_f` the circuit grows while `F`
stays `6×6`, so spectra are stackable across sizes *and* across model families. The legacy default
`n_features = m − 1` coupled them, so every change of circuit size silently changed the Fisher
matrix's dimension.

### 1.2 The artifact depends on the input and the circuit only

This is the structural fix. Legacy `Generator/artifact.py` folded `hash_spec(cfg)` into the dataset
hash, and every Fock teacher's `hash_spec` returned `{"observable": ...}`. So `parity` and `osc` on
an **identical** circuit with **identical** inputs produced two directories and two independent
boson-sampling runs.

In v2 the *distribution* is the artifact; the readout is a cheap downstream stage. Hash fields are
exactly: `format_version, model, m, k, n_features, sample_seed, model_seed, nsample, circuit_spec`.
`dist_hash` **raises** if a model's `circuit_spec` contains an observable-ish key, since that is the
one mistake that would reintroduce the coupling.

`nsample` is part of the *circuit*, not the readout: it changes the stored distribution (each row
becomes an empirical distribution), so two shot budgets are genuinely two datasets.

Two extra fields exist so every observable is re-scorable offline, which the legacy format could
not do: `probs_at_zero` (the `q` that `xent` scores against) and `input_state` (which
`single_output` marks). Both are functions of the circuit alone.

### 1.3 Every model emits a distribution

```python
probs = model.probs(X)        # (N, n_outcomes), rows sum to 1
keys  = model.outcome_keys()  # occupation tuples, aligned to columns
```

Including the two non-Fock models, via a **2-element outcome basis**. With
`keys = [(1,0), (0,1)]`, `parity` over the first `⌈2/2⌉ = 1` mode gives `v = (−1, +1)`, so
`probs @ v = p₁ − p₀`; a model setting `p₁ = (1+s)/2` recovers `s` **exactly** (verified to 2.6e-8,
float32 round-off). So no model needs a special-cased scalar path and the pipeline has one shape.

| kind | `probs` | outcome basis | parameters (measured at `m=6,k=3,n_f=6`) |
|---|---|---|---|
| `photonic` | merlin boson sampling, `prep` × `encoding` | Fock `C(m+k−1,k)` = 56 | 71 = `2m²−1` |
| `fermion` | flavoured `|det|²`, **same circuit** | Fock, occ ≤ `r` | 71 (identical) |
| `quadratic_fock` | `|f(x)ᵀ W φ(n)|²` | Fock | 1008 full-rank; matchable to 71 |
| `mlp_fock` | dense `2·n_fock` head | Fock | 35 328 (exponential in `m`) |
| `qubit` | `|IQP statevector|²` | computational `2ⁿ` = 64 | 96 |
| `mlp` | softmax | 2 outcomes | 744 |
| `analytical` | `p₁ = (1+s)/2` | 2 outcomes | 0 |

`photonic` alone replaces eight legacy classes, because state preparation is a registry choice:
`fock` | `spin` | `spin_magic` cover `model/photonic.py` plus all seven `spoqc*` modules. The spin
preps also *gain* the full observable registry — legacy spin teachers scored with a private
if-chain over four names, so `ent`, `osc`, `prod_parity` and the graph families were unavailable.

---

## 2. Observables: two orthogonal axes

### 2.1 Axis A — functional shape, and the influence function

| class | `T(p)` | influence `ψₙ = ∂T/∂pₙ` | `V_eff = Var_p(ψ)` |
|---|---|---|---|
| `Expectation` | `p·v` | `vₙ` | `Var(v)` — **exact** single-shot |
| `Quadratic` | `pᵀKp` | `2(Kp)ₙ` | `4 Var(Kp)` — asymptotic |
| `ProbFunction` | `Σ vₙ φ(pₙ)` | `vₙ φ′(pₙ)` | `Var(v φ′(p))` — asymptotic |

**All three are in scope for the efficiency metrics.** An earlier design gated them on
`isinstance(obs, Expectation)`, claiming the others "have no variance". That conflated two claims:
no single-shot unbiased estimator with variance `p·v² − (p·v)²` (**true**), and no finite asymptotic
variance (**false**).

For any smooth `T(p)`, since `Σₙ ∂ᵢpₙ = 0` (so `ψ` is defined only up to an additive constant,
harmlessly):

```
∂T/∂xᵢ  =  Σₙ ψₙ ∂ᵢpₙ  =  E[ψ sᵢ]  =  Cov(ψ, sᵢ)                 s = ∂ log p / ∂x
Var(T̂)  ≈  Var_p(ψ) / S      from Cov(p̂ₙ, p̂ₙ′) = (pₙδ − pₙpₙ′)/S
```

**Verified:** `∂T/∂x` matches autograd to 8 decimals; `Var(T̂)·S` vs `Var_p(ψ)` gives 0.0693 vs
0.0685 (dense quadratic) and 0.197 vs 0.187 (entropy) — the residual being exactly the `O(1/S)`
correction of §4.5; and every implemented `ψ` matches autograd to ≤ 2.8e-14 across 11 observables.

For `Quadratic`, `ψ = 2Kp` is the Hájek projection of the order-2 U-statistic, and
`4ζ₁` is the leading term of `Var(T̂) = 4(S−2)/(S(S−1))·ζ₁ + 2/(S(S−1))·ζ₂`. **The factors of 4
cancel in the efficiency ratio**, so a quadratic's `η` equals that of a *linear* observable with the
`x`-dependent score vector `w = Kpₓ` — no new machinery, and free because `p` is stored.

For entropy, `φ(u) = u log u` gives `ψ = log p + 1`, so `V_eff = Var(log p)`: finite and exact.

### 2.2 Axis B — score-vector builders (all produce `Expectation`s)

`v(n)` construction, *not* different measurements. Graph-based and `exp(poly)` observables are
`probs @ v` and hence **genuine expectation values**; only the construction of `v` differs.

- `counting` — `parity`, `majority`, `bunching`, `n_first` (also populate `BASE_SCORERS`)
- `graph` — `connected_<base>`, `maxcc`: outcome as a vertex set of a fixed seeded graph
- `exp_poly` — `(−1)^P(n)`, `cos(P(n))` for a count polynomial `P`
- `marked` — `single_output` (marked contrast), `xent` (weighted `log q`)

**Dropped:** `SelectiveObservable`, `loop_path`'s `__L`/`__P` pre-selection, spoqc's `match{N}_`
prefix. A post-selected mean is a score vector on a smaller support, so it earns no class.

### 2.3 Three identities and a trap, measured at `m=6, k=3`

- `prod_parity_consecutive_pi` ≡ `prod_parity_consecutive` — the `cos(πN) = (−1)^N` identity holding
  numerically. A free cross-check on the angle path.
- `prod_parity_second` ≡ `bunching`. Not coincidence: `Σ_{i≤j} nᵢnⱼ = (k² + Σnᵢ²)/2`, which at
  `k = 3` is even exactly on collision-free outcomes. Reporting both double-counts one measurement.
- **Bare `prod_parity` is degenerate whenever `k < m`**: it defaults to the `full` monomial
  `n₀···n_{m−1}`, needing *every* mode occupied, so with 3 photons in 6 modes it is identically 0
  and `v ≡ +1` — constant score, zero variance, no information. Use `prod_parity__lo3` or the
  geometry-derived `prod_parity_consecutive` (capped at order `min(k,m)` precisely to avoid this).

---

## 3. Analysis A — distribution (`metrics/distribution.py`)

Properties of `pₓ` alone. **No observable.** Differentiation target is the **input** `x` — the
encoded phase vector in `D(x) = diag(e^{ix₀},…,e^{ix_{n_f−1}},1,…,1)` of `U(x) = (W₂D(x)W₁)ᵀ`.

### 3.1 The input Fisher matrix

```
Fᵢⱼ(x)  =  Σ_{n ∈ supp(p)} (1/pₙ) (∂pₙ/∂xᵢ)(∂pₙ/∂xⱼ)
        =  (JᵀJ)ᵢⱼ        with   Jₙᵢ = ∂(2√pₙ)/∂xᵢ = ∂ᵢpₙ/√pₙ
```

**Compute it as a Gram factorisation, never as a matrix product.** Build `J` then take
`svdvals(J)**2` — *not* `eigvalsh(Jᵀ J)`. Forming the Gram squares the condition number: a
resolvable singular-value ratio of `1e-7` becomes an eigenvalue ratio of `1e-14`, at the edge of
double-precision round-off.

Sum over the **support only**; do not clamp `p` and divide. Where `pₙ ≡ 0` identically in `x` those
terms are a genuine `0/0` and must be *masked* — the motivating case being strict free fermions on
the bunched sector (`FermionModel.collision_free_probs`), and, on the shots path, never-observed
outcomes.

Three consequences of choosing `x` over the weights, all load-bearing:

1. `F` is `n_f × n_f` — `6×6` for **every** model and every `(m,k)`. Weight-space would be `72×72`
   at `m=6` and `392×392` at `m=14`: different spaces, no comparison.
2. **No parametric re-implementation needed.** `x` is already the models' input and both readouts
   are differentiable in it (`fermion` builds `U(x)` via `einsum` on `exp(1j·x)`; merlin is a
   differentiable layer). `v2/parametric/` is for the nn learner only.
3. It measures the *labelling function*, which is what learnability is about — the dataset varies
   `x`, not the weights.

**Gauge.** Only *rank* is reparametrisation-invariant; `F → JᵨᵀFJᵨ` is a **congruence**, not a
similarity, so eigenvalues and the condition number are not, and trace normalisation removes *scale*
not gauge. Cross-model eigenvalue comparison is legitimate only in a fixed gauge — which this setup
has by construction, since `x` is the shared physical input in identical radian units for every
model (same rows from `sample_X`). Keep `x` in radians; `x → cx` multiplies `F` by `c²`.

### 3.2 Conditioning on a shared support requires centering

For the boson-vs-determinant comparison, restricting to a common support `C` means **conditioning**,
`q = p/Z_C` with `Z_C = Σ_{n∈C} pₙ`, whose score is the **centered** score:

```
∂ᵢ log q  =  ∂ᵢ log p − ∂ᵢ log Z_C ,      ∂ᵢ log Z_C = E_q[∂ᵢ log p]
F^(C)ᵢⱼ   =  Cov_q(∂ᵢ log p, ∂ⱼ log p)                  ← a COVARIANCE, not E_q[· ·]
```

Omit the centering and you overestimate by the rank-1 outer product `(∂log Z_C)(∂log Z_C)ᵀ`. Not
small here: at `m=6,k=3` the collision-free sector is **20 of 56** outcomes, so `Z_C` is far from 1
*and* strongly `x`-dependent. In Gram form: `q`-weight and `q`-mean-center `J`'s rows before the SVD.

### 3.3 The global-phase mode, and the projection invariant

When `n_features == m`, adding a constant to every `xᵢ` multiplies `D(x)` by a global phase and is
exactly unobservable, so `1/√n_f` is an exact null direction and `rank F = n_f − 1`. When
`n_features < m` the un-encoded modes break the degeneracy.

That would make the headline `(m,k)` sweep inconsistent: `(6,3),(8,4),(10,5)` at `n_f=6` gives
rank 5 at the first point and 6 at the others, so `tr F` sums 5 nonzero eigenvalues at one grid
point and 6 at the rest, and `r_eff` / `exp(H)` inherit the discontinuity.

**Invariant: `F` is always evaluated on the `(n_f − 1)`-dimensional orthogonal complement of
`1/√n_f`.** Project `J → J(I − 11ᵀ/n_f)` before the SVD, at every `(m,k)`, giving a consistently
5-dimensional spectrum. At `m=6` that discards nothing; at `m>6` it discards a genuinely informative
direction, so **report that direction's eigenvalue separately**. The same projection fixes the
`A/D/E` degeneracy of §4.4, so it is one shared invariant, not two patches.

This does **not** settle the `2m²−1` weight count — that needs a weight-space rank. Different gauge
question; do not cross-reference.

### 3.4 Description cost: a count, not a shape

```
log 𝒩(ε)  =  ½ Σᵢ log(1 + λᵢ/ε²)   ≈   ½ Σ_{λᵢ > ε²} log(λᵢ/ε²)
```

Reverse water-filling. Cost = `r_eff(ε)` (above-threshold directions) × average log-excess.

**Spread is strictly more expensive.** At fixed trace `T` with `r` equal above-threshold
eigenvalues, `f(r) = (r/2)log(1 + T/(rε²))` has

```
f′(r) = ½[log(1+x) − x/(1+x)] > 0   for all x = T/(rε²) > 0
```

since `h(x) = log(1+x) − x/(1+x)` has `h(0)=0`, `h′(x) = x/(1+x)² > 0`. And by Jensen,
`Σ log(1+λᵢ/ε²) ≤ n_f log(1 + T/(n_f ε²))` with equality iff flat. So the ordering
`r_eff=1 (cheapest) → gapped → log-uniform → flat (most expensive)` is **proved**, not asserted.

Note magnitude enters only *logarithmically* while the count enters *linearly* — so "big eigenvalues
cost more" is right per-direction and misleading overall. At `k=40, T/ε²=10⁶`: flat ≈ 202 nats,
`r_eff=1` ≈ 6.9 nats.

**The threshold is set by sample size:** `τ_N ~ 2π log N / (γN)`. "Above threshold" means above
`~1/N` in **absolute** terms, not relative to `λ_max` — a spectrum with a fine condition number can
have every eigenvalue below `1/N`.

`ε`-scaling distinguishes the shapes, and it is the *robustness* that matters:

| shape | `r_eff` as `ε → 0` | cost | truncation |
|---|---|---|---|
| gapped (`r` stiff, then cliff) | saturates at `r` | `≈ ½rL` (linear) | survives sharpening `ε` |
| log-uniform (`log λᵢ ≈ log λ₁ − ci`) | grows as `L/c` | `≈ L²/(4c)` (**quadratic**) | tolerance-specific |
| flat | saturates at `n_f` | `≈ ½n_fL` (linear) | survives |

with `L = log(T/ε²)`. Derivation of the log-uniform cost: above threshold when `i < L/c`, so
`½Σ_{i<L/c}(L − ci) = ½[L²/c − L²/2c] = L²/(4c)`.

**At `n_f = 6` the `r_eff(τ)` curve has ≤ 6 steps and cannot distinguish these regimes.** Since
`n_features` is a study invariant, this is a permanent limitation, not a deferred one: emit the
curve as a diagnostic, print **no shape label**, and do not fit a slope to 6 points. What *is* valid
at `n_f=6` — and is the comparison actually wanted — is the cross-model and cross-`(m,k)` comparison
of `tr F`, `r_eff(τ_N)`, `rank`, `exp(H)` at fixed `n_f`.

### 3.5 Report absolute and shape, always both

```
tr F = Σλᵢ            → sample complexity     (ABSOLUTE, in matched physical units)
λᵢ / Σλⱼ              → compressibility, cross-model comparison   (SHAPE)
```

**Normalising away the trace destroys the plateau signal**: if `λᵢ(S) = 2^{−S}μᵢ` the factor cancels
exactly and the normalised spectrum is `S`-independent, so it cannot see uniform collapse.

Secondary: spectral entropy `H = −Σλ̂ᵢ log λ̂ᵢ` with `exp(H) ∈ [1, n_f]`, participation ratio
`(Σλ)²/Σλ²`, numerical rank at a stated tolerance.

**Do not use rank to look for barren plateaus.** A plateau is typically *full rank* with all
eigenvalues uniformly tiny; only magnitudes detect it. Conversely an exact null direction means the
parametrisation is redundant and the manifold is lower-dimensional — learning is *easier* there.
Distinguish exact zeros (a symmetry, provable) from exponentially small ones (a budget). The FIM is
local and blind to discrete symmetries, so full rank never certifies global identifiability
(Rothenberg 1971 gives only the local statement).

### 3.6 What the spectrum does and does not measure

**Measures:** incompressibility of the parametrised family at your sample resolution, and
directional estimability.
**Does not measure:** computational hardness of evaluating or sampling `p`.

Boson sampling is the decisive counterexample and it is *our own model*: the family is indexed by
`U`, i.e. `O(m²)` numbers — polynomially describable, maximally compressible — while each
`p(n) = |Perm(U_{s,n})|²` is `#P`-hard. Knowing the parametrisation exactly buys nothing
computationally.

**So the boson-vs-determinant spectrum comparison is a learnability / effective-input-dimension
statement, and must not be presented as evidence about the `Perm`/`det` separation.** This belongs
in the module docstring, not only here.

Three axes moving in *opposite* directions as `(m,k)` grow:

| quantity | governed by | behaviour |
|---|---|---|
| description complexity | `n_f`; `λᵢ` vs `ε²` | `≤ n_f`, and **decreasing** |
| sample complexity | `1/λ_min` | increasing |
| computational complexity | cost of one `p(n)` | increasing |

Small eigenvalues mean the family is trivial to summarise **and** its parameters unlearnable — the
same statement. Via Pinsker, `TV ≤ (D/2)√λ_max`, so no test on `N ≲ 1/λ_max` samples tells which
`x` was used.

---

## 4. Analysis B — observable (`metrics/observable.py`)

How much of the distribution's input-information a given readout captures. Same differentiation
target (`x`), all exact from the shared score matrix `S = ∂log p/∂x` and the registry's `ψ`.

### 4.1 Efficiency, two readings

```
local  (per direction i):   ρ²(O, sᵢ)  =  Cov(O,sᵢ)² / (Var(O)Var(sᵢ))  =  F_{O,ii} / F_ii
joint  (all directions):    η_O        =  gᵀF⁺g / V_eff ,     gᵢ = Cov(ψ, sᵢ)
```

Since `E[s] = 0` makes `F = Cov(s)` exactly. The local form is a ratio of **diagonal entries**; the
joint form is the squared **multiple** correlation. `η_O ≥ maxᵢ ρ²(O,sᵢ)`, generically strict once
the `sᵢ` are correlated (equality only when `O` lies along a single score component — assert the
inequality, not the strictness). They answer different questions and only the joint one is
congruence-invariant. **Report both.**

**`η ∈ [0,1]` by Cauchy–Schwarz — exact, not asymptotic.** For any `u`:

```
uᵀF_T u = Cov(ψ, uᵀs)² / V_eff ≤ Var(ψ)Var(uᵀs)/V_eff = uᵀFu
```

so `F_T = ggᵀ/V_eff ⪯ F`; then `v = F⁺g` gives
`(gᵀF⁺g)²/V_eff ≤ gᵀF⁺FF⁺g = gᵀF⁺g`, hence `η ≤ 1`. Because this is exact on the discrete
distribution, **a violation is unambiguously a code bug** (Jacobian or pseudo-inverse projection),
never a sampling artifact — which is what makes the test worth having.

Equality iff `ψ` is the score. So **the information-optimal readout is the score itself** — and for
a boson sampler the score's evaluation cost *is* the `#P`-hardness. The optimal readout is precisely
the unusable one, and every implementable observable is inefficient by construction. A statement
about cost, not information: the same describability/simulability separation as §3.6, from the
other side.

### 4.2 Measured `V_eff` — the framework reproduces a known result

| observable | `V_eff` | note |
|---|---|---|
| `parity` | 1.00 | `v = ±1`, so `Var ≈ 1` |
| `n_first` | 0.197 | |
| `ent` | 0.266 | **cheaper than parity** |
| `sq_parity` | 0.0025 | (`η` is scale-invariant, so this is fine) |
| `pairprod` | 0.161 | |
| `xent_parity` | 16.2 | |
| `single_output` | 35.3 | peak-normalised ratio |
| **`osc`** | **1763** | four orders above parity |

`osc` traces directly to `φ′(p) = sin(1/(p+ε)) − p·cos(1/(p+ε))/(p+ε)²` reaching `~1/ε` near
`p ~ ε`. This *quantifies* "hardest family to estimate", which the docstrings previously only
asserted. And `ent` being cheaper than `parity` independently reproduces the shot-correlation table
already recorded in `oscillatory.py` (`ent` 0.82 vs `parity` 0.79 at 100 shots).

### 4.3 Rank-1, and how to read it

`F_O(x) = ggᵀ/V_eff` is **rank 1**: one scalar readout informs one direction in input space at one
point. With any other component treated as a nuisance, the Schur complement is identically zero:

```
F^eff₁₁ = F₁₁ − F_{1r}F_{rr}⁺F_{r1} = F₁₁(1 − ϱ²),   ϱ² = 1 for rank-1   ⟹  F^eff = 0
```

**Scope this as a caveat, not the headline.** This study never poses that inverse problem: the
dataset is `(x, ⟨O⟩ₓ)` and the learner *receives* `x`. Nobody infers `x` from `O`, so there is no
nuisance parameter and `F^eff = 0` is a true fact about a problem the pipeline does not pose.
Reporting it as the headline would replace a useful signal measure with an identically-zero one.

Therefore: report per-direction `F_{O,ii}` and `shots_required` as **signal-carrying** measures,
attach the rank-1 fact as a docstring caveat against estimation-bound readings, and compute the
`A/D/E` summaries on the `x`-**averaged** `F̄_O = E_x[ggᵀ/V_eff]`, which is generically full rank.
(The earlier claim that rank "grows to `min(N,n_f)` when averaged over `x`" is *arithmetically*
right — averaging rank-1 matrices gives full rank — only its interpretation was wrong: averaging
over different `x` cannot resolve several components of *one* `x`.)

### 4.4 The observable set, and the constructed optimum

A single observable is not the right object. For a vector `O = (O₁..O_R)` with covariance `Σ`:

```
F_O = Cov(O,s)ᵀ Σ⁻¹ Cov(O,s) ,     η_O = R² of regressing s on O
O*  = βᵀO ,   β = Σ⁻¹Cov(O, s)          ← projection of the score onto span(O)
```

with `η_{O*} ≥ max_r η_{O_r}` and `Σ⁻¹` automatically discounting redundant observables. Directly
usable: the registry has ~17 families, and every `Expectation` is `probs @ v`, so a linear
combination is *itself* an `Expectation` with score vector `Σ_r β_r v_r` — `O*` is constructible
inside the existing class. Influence functions are linear in the functional, so `ψ* = Σ_r β_r ψ_r`
and the whole machinery works in `ψ`-space regardless of class; the only loss is that a mixed-class
`O*` is a combination of estimators rather than one `Expectation`.

**Scalar summaries** (`k = rank F`), all in `[0,1]` and reparametrisation-invariant because the
`I^{−1/2}` sandwiching cancels the congruence:

| summary | formula | reads as |
|---|---|---|
| A-type | `(1/k) tr(I^{−1/2}F_O I^{−1/2})` | mean fraction captured |
| D-type | `(det F_O / det I)^{1/k}` | confidence-ellipsoid volume ratio |
| E-type | `λ_min(I^{−1/2}F_O I^{−1/2})` | worst-direction fraction |

**Lead with E-type**, since a single blind direction is what §4.3 is about. D-type is what the
distinguishability literature optimises.

**Two implementation traps, both of which would silently produce wrong numbers:**

1. **All three are ill-defined on the raw `F`** at the base config: `rank F = 5 < 6`, so `I^{−1/2}`
   does not exist, D-type is `0/0`, and E-type is `λ_min = 0` *identically for every observable
   set*. Evaluate on `range(F)` with `k → rank F = 5`, using the §3.3 projection.
2. **Pair the averages.** `F_O(x) ⪯ F(x)` pointwise and linearity preserves the PSD order, so
   `E_x[F_O] ⪯ E_x[F] = F̄`. The sandwich must be `F̄^{−1/2}F̄_O F̄^{−1/2}` — both sides over the
   *same* pool. Pairing `F̄_O` with a single-point `F(x₀)` is ill-formed and **can exceed 1 with no
   bug present**.

**Ranges differ by case — assert per case, never one blanket bound:**

```
pointwise rank-1:            tr(F(x)⁺F_O(x))  ∈ [0, 1]
averaged / multi-observable: tr(F̄⁺F̄_O)       ∈ [0, k]      ← since tr(F̄⁺F̄) = rank F̄ = k
every A/D/E summary:                          ∈ [0, 1]
```

That `[0,k]` is exactly *why* A-type carries the `1/k`. Implement one function
`efficiency = tr(F⁺F_O)` **without** the `1/k` and divide at the A-type call site — with `k=5` the
two differ by a factor of 5.

Also: for rank-1 `F_O`, `tr(F⁺F_O) = gᵀF⁺g/V_eff` exactly, so §4.1-joint and A-type-at-`R=1` are one
quantity — implement once.

### 4.5 Shot budget, and where bias bites

```
Var(O|x) = p·v² − (p·v)²                 exact, for an Expectation
shots_requiredᵢ(x) = V_eff / (∂⟨O⟩/∂xᵢ · δ)²
```

An observable whose `shots_required` exceeds the experiment's `S` cannot support learning *whatever
the learner is*, because the labels are noise-dominated at that budget.

**Bias, not variance, is what breaks nonlinear functionals at finite shots.** The plug-in `T(p̂)` is
biased at `O(1/S)`; for entropy-like functionals `~(support−1)/(2S)`, i.e. `−0.275` nats at
`n_out=56, S=100` — ~8% of bare `ent ≈ −3.5`, but larger than `ent_parity ≈ −0.18` outright. This
gates **only** the shot-budget and `R²`-ceiling readings, which assume **zero-mean** label noise: a
biased label adds a term that does not vanish with more training data. The exact `η_T` is
unaffected, because it never samples.

**Degenerate U-statistics:** if `ζ₁ = Var(Kp) = 0` the `1/S` term vanishes, convergence is `O(1/S)`
with a weighted-`χ²` limit, and `V_eff` is not the operative scale. **Flag** `ζ₁/ζ₂ ≈ 0`, do not
exclude. Measured: `sq_parity` ≈ 0.027, `pairprod` ≈ 0.050 — both comfortably non-degenerate.

**The only exclusion is non-differentiability:** `max_prob` has `ψₙ = 1[n = argmax]`, undefined at a
tie, with an upward-biased plug-in. It raises via `is_differentiable = False`.

### 4.6 `G_O`, the gradient-free screen, and its one-directional logic

The law of total variance is **exact**:

```
Var(O)  =  E_x[σ²ₓ]  +  Var_x(μₓ)                 within + between
```

so the two halves are *complementary, not interchangeable* — at fixed total, one large forces the
other small, and neither is a figure of merit alone. The ratio is (an ANOVA F-statistic):

```
G_O  =  Var_x(μₓ) / E_x[σ²ₓ]
```

It needs **no gradients**, only forward evaluations. Its relation to §4.1 is a **one-directional
inequality**, by Poincaré on `x` uniform over an interval of length `D`:

```
Var_x(μₓ)  ≤  (D²/π²) E_x[(∂μ/∂x)²]
```

- `G_O` large **⟹** `F_O` large somewhere. Sound: no global spread without local sensitivity.
- `F_O` large **⇏** `G_O` large. An oscillating `μₓ` has large `E[(μ′)²]` and small `Var_x(μₓ)`
  from cancellation over `U[0,2π]`.

**So `G_O` prioritises work; it must never exclude a cell.** The failure case is in-repo: the
`exp_poly` scorers (`(−1)^{P(n)}`, `cos(P(n))`) have rapidly alternating-sign score vectors, so
their means are the likeliest in the library to oscillate and screen out while being locally
informative. **Rule: always run the gradient path on `exp_poly` regardless of its `G_O`.**

Two further blind spots: **dead zones** (a sharp jump in one sliver plus flatness elsewhere gives
healthy `G_O` while local estimation is hopeless) and **non-injectivity** (`μ_{x₁} = μ_{x₂}` for
well-separated inputs breaks global identifiability while `F_O` looks fine) — so also check
monotonicity of `μₓ` across the grid. And `G_O` is **grid-dependent**: changing the spacing or range
changes the number without changing `μₓ`, so **state the grid**. `η_O(x)` has no such dependence.

### 4.7 The SNR ceiling, and the gate it needs

At global scale `x ~ U[0,2π]^{n_f}` — which is *literally* how the dataset is drawn (`sample_X`) —
`S·G_O` is the dataset's intrinsic SNR and bounds

```
R²  ≲  SNR / (1 + SNR)
```

**This applies only to `R²` measured against SHOT-NOISY test labels, and the default is noiseless.**
Verified: `GenerationConfig.nsample` defaults to `0`, models apply `_shot_sample` only when
`nsample > 0`, and the learner regresses on the stored `soft`. So with default configs the targets
are exact, the ceiling is 1, and the comparison is **vacuous**. Generate at `nsample = S` and score
against those labels, or the prediction is untestable. Do not read a mismatch as a finding before
confirming which labels the denominator used.

Also excluded: `ProbFunction` labels, per §4.5's bias gate.

This supersedes `varsweep_tmp.py`, which computes the numerator (`Var` of `probs @ score_vec` over
`X`) without the denominator that makes it an SNR.

---

## 5. Parameter counting

```
circuit:         2m² − 1        dim U(m) = m² per Haar unitary, less the unobservable global phase
quadratic_fock:  2r(d_x + d_φ)  low-rank W = ABᵀ;  d_x = 2·order·n_f,  d_φ = m(m+1)/2
mlp_fock:        O(n_fock)      structural, and independent of n_features
```

`2m² − 1` is an *upper* bound on the effective count (an output diagonal phase on `W₂` is also
invisible to a Fock measurement), so treat it as nominal. The input-FIM rank does **not** measure
it — different gauge question.

**`quadratic_fock`'s legacy `O(m³)` was an artifact of `n_features = m − 1`**, which made
`d_x = O(m)` multiply `d_φ = O(m²)`. With `n_f` fixed, `d_x` is constant and the count is `O(m²)` —
the circuit's scaling — with no change to the model. Same scaling is not the same count (1008 vs 71
at `m=6`), so the low-rank factorisation gives `r` as the matching dial:

| m | `d_φ` | circuit `2m²−1` | rank-1, order=1 | rank-2, order=1 |
|---|---|---|---|---|
| 6 | 21 | 71 | 66 | 132 |
| 10 | 55 | 199 | 134 | 268 |
| 14 | 105 | 391 | 234 | 468 |

`param_matched=True` solves for the closest `r` and records the *resolved rank* in `circuit_spec`.

---

## 6. The fermion model: matching support without losing the determinant

`p_fermion(n) = |det(U_{s,n})|²` on the **same** circuit — same `W₁,W₂`, same seed, same input
state, same basis, same scoring code. The canonical classical/quantum line: a permanent is `#P`-hard
while a determinant is `O(k³)`.

**Which index picks rows is fixed by the physics, not a convention.** On the circuit matrix
`M = W₂D(x)W₁`, the **input state picks columns and the outcome picks rows**: the submatrix is
`M[T, S]` with `S` the input's occupied modes and `T` the outcome's. Equivalently, on
`U = Mᵀ` — what `sandwich_unitary_at` returns, and what the code indexes — the input picks rows and the
outcome picks columns, since `U[S,T] = M[T,S]ᵀ`. The `.T` *is* the convention, which is why it must not
be dropped: `U[T,S]ᵀ = Uᵀ[S,T]`, so swapping the roles evaluates on the transposed unitary and gives a
different distribution (measured `|det|²` of `0.0585` vs `0.0011` on one `U`, `S`, `T`). What *is* free
is transposing the submatrix once selected, since `Perm` and `det` are transpose-invariant — which is
why `det(U[S,T]) = det(M[T,S])` and the two readings agree. The merlin agreement below is the check
that this is not off by a transpose.

The phase-power construction inherits the same fixing: `col_p` acts on `U[S, j]`, the amplitudes into
outcome mode `j` indexed by the `k` input modes, because it is the *outcome* mode that repeats.

Strict free fermions put **exactly zero** mass on bunched outcomes (a repeated column kills a
determinant), leaving support at `C(m,k)` — 20 of 56 at `m=6,k=3`. A naive comparison against the
boson model then conflates *support size* with *matrix function*.

**What was tried first, and why it was abandoned.** Giving each mode `r` internal states (spinful
fermions) and summing over flavour-conserving assignments does open the bunched sector, but the
interferometer is flavour-blind, so the lifted unitary `U ⊗ I_r` is flavour-diagonal and the `k×k`
determinant **block-factorises** into blocks of size `k/r`. At `r = k` every block is `1×1`: no
determinant survives at all, and the model reduces to
`p(n) = Perm(|U|²)/∏ⱼnⱼ!` — the classical distinguishable-particle distribution. Precisely the
setting with full support was the setting with zero determinant content, and the two goals were in
direct opposition. (The `216` assignments measured at `m=6,k=3,r=3` are `6³`: all maps from 3
labelled particles to 6 modes, which is the tell.) Retired; `model.flavours` is rejected.

**What replaced it.** The `p`-th copy of a repeated column is modified rather than duplicated:

```
col_p(c) = (c/|c|)^p · |c|^(1 + s(1/p − 1)),     s = k/m
```

Integer power on the **phase**, fractional on the **modulus**. The split is load-bearing:

- the phase power breaks the degeneracy — holding it at 1 leaves both columns with identical
  entrywise phases, hence near-proportional (`cond ≈ 26` vs `≈ 5`), and bunched mass collapses to 0.07;
- it must be an **integer**: `arg` is defined mod `2π`, so a fractional phase power is multivalued and
  jumps by `2π/p` when an entry of `U` crosses the branch cut. That flips *one entry*, and `|det|²` is
  invariant only under a *whole-column* phase — measured `0.318 → 0.0038` across `|Δx| = 3.6e-9`, with
  the finite difference then diverging as `1/h`. Integer powers come around and are exactly continuous;
- the modulus power sets bunched mass. Since `|Uᵢⱼ| < 1`, smaller exponent *enhances* bunching — the
  boson direction. `s = 0` undershoots, `s = 1` overshoots.

`s = k/m` (the filling fraction) is why: boson bunched mass falls as modes grow plentiful relative to
photons, and fixed `s` is nearly flat in `m` so cannot track it. Error in mean bunched mass against
the boson model, over `m = 6..12` at `k = 3, 4`:

| | `s = 0` | `s = 0.5` | `s = 1` | `s = k/m` |
|---|---|---|---|---|
| mean \|err\| | 18.0% | 9.5% | 31.4% | **1.8%** |
| worst case | −25.4% | +22.2% | +67.9% | **−5.5%** |

No free parameter. (`s = 1/2` looks good only at `k/m = 1/2`; fitting `s` per config matches exactly
but defines the model by its comparator, and the objective is flat near the optimum anyway.) No
`1/∏ⱼnⱼ!` factor is applied — invisible on the collision-free sector, underived for the bunched one,
and imposing it pushes the rule's error to 32.9%.

**What this buys and what it costs.** Full `C(m+k−1,k)` support from one `O(k³)` determinant, matched
to the boson model on support *and* bunched mass, differentiable, no flavour bookkeeping. But no
quantum state has this amplitude — an elementwise power of an orbital is not an orbital — so the
bunched sector does **not** carry the `Perm`/`det` hardness framing.

The collision-free sector does, and is untouched: every mode has `c = 1`, so `col₁(c) = c` *exactly*.
`FermionModel.collision_free_probs` is `|det U[S,T]|²` on the shared `C(m,k)` support with no
approximation — **run the headline `Perm`-vs-`det` claim there**, and read the full-support readout as
the support-matched comparator.

**`|Perm|²` is computed in pure torch, not through merlin** (`boson_probs_reference`), from the
*same* `sandwich_unitary_at` the determinant path uses — so `Perm` vs `det` is a one-line swap with
every other factor fixed, and both are differentiable in `x`. Verified against merlin: max
difference **6.7e-8**, with identical key ordering. Cost is `k!` per outcome, so cap at `k ≤ 6`
(`k=3`→6 perms, `k=5`→120, `k=7`→5040) and say so; do not silently sample.

---

## 7. Learner interpretation protocol

The fermion arm **calibrates the learner**, and that must be encoded, not left to the reader — a
`perm`-arm failure alone is confounded by architecture and optimiser, which is the standard
objection that kills training-based evidence.

- **Paired design.** Run `perm` and `det` with *everything* fixed — architecture, hyperparameters,
  `n_train`, seeds, split — and report the **paired difference**.
- **Decision logic, in code:**

| `det` | `perm` | reading |
|---|---|---|
| succeeds | fails | the informative cell |
| succeeds | succeeds | no separation at this size |
| fails | fails | **learner inadequate — comparison VOID**, not a hardness signal |
| fails | succeeds | investigate; likely a bug or mismatched control |

Row 3 must be emitted loudly: it is the failure mode that otherwise gets written up as a result.

- **Adjudicate on held-out log-likelihood, not `R²`.** With exact `probs` available, the held-out
  mean log-likelihood against the ideal model is an unbiased KL estimate — stronger and far less
  scale-sensitive than `R²` on scores, at zero extra simulation cost. Keep `R²` as secondary.

**Multiple comparisons:** with `R` observables × `M` grid points you are implicitly selecting a
maximum, and bootstrap bands do not cover that selection. Fix the comparison set in advance, or
split the pool into selection and confirmation halves.

---

## 8. Reporting checklist

**Analysis A**
1. `tr F` in matched physical units (absolute) **and** the normalised spectrum — never one alone
2. `r_eff(τ_N)` with `τ_N` stated; the `r_eff(τ)` curve, with **no shape label** at `n_f=6`
3. `rank F` by eigenvalue ratios, not `matrix_rank`; the excluded `1`-direction eigenvalue separately
4. median + IQR of every statistic across the input pool, not a single `x`
5. conditioned `F^(C)` centered, when comparing on a shared support

**Analysis B**
6. `η_O` (joint) and `ρ²(O,sᵢ)` (local), with the per-case range asserted
7. `V_eff` per observable, and `ζ₁/ζ₂` for the quadratics
8. joint `R²` over the set, constructed `O*`, greedy forward-selection ranking
9. E-type first, on `range(F)` with `k = rank F`, both sides averaged over the same pool
10. `shots_required` labelled as a signal proxy, not an estimation bound
11. `G_O` with the grid stated, as a prioritiser; monotonicity flag for non-injectivity
12. `R²` ceiling **only** against shot-noisy labels, and never for `ProbFunction`
