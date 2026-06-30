# Quantum vs Classical Learning Benchmark — Staged Pipeline

> **Note.** This document describes the **new, config-driven staged architecture**
> (`Generator/`, `model/`, `embedding/`, `kernel/`, `analyzer/`). The original
> flat, CLI-flag layout (`train.py`, `benchmark_6.py`, `data/`, `learner/`) is
> documented in [`README.md`](README.md) and still lives in the tree. The two
> coexist for now; this file covers the pipeline that replaces the inline data
> generation and adds a kernel-methods analysis path.

This benchmark re-implements and extends:

> Havlíček, Córcoles, Temme, Harrow, Kandala, Chow, Gambetta.
> *Supervised learning with quantum-enhanced feature spaces.*
> **Nature 567, 209–212 (2019)** — [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)

It generates synthetic datasets from four **teacher** strategies (photonic-quantum,
qubit-quantum, analytical, MLP), then probes how learnable that data is — both by
training learners and, in the new pipeline, by comparing **kernel embeddings**
(the *Power of Data* experiment, Huang et al. 2021, [arXiv:2011.01938](https://arxiv.org/abs/2011.01938)).

---

## Why a staged pipeline?

The old design generated, filtered and balanced data **inside each learner**, so
two learners could silently see different `X`, and re-running with a different
margin meant regenerating everything. The new design splits the work into
independent stages, each with its own typed YAML config and `python -m <pkg>` CLI,
connected only by **content-hashed artifacts on disk**:

```
                 problem + generation + seeds            split (train/test)
                 ───────────── hashed ───────────►        ──── full pool ────►

  ┌───────────┐   ┌───────────┐   datasets/    ┌───────────┐   ┌───────────┐
  │  model    │──►│ Generator │──► <hash>/  ──► │ embedding │──►│  kernel   │
  │ (teachers)│   │  (draw +  │   data.pt       │ (feature  │   │  (RBF     │
  └───────────┘   │   persist)│   teacher.pt    │   maps,    │   │   Gram)   │
                  └───────────┘   meta.json     │  full X)   │   └─────┬─────┘
                        │                       └─────┬─────┘         │
                        │                             │               ▼
                        └────────────► analyzer ◄─────┴───────► kernel-kernel /
                                  (preview, compare, metrics)    target alignment
                                   prepare = margin/balance DIAGNOSTIC only
```

Three properties fall out of this:

- **Reproducibility by construction.** Same `(sample_seed, teacher_seed,
  generation params)` ⇒ byte-identical pool. The artifact hash *excludes* `size`,
  so growing a pool reuses the same directory and reproduces earlier rows exactly.
- **Nothing is discarded.** `load_split` partitions the *full* pool into
  train/test (deterministic in `split_seed`); margin filtering / class balancing
  are a separate **diagnostic** the analyzer can apply, never a silent load-time
  transform. Downstream learners get every sample and slice by `train_idx`/`test_idx`.
- **Embeddings are pure functions of their key.** Features are computed over the
  full `X`, so a cached blob `embeddings/<dataset_hash>/<name>_<spec-hash>.pt`
  depends only on `(dataset_hash, spec)` — not on any prepare setting.

---

## The five packages

| Package | Stage | Entry point | What it owns |
|---|---|---|---|
| **`model/`** | Teachers | (imported) | The data-generating models + a feature-map view of the quantum ones |
| **`Generator/`** | Generate | `python -m Generator` | Draw a seeded pool, run a teacher, persist raw output; full-pool train/test split |
| **`embedding/`** | Embed | `python -m embedding` | Feature maps over a dataset's **full** `X`, cached per dataset hash |
| **`kernel/`** | Kernel | (imported) | A feature-agnostic Gaussian RBF Gram |
| **`analyzer/`** | Analyze | `python -m analyzer {preview,compare}` | Preview datasets; kernel-kernel & kernel-target metrics; margin/balance diagnostic |
| **`learn/`** | Learn | `python -m learn` | Train an RBF-SVM on each embedding, sliced by the shared split; report test accuracy |

### `model/` — teachers (and quantum feature maps)

A **teacher** maps `X (N, n_features) → soft (N, 1)` or `(N, c)` and *nothing else*
— no labelling, filtering or balancing. Teachers subclass `Teacher`
(`model/base.py`), auto-register into `TEACHERS` by `name`, and describe
themselves via `from_config(cfg)` and `hash_spec(cfg)` (the model-specific knobs
folded into the artifact hash). They are `nn.Module`s, so weights round-trip
through `teacher.pt`.

| `name` | File | Teacher |
|---|---|---|
| `photonic_quantum` | `photonic.py` | `W1 (Haar) → phase-encode(x) → W2 (Haar)` on Merlin/Perceval; observables `parity`/`majority`/`bunching`/`single_output` |
| `qubit_quantum` | `qubit.py` | Havlíček IQP feature map + random variational circuit; `sign(⟨Z^⊗n⟩)` (the paper's setup) |
| `analytical` | `analytical.py` | `sign(Σ_{i<j} sin(k·(x_i − x_j)))` — deterministic closed form |
| `mlp` | `mlp.py` | Random-weight tanh MLP over Fourier features; softmax soft targets |

The quantum modules also expose `QubitFeatureMap` and `PhotonicFeatureMap`, reused
by the embedding stage to build *projected* quantum kernels. `sampler.py` provides
the single shared seeded input draw `sample_X`.

### `Generator/` — generate, split (and a prepare diagnostic)

- **`config.py`** — the typed `ExperimentConfig`. Each section has one job; there
  is **no** `prepare` section and **no** `model_seed` (a learner knob):

  ```yaml
  problem:    { m, k, observable, n_features }   # geometry
  generation: { generator, size, nsample }       # → affect the saved pool (hashed)
  split:      { test_fraction, split_seed }       # the one shared train/test partition
  seeds:      { sample_seed, teacher_seed }       # define X and the labels (hashed)
  ```

- **`generate.py`** — draw `X` via `sample_X`, run the teacher, save. Caches: if an
  artifact ≥ `size` exists it's reused; a smaller one is *extended* (earlier rows
  unchanged). Prints the pool size and class balance.
- **`artifact.py`** — on-disk layout, one dir per logical dataset:

  ```
  datasets/<generator>_m<m>_k<k>_<8-char-hash>/
  ├── data.pt      # {"X", "soft"} — RAW, no label/filter/balance
  ├── teacher.pt   # teacher state_dict (if available)
  └── meta.json    # provenance + the model-specific hash spec + current size
  ```

- **`prepare.py`** — labelling inferred from the *soft output shape* (`(N, 1)` →
  `y = (soft ≥ 0)`, margin `|soft|`; `(N, c)` → `y = argmax`, margin = top-two gap),
  plus a **diagnostic** `prepare_indices` / `prepare` that returns the rows
  surviving a margin filter + optional class balance. This is *not* on the
  load/train path — the analyzer invokes it to report separability.
- **`split.py`** — `load_split(...)`, the one shared loader. Partitions the **full**
  pool into a deterministic seeded `Split` and exposes `train_idx`/`test_idx`
  (indices into the raw pool order, so embeddings can be sliced by them). No
  filtering, no balancing.
- **`seeding.py`** — `seed_everything`.

### `embedding/` — kernel representations of a dataset

An *embedding* is a feature map `features(X) → (N, d)` plus the params that
identify it for caching. The RBF kernel is applied later, so an embedding is fully
described by *how the features are produced*:

| `type` | Features | Source |
|---|---|---|
| `rbf` | raw angles `X` | Huang universal baseline |
| `fourier_rbf` | `[sin(jx), cos(jx)]_{j=1..order}` | periodicity-aware |
| `qubit_projected` | 1-qubit reduced-state Pauli expectations `⟨X_q⟩,⟨Y_q⟩,⟨Z_q⟩ → 3n` | `model.qubit.QubitFeatureMap` |
| `photonic_projected` | occupation moments `⟨n_i⟩ + ⟨n_i n_j⟩ → m(m+1)/2` | `model.photonic.PhotonicFeatureMap` |

An **embedding config** references a data config and lists the kernels; an
embedding seed equal to the dataset's `teacher_seed` is tagged **matched** (the
rigged sanity cell), any other seed is the honest unmatched ensemble. Blobs are
cached separately from datasets:

```
embeddings/<dataset_hash>/<embedding-name>_<spec-hash>.pt
```

(Fidelity kernels `|⟨φ|φ'⟩|²` were intentionally dropped — not hardware-realizable
and the only thing that needed an N×N Gram on disk.)

### `kernel/` — the Gram

A single Gaussian RBF over a precomputed feature matrix (`rbf.py`), agnostic to
where features came from. `gamma="median"` fits the median-pairwise-distance
heuristic on the training call and reuses it for cross-Grams so train/test
kernels stay consistent. `cache.py` turns a stored embedding blob into a Gram.

### `analyzer/` — preview, compare, diagnostics

Two subcommands (`python -m analyzer {preview,compare}`; a bare `--config` defaults
to `preview` for backward compatibility):

- **`loader.py`** (`preview`) — resolve the artifact a config points to and preview
  the raw pool, reporting class counts and (with `--min-margin`) how many samples
  clear that margin. Read-only; never filters the stored pool.
- **`compare.py`** (`compare`) — build/load cached embeddings, RBF each into a Gram,
  and print the kernel-kernel **CKA** matrix (names auto-tagged by seed /
  matched-unmatched). `--target` adds kernel-target alignment. `--min-margin` /
  `--balanced` opt into the **separability diagnostic** (restrict to the
  margin-filtered / balanced rows). `-n` caps rows for the O(N²) Gram.
- **`metrics.py`** — label-free **centered kernel alignment** (`kernel_alignment`,
  `kernel_kernel_matrix`), plus the Huang Power-of-Data metrics: **kernel-target
  alignment** (`target_alignment`) and the **geometric difference**
  `g(K_classical ‖ K_quantum)` (`geometric_difference`) — a large `g` is the
  precondition for a potential quantum advantage on the data.

### `learn/` — train a classifier on an embedding

The downstream consumer the decoupling enables. `svm.py` (`run_svm`) builds/loads
the embeddings (full pool), takes labels `y = derive_labels(soft)` from the
teacher output, slices each feature matrix by the shared `train_idx` / `test_idx`,
and fits an **RBF-kernel SVM** (sklearn `SVC`) per embedding — reporting train/test
accuracy. Nothing is re-embedded. `--n-train` / `--n-test` cap the rows (the SVM is
~O(N²)); `--min-margin` / `--balanced` opt into the same separability diagnostic.

---

## Prerequisites

```bash
pip install -e .                                  # install merlin
pip install perceval-quandela torch scikit-learn matplotlib pyyaml
```

Run everything from this directory (`papers/quantum_feature_spaces/`). Each stage
adds the paper root to `sys.path` when run as a bare script, so both
`python -m Generator ...` and `python Generator/generate.py ...` work.

---

## Quick start

### 1. Generate a dataset

```bash
python -m Generator --config configs/example_photonic.yaml
# add --force to regenerate from scratch, --out-root to change the artifact root
```

Writes `datasets/<generator>_m<m>_k<k>_<hash>/` and prints the pool size + class balance.

### 2. Preview it

```bash
python -m analyzer preview --config configs/example_photonic.yaml -n 10 --min-margin 0.1
# (a bare `--config` without `preview` also works)
```

### 3. Build embeddings

```bash
python -m embedding --config configs/embed_example.yaml
# writes embeddings/<dataset_hash>/...  (requires the dataset from step 1)
```

### 4. Compare kernels

```bash
python -m analyzer compare --config configs/embed_example.yaml --target
```

Loads the cached embeddings (build them in step 3 first), RBFs each into a Gram,
and prints the pairwise **centered kernel alignment** matrix (`*` = matched
teacher seed). `--target` additionally reports each kernel's **kernel-target
alignment** (how learnable the data is for that kernel).

The RBF Gram is O(N²), so the rows are capped at `-n 2000` by default; pass a
larger `-n`, or `-n 0` to use all rows (watch memory). `--min-margin` / `--balanced`
restrict the comparison to the separability-diagnostic subset. `--force` recomputes
embeddings instead of loading the cache.

Equivalently, from Python:

```python
from analyzer import compare_from_embeddings, print_matrix

names, M = compare_from_embeddings("configs/embed_example.yaml")
print_matrix(names, M)        # pairwise centered kernel alignment; *=matched seed
```

### 5. Train a learner on the embeddings

```bash
python -m learn --config configs/embed_example.yaml --n-train 1500 --n-test 1000
```

Builds/loads the embeddings, takes labels from the teacher, slices each feature
matrix by the shared split, and fits an RBF-SVM per embedding:

```
[learn] dataset f70af2d5  RBF-SVM C=1.0  n_train=1500 n_test=1000

               embedding   dim   train    test
             fourier_rbf    30   0.964   0.804
     qubit_projected@42*    15   0.742   0.509
                     rbf     5   0.580   0.507
       ...
```

Equivalently, the manual slicing the learn stage does for you:

```python
from Generator import load_config, generate, load_split

cfg   = load_config("configs/example_photonic.yaml")
path  = generate(cfg)                              # draw + save (or reuse cache)
split = load_split(path,                           # full pool -> train/test indices
                   test_fraction=cfg.split.test_fraction,
                   split_seed=cfg.split.split_seed)

# embeddings are stored over the FULL X, so slice them by the same indices:
#   F = blob["data"]
#   svm.fit(F[split.train_idx], split.y_train)
#   svm.score(F[split.test_idx], split.y_test)
```

---

## Example configs

**`configs/example_photonic.yaml`** — a data (generation) config:

```yaml
problem:
  m: 6
  k: 3
  observable: parity          # parity | majority | bunching | single_output
  n_features: null            # null -> m-1

generation:
  generator: mlp              # photonic_quantum | qubit_quantum | analytical | mlp
  size: 100000                # raw pool size (nothing discarded at load time)
  nsample: 0                  # 0 = exact; >0 = finite-shot simulation

split:                        # the one shared train/test partition (full pool)
  test_fraction: 0.20
  split_seed: 0

seeds:
  sample_seed: 42             # the input points X       (in the hash)
  teacher_seed: 42            # the teacher's weights     (in the hash)
```

Margin filtering / class balancing are a load-time **diagnostic** (run via
`analyzer preview --min-margin …` or `compare --min-margin … --balanced`), not a
generation knob; `model_seed` belongs to the (future) learn stage, not here.

**`configs/embed_example.yaml`** — an embedding config referencing the data config:

```yaml
dataset: configs/example_photonic.yaml

embeddings:
  - {type: rbf}
  - {type: fourier_rbf, fourier_order: 3}
  - {type: qubit_projected, depth: 3, seed: 42}   # 42 == teacher_seed -> matched
  - {type: qubit_projected, depth: 3, seed: 7}    # unmatched
```

---

## Config field reference

### `problem`
| Field | Default | Description |
|---|---|---|
| `m` | 6 | Optical modes (photonic) / qubit count (qubit) |
| `k` | 3 | Photons / teacher depth / Fourier frequency |
| `observable` | `parity` | Photonic observable: `parity`, `majority` (even `m`), `bunching`, `single_output` |
| `n_features` | `null` (→ `m-1`) | Encoded input dimension; must be ≤ `m-1` |

### `generation`
| Field | Default | Description |
|---|---|---|
| `generator` | `photonic_quantum` | `photonic_quantum`, `qubit_quantum`, `analytical`, `mlp` |
| `size` | 10000 | Raw pool size (excluded from the artifact hash) |
| `nsample` | 0 | `0` = exact probabilities; `>0` = finite-shot simulation |

### `split` (the shared train/test partition over the full pool)
| Field | Default | Description |
|---|---|---|
| `test_fraction` | 0.20 | Fraction held out for test |
| `split_seed` | 0 | Seeds the (prepare-independent) train/test permutation |

### `seeds`
| Field | Default | Hashed? | Description |
|---|---|---|---|
| `sample_seed` | 42 | yes | The input points `X` |
| `teacher_seed` | 42 | yes | The teacher's weights (labelling fn) |

### Diagnostic (CLI flags, **not** config — analyzer only)
| Flag | Default | Description |
|---|---|---|
| `--min-margin` | 0.0 | Keep only samples whose confidence ≥ this |
| `--balanced` | off | Subsample every class to the minority count |

---

## Mapping from the old layout

| Old (`README.md`) | New |
|---|---|
| `data/photonic_quantum.py`, `qubit_quantum.py`, `analytical.py`, `mlp.py` | `model/photonic.py`, `qubit.py`, `analytical.py`, `mlp.py` (now `Teacher` subclasses) |
| `data/_resample.py` (iterative re-draw) | `Generator/prepare.py` (a *diagnostic* filter/balance — no longer on the load/train path) |
| inline generation inside each learner | `Generator/generate.py` + on-disk hashed artifacts |
| `train.py --m … --observable … --min-margin …` flags | YAML `ExperimentConfig` (`configs/*.yaml`) |
| SVM Fourier features | `embedding/` (`fourier_rbf`) + `kernel/` (RBF) |
| *(none)* | `embedding/` projected quantum kernels, `kernel/` Gram, `analyzer/` Power-of-Data metrics |

> **Partially ported:** the pipeline now covers **generate → embed → kernel-compare
> → learn**, where `learn/` trains an RBF-SVM on each embedding (sliced by the
> shared split). Still missing from the old `learner/` package: the photonic/qubit
> *variational* learners and the minimum-model-size-to-90% search. The SVM learn
> stage is the first, simplest member of that stage.

---

## Testing

Each package ships its own tests:

```bash
pytest Generator/tests model/tests embedding/tests kernel/tests analyzer/tests learn/tests
```