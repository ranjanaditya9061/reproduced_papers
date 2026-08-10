"""Generation: draw the input pool, then populate a branch of the store.

    python -m pipeline.generate --config configs/photonic.yaml
    python -m pipeline.generate --configs-dir configs

Both branches hang off one circuit identity (:func:`~pipeline.artifact.circuit_hash`), which covers
the input and the circuit and **nothing else**, so they share one directory::

    datasets/<circuit_hash>/exact/                dist.npz   keys + probs + probs_at_zero
    datasets/<circuit_hash>/counts/<shot_tag>/    shots.npz  keys + per-shot sequence

**Both directions extend rather than recompute.**

* *rows* -- ``sample_X`` is prefix-stable and the model deterministic in ``model_seed``, so raising
  ``generation.size`` simulates only the new rows; the reused prefix is byte-identical.
* *shots* -- the draw is seeded on the shot offset, so raising ``generation.shots`` simulates only
  the new shots and appends them.  Going 20k -> 30k costs a 10k draw, not a re-simulation.

**Lowering ``generation.shots`` is not a generation job at all.**  The store keeps the shots in draw
order, so a smaller budget is a prefix crop in :func:`~pipeline.shots.load_shots` -- the actual first
N shots, not a resample.  That is why the store is never shrunk here: it only ever grows, and every
budget below its width stays exactly reproducible from it.

Both work because neither ``size`` nor the budget is hashed.  With ``nsample`` in the hash, 20k and
30k were different directories, so the second run re-ran the exact simulation (76% of the work,
producing bit-identical output), redrew from shot zero, and still did not yield an extension of the
first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):                    # allow `python pipeline/generate.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pipeline"

import torch

from config import ExperimentConfig, load_config
from model import build_model, sample_X

from .artifact import DIST_FILENAME, SHOTS_FILENAME, exact_path, load_meta
from .distribution import check_size, load_dist, save_dist
from .shots import load_shots, save_shots, shots_path, to_seqs


def _resolve_n_outcomes(model, X: torch.Tensor) -> int:
    """``model.n_outcomes``, simulating a single row first if the prep needs one to discover its
    basis (``spin``/``spin_magic`` -- see ``model.photonic.PhotonicModel.outcome_keys``).

    One row is enough: :meth:`~circuit.prep.StatePrep.outcome_keys` only needs *a* backend result
    to read the basis off of, and it is cheap relative to the full-size call this guards --
    exactly the reverse of calling :meth:`~model.base.DistributionModel.probs` on the whole pool
    just to size the pre-flight check that is supposed to run *before* that call.
    """
    try:
        return model.n_outcomes
    except RuntimeError:
        model.probs(X[:1])
        return model.n_outcomes


def generate_exact(cfg: ExperimentConfig, *, root: str | Path = "datasets",
                   force: bool = False) -> Path:
    """Create, reuse or row-extend the exact branch; return its directory."""
    model = build_model(cfg)
    path = exact_path(cfg, model, root)
    target = int(cfg.generation.size)
    X = sample_X(target, cfg.problem.n_features, cfg.seeds.sample_seed)
    n_out = _resolve_n_outcomes(model, X)
    check_size(target, n_out, cfg.generation.max_dist_bytes, m=cfg.problem.m, k=cfg.problem.k)

    have, cached = 0, None
    if not force and (path / DIST_FILENAME).exists():
        have = int(load_meta(path)["size"])
        if have >= target:
            print(f"[exact] cache hit: {path} ({have} rows, {n_out} outcomes)")
            return path
        cached = load_dist(path).probs
        print(f"[exact] row-extending {have} -> {target}")

    probs = model.probs(X[have:])                       # exact, always; chunked internally
    if cached is not None:
        probs = torch.cat([cached, probs], dim=0)

    save_dist(path, cfg, model, probs)
    print(f"[exact] wrote {path}  ({target} rows, {n_out} outcomes, "
          f"{model.n_model_parameters()} model parameters)")
    return path


def generate_shots(cfg: ExperimentConfig, *, root: str | Path = "datasets",
                   method: str = "clifford", force: bool = False) -> Path | None:
    """Create, reuse or extend the counts branch; ``None`` if no shots were requested.

    Only the missing shots and the missing rows are drawn, and they are **appended** to the stored
    sequence -- the stored prefix is never rewritten, so every smaller budget stays exactly
    reproducible from the same store.  A budget *below* what is stored is not a generation job at
    all: :func:`~pipeline.shots.load_shots` crops to it.
    """
    target_shots = int(cfg.generation.shots)
    if target_shots <= 0:
        return None

    model = build_model(cfg)
    if not model.supports_shots:
        raise NotImplementedError(
            f"generation.shots > 0 but model kind={cfg.model.kind!r} prep={cfg.model.prep!r} is a "
            "probability-distribution model.")

    path = shots_path(cfg, model, method=method, root=root)

    # X comes from the shared sampler, NOT from the exact branch: the counts branch must not depend
    # on the exact distribution existing at all.
    n_rows = int(cfg.generation.size)
    X = sample_X(n_rows, cfg.problem.n_features, cfg.seeds.sample_seed)
    seed = cfg.seeds.shot_seed

    seqs, have_shots, have_rows = None, 0, 0
    if not force and (path / SHOTS_FILENAME).exists():
        keys, seq, _ = load_shots(path)
        have_rows, have_shots = seq.shape
        if have_rows >= n_rows and have_shots >= target_shots:
            print(f"[shots] cache hit: {path} ({have_shots} shots/row, {have_rows} rows, "
                  f"{len(keys)} observed outcomes)")
            return path
        seqs = to_seqs(keys, seq)
        if have_shots < target_shots:
            print(f"[shots] extending shots {have_shots} -> {target_shots}")
        if have_rows < n_rows:
            print(f"[shots] row-extending {have_rows} -> {n_rows}")

    if seqs is None:
        seqs = model.shot_counts(X, shots=target_shots, rows=range(n_rows), shot_seed=seed)
        final_shots = target_shots
    else:
        # Never shrink a store: a smaller target is served by cropping on load.
        final_shots = max(have_shots, target_shots)
        tail = final_shots - have_shots
        if tail:                               # the new shots, over the rows already stored
            seqs = [old + new for old, new in zip(seqs, model.shot_counts(
                X, shots=tail, offset=have_shots, rows=range(have_rows), shot_seed=seed))]
        if have_rows < n_rows:                 # new rows, built from the same offsets as the old
            new_rows = range(have_rows, n_rows)
            fresh = model.shot_counts(X, shots=have_shots, rows=new_rows, shot_seed=seed)
            if tail:
                fresh = [a + b for a, b in zip(fresh, model.shot_counts(
                    X, shots=tail, offset=have_shots, rows=new_rows, shot_seed=seed))]
            seqs = seqs + fresh

    save_shots(path, seqs, cfg, model, method=method)
    print(f"[shots] wrote {path}  ({final_shots} shots/row, {len(seqs)} rows, "
          f"seed={seed}, method={method})")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--configs-dir")
    ap.add_argument("--root", default="datasets", help="store root; both branches live under it")
    ap.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = ap.parse_args(argv)

    paths = ([Path(args.config)] if args.config else
             sorted(p for p in Path(args.configs_dir).glob("*.yaml") if not p.name.startswith("_")))

    for p in paths:
        print(f"--- {p}")
        cfg = load_config(p)
        if cfg.generation.shots:
            saved = generate_shots(cfg, root=args.root, force=args.force)
        else:
            saved = generate_exact(cfg, root=args.root, force=args.force)
        print("Data saved to:", saved)


if __name__ == "__main__":
    main()
