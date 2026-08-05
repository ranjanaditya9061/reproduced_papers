"""Generation: draw the input pool, then populate either or both branches.

    python -m pipeline.generate --config configs/photonic.yaml
    python -m pipeline.generate --configs-dir configs

Two branches hang off one circuit identity (:func:`~pipeline.artifact.circuit_hash`), which
covers the input and the circuit and **nothing else**::

    datasets_<...>_<circuit_hash>/dist.npz          exact probs      (when it fits)
    shots_<circuit_hash>/<shot_hash>/counts.npz     cumulative int counts

**Both directions extend rather than recompute.**

* *rows* -- ``sample_X`` is prefix-stable and the model deterministic in ``model_seed``, so raising
  ``generation.size`` simulates only the new rows; the reused prefix is byte-identical.
* *shots* -- blocks are nested and additive, so raising ``generation.shots`` draws only the new
  blocks and adds their counts.  Going 20k -> 30k costs one block draw instead of a full
  re-simulation, and the 20k answer stays recoverable from the 30k store.

The old behaviour is what makes that worth stating: with ``nsample`` in the hash, 20k and 30k were
different directories, so the second run re-ran the exact simulation (76% of the work, producing
bit-identical output), redrew from shot zero, and still did not yield an extension of the first.
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
from .artifact import DIST_FILENAME, artifact_path, load_meta, save_circuit, save_meta
from .distribution import check_size, load_dist, write_dist
from .shots import (COUNTS_FILENAME, load_shots, merge_shots, n_blocks_for, observed_keys,
                    realised_shots, save_shots, shot_spec, shots_path)


def generate_exact(cfg: ExperimentConfig, *, out_root: str | Path = "datasets_v2",
                   force: bool = False) -> Path:
    """Create, reuse or row-extend the exact-distribution branch; return its directory."""
    model = build_model(cfg)
    path = artifact_path(cfg, model, out_root)
    target = int(cfg.generation.size)
    n_out = model.n_outcomes
    check_size(target, n_out, cfg.generation.max_dist_bytes, m=cfg.problem.m, k=cfg.problem.k)

    have, cached = 0, None
    if not force and (path / DIST_FILENAME).exists():
        have = int(load_meta(path)["size"])
        if have >= target:
            print(f"[exact] cache hit: {path} ({have} rows, {n_out} outcomes)")
            return path
        cached = load_dist(path).probs
        print(f"[exact] row-extending {have} -> {target}")

    X = sample_X(target, cfg.problem.n_features, cfg.seeds.sample_seed)
    probs = model.probs(X[have:])                       # exact, always; chunked internally
    if cached is not None:
        probs = torch.cat([cached, probs], dim=0)

    write_dist(path, probs=probs, keys=model.outcome_keys(),
               probs_at_zero=model.probs_at_zero())
    save_meta(path, cfg, model, size=target, n_out=n_out, exact_available=True)
    save_circuit(path, model)
    print(f"[exact] wrote {path}  ({target} rows, {n_out} outcomes, "
          f"{model.n_model_parameters()} model parameters)")
    return path


def generate_shots(cfg: ExperimentConfig, *, shots_root: str | Path = "shots_v2",
                   method: str = "clifford", force: bool = False) -> Path | None:
    """Create or **extend** the shots branch.  Returns its directory, or ``None`` if not requested.

    Only the missing blocks are drawn; existing counts are added to.  This is the payoff of keeping
    ``n_blocks`` out of the circuit hash.
    """
    target_blocks = n_blocks_for(cfg.generation.shots)
    if target_blocks == 0:
        return None

    model = build_model(cfg)
    path = shots_path(cfg, model, method=method, shots_root=shots_root)
    spec = shot_spec(cfg, method=method)

    if int(cfg.generation.shots) != realised_shots(target_blocks):
        print(f"[shots] {cfg.generation.shots} rounded up to {realised_shots(target_blocks)} "
              f"({target_blocks} x {spec['block']}): counts are additive only at block boundaries")

    if method != "clifford":
        raise NotImplementedError(
            f"shot method {method!r} is not implemented.")

    if not model.supports_shots:
        raise NotImplementedError(
            f"generation.shots > 0 but model kind={cfg.model.kind!r} prep={cfg.model.prep!r} is a "
            "probability-distribution model.")

    # X comes from the shared sampler, NOT from the exact artifact: the shots branch must not
    # depend on the exact distribution existing at all.
    n_rows = int(cfg.generation.size)
    X = sample_X(n_rows, cfg.problem.n_features, cfg.seeds.sample_seed)

    have_blocks, have_rows, rows_data = 0, 0, None
    if not force and (path / COUNTS_FILENAME).exists():
        rows_data, meta = load_shots(path)
        have_blocks, have_rows = int(meta["n_blocks"]), len(rows_data)
        if have_rows >= n_rows and have_blocks >= target_blocks:
            print(f"[shots] cache hit: {path} ({realised_shots(have_blocks)} shots/row, "
                  f"{have_rows} rows, {len(observed_keys(rows_data))} observed outcomes)")
            return path
        if have_blocks < target_blocks:
            print(f"[shots] extending blocks {have_blocks} -> {target_blocks} "
                  f"({realised_shots(have_blocks)} -> {realised_shots(target_blocks)} shots/row)")
        if have_rows < n_rows:
            print(f"[shots] row-extending {have_rows} -> {n_rows}")

    if rows_data is None:
        rows_data = model.shot_counts(X, blocks=range(target_blocks),
                                     shot_seed=cfg.seeds.shot_seed)
    else:
        if target_blocks > have_blocks:        # missing blocks, over the rows already stored
            rows_data = merge_shots(rows_data, model.shot_counts(
                X, blocks=range(have_blocks, target_blocks), rows=range(have_rows),
                shot_seed=cfg.seeds.shot_seed))
        if have_rows < n_rows:                 # every target block, over the new rows only
            rows_data = rows_data + model.shot_counts(
                X, blocks=range(target_blocks), rows=range(have_rows, n_rows),
                shot_seed=cfg.seeds.shot_seed)

    save_shots(path, rows_data, spec, n_blocks=target_blocks)
    print(f"[shots] wrote {path}  ({realised_shots(target_blocks)} shots/row, "
          f"{len(observed_keys(rows_data))} observed outcomes, seed={spec['shot_seed']}, "
          f"method={method})")
    return path


def _configs(args) -> list[Path]:
    if args.config:
        return [Path(args.config)]
    return sorted(p for p in Path(args.configs_dir).glob("*.yaml") if not p.name.startswith("_"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--configs-dir")
    ap.add_argument("--out-root", default="datasets_v2")
    ap.add_argument("--shots-root", default="shots_v2")
    ap.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = ap.parse_args(argv)

    for p in _configs(args):
        print(f"--- {p}")
        cfg = load_config(p)
        out_root=args.out_root
        shots_root=args.shots_root
        force=args.force

        if cfg.generation.shots:
            saved_path = generate_shots(cfg, shots_root=shots_root, force=force)
        else:
            saved_path = generate_exact(cfg, out_root=out_root, force=force)

        print("Data saved to:", saved_path)


if __name__ == "__main__":
    main()
