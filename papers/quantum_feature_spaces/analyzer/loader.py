"""Load a saved dataset from its experiment config and preview a few datapoints.

The analyzer never generates data; it resolves the artifact a config points to
(same content hash as the generation stage) and reads the **raw** pool through
:func:`Generator.load_raw`.  Previewing the raw pool (not the filtered/balanced
train/test view) means the preview never depends on ``min_margin`` and always
shows what is actually stored; it still reports how many samples survive the
config's margin.

CLI::

    python -m analyzer --config configs/example_photonic.yaml [-n 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Support both `python -m analyzer` and `python analyzer/loader.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "analyzer"

from Generator import artifact_path, load_config, load_raw
from Generator.prepare import derive_confidence, derive_labels


def load_dataset(config_path: str | Path, out_root: str | Path = "datasets"):
    """Resolve and load the raw ``(X, soft, meta)`` pool a config points to.

    Raises ``FileNotFoundError`` if the artifact has not been generated yet.
    """
    cfg = load_config(config_path)
    path = artifact_path(cfg, out_root)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset artifact at {path}.\n"
            f"Generate it first:  python -m Generator --config {config_path}"
        )
    return load_raw(path)


def preview(config_path: str | Path, n: int = 10, min_margin: float = 0.0,
            out_root: str | Path = "datasets"):
    """Load the dataset for ``config_path`` and print its first ``n`` datapoints.

    ``min_margin`` is only a diagnostic readout (how many samples are cleanly
    separable); it does not filter the stored pool.
    """
    cfg = load_config(config_path)
    X, soft, meta = load_dataset(config_path, out_root)

    y = derive_labels(soft)
    conf = derive_confidence(soft)
    n_survive = int((conf >= min_margin).sum())
    pool = X.shape[0]

    print(f"Loaded {artifact_path(cfg, out_root).name}")
    print(f"  generator={cfg.generation.generator}  m={cfg.problem.m}  k={cfg.problem.k}  "
          f"n_features={cfg.resolved_n_features}")
    print(f"  pool={pool}  classes: 0->{int((y == 0).sum())}  1->{int((y == 1).sum())}")
    print(f"  survive min_margin={min_margin}: {n_survive}/{pool} ({n_survive / max(pool, 1):.1%})")

    n = min(n, pool)
    print(f"  first {n} raw datapoints  (X | y | soft):")
    for i in range(n):
        xs = "[" + ", ".join(f"{v:6.3f}" for v in X[i].tolist()) + "]"
        softs = ", ".join(f"{v:+.4f}" for v in soft[i].tolist())
        print(f"    {i:2d}: {xs}  y={int(y[i])}  soft=[{softs}]")
    return X, soft, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load a dataset from its config and preview datapoints.")
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument("--out-root", default="datasets", help="root dir for artifacts")
    parser.add_argument("-n", type=int, default=10, help="number of datapoints to print")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="diagnostic: report how many samples clear this margin")
    args = parser.parse_args(argv)
    preview(args.config, n=args.n, min_margin=args.min_margin, out_root=args.out_root)


if __name__ == "__main__":
    main()