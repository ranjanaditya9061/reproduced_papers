"""Compare spoqc spin entanglers on a single datapoint.

One teacher (:class:`model.spoqc.SpoqcPhotonicTeacher`), varying only ``cx_pairs``
(the config knob that selects the CX entangler in the spin prep).  Runs each on the
same input ``x`` and prints ``soft`` per observable.

    python model/spoqc_tester.py [--seed 42] [--m 6] [--k 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

# make the paper packages importable when run as a bare script
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import SpoqcPhotonicTeacher
from model.sampler import sample_X

# label -> cx_pairs
VARIANTS = {
    "base (none)": None,
    "one  (0-1)": [[0, 1]],
    "two  (chain)": [[0, 1], [1, 2]],
    "three(ring)": [[0, 1], [1, 2], [2, 0]],
}
OBSERVABLES = ("parity", "majority", "bunching")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the spoqc teacher with different cx_pairs on one datapoint.")
    ap.add_argument("--seed", type=int, default=42, help="teacher seed (Rx/Ry twists)")
    ap.add_argument("--m", type=int, default=6, help="optical modes (2*k <= m)")
    ap.add_argument("--k", type=int, default=3, help="qubits / photons")
    ap.add_argument("--n-features", type=int, default=None, help="input dim (default m-1)")
    ap.add_argument("--sample-seed", type=int, default=42, help="seed for the single input x")
    args = ap.parse_args(argv)

    nf = args.n_features if args.n_features is not None else args.m - 1
    X = sample_X(1, nf, args.sample_seed)
    print(f"x = [{', '.join(f'{v:.3f}' for v in X[0].tolist())}]   "
          f"(m={args.m}, k={args.k}, seed={args.seed})\n")

    w = max(len(name) for name in VARIANTS)
    header = f"  {'cx_pairs':<{w}}  " + "  ".join(f"{o:>9}" for o in OBSERVABLES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, cx in VARIANTS.items():
        cells = []
        for obs in OBSERVABLES:
            teacher = SpoqcPhotonicTeacher(m=args.m, k=args.k, n_features=nf,
                                           observable=obs, seed=args.seed, cx_pairs=cx)
            cells.append(f"{float(teacher(X)[0, 0]):>+9.4f}")
        print(f"  {label:<{w}}  " + "  ".join(cells))


if __name__ == "__main__":
    main()