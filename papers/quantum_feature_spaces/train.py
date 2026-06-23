"""Unified training CLI — dispatches to the right learner (mlp or quantum).

Usage examples
--------------
# MLP learner on quantum-generated data
python work/train.py --learner mlp --generator quantum --m 6 --k 3

# Quantum learner on quantum-generated data
python work/train.py --learner quantum --generator quantum --m 6 --k 3

# Cross-evaluation: quantum learner on analytical data
python work/train.py --learner quantum --generator analytical --m 6 --k 3 --depths 1 2 3

# MLP learner with explicit architecture grid
python work/train.py --learner mlp --generator mlp --m 6 --k 3 --hidden-sizes 64 "64,64" "128,64,32"
"""

from __future__ import annotations

import argparse
from typing import Callable

from data import GENERATORS
from learner.mlp import find_min_hidden_size, HiddenLayerSpec
from learner.photonic_quantum import find_min_depth
from learner.qubit_quantum import find_min_qubit_depth
from learner.svm import find_min_svm


# ---------------------------------------------------------------------------
# Learners registry  (mirrors datasets.GENERATORS)
# ---------------------------------------------------------------------------

LearnerFn = Callable[..., None]

LEARNERS: dict[str, LearnerFn] = {
    "mlp":              find_min_hidden_size,
    "photonic_quantum": find_min_depth,
    "qubit_quantum":    find_min_qubit_depth,
    "svm":              find_min_svm,
}


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_hidden_layer_spec(s: str) -> HiddenLayerSpec:
    """Parse a hidden-layer spec from a CLI string.

    '64'       → 64            (single layer)
    '64,32'    → (64, 32)      (two layers)
    '128,64,32' → (128, 64, 32)
    """
    parts = [int(x) for x in s.split(",")]
    return parts[0] if len(parts) == 1 else tuple(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generic training CLI. Select a learner architecture (mlp or quantum) "
            "and a dataset generator (quantum, analytical, mlp)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required / core ---
    parser.add_argument(
        "--learner",
        choices=list(LEARNERS),
        default="mlp",
        help="Learner architecture to train",
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default="quantum",
        help="Dataset generation strategy",
    )
    parser.add_argument("--m", type=int, default=6, help="Number of optical modes")
    parser.add_argument("--k", type=int, default=3, help="Complexity parameter (photons / depth / frequency)")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size for gradient-based training (default: 64).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 3e-3 for mlp, 1e-2 for quantum)")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="If set, subsample the majority class so the dataset is exactly 50/50. "
             "Recommended for the quantum generator, where Haar-random circuits "
             "can give skewed splits at small dataset sizes.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.2,
        metavar="M",
        help=(
            "Drop samples whose decision confidence is below M (in [0, 1]). "
            "For quantum: |parity_expectation| >= M. "
            "For analytical: |normalised_score| >= M. "
            "For mlp: |p1 - p0| >= M. "
            "Reproduces the 0.3 separation gap in Havlíček et al. (2019). "
            "Default: 0.0 (keep all samples)."
        ),
    )
    parser.add_argument(
        "--bail-threshold",
        type=float,
        default=0.30,
        metavar="FRAC",
        help=(
            "Resampling bail-out: if the post-balance yield from the first batch "
            "is below this fraction (default 0.30), return a short dataset rather "
            "than iteratively resampling. Set to 0 to disable the bail-out and "
            "always iterate to --dataset-size — at the cost of potentially many "
            "raw draws when --min-margin is aggressive or class skew is severe."
        ),
    )
    parser.add_argument(
        "--max-resample-iter",
        type=int,
        default=10,
        metavar="N",
        help="Max resample iterations after the first batch. Default 10.",
    )

    # --- svm-specific ---
    svm_group = parser.add_argument_group("SVM learner options")
    svm_group.add_argument(
        "--C-values",
        type=float,
        nargs="+",
        default=None,
        metavar="C",
        help=(
            "SVM regularisation values to sweep (ascending). "
            "Default: [0.01, 0.1, 1, 10, 100, 1000]."
        ),
    )
    svm_group.add_argument(
        "--svm-kernel",
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="SVM kernel type (default: rbf).",
    )
    svm_group.add_argument(
        "--fourier-order",
        type=int,
        default=3,
        metavar="F",
        help=(
            "Fourier expansion order applied to inputs before the SVM kernel. "
            "0 = raw angle features. Default: 3 (yields sin/cos at frequencies 1–3)."
        ),
    )

    # --- mlp-specific ---
    mlp_group = parser.add_argument_group("MLP learner options")
    mlp_group.add_argument(
        "--hidden-sizes",
        type=_parse_hidden_layer_spec,
        nargs="+",
        default=None,
        metavar="SPEC",
        help=(
            "Hidden-layer specs to try. Each spec is either an int (e.g. 64) "
            "or a comma-separated list of ints (e.g. 64,32). "
            "Default: [2, 4, 8, 16, 32, 64, 128, 256]."
        ),
    )

    # --- quantum generator options (photonic + qubit) ---
    ph_group = parser.add_argument_group("Quantum generator options (photonic + qubit)")
    ph_group.add_argument(
        "--observable",
        choices=["parity", "majority", "bunching", "single_output"],
        default="parity",
        help=(
            "Photonic teacher observable (photonic_quantum generator only). "
            "'parity': (-1)^{n_left}. "
            "'majority': sign(n_left - n_right). Requires even m. "
            "'bunching': P(anti-bunched) - P(bunched). "
            "'single_output': P(output=input_state) - P(output=reversed_input_state). "
            "Default: parity."
        ),
    )
    ph_group.add_argument(
        "--learner_encoding",
        choices=["parity", "majority", "bunching", "single_output"],
        default="parity",
        help=(
            "Photonic teacher learner encoding (photonic_quantum generator only). "
            "'parity': (-1)^{n_left}. "
            "'majority': sign(n_left - n_right). Requires even m. "
            # "'bunching': P(anti-bunched) - P(bunched). "
            # "'single_output': P(output=input_state) - P(output=reversed_input_state). "
            "Default: parity."
        ),
    )
    ph_group.add_argument(
        "--nshots",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of measurement shots for quantum generators. "
            "0 (default) = exact computation (analytical probabilities / exact parity expectation). "
            ">0 = finite-shot simulation: photonic uses Merlin's shot sampler; "
            "qubit adds Gaussian noise with std=sqrt((1−E²)/N) to the exact expectation."
        ),
    )
    ph_group.add_argument(
        "--n-features",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of data-encoded features (photonic_quantum generator/learner only). "
            "Decouples the input dimension from the circuit size m: n_features modes "
            "receive phase encoding while the full m-mode Fock space is used for "
            "measurement. Default: m-1 (all modes except the last)."
        ),
    )

    # --- quantum-specific ---
    q_group = parser.add_argument_group("Quantum learner options")
    q_group.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help="Circuit depths (re-uploading layers) to try. Default: [1, 2, 3, 4].",
    )
    q_group.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="M_CIRC",
        help=(
            "Interferometer sizes (number of modes) to try. "
            "Must be >= m (dataset modes). "
            "Only the first m-1 modes receive phase encoding; "
            "extra modes increase expressivity with evenly-spaced photons. "
            "Default: [m] (circuit matches dataset)."
        ),
    )
    q_group.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "mse", "hloss"],
        help=(
            "Loss for the quantum learner. "
            "'ce' = cross-entropy on hard labels (default). "
            "'mse' = regress learner parity expectation against teacher's "
            "(only with --generator quantum; the one case where distillation applies). "
            "'hloss' = Havlíček et al. sigmoid loss: -log σ(parity_exp + b) "
            "with a trainable scalar bias b; uses only hard labels, no teacher "
            "soft targets required."
        ),
    )
    q_group.add_argument(
        "--optimizer",
        default="Adam",
        choices=["Adam", "AdamW", "SGD", "SPSA", "CMA", "Powell", "NelderMead", "COBYLA"],
        help=(
            "Optimizer for the quantum learner. "
            "Gradient-based: Adam (default), AdamW, SGD. "
            "SPSA: Simultaneous Perturbation Stochastic Approximation (Spall 1992) — "
            "2 forward passes per mini-batch, gradient-free, noise-robust; "
            "matches the optimizer used in Havlíček et al. (2019). "
            "Gradient-free (epochs = max function evaluations): "
            "CMA, Powell, NelderMead, COBYLA."
        ),
    )
    q_group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Stop training if val_acc has not improved for N epochs (evaluated every 10). "
            "Default: 60. Set to a large value (e.g. 99999) to disable."
        ),
    )

    cli = parser.parse_args()

    # Validate n_features vs m
    if cli.n_features is not None and cli.n_features > cli.m - 1:
        parser.error(
            f"--n-features {cli.n_features} must be ≤ m-1 = {cli.m - 1}. "
            f"Only the first n_features modes receive phase encoding; "
            f"the last mode is unencoded to remove global-phase redundancy."
        )

    # Resolve default LR per learner
    lr = cli.lr if cli.lr is not None else (3e-3 if cli.learner == "mlp" else 1e-2)

    common = dict(
        m=cli.m,
        k=cli.k,
        generator=cli.generator,
        target_accuracy=cli.target_accuracy,
        dataset_size=cli.dataset_size,
        epochs=cli.epochs,
        lr=lr,
        batch_size=cli.batch_size,
        data_seed=cli.data_seed,
        model_seed=cli.model_seed,
    )

    if cli.learner == "mlp":
        find_min_hidden_size(**common, hidden_sizes=cli.hidden_sizes,
                             balanced=cli.balanced, min_margin=cli.min_margin,
                             bail_threshold=cli.bail_threshold,
                             max_resample_iter=cli.max_resample_iter,
                             early_stopping_patience=cli.early_stopping_patience)
    elif cli.learner == "svm":
        find_min_svm(**common, C_values=cli.C_values,
                     kernel=cli.svm_kernel, fourier_order=cli.fourier_order,
                     balanced=cli.balanced, min_margin=cli.min_margin,
                     bail_threshold=cli.bail_threshold,
                     max_resample_iter=cli.max_resample_iter)
    elif cli.learner == "qubit_quantum":
        find_min_qubit_depth(**common, depths=cli.depths,
                             optimizer_name=cli.optimizer,
                             balanced=cli.balanced, loss=cli.loss,
                             min_margin=cli.min_margin,
                             bail_threshold=cli.bail_threshold,
                             max_resample_iter=cli.max_resample_iter,
                             nsample=cli.nshots)
    else:
        find_min_depth(**common, depths=cli.depths, m_circuits=cli.sizes,
                       optimizer_name=cli.optimizer,
                       balanced=cli.balanced, loss=cli.loss,
                       min_margin=cli.min_margin,
                       bail_threshold=cli.bail_threshold,
                       max_resample_iter=cli.max_resample_iter,
                       early_stopping_patience=cli.early_stopping_patience,
                       observable=cli.observable,
                       data_encoding = cli.observable,
                       learner_encoding = cli.learner_encoding,
                       n_features=cli.n_features,
                       nsample=cli.nshots)
