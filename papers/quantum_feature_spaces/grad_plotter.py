import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

def diff_stats(grads: np.ndarray) -> np.ndarray:
    print(grads.shape)
    # 1. Absolute diff along the column axis (axis=1)
    diff = np.abs(np.diff(grads, axis=0))   # shape: (N, M‑1)
    print(diff.shape)

    means = diff.mean(axis=0)
    maxs  = diff.max(axis=0)
    mins  = diff.min(axis=0)

    return means, maxs, mins

def plot_grad_over_batches(
        grad_file_name,   # change if you used .json or .pkl
        best_val_acc,
        data,
        learner,
        ylabel: str = r"Grad[0] value",
        xlabel: str = "Batch index",
        linewidth: float = 1.5,
        marker: str = None,
        grid: bool = True,
        **plot_kwargs,
):
    """
    Load a saved list of scalar gradient values, plot them, and write a PNG.
    
    Parameters
    ----------
    runs_dir        : Path to the folder that contains the file.
    grad_file_name  : Name of the file with the saved list. Must match the
                      one you used in `train_and_record`.
    output_path     : Where the image will be written.
    title, ylabel, ... : Plot aesthetics.
    plot_kwargs     : Additional kwargs forwarded to plt.plot().
    """
    
    base_dir =  "./grad_data"
    grad_path = Path(base_dir) / f"{grad_file_name}.npy"
    output_path = Path(base_dir) / f"{grad_file_name}.jpg"

    if not grad_path.exists():
        raise FileNotFoundError(f"Could not find {grad_path}")
    
    # --- load ---
    if grad_path.suffix == ".npy":
        grads = np.load(grad_path)
    elif grad_path.suffix == ".json":
        with open(grad_path, "r") as fp:
            grads = json.load(fp)
    elif grad_path.suffix == ".pkl":
        with open(grad_path, "rb") as fp:
            grads = pickle.load(fp)
    else:
        raise ValueError(f"Unsupported file type: {grad_path.suffix}")
    
    means, maxs, mins = diff_stats(grads)
    print(np.mean(means), np.max(means), np.min(means))

    # grads = grads[:100]
    # --- plot ---
    plt.figure(figsize=(12, 4.5))
    plt.plot(
        range(1, len(grads)+1),
        grads,
        lw=linewidth,
        marker=marker,
        **plot_kwargs,
    )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{data} Data, {learner} Learner, Accuracy = {best_val_acc:.4f}")
    if grid:
        plt.grid(alpha=0.4)

    # --- save & close ---
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import re

# ---- 1️⃣  Pattern that captures every field ----------
# pattern is essentially:
#   {m}_{k}grad_{data}_data_{learner}_learner_{acc}_best_acc.npy
#   groups → 1:m, 2:k, 3:data, 4:learner, 5:acc
FILE_NAME_RE = re.compile(
    r"^(?P<m>\d+)_(?P<k>\d+)grad_(?P<data>\w+)_data_(?P<learner>\w+)_learner_(?P<acc>\d+)_best_acc\.npy$"
)

def parse_grad_file_name(path: Path):
    """Return dict with parsed fields or None if the name does not match."""
    match = FILE_NAME_RE.match(path.name)
    if not match:
        return None

    m   = int(match.group("m"))
    k   = int(match.group("k"))
    data   = match.group("data")
    learner= match.group("learner")
    acc    = int(match.group("acc")) / 10_000     # undo the scaling

    return {
        "m"        : m,
        "k"        : k,
        "filename" : path.stem,
        "data"     : data,
        "learner"  : learner,
        "accuracy" : acc,
    }


from pathlib import Path

folder = Path("./grad_data")


for f in folder.glob("*.npy"):
    print(f)
    meta = parse_grad_file_name(f)
    if meta is None:
        print(f"⚠️  Skipping unrecognised file {f}")
        continue

    # optional: load the gradient
    # grad_arr = np.load(f)          # uncomment if you need the array
    print(meta)

    plot_grad_over_batches(meta["filename"], meta["accuracy"], meta["data"], meta["learner"])

# m = 4
# k = 3

# file_names = ["img_grad_majority_data_majority_learner", "img_grad_parity_data_majority_learner",
#               "img_grad_majority_data_parity_learner", "img_grad_parity_data_parity_learner"]
# datas = ["Majority", "Parity", "Majority", "Parity"]
# learners = ["Majority", "Majority", "Parity",  "Parity"]

# best_val_accs = [1.000, 0.8587, 1.000 ,0.8675]

# for file_name, best_val_acc, data, learner in zip(file_names, best_val_accs, datas, learners):
    
#     plot_grad_over_batches(file_name, best_val_acc, data, learner)