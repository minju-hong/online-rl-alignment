from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

import regret as reg


def _load_summary_arrays(summary_npz_path: str | Path) -> dict[str, np.ndarray]:
    arrays, _ = reg.load_npz(summary_npz_path)
    return arrays


def _algo_names_from_summary(arrays: dict[str, np.ndarray]) -> list[str]:
    if "algos" in arrays:
        return [str(x) for x in np.asarray(arrays["algos"]).reshape(-1)]

    names = []
    for key in arrays:
        if key.startswith("mnreg_cum__"):
            names.append(key.split("__", maxsplit=1)[1])
    return sorted(names)


def plot_single_algorithm(summary_npz_path, algo_name, save_dir):
    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    key = f"mnreg_cum__{algo_name}"
    if key not in arrays:
        raise KeyError(f"Algorithm '{algo_name}' not found in summary file.")

    y = np.asarray(arrays[key], dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != t.shape[0]:
        raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")

    mean = y.mean(axis=0)
    std = y.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, mean, label=f"{algo_name} mean")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, label=f"{algo_name} +/- 1 std")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Max-Nash Regret")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_name}_test_regret.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_combined_regret(summary_npz_path, save_dir):
    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    algos = _algo_names_from_summary(arrays)
    if not algos:
        raise ValueError("No algorithms found in summary file.")

    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in algos:
        key = f"mnreg_cum__{algo}"
        y = np.asarray(arrays[key], dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.ndim != 2 or y.shape[1] != t.shape[0]:
            raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")
        reg.plot_mean_cum_regret(t, y, label=algo, ax=ax)

    ax.legend()
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_regret.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_eta_scaling(eta_dict: Mapping[float, float], save_dir):
    if not eta_dict:
        raise ValueError("eta_dict must not be empty.")

    pairs = sorted((float(eta), float(regret)) for eta, regret in eta_dict.items())
    etas = np.asarray([p[0] for p in pairs], dtype=float)
    final_regrets = np.asarray([p[1] for p in pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, final_regrets, marker="o")
    ax.set_xlabel("eta")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eta_scaling.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
