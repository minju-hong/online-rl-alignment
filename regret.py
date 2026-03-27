# regret.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s


def payoff_matrix(Phi: np.ndarray, Theta: np.ndarray, mu) -> np.ndarray:
    """
    G[i,j] = mu(phi_i^T Theta phi_j)
    Phi: (K,d), Theta: (d,d)
    """
    Phi = np.asarray(Phi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    Z = Phi @ Theta @ Phi.T
    G = mu(Z)
    G = np.asarray(G, dtype=float)
    K = Phi.shape[0]
    if G.shape != (K, K):
        raise ValueError(f"mu(Z) must return shape (K,K). Got {G.shape}.")
    return G


def max_nash_regret_increments(
    pi1_seq: np.ndarray,
    G_star: np.ndarray,
    value_star: float = 0.5,
    clip_nonneg: bool = True,
) -> np.ndarray:
    """
    inc_t = max_{pi in simplex} [ value_star - J(pi1_t, pi) ]
         = value_star - min_{j in [K]} (pi1_t^T G_star)[j]
    pi1_seq: (T,K)
    G_star:  (K,K)
    """
    pi1_seq = np.asarray(pi1_seq, dtype=float)
    if pi1_seq.ndim != 2:
        raise ValueError(f"pi1_seq must be 2D (T,K). Got {pi1_seq.shape}")
    T, K = pi1_seq.shape
    if G_star.shape != (K, K):
        raise ValueError(f"G_star must be (K,K)=({K},{K}). Got {G_star.shape}")

    inc = np.zeros(T, dtype=float)
    for t in range(T):
        pi1 = normalize_probs(pi1_seq[t])
        pay_vs_pure = pi1 @ G_star          # (K,)
        worst_case = float(np.min(pay_vs_pure))
        inc_t = float(value_star - worst_case)
        if clip_nonneg:
            inc_t = max(0.0, inc_t)
        inc[t] = inc_t
    return inc


def cumulative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.cumsum(x)


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, indent=2, sort_keys=True)
    np.savez_compressed(path, **arrays, meta_json=np.array(meta_json))
    return path


def load_npz(path: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
        meta_json = str(data["meta_json"])
    meta = json.loads(meta_json)
    return arrays, meta


def plot_mean_cum_regret(
    t: np.ndarray,
    cum_regret_by_seed: np.ndarray,
    label: str,
    ax=None,
):
    """
    cum_regret_by_seed: (n_seeds, T)
    """
    t = np.asarray(t)
    Y = np.asarray(cum_regret_by_seed, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != len(t):
        raise ValueError("cum_regret_by_seed must be (n_seeds, T) matching t length.")

    mean = Y.mean(axis=0)
    if Y.shape[0] >= 2:
        stderr = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0])
        ci95 = 1.96 * stderr
    else:
        ci95 = np.zeros_like(mean)

    if ax is None:
        ax = plt.gca()
    ax.plot(t, mean, label=label)
    ax.fill_between(t, mean - ci95, mean + ci95, alpha=0.2)
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Max-Nash Regret")
    ax.grid(True, alpha=0.3)
    return ax
