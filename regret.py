from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import the Best Response calculator from your solvers file
from solvers import compute_best_response


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
    """
    Phi = np.asarray(Phi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    Z = Phi @ Theta @ Phi.T
    G = mu(Z)
    return np.asarray(G, dtype=float)


# ---------------------------------------------------------
# Regularized Game Math (Matches Paper Definition 3.5)
# ---------------------------------------------------------

def regularizer_penalty(p: np.ndarray, reg_type: str) -> float:
    """
    Computes psi(p) for the given regularizer type.
    """
    K = len(p)
    p_safe = np.clip(p, 1e-12, 1.0)
    
    if reg_type == 'reverse_kl':
        return float(np.sum(p_safe * np.log(K * p_safe)))
    elif reg_type == 'tsallis':
        return float(0.5 * np.sum(p**2))
    elif reg_type == 'chi_squared':
        return float(0.5 * K * np.sum((p - 1.0/K)**2))
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")


def compute_dual_gap(p: np.ndarray, G_star: np.ndarray, eta: float, reg_type: str) -> float:
    """
    Computes the Symmetric Dual Gap:
    DGap_eta(p) = 1/2 - min_q [ p^T G_star q - (1/eta)*psi(p) + (1/eta)*psi(q) ]
    """
    # 1. Center the payoff matrix to use the symmetric best response logic
    G_tilde = G_star - 0.5
    
    # 2. Find the opponent's best response q* using your existing solver
    q_star = compute_best_response(p, G_tilde, eta, reg_type)
    
    # 3. Calculate the full regularized objective J_eta(p, q*)
    j_val = p @ G_star @ q_star
    psi_p = regularizer_penalty(p, reg_type)
    psi_q = regularizer_penalty(q_star, reg_type)
    
    j_eta = j_val - (1.0 / eta) * psi_p + (1.0 / eta) * psi_q
    
    # 4. Return Dual Gap
    gap = 0.5 - j_eta
    
    # Clip tiny negative numerical artifacts to 0
    return max(0.0, float(gap))

# ---------------------------------------------------------
# Regret Trackers
# ---------------------------------------------------------

def mbr_regret(pi1_seq: np.ndarray, G_star: np.ndarray, eta: float, reg_type: str) -> np.ndarray:
    """
    Max-Best-Response Regret: sum_t DGap_eta(pi^1_t)
    """
    T = len(pi1_seq)
    inc = np.zeros(T)
    for t in range(T):
        p1 = normalize_probs(pi1_seq[t])
        inc[t] = compute_dual_gap(p1, G_star, eta, reg_type)
    return inc


def abr_regret(pi1_seq: np.ndarray, pi2_seq: np.ndarray, G_star: np.ndarray, eta: float, reg_type: str) -> np.ndarray:
    """
    Average-Best-Response Regret: sum_t [DGap_eta(pi^1_t) + DGap_eta(pi^2_t)]
    """
    T = len(pi1_seq)
    inc = np.zeros(T)
    for t in range(T):
        p1 = normalize_probs(pi1_seq[t])
        p2 = normalize_probs(pi2_seq[t])
        
        gap1 = compute_dual_gap(p1, G_star, eta, reg_type)
        gap2 = compute_dual_gap(p2, G_star, eta, reg_type)
        
        inc[t] = gap1 + gap2
    return inc


def cumulative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.cumsum(x)


# ---------------------------------------------------------
# Backward-compatible utilities used by test scripts
# ---------------------------------------------------------
def max_nash_regret_increments(
    pi1_seq: np.ndarray,
    G_star: np.ndarray,
    value_star: float = 0.5,
    clip_nonneg: bool = True,
) -> np.ndarray:
    """
    Legacy MN regret used by ETC scripts:
      inc_t = value_star - min_j (pi1_t^T G_star)[j]
    """
    pi1_seq = np.asarray(pi1_seq, dtype=float)
    if pi1_seq.ndim != 2:
        raise ValueError(f"pi1_seq must be 2D (T,K). Got {pi1_seq.shape}")
    T, K = pi1_seq.shape
    G_star = np.asarray(G_star, dtype=float)
    if G_star.shape != (K, K):
        raise ValueError(f"G_star must be (K,K)=({K},{K}). Got {G_star.shape}")

    inc = np.zeros(T, dtype=float)
    for t in range(T):
        p1 = normalize_probs(pi1_seq[t])
        worst_case = float(np.min(p1 @ G_star))
        val = float(value_star - worst_case)
        inc[t] = max(0.0, val) if clip_nonneg else val
    return inc


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
    """
    Save arrays + json metadata in a single compressed npz file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, indent=2, sort_keys=True)
    np.savez_compressed(path, **arrays, meta_json=np.array(meta_json))
    return path


def load_npz(path: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load arrays + json metadata from save_npz format.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
        meta_json = str(data["meta_json"]) if "meta_json" in data.files else "{}"
    meta = json.loads(meta_json)
    return arrays, meta