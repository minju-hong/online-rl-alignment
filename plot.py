from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

# A standard list of clear academic-style markers
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
# A professional color cycle
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _load_summary_arrays(summary_npz_path: str | Path) -> dict[str, np.ndarray]:
    """Loads the arrays directly to keep dependencies clean"""
    with np.load(summary_npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
    return arrays


# --- [CHANGE]: Added regret_type parameter to dynamically filter keys ---
def _algo_names_from_summary(arrays: dict[str, np.ndarray], regret_type: str = "mbr") -> list[str]:
    """Extracts algorithm names from summary array keys based on regret type"""
    if "algos" in arrays:
        return [str(x) for x in np.asarray(arrays["algos"]).reshape(-1)]

    names = []
    prefix = f"{regret_type}_cum__"
    fallback_prefix = "mnreg_cum__"
    
    for key in arrays:
        if key.startswith(prefix):
            names.append(key.split("__", maxsplit=1)[1])
        # Only fallback to old mnreg keys if we are specifically looking for mbr
        elif key.startswith(fallback_prefix) and regret_type == "mbr":
            names.append(key.split("__", maxsplit=1)[1])
            
    return sorted(list(set(names))) 


# --- [CHANGE]: Added regret_type parameter to dynamically build the search key ---
def _resolve_cum_key(arrays: dict[str, np.ndarray], algo_name: str, regret_type: str = "mbr") -> str:
    """
    Resolve cumulative-regret key dynamically based on the requested regret type.
    """
    candidates = [f"{regret_type}_cum__{algo_name}"]
    
    if regret_type == "mbr":
        candidates.append(f"mnreg_cum__{algo_name}") # Legacy fallback
        
    for key in candidates:
        if key in arrays:
            return key
            
    raise KeyError(f"Algorithm '{algo_name}' with regret type '{regret_type}' not found in summary file.")


def _setup_academic_plot():
    """Configures matplotlib for a clean, professional paper-like look"""
    plt.rcParams.update({
        'font.size': 11,           
        'axes.labelsize': 13,      
        'axes.titlesize': 14,      
        'legend.fontsize': 10,     
        'xtick.labelsize': 11,     
        'ytick.labelsize': 11,     
        'lines.linewidth': 1.5,    
        'lines.markersize': 5,     
    })


# --- [CHANGE]: Added regret_type parameter (defaults to "mbr") ---
def plot_single_algorithm(summary_npz_path, algo_name, save_dir, regret_type="mbr"):
    """Plots mean and std deviation for a single algorithm using error bars."""
    _setup_academic_plot()
    
    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    
    # Resolves using the new parameter
    key = _resolve_cum_key(arrays, algo_name, regret_type)

    y = np.asarray(arrays[key], dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != t.shape[0]:
        raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")

    mean = y.mean(axis=0)
    std_err = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    step = max(1, len(t) // 15)

    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.errorbar(
        t, mean, yerr=std_err, 
        label=algo_name,
        color=COLORS[0],
        marker=MARKERS[0], markersize=5, markevery=step,
        capsize=4, capthick=1.5, elinewidth=1.5, errorevery=step,
        linewidth=1.5
    )
    
    ax.set_xlabel("Time step (t)", fontweight='bold')
    
    # --- [CHANGE]: Dynamically updates the Y-axis label (e.g., "Cumulative MBR Regret") ---
    ax.set_ylabel(f"Cumulative {regret_type.upper()} Regret", fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- [CHANGE]: Saves file with the regret type in the name ---
    out_path = out_dir / f"{algo_name}_{regret_type}_regret.png"
    fig.savefig(out_path, dpi=300) 
    plt.close(fig)
    return out_path


# --- [CHANGE]: Added regret_type parameter (defaults to "mbr") ---
def plot_combined_regret(summary_npz_path, save_dir, regret_type="mbr"):
    """Plots multiple algorithm results on the same figure using error bars and markers."""
    _setup_academic_plot()
    
    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    algos = _algo_names_from_summary(arrays, regret_type)
    
    if not algos:
        raise ValueError(f"No algorithms found in summary file for regret type: {regret_type}")

    step = max(1, len(t) // 15)
    fig, ax = plt.subplots(figsize=(7, 4))
    
    for i, algo in enumerate(algos):
        key = _resolve_cum_key(arrays, algo, regret_type)
        
        y = np.asarray(arrays[key], dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
            
        mean = y.mean(axis=0)
        std_err = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
        
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.errorbar(
            t, mean, yerr=std_err, 
            label=algo,
            color=color,
            marker=marker, markersize=5, markevery=step,
            capsize=4, capthick=1.5, elinewidth=1.5, errorevery=step,
            linewidth=1.5
        )

    ax.set_xlabel("Time step (t)", fontweight='bold')
    
    # --- [CHANGE]: Dynamically updates the Y-axis label ---
    ax.set_ylabel(f"Cumulative {regret_type.upper()} Regret", fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- [CHANGE]: Saves file with the regret type in the name ---
    out_path = out_dir / f"combined_{regret_type}_regret.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --- [CHANGE]: Added regret_type parameter (defaults to "mbr") ---
def plot_eta_scaling(eta_dict: Mapping[float, float], save_dir, regret_type="mbr"):
    """Plots final regret across a range of regularization strengths (log-scale X)."""
    _setup_academic_plot()
    
    if not eta_dict:
        raise ValueError("eta_dict must not be empty.")

    pairs = sorted((float(eta), float(regret)) for eta, regret in eta_dict.items())
    etas = np.asarray([p[0] for p in pairs], dtype=float)
    final_regrets = np.asarray([p[1] for p in pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, final_regrets, marker="o", color=COLORS[0], linewidth=1.5, markersize=6)
    
    ax.set_xlabel("Regularization Strength (eta)", fontweight='bold')
    
    # --- [CHANGE]: Dynamically updates the Y-axis label ---
    ax.set_ylabel(f"Final Cumulative {regret_type.upper()} Regret", fontweight='bold')
    
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- [CHANGE]: Saves file with the regret type in the name ---
    out_path = out_dir / f"eta_scaling_{regret_type}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_combined_regret_shaded(summary_npz_path, save_dir, regret_type="mbr"):
    """
    Plots multiple algorithm results on the same figure using shaded regions 
    instead of error bars to represent the standard error/deviation.
    """
    # Uses your existing setup and data loading functions
    _setup_academic_plot()
    
    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    algos = _algo_names_from_summary(arrays, regret_type)
    
    if not algos:
        raise ValueError(f"No algorithms found in summary file for regret type: {regret_type}")

    fig, ax = plt.subplots(figsize=(7, 4))
    
    for i, algo in enumerate(algos):
        key = _resolve_cum_key(arrays, algo, regret_type)
        
        y = np.asarray(arrays[key], dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
            
        mean = y.mean(axis=0)
        
        # Note: I kept this as Standard Error to match your original math. 
        # If you want pure Standard Deviation, remove the division by np.sqrt(y.shape[0])
        std_spread = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
        
        color = COLORS[i % len(COLORS)]

        # 1. Plot the solid mean line (removed markers for a cleaner look with shading)
        ax.plot(
            t, mean, 
            label=algo,
            color=color,
            linewidth=2.0 
        )
        
        # 2. Shade the area around the mean
        ax.fill_between(
            t, 
            mean - std_spread, 
            mean + std_spread, 
            color=color, 
            alpha=0.2,       # Controls transparency (0.0 is invisible, 1.0 is solid)
            edgecolor=None   # Removes the harsh border line around the shading
        )

    ax.set_xlabel("Time step (t)", fontweight='bold')
    ax.set_ylabel(f"Cumulative {regret_type.upper()} Regret", fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Saves with a new "_shaded" suffix so it doesn't overwrite your original plot
    out_path = out_dir / f"combined_{regret_type}_regret_shaded.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path