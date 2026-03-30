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


def plot_theta_frob_error(summary_npz_path, algo_name, save_dir):
    """
    Plot ||Theta_hat_t - Theta_star||_F per iteration from summary arrays.
    Expects key: theta_frob_err__{algo_name} with shape (n_seeds, T) or (T,).
    """
    _setup_academic_plot()

    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=int).reshape(-1)
    key = f"theta_frob_err__{algo_name}"
    if key not in arrays:
        raise KeyError(f"Missing key '{key}' in summary file.")

    y = np.asarray(arrays[key], dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != t.shape[0]:
        raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")

    mean = y.mean(axis=0)
    if y.shape[0] > 1:
        spread = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    else:
        spread = np.zeros_like(mean)

    step = max(1, len(t) // 15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        t, mean, yerr=spread,
        label=f"{algo_name} theta error",
        color=COLORS[0],
        marker=MARKERS[0], markersize=5, markevery=step,
        capsize=4, capthick=1.5, elinewidth=1.5, errorevery=step,
        linewidth=1.5
    )
    ax.set_xlabel("Time step (t)", fontweight='bold')
    ax.set_ylabel(r"$\|\hat{\Theta}_t - \Theta^\star\|_F$", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_name}_theta_frob_error.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_loglog_regret_with_fit(
    summary_npz_path,
    algo_name,
    save_dir,
    regret_type="mbr",
    fit_start_t=10,
    suffix: str = "",
):
    """
    Plot cumulative regret on log-log axes and overlay best linear fit in log-space.
    Returns (out_path, slope, r2).
    """
    _setup_academic_plot()

    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=float).reshape(-1)
    key = _resolve_cum_key(arrays, algo_name, regret_type)
    y = np.asarray(arrays[key], dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != t.shape[0]:
        raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")

    y_mean = y.mean(axis=0)
    if y.shape[0] > 1:
        y_spread = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    else:
        y_spread = np.zeros_like(y_mean)

    mask = (t >= float(fit_start_t)) & (y_mean > 0)
    if np.sum(mask) < 3:
        raise ValueError("Not enough positive points for log-log fit.")

    xlog = np.log(t[mask])
    ylog = np.log(y_mean[mask])
    slope, intercept = np.polyfit(xlog, ylog, 1)
    yhat_log = slope * xlog + intercept
    ss_tot = float(np.sum((ylog - ylog.mean()) ** 2))
    ss_res = float(np.sum((ylog - yhat_log) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    yfit = np.exp(intercept) * np.power(t[mask], slope)

    fig, ax = plt.subplots(figsize=(7, 4))
    y_lo = np.clip(y_mean - y_spread, 1e-12, None)
    y_hi = np.clip(y_mean + y_spread, 1e-12, None)
    ax.loglog(t, y_mean, label=f"{algo_name} mean regret", color=COLORS[0], linewidth=1.8)
    ax.fill_between(t, y_lo, y_hi, color=COLORS[0], alpha=0.2, label="mean ± SE")
    ax.loglog(t[mask], yfit, "--", label=f"fit slope={slope:.3f}, R^2={r2:.3f}", color=COLORS[1], linewidth=1.5)
    ax.set_xlabel("Time step (t)", fontweight="bold")
    ax.set_ylabel(f"Cumulative {regret_type.upper()} Regret", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_name}_{regret_type}_regret_loglog_fit{suffix}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path, float(slope), float(r2)


def plot_eta_vs_slope(eta_to_slope: Mapping[float, float], save_dir, regret_type="mbr"):
    """
    Plot slope of log-log regret fit versus eta (log x-axis).
    """
    _setup_academic_plot()
    if not eta_to_slope:
        raise ValueError("eta_to_slope must not be empty.")

    pairs = sorted((float(k), float(v)) for k, v in eta_to_slope.items())
    etas = np.asarray([p[0] for p in pairs], dtype=float)
    slopes = np.asarray([p[1] for p in pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, slopes, marker="o", color=COLORS[0], linewidth=1.7, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Strength (eta)", fontweight="bold")
    ax.set_ylabel(f"Log-Log Slope of {regret_type.upper()} Regret", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eta_vs_loglog_slope_{regret_type}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _fit_regret_vs_features(t: np.ndarray, y: np.ndarray, feature: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Fit y ~= a * feature + b and return (a, r2, yhat).
    """
    X = np.vstack([feature, np.ones_like(feature)]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(beta[0]), r2, yhat


def plot_regret_with_fit_metrics(
    summary_npz_path,
    algo_name,
    save_dir,
    regret_type="mbr",
    fit_start_t=10,
    suffix: str = "",
):
    """
    Plain t-vs-regret plot with mean±SE across seeds and independent fit metrics:
      - Reg ~ a_log * log(t) + b
      - Reg ~ a_sqrt * sqrt(t) + b
    Returns (out_path, metrics_dict).
    """
    _setup_academic_plot()

    arrays = _load_summary_arrays(summary_npz_path)
    t = np.asarray(arrays["t"], dtype=float).reshape(-1)
    key = _resolve_cum_key(arrays, algo_name, regret_type)
    y = np.asarray(arrays[key], dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != t.shape[0]:
        raise ValueError(f"Expected '{key}' shape (n_seeds, T), got {y.shape}.")

    mean = y.mean(axis=0)
    if y.shape[0] > 1:
        spread = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    else:
        spread = np.zeros_like(mean)

    mask = t >= float(fit_start_t)
    if np.sum(mask) < 3:
        raise ValueError("Not enough points for fit metrics.")

    tt = t[mask]
    yy_mean = mean[mask]
    # Keep mean-curve fit for reference
    slope_log_mean_curve, r2_log_mean_curve, _ = _fit_regret_vs_features(tt, yy_mean, np.log(tt))
    slope_sqrt_mean_curve, r2_sqrt_mean_curve, _ = _fit_regret_vs_features(tt, yy_mean, np.sqrt(tt))

    # Per-seed fits (preferred statistics across random trials)
    slope_log_per_seed: list[float] = []
    r2_log_per_seed: list[float] = []
    slope_sqrt_per_seed: list[float] = []
    r2_sqrt_per_seed: list[float] = []
    for i in range(y.shape[0]):
        yy = y[i, mask]
        sl_log, rr_log, _ = _fit_regret_vs_features(tt, yy, np.log(tt))
        sl_sqrt, rr_sqrt, _ = _fit_regret_vs_features(tt, yy, np.sqrt(tt))
        slope_log_per_seed.append(float(sl_log))
        r2_log_per_seed.append(float(rr_log))
        slope_sqrt_per_seed.append(float(sl_sqrt))
        r2_sqrt_per_seed.append(float(rr_sqrt))

    slope_log_mean = float(np.mean(slope_log_per_seed))
    slope_log_std = float(np.std(slope_log_per_seed, ddof=1)) if len(slope_log_per_seed) > 1 else 0.0
    r2_log_mean = float(np.mean(r2_log_per_seed))
    r2_log_std = float(np.std(r2_log_per_seed, ddof=1)) if len(r2_log_per_seed) > 1 else 0.0
    slope_sqrt_mean = float(np.mean(slope_sqrt_per_seed))
    slope_sqrt_std = float(np.std(slope_sqrt_per_seed, ddof=1)) if len(slope_sqrt_per_seed) > 1 else 0.0
    r2_sqrt_mean = float(np.mean(r2_sqrt_per_seed))
    r2_sqrt_std = float(np.std(r2_sqrt_per_seed, ddof=1)) if len(r2_sqrt_per_seed) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, mean, color=COLORS[0], linewidth=1.8, label=f"{algo_name} mean regret")
    ax.fill_between(t, mean - spread, mean + spread, color=COLORS[0], alpha=0.2, label="mean ± SE")
    ax.set_xlabel("Time step (t)", fontweight="bold")
    ax.set_ylabel(f"Cumulative {regret_type.upper()} Regret", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    metric_text = (
        f"log(t) fit (seed-avg): slope={slope_log_mean:.4f}±{slope_log_std:.4f}, "
        f"R^2={r2_log_mean:.4f}±{r2_log_std:.4f}\n"
        f"sqrt(t) fit (seed-avg): slope={slope_sqrt_mean:.4f}±{slope_sqrt_std:.4f}, "
        f"R^2={r2_sqrt_mean:.4f}±{r2_sqrt_std:.4f}"
    )
    ax.text(
        0.02, 0.98, metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
    )

    fig.tight_layout()
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_name}_{regret_type}_regret_plain_fit{suffix}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    metrics = {
        # Per-seed arrays
        "slope_log_t_per_seed": [float(x) for x in slope_log_per_seed],
        "r2_log_t_per_seed": [float(x) for x in r2_log_per_seed],
        "slope_sqrt_t_per_seed": [float(x) for x in slope_sqrt_per_seed],
        "r2_sqrt_t_per_seed": [float(x) for x in r2_sqrt_per_seed],
        # Aggregates used for cross-eta plots/tables
        "slope_log_t_mean": slope_log_mean,
        "slope_log_t_std": slope_log_std,
        "r2_log_t_mean": r2_log_mean,
        "r2_log_t_std": r2_log_std,
        "slope_sqrt_t_mean": slope_sqrt_mean,
        "slope_sqrt_t_std": slope_sqrt_std,
        "r2_sqrt_t_mean": r2_sqrt_mean,
        "r2_sqrt_t_std": r2_sqrt_std,
        # Mean-curve fit retained for reference
        "slope_log_t_mean_curve": float(slope_log_mean_curve),
        "r2_log_t_mean_curve": float(r2_log_mean_curve),
        "slope_sqrt_t_mean_curve": float(slope_sqrt_mean_curve),
        "r2_sqrt_t_mean_curve": float(r2_sqrt_mean_curve),
    }
    return out_path, metrics


def plot_eta_vs_two_r2(
    eta_to_r2_log: Mapping[float, float],
    eta_to_r2_sqrt: Mapping[float, float],
    save_dir,
    eta_to_r2_log_std: Mapping[float, float] | None = None,
    eta_to_r2_sqrt_std: Mapping[float, float] | None = None,
    regret_type="mbr",
):
    """
    Plot eta vs R^2 from:
      - Reg ~ a log(t)+b
      - Reg ~ c sqrt(t)+d
    """
    _setup_academic_plot()
    if not eta_to_r2_log or not eta_to_r2_sqrt:
        raise ValueError("R^2 mappings must not be empty.")

    keys = sorted(set(float(k) for k in eta_to_r2_log) & set(float(k) for k in eta_to_r2_sqrt))
    etas = np.asarray(keys, dtype=float)
    r2_log = np.asarray([float(eta_to_r2_log[k]) for k in keys], dtype=float)
    r2_sqrt = np.asarray([float(eta_to_r2_sqrt[k]) for k in keys], dtype=float)
    if eta_to_r2_log_std is not None:
        r2_log_std = np.asarray([float(eta_to_r2_log_std.get(k, 0.0)) for k in keys], dtype=float)
    else:
        r2_log_std = np.zeros_like(r2_log)
    if eta_to_r2_sqrt_std is not None:
        r2_sqrt_std = np.asarray([float(eta_to_r2_sqrt_std.get(k, 0.0)) for k in keys], dtype=float)
    else:
        r2_sqrt_std = np.zeros_like(r2_sqrt)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        etas, r2_log, yerr=r2_log_std,
        marker="o", color=COLORS[0], linewidth=1.7, markersize=6,
        capsize=3, elinewidth=1.2, label="R^2 for Reg~a log(t)+b"
    )
    ax.errorbar(
        etas, r2_sqrt, yerr=r2_sqrt_std,
        marker="s", color=COLORS[1], linewidth=1.7, markersize=6,
        capsize=3, elinewidth=1.2, label="R^2 for Reg~c sqrt(t)+d"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Strength (eta)", fontweight="bold")
    ax.set_ylabel(f"R^2 ({regret_type.upper()} regret fits)", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eta_vs_r2_{regret_type}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path