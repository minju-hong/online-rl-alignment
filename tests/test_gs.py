from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust this import if your file is named ours_gs.py instead of gs.py
from algo.gs import gs_s2p_cvxpy
from env import GBPMEnv, mu_linear, mu_logistic
import plot
import regret as reg

# --- IMPORT FROM YOUR NEW UTILS FILE ---
from utils import ProgressBar, now_stamp

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
T = 10000
K = 40
d = 10
r = 1
S = 10.0
mu_name = "logistic"  # "logistic" or "linear"
phi_mode = "random"  # "basis" or "random"
instance_seed = 0
seeds = list(range(20))

# GS-specific knobs.
gs_eta = 1000.0    
reg_type = "reverse_kl"  
update_freq = 2   
run_eta_sweep = True
eta_grid = np.logspace(-2, 3, num=10)
parallel_seeds = True
n_jobs = max(1, min(8, (os.cpu_count() or 1)))

# CVX options.
cvx_solver = "SCS"
cvx_max_iters = 10_000
cvx_verbose = False

# Theta estimator options.
theta_estimator = "onepass_ons"  # logistic: "onepass_ons" only, linear: "cvxpy"
ons_a0 = 1.0
ons_step_size = 1.0
ons_hess_floor = 1e-6
ons_hess_cap = 0.25

algo_name = "gs_cvxpy"
base_out_dir = Path("results") / "test_gs"


def one_hot_from_actions(a_seq: np.ndarray, n_arms: int) -> np.ndarray:
    """Fallback utility if the agent doesn't return pi1_seq natively."""
    a_seq = np.asarray(a_seq, dtype=int).reshape(-1)
    horizon = len(a_seq)
    out = np.zeros((horizon, n_arms), dtype=float)
    out[np.arange(horizon), a_seq] = 1.0
    return out


def run_one_seed(seed: int, eta_value: float, *, enable_progress: bool = True) -> dict[str, Any]:
    mu = mu_logistic if mu_name == "logistic" else mu_linear
    env_seed = instance_seed + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_seed, mu=mu, phi_mode=phi_mode)

    if enable_progress:
        progress = ProgressBar(T, prefix=f"Running GS Seed {seed:2d}")
        original_step = env.step
        step_counter = [0]

        def step_wrapper(pi1, pi2):
            res = original_step(pi1, pi2)
            step_counter[0] += 1
            progress.update(step_counter[0])
            return res

        env.step = step_wrapper

    rho = np.ones(env.K) / env.K # Uniform
    out = gs_s2p_cvxpy(
        env,
        T=T,
        rho=rho,
        mu=env.mu,
        link_type=mu_name,
        eta=float(eta_value),
        reg_type=reg_type,
        S=S,
        update_freq=update_freq,
        episode_seed=seed,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
        theta_estimator=theta_estimator,
        ons_a0=ons_a0,
        ons_step_size=ons_step_size,
        ons_hess_floor=ons_hess_floor,
        ons_hess_cap=ons_hess_cap,
    )

    # --- EXPECTED REGRET FIX ---
    if "pi1_seq" in out:
        pi1_seq = np.asarray(out["pi1_seq"], dtype=float)
    else:
        # Fallback to noisy one-hot if 'pi1_seq' hasn't been added to gs.py yet
        a1_seq = np.asarray([step[1] for step in out["traj"]], dtype=int)
        pi1_seq = one_hot_from_actions(a1_seq, env.K)

    # Calculate ground truth payoff matrix
    g_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)
    
    # --- EVALUATE REGULARIZED MBR REGRET ---
    mbr_inc = reg.mbr_regret(pi1_seq, g_star, eta=float(eta_value), reg_type=reg_type)
    mbr_cum = reg.cumulative(mbr_inc)

    return {
        "seed": int(seed),
        "t": np.arange(1, T + 1, dtype=int),
        "mbr_inc": np.asarray(mbr_inc, dtype=float),
        "mbr_cum": np.asarray(mbr_cum, dtype=float),
        "theta_frob_err": np.asarray(out["theta_frob_err_seq"], dtype=float),
        "ne_residual": np.asarray(out["ne_residual_seq"], dtype=float),
    }


def run_one_seed_task(task: tuple[int, float]) -> tuple[int, dict[str, Any], float]:
    seed, eta_value = task
    t0 = time.perf_counter()
    result = run_one_seed(seed, eta_value, enable_progress=False)
    dt = time.perf_counter() - t0
    return int(seed), result, float(dt)


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any], eta_value: float) -> Path:
    algo_dir = run_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / (
        f"seed_{result['seed']}_d{d}_r{r}_eta{eta_value:g}_mu{mu_name}_reg{reg_type}_est{theta_estimator}.npz"
    )
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        "mbr_inc": np.asarray(result["mbr_inc"], dtype=float),
        "mbr_cum": np.asarray(result["mbr_cum"], dtype=float),
        "theta_frob_err": np.asarray(result["theta_frob_err"], dtype=float),
        "ne_residual": np.asarray(result["ne_residual"], dtype=float),
    }
    meta2 = dict(meta)
    meta2.update({"algo": algo_name, "seed": int(result["seed"])})
    reg.save_npz(path, arrays=arrays, meta=meta2)
    return path


def main() -> None:
    if not seeds:
        raise ValueError("Need at least one seed in `seeds`.")
    if d < 2 * r:
        raise ValueError(f"Need d >= 2*r. Got d={d}, r={r}.")
    if update_freq < 1:
        raise ValueError("Need update_freq >= 1.")
    if n_jobs < 1:
        raise ValueError("Need n_jobs >= 1.")
    if phi_mode not in {"basis", "random"}:
        raise ValueError(f"phi_mode must be 'basis' or 'random'. Got {phi_mode!r}.")

    run_name = (
        f"d{d}_r{r}_K{K}_S{S:g}_{phi_mode}_{mu_name}_{reg_type}_{now_stamp()}"
    )
    run_dir = base_out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_eta_sweep:
        manifest_path = run_dir / (
            f"manifest_d{d}_r{r}_etaSweep_mu{mu_name}_reg{reg_type}_est{theta_estimator}.jsonl"
        )
    else:
        manifest_path = run_dir / (
            f"manifest_d{d}_r{r}_eta{gs_eta:g}_mu{mu_name}_reg{reg_type}_est{theta_estimator}.jsonl"
        )

    meta_common = {
        "tag": "test_gs",
        "algo": algo_name,
        "run_dir": str(run_dir),
        "created_at": now_stamp(),
        "T": int(T),
        "K": int(K),
        "d": int(d),
        "r": int(r),
        "S": float(S),
        "instance_seed": int(instance_seed),
        "mu": mu_name,
        "phi_mode": phi_mode,
        "seeds": list(seeds),
        "eta": float(gs_eta),
        "run_eta_sweep": bool(run_eta_sweep),
        "eta_grid": [float(x) for x in eta_grid],
        "parallel_seeds": bool(parallel_seeds),
        "n_jobs": int(n_jobs),
        "reg_type": reg_type,
        "update_freq": int(update_freq),
        "cvx_solver": cvx_solver,
        "cvx_max_iters": int(cvx_max_iters),
        "cvx_verbose": bool(cvx_verbose),
        "theta_estimator": theta_estimator,
        "ons_a0": float(ons_a0),
        "ons_step_size": float(ons_step_size),
        "ons_hess_floor": float(ons_hess_floor),
        "ons_hess_cap": float(ons_hess_cap),
        "regret_def": "Regularized Max-Best-Response (MBR) Regret",
        "pi1_seq_note": "Expected policy probabilities.",
    }

    eta_to_r2_log_t_raw: dict[float, float] = {}
    eta_to_r2_sqrt_t_raw: dict[float, float] = {}
    eta_to_r2_log_t_raw_std: dict[float, float] = {}
    eta_to_r2_sqrt_t_raw_std: dict[float, float] = {}
    eta_to_final_regret: dict[float, float] = {}
    eta_to_final_regret_std: dict[float, float] = {}
    eta_summary_rows: list[dict[str, Any]] = []

    sweep_values = [float(gs_eta)] if not run_eta_sweep else [float(x) for x in eta_grid]
    t = np.arange(1, T + 1, dtype=int)

    for eta_value in sweep_values:
        print(f"\n=== ETA {eta_value:g} ===")
        all_inc: list[np.ndarray] = []
        all_cum: list[np.ndarray] = []
        all_theta_err: list[np.ndarray] = []
        all_ne_resid: list[np.ndarray] = []

        results_by_seed: dict[int, tuple[dict[str, Any], float]] = {}
        if parallel_seeds and len(seeds) > 1:
            workers = min(int(n_jobs), len(seeds))
            tasks = [(int(seed), float(eta_value)) for seed in seeds]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for seed_out, result_out, sec_out in ex.map(run_one_seed_task, tasks):
                    results_by_seed[int(seed_out)] = (result_out, float(sec_out))
        else:
            for seed in seeds:
                t0 = time.perf_counter()
                result = run_one_seed(seed, eta_value, enable_progress=True)
                results_by_seed[int(seed)] = (result, float(time.perf_counter() - t0))

        # Preserve deterministic ordering by seed.
        for seed in seeds:
            result, seconds = results_by_seed[int(seed)]
            saved_path = save_per_seed(run_dir, result, meta_common, eta_value)
            all_inc.append(np.asarray(result["mbr_inc"], dtype=float))
            all_cum.append(np.asarray(result["mbr_cum"], dtype=float))
            all_theta_err.append(np.asarray(result["theta_frob_err"], dtype=float))
            all_ne_resid.append(np.asarray(result["ne_residual"], dtype=float))

            rec = {
                "algo": algo_name,
                "eta": float(eta_value),
                "seed": int(seed),
                "file": str(saved_path),
                "seconds": round(float(seconds), 4),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        arrays = {
            "t": t,
            "seeds": np.asarray(seeds, dtype=int),
            "algos": np.asarray([algo_name], dtype=str),
            f"mbr_inc__{algo_name}": np.stack(all_inc, axis=0),
            f"mbr_cum__{algo_name}": np.stack(all_cum, axis=0),
            f"theta_frob_err__{algo_name}": np.stack(all_theta_err, axis=0),
            f"ne_residual__{algo_name}": np.stack(all_ne_resid, axis=0),
        }
        summary_path = run_dir / (
            f"summary_d{d}_r{r}_eta{eta_value:g}_mu{mu_name}_reg{reg_type}_est{theta_estimator}.npz"
        )
        meta_eta = dict(meta_common)
        meta_eta["eta"] = float(eta_value)
        reg.save_npz(summary_path, arrays=arrays, meta=meta_eta)

        theta_err_path = plot.plot_theta_frob_error(summary_path, algo_name, run_dir)
        plain_fit_path, fit_metrics = plot.plot_regret_with_fit_metrics(
            summary_path,
            algo_name,
            run_dir,
            regret_type="mbr",
            fit_start_t=10,
            suffix=f"_eta{eta_value:g}",
        )
        print(f"[saved] summary: {summary_path}")
        print(f"[saved] theta-error plot: {theta_err_path}")
        print(
            f"[saved] plain regret+fit plot: {plain_fit_path} "
            f"(log-slope={fit_metrics['slope_log_t_mean']:.4f}±{fit_metrics['slope_log_t_std']:.4f}, "
            f"log-R2={fit_metrics['r2_log_t_mean']:.4f}±{fit_metrics['r2_log_t_std']:.4f}, "
            f"sqrt-slope={fit_metrics['slope_sqrt_t_mean']:.4f}±{fit_metrics['slope_sqrt_t_std']:.4f}, "
            f"sqrt-R2={fit_metrics['r2_sqrt_t_mean']:.4f}±{fit_metrics['r2_sqrt_t_std']:.4f})"
        )

        finals = [float(y[-1]) for y in all_cum]
        ne_end = [float(y[-1]) for y in all_ne_resid]
        eta_to_final_regret[float(eta_value)] = float(np.mean(finals))
        eta_to_final_regret_std[float(eta_value)] = (
            float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0
        )
        r2_log_vals = np.asarray(fit_metrics["r2_log_t_per_seed"], dtype=float)
        r2_sqrt_vals = np.asarray(fit_metrics["r2_sqrt_t_per_seed"], dtype=float)
        eta_to_r2_log_t_raw[float(eta_value)] = float(np.mean(r2_log_vals))
        eta_to_r2_sqrt_t_raw[float(eta_value)] = float(np.mean(r2_sqrt_vals))
        eta_to_r2_log_t_raw_std[float(eta_value)] = (
            float(np.std(r2_log_vals, ddof=1)) if r2_log_vals.size > 1 else 0.0
        )
        eta_to_r2_sqrt_t_raw_std[float(eta_value)] = (
            float(np.std(r2_sqrt_vals, ddof=1)) if r2_sqrt_vals.size > 1 else 0.0
        )
        eta_summary_rows.append(
            {
                "eta": float(eta_value),
                "summary_path": str(summary_path),
                "plain_fit_plot_path": str(plain_fit_path),
                "theta_plot_path": str(theta_err_path),
                "slope_log_t_mean": float(fit_metrics["slope_log_t_mean"]),
                "slope_log_t_std": float(fit_metrics["slope_log_t_std"]),
                "r2_log_t_mean": float(fit_metrics["r2_log_t_mean"]),
                "r2_log_t_std": float(fit_metrics["r2_log_t_std"]),
                "slope_sqrt_t_mean": float(fit_metrics["slope_sqrt_t_mean"]),
                "slope_sqrt_t_std": float(fit_metrics["slope_sqrt_t_std"]),
                "r2_sqrt_t_mean": float(fit_metrics["r2_sqrt_t_mean"]),
                "r2_sqrt_t_std": float(fit_metrics["r2_sqrt_t_std"]),
                "slope_log_t_per_seed": fit_metrics["slope_log_t_per_seed"],
                "r2_log_t_per_seed": fit_metrics["r2_log_t_per_seed"],
                "slope_sqrt_t_per_seed": fit_metrics["slope_sqrt_t_per_seed"],
                "r2_sqrt_t_per_seed": fit_metrics["r2_sqrt_t_per_seed"],
                "plot_r2_log_t_mean_raw": eta_to_r2_log_t_raw[float(eta_value)],
                "plot_r2_log_t_std_raw": eta_to_r2_log_t_raw_std[float(eta_value)],
                "plot_r2_sqrt_t_mean_raw": eta_to_r2_sqrt_t_raw[float(eta_value)],
                "plot_r2_sqrt_t_std_raw": eta_to_r2_sqrt_t_raw_std[float(eta_value)],
                "slope_log_t_mean_curve": float(fit_metrics["slope_log_t_mean_curve"]),
                "r2_log_t_mean_curve": float(fit_metrics["r2_log_t_mean_curve"]),
                "slope_sqrt_t_mean_curve": float(fit_metrics["slope_sqrt_t_mean_curve"]),
                "r2_sqrt_t_mean_curve": float(fit_metrics["r2_sqrt_t_mean_curve"]),
                "final_regret_mean": float(np.mean(finals)),
                "final_regret_per_seed": finals,
                "final_ne_residual_mean": float(np.mean(ne_end)),
                "final_ne_residual_per_seed": ne_end,
            }
        )

    r2_plot_raw_path = plot.plot_eta_vs_two_r2(
        eta_to_r2_log_t_raw,
        eta_to_r2_sqrt_t_raw,
        run_dir,
        eta_to_r2_log_t_raw_std,
        eta_to_r2_sqrt_t_raw_std,
        regret_type="mbr",
        suffix="_raw",
    )
    regret_vs_eta_path = plot.plot_eta_vs_regret_with_errorbars(
        eta_to_final_regret,
        run_dir,
        eta_to_regret_std=eta_to_final_regret_std,
        regret_type="mbr",
        t_value=int(T),
    )
    regret_vs_eta_theory_loglog_path = plot.plot_eta_vs_regret_theory_bound_loglog(
        eta_to_final_regret,
        run_dir,
        eta_to_regret_std=eta_to_final_regret_std,
        regret_type="mbr",
        t_value=int(T),
    )
    eta_json = run_dir / "eta_summary.json"
    with open(eta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "T": int(T),
                "eta_grid": [float(x) for x in sweep_values],
                "algo": algo_name,
                "regret_type": "mbr",
                "rows": sorted(eta_summary_rows, key=lambda r: r["eta"]),
            },
            f,
            indent=2,
        )

    print(f"\n[saved] manifest: {manifest_path}")
    print(f"[saved] eta-vs-R2 plot (raw): {r2_plot_raw_path}")
    print(f"[saved] eta-vs-regret plot (t={T}): {regret_vs_eta_path}")
    print(f"[saved] eta-vs-regret theory log-log plot (t={T}): {regret_vs_eta_theory_loglog_path}")
    print(f"[saved] eta summary json: {eta_json}")

if __name__ == "__main__":
    main()