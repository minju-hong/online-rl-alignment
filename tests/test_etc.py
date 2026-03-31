from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust this import if your file is named ours_etc.py
from algo.etc import etc_s2p_cvxpy
from env import GBPMEnv, mu_linear, mu_logistic
import plot
import regret as reg
from solvers import compute_rho_E

from utils import ProgressBar, now_stamp

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
T = 1000            
T0 = 100            
K = 20
d = 5             
r = 1              
S = 5.0
mu_name = "logistic"  
phi_mode = "random"  # "basis" or "random"
instance_seed = 0
seeds = [0, 1, 2, 3, 4]

# ETC-specific knobs.
lam = 0.05
frob_bound = None
cvx_solver = "SCS"
cvx_max_iters = 10_000
cvx_verbose = False
commit_symmetric = True

# --- EVALUATION KNOBS ---
etc_eta = 1.0          
reg_type = "reverse_kl"  

# --- [CHANGE]: The Toggle! ---
eval_regret = "mbr"  # Switch between "mbr" and "abr"

algo_name = "etc_cvxpy"
base_out_dir = Path("results") / "test_etc"


def run_one_seed(seed: int) -> dict[str, Any]:
    mu = mu_logistic if mu_name == "logistic" else mu_linear
    env_seed = instance_seed + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_seed, mu=mu, phi_mode=phi_mode)

    rho = compute_rho_E(env.Phi, solver=cvx_solver)[0]

    progress = ProgressBar(T, prefix=f"Running ETC Seed {seed:2d}")
    original_step = env.step
    step_counter = [0]
    
    def step_wrapper(pi1, pi2):
        res = original_step(pi1, pi2)
        step_counter[0] += 1
        progress.update(step_counter[0])
        return res
        
    env.step = step_wrapper

    out = etc_s2p_cvxpy(
        env,
        T=T,
        T0=T0,
        rho=rho,
        mu=env.mu,
        link_type=mu_name,
        eta=etc_eta,          
        reg_type=reg_type,    
        lam=lam,
        frob_bound=frob_bound,
        episode_seed=seed,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
        commit_symmetric=commit_symmetric,
    )

    # Grab both policy sequences from the algorithm
    pi1_seq = np.asarray(out["pi1_seq"], dtype=float)
    pi2_seq = np.asarray(out["pi2_seq"], dtype=float)

    g_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)
    theta_hat = np.asarray(out["Theta_hat"], dtype=float)
    theta_frob_err = float(np.linalg.norm(theta_hat - env.Theta_star, ord="fro"))
    
    # --- [CHANGE]: Dynamically route the regret calculation ---
    if eval_regret == "mbr":
        inc = reg.mbr_regret(pi1_seq, g_star, eta=etc_eta, reg_type=reg_type)
    elif eval_regret == "abr":
        inc = reg.abr_regret(pi1_seq, pi2_seq, g_star, eta=etc_eta, reg_type=reg_type)
    else:
        raise ValueError(f"Unknown eval_regret type: {eval_regret}")
        
    cum = reg.cumulative(inc)

    return {
        "seed": int(seed),
        "t": np.arange(1, T + 1, dtype=int),
        f"{eval_regret}_inc": np.asarray(inc, dtype=float),
        f"{eval_regret}_cum": np.asarray(cum, dtype=float),
        "theta_frob_err_final": theta_frob_err,
    }


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any]) -> Path:
    algo_dir = run_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / (
        f"seed_{result['seed']}_d{d}_r{r}_K{K}_S{S:g}_phi{phi_mode}_"
        f"mu{mu_name}_reg{reg_type}_eta{etc_eta:g}_lam{lam:g}.npz"
    )
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        f"{eval_regret}_inc": np.asarray(result[f"{eval_regret}_inc"], dtype=float),
        f"{eval_regret}_cum": np.asarray(result[f"{eval_regret}_cum"], dtype=float),
        "theta_frob_err_final": np.asarray(result["theta_frob_err_final"], dtype=float),
    }
    meta2 = dict(meta)
    meta2.update({"algo": algo_name, "seed": int(result["seed"])})
    reg.save_npz(path, arrays=arrays, meta=meta2)
    return path


def main() -> None:
    if not seeds:
        raise ValueError("Need at least one seed in `seeds`.")
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T.")
    if phi_mode not in {"basis", "random"}:
        raise ValueError(f"phi_mode must be 'basis' or 'random'. Got {phi_mode!r}.")

    run_name = (
        f"d{d}_r{r}_K{K}_S{S:g}_{phi_mode}_{mu_name}_{reg_type}_"
        f"eta{etc_eta:g}_lam{lam:g}_{now_stamp()}"
    )
    run_dir = base_out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / (
        f"manifest_d{d}_r{r}_K{K}_S{S:g}_phi{phi_mode}_mu{mu_name}_"
        f"reg{reg_type}_eta{etc_eta:g}_lam{lam:g}.jsonl"
    )

    meta_common = {
        "tag": "test_etc",
        "algo": algo_name,
        "run_dir": str(run_dir),
        "created_at": now_stamp(),
        "T": int(T),
        "T0": int(T0),
        "K": int(K),
        "d": int(d),
        "r": int(r),
        "S": float(S),
        "instance_seed": int(instance_seed),
        "mu": mu_name,
        "phi_mode": phi_mode,
        "seeds": list(seeds),
        "etc_eta": float(etc_eta),
        "lam": float(lam),
        "frob_bound": frob_bound,
        "cvx_solver": cvx_solver,
        "cvx_max_iters": int(cvx_max_iters),
        "cvx_verbose": bool(cvx_verbose),
        "commit_symmetric": bool(commit_symmetric),
        "eta": float(etc_eta),
        "reg_type": reg_type,
        "eval_regret": eval_regret,
        "regret_def": f"Regularized {eval_regret.upper()} Regret",
    }

    all_inc: list[np.ndarray] = []
    all_cum: list[np.ndarray] = []
    all_theta_frob_final: list[float] = []
    
    for seed in seeds:
        t0 = time.perf_counter()
        result = run_one_seed(seed)
        saved_path = save_per_seed(run_dir, result, meta_common)
        
        # Dynamically grab the right key
        all_inc.append(np.asarray(result[f"{eval_regret}_inc"], dtype=float))
        all_cum.append(np.asarray(result[f"{eval_regret}_cum"], dtype=float))
        all_theta_frob_final.append(float(result["theta_frob_err_final"]))

        print(
            f"[seed {seed:2d}] final theta Frobenius error: "
            f"{float(result['theta_frob_err_final']):.6f}"
        )

        rec = {
            "algo": algo_name,
            "seed": int(seed),
            "file": str(saved_path),
            "seconds": round(time.perf_counter() - t0, 4),
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    t = np.arange(1, T + 1, dtype=int)
    arrays = {
        "t": t,
        "seeds": np.asarray(seeds, dtype=int),
        "algos": np.asarray([algo_name], dtype=str),
        f"{eval_regret}_inc__{algo_name}": np.stack(all_inc, axis=0),
        f"{eval_regret}_cum__{algo_name}": np.stack(all_cum, axis=0),
        f"theta_frob_err_final__{algo_name}": np.asarray(all_theta_frob_final, dtype=float),
    }
    summary_path = run_dir / (
        f"summary_d{d}_r{r}_K{K}_S{S:g}_phi{phi_mode}_mu{mu_name}_"
        f"reg{reg_type}_eta{etc_eta:g}_lam{lam:g}.npz"
    )
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)

    # --- [CHANGE]: Pass the regret_type to the plotter so it knows what to plot! ---
    single_path = plot.plot_single_algorithm(summary_path, algo_name, run_dir, regret_type=eval_regret)
    
    print(f"\n[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")
    print(f"[saved] plot: {single_path}")
    theta_frob_mean = float(np.mean(all_theta_frob_final))
    theta_frob_std = (
        float(np.std(all_theta_frob_final, ddof=1)) if len(all_theta_frob_final) > 1 else 0.0
    )
    print(
        f"[stats] final theta Frobenius error: "
        f"mean={theta_frob_mean:.6f}, std={theta_frob_std:.6f} "
        f"(n={len(all_theta_frob_final)})"
    )


if __name__ == "__main__":
    main()