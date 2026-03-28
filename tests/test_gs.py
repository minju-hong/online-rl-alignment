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
T = 100            
K = 10
d = 40
r = 1
S = 4.0
mu_name = "logistic"  # "logistic" or "linear"
instance_seed = 0
seeds = [0, 1, 2, 3, 4]

# GS-specific knobs.
gs_eta = 1       
reg_type = "reverse_kl"  
update_freq = 2    

# CVX options.
cvx_solver = "SCS"
cvx_max_iters = 10_000
cvx_verbose = False

algo_name = "gs_cvxpy"
base_out_dir = Path("results") / "test_gs"


def one_hot_from_actions(a_seq: np.ndarray, n_arms: int) -> np.ndarray:
    """Fallback utility if the agent doesn't return pi1_seq natively."""
    a_seq = np.asarray(a_seq, dtype=int).reshape(-1)
    horizon = len(a_seq)
    out = np.zeros((horizon, n_arms), dtype=float)
    out[np.arange(horizon), a_seq] = 1.0
    return out


def run_one_seed(seed: int) -> dict[str, Any]:
    mu = mu_logistic if mu_name == "logistic" else mu_linear
    env_seed = instance_seed + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_seed, mu=mu)

    # --- PROGRESS BAR INJECTION ---
    progress = ProgressBar(T, prefix=f"Running GS Seed {seed:2d}")
    original_step = env.step
    step_counter = [0]
    
    def step_wrapper(pi1, pi2):
        res = original_step(pi1, pi2)
        step_counter[0] += 1
        progress.update(step_counter[0])
        return res
        
    env.step = step_wrapper
    # ------------------------------

    rho = np.ones(env.K) / env.K # Uniform
    out = gs_s2p_cvxpy(
        env,
        T=T,
        rho=rho,
        mu=env.mu,
        link_type=mu_name,
        eta=gs_eta,
        reg_type=reg_type,
        S=S,
        update_freq=update_freq,
        episode_seed=seed,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
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
    mbr_inc = reg.mbr_regret(pi1_seq, g_star, eta=gs_eta, reg_type=reg_type)
    mbr_cum = reg.cumulative(mbr_inc)

    return {
        "seed": int(seed),
        "t": np.arange(1, T + 1, dtype=int),
        "mbr_inc": np.asarray(mbr_inc, dtype=float),
        "mbr_cum": np.asarray(mbr_cum, dtype=float),
    }


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any]) -> Path:
    algo_dir = run_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / f"seed_{result['seed']}_d{d}_r{r}.npz"
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        "mbr_inc": np.asarray(result["mbr_inc"], dtype=float),
        "mbr_cum": np.asarray(result["mbr_cum"], dtype=float),
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

    run_name = f"d{d}_r{r}_eta{gs_eta:g}_{now_stamp()}"
    run_dir = base_out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"manifest_d{d}_r{r}_eta{gs_eta:g}.jsonl"

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
        "seeds": list(seeds),
        "eta": float(gs_eta),
        "reg_type": reg_type,
        "update_freq": int(update_freq),
        "cvx_solver": cvx_solver,
        "cvx_max_iters": int(cvx_max_iters),
        "cvx_verbose": bool(cvx_verbose),
        "regret_def": "Regularized Max-Best-Response (MBR) Regret",
        "pi1_seq_note": "Expected policy probabilities.",
    }

    all_inc: list[np.ndarray] = []
    all_cum: list[np.ndarray] = []
    
    for seed in seeds:
        t0 = time.perf_counter()
        result = run_one_seed(seed)
        saved_path = save_per_seed(run_dir, result, meta_common)
        all_inc.append(np.asarray(result["mbr_inc"], dtype=float))
        all_cum.append(np.asarray(result["mbr_cum"], dtype=float))

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
        f"mbr_inc__{algo_name}": np.stack(all_inc, axis=0),
        f"mbr_cum__{algo_name}": np.stack(all_cum, axis=0),
    }
    summary_path = run_dir / f"summary_d{d}_r{r}_eta{gs_eta:g}.npz"
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)

    single_path = plot.plot_single_algorithm(summary_path, algo_name, run_dir)
    print(f"\n[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")
    print(f"[saved] plot: {single_path}")

if __name__ == "__main__":
    main()