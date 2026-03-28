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

from algo.omg import omg_s2p_cvxpy
from env import GBPMEnv, mu_linear
import plot
import regret as reg
from utils import ProgressBar, now_stamp

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
T = 300            
K = 10
d = 10             
r = 1              
S = 4.0
mu_name = "linear"  # OMG strictly requires linear for ridge regression
instance_seed = 0
seeds = [0, 1, 2, 3, 4]

# OMG-specific knobs.
omg_eta = 4.0       
reg_type = "reverse_kl"
lam = 1.0           # Ridge regularization
alpha = 1.0         # Optimistic bonus scale
update_freq = 2

cvx_solver = "SCS"

# --- EVALUATION KNOBS ---
eval_regret = "mbr"  # Switch between "mbr" and "abr"

algo_name = "omg_cvxpy"
base_out_dir = Path("results") / "test_omg"


def run_one_seed(seed: int) -> dict[str, Any]:
    # Strictly enforce linear mu for OMG
    mu = mu_linear
    env_seed = instance_seed + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_seed, mu=mu)

    progress = ProgressBar(T, prefix=f"Running OMG Seed {seed:2d}")
    original_step = env.step
    step_counter = [0]
    
    def step_wrapper(pi1, pi2):
        res = original_step(pi1, pi2)
        step_counter[0] += 1
        progress.update(step_counter[0])
        return res
        
    env.step = step_wrapper

    out = omg_s2p_cvxpy(
        env,
        T=T,
        mu=env.mu,
        eta=omg_eta,
        reg_type=reg_type,
        lam=lam,
        alpha=alpha,
        update_freq=update_freq,
        episode_seed=seed,
        cvx_solver=cvx_solver
    )

    # Grab both policy sequences from the algorithm
    pi1_seq = np.asarray(out["pi1_seq"], dtype=float)
    pi2_seq = np.asarray(out["pi2_seq"], dtype=float)

    g_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)
    
    if eval_regret == "mbr":
        inc = reg.mbr_regret(pi1_seq, g_star, eta=omg_eta, reg_type=reg_type)
    elif eval_regret == "abr":
        inc = reg.abr_regret(pi1_seq, pi2_seq, g_star, eta=omg_eta, reg_type=reg_type)
    else:
        raise ValueError(f"Unknown eval_regret type: {eval_regret}")
        
    cum = reg.cumulative(inc)

    return {
        "seed": int(seed),
        "t": np.arange(1, T + 1, dtype=int),
        f"{eval_regret}_inc": np.asarray(inc, dtype=float),
        f"{eval_regret}_cum": np.asarray(cum, dtype=float),
    }


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any]) -> Path:
    algo_dir = run_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / f"seed_{result['seed']}_d{d}_r{r}.npz"
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        f"{eval_regret}_inc": np.asarray(result[f"{eval_regret}_inc"], dtype=float),
        f"{eval_regret}_cum": np.asarray(result[f"{eval_regret}_cum"], dtype=float),
    }
    meta2 = dict(meta)
    meta2.update({"algo": algo_name, "seed": int(result["seed"])})
    reg.save_npz(path, arrays=arrays, meta=meta2)
    return path


def main() -> None:
    if not seeds:
        raise ValueError("Need at least one seed in `seeds`.")

    run_name = f"d{d}_r{r}_{now_stamp()}"
    run_dir = base_out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"manifest_d{d}_r{r}.jsonl"

    meta_common = {
        "tag": "test_omg",
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
        "eta": float(omg_eta),
        "reg_type": reg_type,
        "lam": float(lam),
        "alpha": float(alpha),
        "update_freq": int(update_freq),
        "cvx_solver": cvx_solver,
        "eval_regret": eval_regret,
        "regret_def": f"Regularized {eval_regret.upper()} Regret",
    }

    all_inc: list[np.ndarray] = []
    all_cum: list[np.ndarray] = []
    
    for seed in seeds:
        t0 = time.perf_counter()
        result = run_one_seed(seed)
        saved_path = save_per_seed(run_dir, result, meta_common)
        
        all_inc.append(np.asarray(result[f"{eval_regret}_inc"], dtype=float))
        all_cum.append(np.asarray(result[f"{eval_regret}_cum"], dtype=float))

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
    }
    summary_path = run_dir / f"summary_d{d}_r{r}.npz"
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)

    single_path = plot.plot_single_algorithm(summary_path, algo_name, run_dir, regret_type=eval_regret)
    
    print(f"\n[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")
    print(f"[saved] plot: {single_path}")


if __name__ == "__main__":
    main()