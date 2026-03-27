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

from algo.gs import gs_s2p_cvxpy
from env import GBPMEnv, mu_linear, mu_logistic
import plot
import regret as reg
from solvers import compute_rho_E


# Edit values here directly.
T = 100
K = 10
d = 10
r = 2
S = 4.0
mu_name = "logistic"  # "logistic" or "linear"
instance_seed = 0
seeds = [0, 1, 2, 3, 4]

# GS-specific knobs.
# Change this value to control GS eta.
gs_eta = 1.0
reg_type = "reverse_kl"  # "reverse_kl", "chi_squared", "tsallis"
update_freq = 10

# CVX options.
cvx_solver = "SCS"
cvx_max_iters = 10_000
cvx_verbose = False

algo_name = "gs_cvxpy"
base_out_dir = Path("results") / "test_gs"


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def one_hot_from_actions(a_seq: np.ndarray, n_arms: int) -> np.ndarray:
    a_seq = np.asarray(a_seq, dtype=int).reshape(-1)
    horizon = len(a_seq)
    out = np.zeros((horizon, n_arms), dtype=float)
    out[np.arange(horizon), a_seq] = 1.0
    return out


def run_one_seed(seed: int) -> dict[str, Any]:
    mu = mu_logistic if mu_name == "logistic" else mu_linear
    env_seed = instance_seed + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_seed, mu=mu)

    rho = compute_rho_E(env.Phi, solver=cvx_solver)[0]
    out = gs_s2p_cvxpy(
        env,
        T=T,
        rho=rho,
        mu=env.mu,
        eta=gs_eta,
        reg_type=reg_type,
        S=S,
        update_freq=update_freq,
        episode_seed=seed,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
    )

    # GS returns trajectory of sampled actions. For regret evaluation here,
    # use the realized player-1 actions as one-hot mixed strategies.
    a1_seq = np.asarray([step[1] for step in out["traj"]], dtype=int)
    pi1_seq = one_hot_from_actions(a1_seq, env.K)

    g_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)
    mnreg_inc = reg.max_nash_regret_increments(pi1_seq, g_star, value_star=0.5)
    mnreg_cum = reg.cumulative(mnreg_inc)

    return {
        "seed": int(seed),
        "t": np.arange(1, T + 1, dtype=int),
        "mnreg_inc": np.asarray(mnreg_inc, dtype=float),
        "mnreg_cum": np.asarray(mnreg_cum, dtype=float),
    }


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any]) -> Path:
    algo_dir = run_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / f"seed_{result['seed']}_d{d}_r{r}.npz"
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        "mnreg_inc": np.asarray(result["mnreg_inc"], dtype=float),
        "mnreg_cum": np.asarray(result["mnreg_cum"], dtype=float),
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
        "regret_def": "MNReg(T)=sum_t (1/2 - min_j (pi1_t^T G_star)[j])",
        "pi1_seq_note": "Computed from realized sampled actions in GS trajectory.",
    }

    all_inc: list[np.ndarray] = []
    all_cum: list[np.ndarray] = []
    for seed in seeds:
        t0 = time.perf_counter()
        print(f"Running GS seed={seed}")
        result = run_one_seed(seed)
        saved_path = save_per_seed(run_dir, result, meta_common)
        all_inc.append(np.asarray(result["mnreg_inc"], dtype=float))
        all_cum.append(np.asarray(result["mnreg_cum"], dtype=float))

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
        f"mnreg_inc__{algo_name}": np.stack(all_inc, axis=0),
        f"mnreg_cum__{algo_name}": np.stack(all_cum, axis=0),
    }
    summary_path = run_dir / f"summary_d{d}_r{r}_eta{gs_eta:g}.npz"
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)

    single_path = plot.plot_single_algorithm(summary_path, algo_name, run_dir)
    print(f"[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")
    print(f"[saved] plot: {single_path}")


if __name__ == "__main__":
    main()
