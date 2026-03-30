from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from env import GBPMEnv, mu_logistic, mu_linear
import regret as reg
from algo.etc import etc_s2p_cvxpy
from solvers import compute_rho_E


# Small hardcoded test configuration.
T = 50
T0 = 10
K = 10
d = 5
seeds = [0, 1]
algos = ["etc_cvxpy"]

# Optional defaults kept from run.py.
r = 1
S = 4.0
instance_seed = 0
mu_name = "logistic"
lam = 0.01
frob_bound = None
cvx_solver = "SCS"
cvx_max_iters = 10_000
cvx_verbose = False
commit_symmetric = False
out_dir = "results"
tag = "test_pipeline"


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def algo_etc_cvxpy(
    env: GBPMEnv,
    *,
    T: int,
    T0: int,
    episode_seed: int,
    lam: float,
    frob_bound: float | None,
    cvx_solver: str,
    cvx_max_iters: int,
    cvx_verbose: bool,
    commit_symmetric: bool,
    **kwargs,
) -> dict[str, Any]:
    if "rho" not in kwargs:
        raise ValueError("algo_etc_cvxpy expects rho to be provided (rho=...).")

    rho = np.asarray(kwargs["rho"], dtype=float).reshape(-1)
    if rho.shape[0] != env.K:
        raise ValueError(f"rho has shape {rho.shape}, expected ({env.K},)")
    if np.any(rho < -1e-8) or abs(rho.sum() - 1.0) > 1e-6:
        raise ValueError(f"Invalid rho passed to algo_etc_cvxpy: min={rho.min()}, sum={rho.sum()}")

    out = etc_s2p_cvxpy(
        env,
        T=T,
        T0=T0,
        rho=rho,
        mu=env.mu,
        lam=lam,
        frob_bound=frob_bound,
        episode_seed=episode_seed,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
        commit_symmetric=commit_symmetric,
    )

    pi1_seq = np.zeros((T, env.K), dtype=float)
    pi1_seq[:T0, :] = rho
    pi_commit = out["pi_hat"] if commit_symmetric else out["pi1_hat"]
    pi1_seq[T0:, :] = pi_commit

    return {"pi1_seq": pi1_seq, "name": "etc_cvxpy", "details": out}


ALGO_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "etc_cvxpy": algo_etc_cvxpy,
}


def one_hot_from_actions(a_seq: np.ndarray, n_arms: int) -> np.ndarray:
    a_seq = np.asarray(a_seq, dtype=int).reshape(-1)
    horizon = len(a_seq)
    out = np.zeros((horizon, n_arms), dtype=float)
    out[np.arange(horizon), a_seq] = 1.0
    return out


def run_one_algo_one_seed(
    algo_name: str,
    seed: int,
    *,
    T: int,
    T0: int,
    K: int,
    d: int,
    r: int,
    S: float,
    instance_seed: int,
    mu_name: str,
    lam: float,
    frob_bound: float | None,
    cvx_solver: str,
    cvx_max_iters: int,
    cvx_verbose: bool,
    commit_symmetric: bool,
) -> dict[str, Any]:
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{algo_name}'. Available: {sorted(ALGO_REGISTRY)}")

    mu = mu_logistic if mu_name == "logistic" else mu_linear
    env_instance_seed = int(instance_seed) + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_instance_seed, mu=mu)

    rho = compute_rho_E(env.Phi, solver=cvx_solver)[0]
    sigma = env.Phi.T @ (rho[:, None] * env.Phi)
    eigs = np.linalg.eigvalsh(sigma)
    rank = int(np.sum(eigs > 1e-10))
    print(
        f"[diag] Sigma under rho_E: min_eig={eigs[0]:.3e}, "
        f"max_eig={eigs[-1]:.3e}, rank={rank}/{env.d}, K={env.K}, d={env.d}"
    )

    g_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)
    out = ALGO_REGISTRY[algo_name](
        env,
        T=T,
        T0=T0,
        rho=rho,
        episode_seed=seed,
        lam=lam,
        frob_bound=frob_bound,
        cvx_solver=cvx_solver,
        cvx_max_iters=cvx_max_iters,
        cvx_verbose=cvx_verbose,
        commit_symmetric=commit_symmetric,
    )

    if "pi1_seq" in out:
        pi1_seq = np.asarray(out["pi1_seq"], dtype=float)
    elif "a1_seq" in out:
        pi1_seq = one_hot_from_actions(out["a1_seq"], env.K)
    else:
        raise ValueError(f"Algorithm '{algo_name}' must return 'pi1_seq' or 'a1_seq'.")

    inc = reg.max_nash_regret_increments(pi1_seq, g_star, value_star=0.5)
    cum = reg.cumulative(inc)

    return {
        "seed": int(seed),
        "algo": algo_name,
        "t": np.arange(1, T + 1),
        "mnreg_inc": inc,
        "mnreg_cum": cum,
        "pi1_seq": pi1_seq,
    }


def save_per_seed(run_dir: Path, result: dict[str, Any], meta: dict[str, Any]) -> Path:
    algo = result["algo"]
    seed = result["seed"]
    algo_dir = run_dir / algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    path = algo_dir / f"seed_{seed}.npz"
    arrays = {
        "t": np.asarray(result["t"], dtype=int),
        "mnreg_inc": np.asarray(result["mnreg_inc"], dtype=float),
        "mnreg_cum": np.asarray(result["mnreg_cum"], dtype=float),
    }
    meta2 = dict(meta)
    meta2.update({"algo": algo, "seed": seed})
    reg.save_npz(path, arrays=arrays, meta=meta2)
    return path


def main() -> None:
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T.")

    run_dir = Path(out_dir) / f"{tag}_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        "tag": tag,
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
        "algos": list(algos),
        "seeds": list(seeds),
        "lam": float(lam),
        "frob_bound": frob_bound,
        "cvx_solver": cvx_solver,
        "cvx_max_iters": int(cvx_max_iters),
        "cvx_verbose": bool(cvx_verbose),
        "commit_symmetric": bool(commit_symmetric),
        "regret_def": "MNReg(T)=sum_t (1/2 - min_j (pi1_t^T G_star)[j])",
    }

    manifest_path = run_dir / "manifest.jsonl"
    all_cum: dict[str, list[np.ndarray]] = {a: [] for a in algos}
    all_inc: dict[str, list[np.ndarray]] = {a: [] for a in algos}

    for algo in algos:
        for seed in seeds:
            t0 = time.perf_counter()
            print(f"\n== Running algo='{algo}' seed={seed} ==")
            result = run_one_algo_one_seed(
                algo,
                seed,
                T=T,
                T0=T0,
                K=K,
                d=d,
                r=r,
                S=S,
                instance_seed=instance_seed,
                mu_name=mu_name,
                lam=lam,
                frob_bound=frob_bound,
                cvx_solver=cvx_solver,
                cvx_max_iters=cvx_max_iters,
                cvx_verbose=cvx_verbose,
                commit_symmetric=commit_symmetric,
            )
            saved_path = save_per_seed(run_dir, result, meta_common)
            rec = {
                "algo": algo,
                "seed": int(seed),
                "file": str(saved_path),
                "seconds": round(time.perf_counter() - t0, 4),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            all_inc[algo].append(np.asarray(result["mnreg_inc"], dtype=float))
            all_cum[algo].append(np.asarray(result["mnreg_cum"], dtype=float))

    t = np.arange(1, T + 1, dtype=int)
    arrays: dict[str, np.ndarray] = {
        "t": t,
        "seeds": np.asarray(seeds, dtype=int),
        "algos": np.asarray(algos, dtype=str),
    }
    for algo in algos:
        arrays[f"mnreg_inc__{algo}"] = np.stack(all_inc[algo], axis=0)
        arrays[f"mnreg_cum__{algo}"] = np.stack(all_cum[algo], axis=0)

    summary_path = run_dir / "summary.npz"
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)
    print(f"\n[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")

    import plot

    plot.plot_single_algorithm(summary_path, "etc_cvxpy", run_dir)
    plot.plot_combined_regret(summary_path, run_dir)
    print(f"[saved] plots in: {run_dir}")


if __name__ == "__main__":
    main()
