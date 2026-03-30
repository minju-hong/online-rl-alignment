# run.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from env import GBPMEnv, mu_logistic, mu_linear
import regret as reg

from algo.etc import etc_s2p_cvxpy
from solvers import compute_rho_E


# -----------------------------
# Progress logging (10% ticks)
# -----------------------------
class TenPercentProgress:
    def __init__(self, total: int, *, prefix: str = ""):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.start = time.perf_counter()
        self.last_tick_time = self.start
        self.next_pct = 10

    def update(self, done: int):
        done = int(done)
        pct = int(round(100 * done / self.total))
        pct = min(100, max(0, pct))

        if pct >= self.next_pct or done >= self.total:
            now = time.perf_counter()
            elapsed = now - self.start
            delta = now - self.last_tick_time
            self.last_tick_time = now

            rate = elapsed / done if done > 0 else None
            eta = rate * (self.total - done) if rate is not None else None

            bar_len = 20
            filled = int(round(bar_len * pct / 100))
            bar = "#" * filled + "-" * (bar_len - filled)

            eta_str = f"{eta:.1f}s" if eta is not None else "?"
            print(f"{self.prefix}[{bar}] {pct:3d}%  ({done}/{self.total})  +{delta:.1f}s  elapsed {elapsed:.1f}s  eta {eta_str}")

            while self.next_pct <= pct:
                self.next_pct += 10


def now_stamp() -> str:
    # Asia/Seoul already in your system setting; localtime is fine
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


# -----------------------------
# Algorithms
# -----------------------------


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
) -> Dict[str, Any]:

    if "rho" not in kwargs:
        raise ValueError("algo_etc_cvxpy expects rho to be provided (rho=...).")

    rho = np.asarray(kwargs["rho"], dtype=float).reshape(-1)
    if rho.shape[0] != env.K:
        raise ValueError(f"rho has shape {rho.shape}, expected ({env.K},)")
    # (Optional) quick sanity
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


ALGO_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "etc_cvxpy": algo_etc_cvxpy,
}


def one_hot_from_actions(a_seq: np.ndarray, K: int) -> np.ndarray:
    a_seq = np.asarray(a_seq, dtype=int).reshape(-1)
    T = len(a_seq)
    out = np.zeros((T, K), dtype=float)
    out[np.arange(T), a_seq] = 1.0
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
) -> Dict[str, Any]:
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{algo_name}'. Available: {sorted(ALGO_REGISTRY)}")

    mu = mu_logistic if mu_name == "logistic" else mu_linear

    env_instance_seed = int(instance_seed) + 1_000_003 * int(seed)
    env = GBPMEnv(K=K, d=d, r=r, S=S, instance_seed=env_instance_seed, mu=mu)
    rho = compute_rho_E(env.Phi, solver=cvx_solver)[0]  # same rho you use in algo_etc_cvxpy
    Sigma = env.Phi.T @ (rho[:, None] * env.Phi)  # (d,d) = Phi^T diag(rho) Phi
    eigs = np.linalg.eigvalsh(Sigma)
    rank = int(np.sum(eigs > 1e-10))
    print(
    f"[diag] Sigma under rho_E: "
    f"min_eig={eigs[0]:.3e}, max_eig={eigs[-1]:.3e}, "
    f"rank={rank}/{env.d}, K={env.K}, d={env.d}"
    )

    # True payoff matrix used in regret
    G_star = reg.payoff_matrix(env.Phi, env.Theta_star, env.mu)

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

    inc = reg.max_nash_regret_increments(pi1_seq, G_star, value_star=0.5)
    cum = reg.cumulative(inc)

    return {
        "seed": int(seed),
        "algo": algo_name,
        "t": np.arange(1, T + 1),
        "mnreg_inc": inc,
        "mnreg_cum": cum,
        "pi1_seq": pi1_seq,
    }


def save_per_seed(run_dir: Path, result: Dict[str, Any], meta: Dict[str, Any]) -> Path:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--tag", type=str, default="exp")

    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--T0", type=int, default=50)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--S", type=float, default=4.0)
    ap.add_argument("--instance_seed", type=int, default=0)

    ap.add_argument("--mu", type=str, choices=["logistic", "linear"], default="logistic")
    ap.add_argument("--algos", nargs="+", default=["etc_cvxpy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    # ETC/CVXPY options
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--frob_bound", type=float, default=None)
    ap.add_argument("--cvx_solver", type=str, default="SCS")
    ap.add_argument("--cvx_max_iters", type=int, default=10_000)
    ap.add_argument("--cvx_verbose", action="store_true")
    ap.add_argument("--commit_symmetric", action="store_true")

    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    T = int(args.T)
    T0 = int(args.T0)
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T.")

    run_dir = Path(args.out_dir) / f"{args.tag}_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        "tag": args.tag,
        "run_dir": str(run_dir),
        "created_at": now_stamp(),
        "T": T,
        "T0": T0,
        "K": int(args.K),
        "d": int(args.d),
        "r": int(args.r),
        "S": float(args.S),
        "instance_seed": int(args.instance_seed),
        "mu": args.mu,
        "algos": list(args.algos),
        "seeds": list(args.seeds),
        "lam": float(args.lam),
        "frob_bound": args.frob_bound,
        "cvx_solver": args.cvx_solver,
        "cvx_max_iters": int(args.cvx_max_iters),
        "cvx_verbose": bool(args.cvx_verbose),
        "commit_symmetric": bool(args.commit_symmetric),
        "regret_def": "MNReg(T)=sum_t (1/2 - min_j (pi1_t^T G_star)[j])",
    }

    # Manifest for later browsing
    manifest_path = run_dir / "manifest.jsonl"

    total_tasks = len(args.algos) * len(args.seeds)
    overall = TenPercentProgress(total_tasks, prefix="OVERALL ")

    # Accumulate for immediate plotting + summary saving
    all_cum: Dict[str, List[np.ndarray]] = {a: [] for a in args.algos}
    all_inc: Dict[str, List[np.ndarray]] = {a: [] for a in args.algos}

    done = 0
    for algo in args.algos:
        for seed in args.seeds:
            t0 = time.perf_counter()
            print(f"\n== Running algo='{algo}' seed={seed} ==")

            result = run_one_algo_one_seed(
                algo, seed,
                T=T, T0=T0,
                K=int(args.K), d=int(args.d), r=int(args.r), S=float(args.S),
                instance_seed=int(args.instance_seed),
                mu_name=args.mu,
                lam=float(args.lam),
                frob_bound=args.frob_bound,
                cvx_solver=args.cvx_solver,
                cvx_max_iters=int(args.cvx_max_iters),
                cvx_verbose=bool(args.cvx_verbose),
                commit_symmetric=bool(args.commit_symmetric),
            )

            saved_path = save_per_seed(run_dir, result, meta_common)

            # Append to manifest
            rec = {
                "algo": algo,
                "seed": int(seed),
                "file": str(saved_path),
                "seconds": round(time.perf_counter() - t0, 4),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            all_inc[algo].append(result["mnreg_inc"])
            all_cum[algo].append(result["mnreg_cum"])

            done += 1
            overall.update(done)

    # Save a quick "summary.npz" for plotting (stacked arrays)
    t = np.arange(1, T + 1, dtype=int)
    arrays = {"t": t, "seeds": np.array(args.seeds, dtype=int), "algos": np.array(args.algos, dtype=str)}
    for algo in args.algos:
        arrays[f"mnreg_inc__{algo}"] = np.stack(all_inc[algo], axis=0)   # (n_seeds, T)
        arrays[f"mnreg_cum__{algo}"] = np.stack(all_cum[algo], axis=0)   # (n_seeds, T)
    summary_path = run_dir / "summary.npz"
    reg.save_npz(summary_path, arrays=arrays, meta=meta_common)
    print(f"\n[saved] summary: {summary_path}")
    print(f"[saved] manifest: {manifest_path}")

    # Immediate plot (mean cum regret across seeds) + save figure
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in args.algos:
        Y = np.stack(all_cum[algo], axis=0)
        mean = Y.mean(axis=0)
        if Y.shape[0] > 1:
            spread = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0])
        else:
            spread = np.zeros_like(mean)
        ax.plot(t, mean, label=algo)
        ax.fill_between(t, mean - spread, mean + spread, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    fig_path = run_dir / "mnreg_cum_mean.png"
    fig.savefig(fig_path, dpi=200)
    print(f"[saved] plot: {fig_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
