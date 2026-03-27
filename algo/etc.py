import numpy as np
import cvxpy as cp

from solvers import bilinear_solver_unreg

def estimate_theta_cvxpy(
    phi1_list, 
    phi2_list,
    r_list,
    *,
    lam: float = 0.5,
    skew: bool = True,
    frob_bound: float | None = None,
    solver: str = "SCS",
    max_iters: int = 10_000,
    verbose: bool = False,
):
    """
    Convex nuclear-norm-regularized logistic MLE (solved once via CVXPY):
      minimize (1/T0) sum_t [log(1+exp(z_t)) - r_t z_t] + lam ||Theta||_*
      s.t. Theta + Theta^T = 0   (if skew)
           ||Theta||_F <= frob_bound (optional)

    z_t = <Theta, X_t>, X_t = phi1_t phi2_t^T.
    """
    T0 = len(r_list)
    if T0 == 0:
        raise ValueError("Need at least one sample to estimate Theta.")

    phi1_list = [np.asarray(v, dtype=float).reshape(-1) for v in phi1_list]
    phi2_list = [np.asarray(v, dtype=float).reshape(-1) for v in phi2_list]
    r = np.asarray(r_list, dtype=float).reshape(-1)

    d = phi1_list[0].shape[0]
    if any(v.shape[0] != d for v in phi1_list) or any(v.shape[0] != d for v in phi2_list):
        raise ValueError("All phi vectors must have the same dimension d.")

    X_list = [np.outer(phi1_list[t], phi2_list[t]) for t in range(T0)]

    Theta = cp.Variable((d, d))
    constraints = []
    if skew:
        constraints.append(Theta + Theta.T == 0)
    if frob_bound is not None:
        constraints.append(cp.norm(Theta, "fro") <= float(frob_bound))

    loss_terms = []
    for t in range(T0):
        z_t = cp.sum(cp.multiply(Theta, X_list[t]))   # <Theta, X_t>
        loss_terms.append(cp.logistic(z_t) - r[t] * z_t)
    loss = (1.0 / T0) * cp.sum(loss_terms)

    reg = float(lam) * cp.normNuc(Theta) if lam > 0 else 0.0
    prob = cp.Problem(cp.Minimize(loss + reg), constraints)
    prob.solve(solver=solver, max_iters=max_iters, verbose=verbose)

    if Theta.value is None:
        raise RuntimeError(f"CVXPY failed. status={prob.status}")

    return np.asarray(Theta.value, dtype=float)


def _get_phi(env, x, a):
    Phi = env.Phi
    if Phi.ndim == 2:
        return Phi[a]
    if Phi.ndim == 3:
        return Phi[x, a]
    raise ValueError(f"env.Phi must be 2D or 3D, got {Phi.shape}")


def etc_s2p_cvxpy(
    env,
    *,
    T: int,
    T0: int,
    rho,                    # exploration policy (K,) or callable -> (K,)
    mu,                     # <<< you pass in mu here (e.g. env.mu_logistic or your own)
    lam: float = 100.0,
    frob_bound: float | None = None,
    episode_seed: int = 0,
    cvx_solver: str = "SCS",
    cvx_max_iters: int = 10_000,
    cvx_verbose: bool = False,
    commit_symmetric: bool = True,  # if True commit to (pi_hat, pi_hat)
):
    """
    Explore-Then-Commit algorithm using:
      - env.step(...) for data collection
      - CVXPY for Theta_hat (eq 6-7)
      - your bilinear_solver_unreg(..., mu=mu) for Nash equilibrium (eq 5)

    Returns dict with Theta_hat, equilibrium policies, and trajectory.
    """
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T.")

    # Make sure mu can be used inside bilinear_solver_unreg even if scalar-only

    env.reset_episode(T=T, episode_seed=episode_seed)

    # ---- Exploration ----
    phi1_list, phi2_list, r_list = [], [], []
    traj = []

    for _ in range(T0):
        x, a1, a2, r, p = env.step(rho, rho)
        traj.append((x, a1, a2, r, p))
        phi1_list.append(_get_phi(env, x, a1))
        phi2_list.append(_get_phi(env, x, a2))
        r_list.append(float(r))

    # ---- Estimate Theta_hat ----
    Theta_hat = estimate_theta_cvxpy(
        phi1_list, phi2_list, r_list,
        lam=lam,
        skew=True,
        frob_bound=frob_bound,
        solver=cvx_solver,
        max_iters=cvx_max_iters,
        verbose=cvx_verbose,
    )

    # ---- Nash equilibrium (Eq. 5) ----
    if env.Phi.ndim == 3:
        Phi_table = env.Phi.mean(axis=0)   # (K,d) fallback for contextual
    else:
        Phi_table = env.Phi               # (K,d)

    # IMPORTANT: call YOUR solver signature (keyword-only mu)
    pi1_hat, pi2_hat, v_hat, G_hat = bilinear_solver_unreg(Phi_table, Theta_hat, mu=mu)

    # ---- Commit ----
    if commit_symmetric:
        pi_commit_1 = pi1_hat
        pi_commit_2 = pi1_hat
    else:
        pi_commit_1 = pi1_hat
        pi_commit_2 = pi2_hat

    for _ in range(T0, T):
        x, a1, a2, r, p = env.step(pi_commit_1, pi_commit_2)
        traj.append((x, a1, a2, r, p))

    return {
        "Theta_hat": Theta_hat,
        "pi1_hat": pi1_hat,
        "pi2_hat": pi2_hat,
        "pi_hat": pi1_hat,
        "v_hat": v_hat,
        "G_hat": G_hat,
        "traj": traj,
    }
