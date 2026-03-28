import numpy as np
import cvxpy as cp

# --- [CHANGE]: Import the regularized solver! ---
from solvers import bilinear_solver_reg

def estimate_theta_cvxpy(
    phi1_list, 
    phi2_list,
    r_list,
    *,
    lam: float = 0.5,
    skew: bool = True,
    frob_bound: float | None = None,
    link_type: str = "logistic",
    solver: str = "SCS",
    max_iters: int = 10_000,
    verbose: bool = False,
):
    """
    Convex nuclear-norm-regularized MLE (solved once via CVXPY)
    Supports both Logistic and Linear links.
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

    if link_type == "logistic":
        loss_terms = []
        for t in range(T0):
            z_t = cp.sum(cp.multiply(Theta, X_list[t]))   
            loss_terms.append(cp.logistic(z_t) - r[t] * z_t)
        loss = (1.0 / T0) * cp.sum(loss_terms)
        
    elif link_type == "linear":
        X_stack = np.array([x.flatten() for x in X_list])
        targets = 4.0 * (r - 0.5)
        z = X_stack @ cp.vec(Theta)
        loss = (1.0 / T0) * cp.sum_squares(z - targets)
        
    else:
        raise ValueError(f"Unknown link_type: {link_type}")

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
    rho,                    
    mu,                     
    link_type: str = "logistic", 
    eta: float = 1.0,               # --- [CHANGE]: Added eta parameter
    reg_type: str = 'reverse_kl',   # --- [CHANGE]: Added reg_type parameter
    lam: float = 100.0,
    frob_bound: float | None = None,
    episode_seed: int = 0,
    cvx_solver: str = "SCS",
    cvx_max_iters: int = 10_000,
    cvx_verbose: bool = False,
    commit_symmetric: bool = True,  
):
    """
    Explore-Then-Commit algorithm.
    """
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T.")

    env.reset_episode(T=T, episode_seed=episode_seed)

    phi1_list, phi2_list, r_list = [], [], []
    traj = []
    
    pi1_seq = []
    pi2_seq = []
    
    if callable(rho):
        rho_arr = np.asarray(rho(0), dtype=float) 
    else:
        rho_arr = np.asarray(rho, dtype=float)

    # ---- Exploration Phase ----
    for _ in range(T0):
        pi1_seq.append(rho_arr.copy())
        pi2_seq.append(rho_arr.copy())
        
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
        link_type=link_type,  
        solver=cvx_solver,
        max_iters=cvx_max_iters,
        verbose=cvx_verbose,
    )

    # ---- Nash equilibrium (Eq. 5) ----
    if env.Phi.ndim == 3:
        Phi_table = env.Phi.mean(axis=0)   
    else:
        Phi_table = env.Phi               

    # --- [CHANGE]: Solve the Regularized game using the provided eta and reg_type ---
    pi1_hat, pi2_hat, v_hat, G_hat = bilinear_solver_reg(
        Phi_table, Theta_hat, mu=mu, eta=eta, reg_type=reg_type
    )

    # ---- Commit Phase ----
    if commit_symmetric:
        pi_commit_1 = pi1_hat
        pi_commit_2 = pi1_hat
    else:
        pi_commit_1 = pi1_hat
        pi_commit_2 = pi2_hat

    for _ in range(T0, T):
        pi1_seq.append(pi_commit_1.copy())
        pi2_seq.append(pi_commit_2.copy())
        
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
        "pi1_seq": pi1_seq,
        "pi2_seq": pi2_seq,
    }