import numpy as np
import cvxpy as cp

# We assume you will create a regularized solver in solvers.py that accepts reg_type
from solvers import bilinear_solver_reg 

def estimate_theta_gs_cvxpy(
    phi1_list, 
    phi2_list,
    r_list,
    *,
    S: float,
    solver: str = "SCS",
    max_iters: int = 10_000,
    verbose: bool = False,
):
    """
    Constrained MLE for Greedy Sampling (no nuclear norm penalty):
      minimize sum_t [log(1+exp(z_t)) - r_t z_t]
      s.t. Theta + Theta^T = 0
           ||Theta||_F <= S
           
    z_t = <Theta, X_t>, X_t = phi1_t phi2_t^T.
    """
    T_current = len(r_list)
    if T_current == 0:
        raise ValueError("Need at least one sample to estimate Theta.")

    phi1_list = [np.asarray(v, dtype=float).reshape(-1) for v in phi1_list]
    phi2_list = [np.asarray(v, dtype=float).reshape(-1) for v in phi2_list]
    r = np.asarray(r_list, dtype=float).reshape(-1)

    d = phi1_list[0].shape[0]

    X_list = [np.outer(phi1_list[t], phi2_list[t]) for t in range(T_current)]

    Theta = cp.Variable((d, d))
    
    # GS Constraints: Skew-symmetry AND Frobenius norm bound S
    constraints = [
        Theta + Theta.T == 0,
        cp.norm(Theta, "fro") <= float(S)
    ]

    loss_terms = []
    for t in range(T_current):
        z_t = cp.sum(cp.multiply(Theta, X_list[t]))   # <Theta, X_t>
        loss_terms.append(cp.logistic(z_t) - r[t] * z_t)
        
    # Standard MLE (no regularization penalty in the objective for GS)
    loss = cp.sum(loss_terms) 

    prob = cp.Problem(cp.Minimize(loss), constraints)
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


def gs_s2p_cvxpy(
    env,
    *,
    T: int,
    rho,                    # exploration policy (K,) or callable
    mu,                     # link function (e.g., env.mu_logistic)
    eta: float = 1.0,       # Regularization strength
    reg_type: str = 'reverse_kl', # Which regularizer to use for NE ('reverse_kl', 'chi_squared', 'tsallis')
    S: float = 4.0,         # Frobenius norm bound
    update_freq: int = 1,   # How often to run CVXPY (set > 1 to speed up simulations)
    episode_seed: int = 0,
    cvx_solver: str = "SCS",
    cvx_max_iters: int = 10_000,
    cvx_verbose: bool = False,
):
    """
    Greedy Sampling algorithm.
    - Player 1 plays the greedy regularized NE policy w.r.t Theta_hat.
    - Player 2 explores using rho.
    - Updates Theta_hat iteratively.
    """
    env.reset_episode(T=T, episode_seed=episode_seed)

    phi1_list, phi2_list, r_list = [], [], []
    traj = []

    # Initialize pi_hat_1 as the exploration policy rho (Algorithm line 2)
    if callable(rho):
        pi1_hat = np.asarray(rho(0), dtype=float) 
    else:
        pi1_hat = np.asarray(rho, dtype=float)
    
    Theta_hat = np.zeros((env.d, env.d))
    v_hat, G_hat = None, None

    for t in range(T):
        # 1. Take a step: Player 1 uses pi1_hat, Player 2 uses rho
        x, a1, a2, r, p = env.step(pi1_hat, rho)
        traj.append((x, a1, a2, r, p))
        
        # 2. Record the history
        phi1_list.append(_get_phi(env, x, a1))
        phi2_list.append(_get_phi(env, x, a2))
        r_list.append(float(r))

        # 3. Update Theta_hat and NE policy periodically (Algorithm lines 6-7)
        # (We use update_freq so you don't have to wait hours for T=1000 if CVXPY is slow)
        if (t + 1) % update_freq == 0:
            Theta_hat = estimate_theta_gs_cvxpy(
                phi1_list, phi2_list, r_list,
                S=S,
                solver=cvx_solver,
                max_iters=cvx_max_iters,
                verbose=cvx_verbose,
            )

            if env.Phi.ndim == 3:
                Phi_table = env.Phi.mean(axis=0)   
            else:
                Phi_table = env.Phi               

            # Compute the Regularized Nash Equilibrium
            # IMPORTANT: We pass eta and reg_type here so the solver handles the math!
            pi1_hat, pi2_hat, v_hat, G_hat = bilinear_solver_reg(
                Phi_table, 
                Theta_hat, 
                mu=mu, 
                eta=eta, 
                reg_type=reg_type
            )

    return {
        "Theta_hat": Theta_hat,
        "pi1_hat": pi1_hat,  # Final player 1 policy
        "pi2_hat": pi2_hat,  # Final player 2 policy
        "pi_hat": pi1_hat,   # Alias for consistency
        "v_hat": v_hat,
        "G_hat": G_hat,
        "traj": traj,
    }