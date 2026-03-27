import numpy as np
from solvers import bilinear_solver_reg, compute_best_response, estimate_theta_ridge_cvxpy

def _get_phi(env, x, a):
    Phi = env.Phi
    if Phi.ndim == 2:
        return Phi[a]
    if Phi.ndim == 3:
        return Phi[x, a]
    raise ValueError(f"env.Phi must be 2D or 3D, got {Phi.shape}")

def omg_s2p_cvxpy(
    env,
    *,
    T: int,
    mu,                     # MUST be mu_linear for OMG theory to hold
    eta: float = 1.0,       # Regularization strength
    reg_type: str = 'reverse_kl', 
    lam: float = 1.0,       # Ridge regularization parameter
    alpha: float = 1.0,     # Optimistic Bonus scaling factor
    update_freq: int = 1,   
    episode_seed: int = 0,
    cvx_solver: str = "SCS"
):
    """
    Optimistic Matrix Game (Nayak et al., 2025) adapted for self-play GBPM.
    """
    env.reset_episode(T=T, episode_seed=episode_seed)

    phi1_list, phi2_list, r_list = [], [], []
    traj = []

    K, d = env.K, env.d
    
    # Initialize the inverse covariance matrix V^{-1} for the bonus calculation
    # Since features are dxd matrices flattened to d^2, V is d^2 x d^2
    V_inv = np.eye(d * d) / lam
    
    # Pre-calculate all pairwise flattened feature combinations for fast bonus math
    if env.Phi.ndim == 3:
        Phi_table = env.Phi.mean(axis=0)   
    else:
        Phi_table = env.Phi
        
    # X_all shape: (K*K, d^2)
    X_all = np.einsum('id,jd->ijd', Phi_table, Phi_table).reshape(K * K, d * d)

    Theta_hat = np.zeros((d, d))
    pi1_hat = np.ones(K) / K
    pi2_hat = np.ones(K) / K
    v_hat, G_hat = None, None

    for t in range(T):
        # 1. Take a step using the optimistic policies
        x, a1, a2, r, p = env.step(pi1_hat, pi2_hat)
        traj.append((x, a1, a2, r, p))
        
        phi1 = _get_phi(env, x, a1)
        phi2 = _get_phi(env, x, a2)
        
        phi1_list.append(phi1)
        phi2_list.append(phi2)
        r_list.append(float(r))

        # 2. Update V_inv efficiently using Sherman-Morrison
        x_t = np.outer(phi1, phi2).reshape(-1)
        denom = 1.0 + x_t.T @ V_inv @ x_t
        V_inv -= (V_inv @ np.outer(x_t, x_t) @ V_inv) / denom

        # 3. Update Theta_hat and Policies periodically
        if (t + 1) % update_freq == 0:
            Theta_hat = estimate_theta_ridge_cvxpy(
                phi1_list, phi2_list, r_list,
                lam=lam,
                solver=cvx_solver
            )

            # Step 5 of OMG: Find Base NE (pi_t)
            pi_t, _, v_hat, G_hat = bilinear_solver_reg(
                Phi_table, Theta_hat, mu=mu, eta=eta, reg_type=reg_type
            )
            
            # Center the base payoff (skew-symmetric space)
            G_base = G_hat - 0.5

            # Step 6: Compute the Optimistic Bonus Matrix B_t
            # B_t(i,j) = alpha * sqrt( X_ij^T * V_inv * X_ij )
            variance = np.sum((X_all @ V_inv) * X_all, axis=1).reshape(K, K)
            B_t = alpha * np.sqrt(np.clip(variance, 0, None))

            # Step 7: Compute Optimistic Best Responses
            # Player 1 maximizes Base Payoff + Bonus
            pi1_hat = compute_best_response(pi_t, G_base + B_t, eta, reg_type)
            
            # Player 2 minimizes Base Payoff (which is maximizing -Base Payoff) + Bonus
            pi2_hat = compute_best_response(pi_t, -G_base + B_t.T, eta, reg_type)

    return {
        "Theta_hat": Theta_hat,
        "pi1_hat": pi1_hat,
        "pi2_hat": pi2_hat,
        "pi_hat": pi1_hat, # Keeping alias for compatibility
        "v_hat": v_hat,
        "G_hat": G_hat,
        "traj": traj,
    }