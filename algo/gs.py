import numpy as np
import cvxpy as cp

from solvers import (
    bilinear_solver_reg,
    fixed_point_residual,
    init_theta_logistic_onepass_ons,
    update_theta_logistic_onepass_ons,
)

def estimate_theta_gs_cvxpy(
    phi1_list, 
    phi2_list,
    r_list,
    *,
    S: float,
    link_type: str = "logistic", 
    solver: str = "SCS",
    max_iters: int = 10_000,
    verbose: bool = False,
):
    """
    Constrained CVXPY estimator for the linear link only.
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
    constraints = [
        Theta + Theta.T == 0,
        cp.norm(Theta, "fro") <= float(S)
    ]

    if link_type == "linear":
        # Linear link requires Least Squares Loss. 
        # Target inversion: r_t = 0.5 + 0.125*z => z = 8(r_t - 0.5)
        X_stack = np.array([x.flatten(order='F') for x in X_list])
        targets = 8.0 * (r - 0.5)
        z = X_stack @ cp.vec(Theta, order='F')
        loss = cp.sum_squares(z - targets)
        
    else:
        raise ValueError(f"Unknown link_type: {link_type}")

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
    rho,                    
    mu,                     
    link_type: str = "logistic", 
    eta: float = 1.0,       
    reg_type: str = 'reverse_kl', 
    S: float = 4.0,         
    update_freq: int = 1,   
    episode_seed: int = 0,
    cvx_solver: str = "SCS",
    cvx_max_iters: int = 10_000,
    cvx_verbose: bool = False,
    theta_estimator: str = "onepass_ons",
    ons_a0: float = 1.0,
    ons_step_size: float = 1.0,
    ons_hess_floor: float = 1e-6,
    ons_hess_cap: float = 0.25,
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
    
    # Initialize the list to store expected policies
    pi1_seq = []
    theta_frob_err_seq = []
    ne_residual_seq = []
    last_ne_residual = float("nan")

    if callable(rho):
        pi1_hat = np.asarray(rho(0), dtype=float) 
    else:
        pi1_hat = np.asarray(rho, dtype=float)
    
    Theta_hat = np.zeros((env.d, env.d))
    pi2_hat = pi1_hat.copy()
    v_hat, G_hat = None, None

    theta_star = getattr(env, "Theta_star", None)
    if link_type == "logistic" and theta_estimator != "onepass_ons":
        raise ValueError(
            f"For logistic link, only theta_estimator='onepass_ons' is supported. Got '{theta_estimator}'."
        )
    if link_type == "linear" and theta_estimator != "cvxpy":
        raise ValueError(
            f"For linear link, use theta_estimator='cvxpy'. Got '{theta_estimator}'."
        )

    ons_state = None
    if link_type == "logistic" and theta_estimator == "onepass_ons":
        ons_state = init_theta_logistic_onepass_ons(
            env.d,
            a0=ons_a0,
            step_size=ons_step_size,
            frob_bound=S,
            hess_floor=ons_hess_floor,
            hess_cap=ons_hess_cap,
            warm_start=Theta_hat,
        )

    for t in range(T):
        # Save the expected policy BEFORE taking the step
        pi1_seq.append(pi1_hat.copy())
        
        # 1. Take a step
        x, a1, a2, r, p = env.step(pi1_hat, rho)
        traj.append((x, a1, a2, r, p))
        
        # 2. Record the history
        phi1_list.append(_get_phi(env, x, a1))
        phi2_list.append(_get_phi(env, x, a2))
        r_list.append(float(r))

        # One-pass ONS updates parameter online each round.
        if ons_state is not None:
            Theta_hat, _ = update_theta_logistic_onepass_ons(
                ons_state, phi1_list[-1], phi2_list[-1], r_list[-1]
            )

        # 3. Update Theta_hat
        if (t + 1) % update_freq == 0:
            if link_type == "logistic":
                # Theta_hat already updated online above.
                pass
            else:
                Theta_hat = estimate_theta_gs_cvxpy(
                    phi1_list, phi2_list, r_list,
                    S=S,
                    link_type=link_type,  # Wired this up to the solver
                    solver=cvx_solver,
                    max_iters=cvx_max_iters,
                    verbose=cvx_verbose,
                )

            if env.Phi.ndim == 3:
                Phi_table = env.Phi.mean(axis=0)   
            else:
                Phi_table = env.Phi               

            pi1_hat, pi2_hat, v_hat, G_hat = bilinear_solver_reg(
                Phi_table, 
                Theta_hat, 
                mu=mu, 
                eta=eta, 
                reg_type=reg_type,
                ref_policy = rho
            )
            last_ne_residual = fixed_point_residual(
                pi1_hat,
                G_hat,
                eta=eta,
                reg_type=reg_type,
                ref_policy=rho,
            )
            if not np.isfinite(last_ne_residual) or last_ne_residual > 1e-4:
                print(
                    f"[warn] large NE residual at t={t+1}: "
                    f"resid={last_ne_residual:.3e}, eta={eta:g}, reg={reg_type}"
                )

        if theta_star is not None:
            theta_frob_err_seq.append(float(np.linalg.norm(Theta_hat - theta_star, ord="fro")))
        else:
            theta_frob_err_seq.append(float("nan"))
        ne_residual_seq.append(float(last_ne_residual))

    return {
        "Theta_hat": Theta_hat,
        "pi1_hat": pi1_hat,  
        "pi2_hat": pi2_hat,  
        "pi_hat": pi1_hat,   
        "v_hat": v_hat,
        "G_hat": G_hat,
        "traj": traj,
        "pi1_seq": pi1_seq,  # Returned the sequence for regret evaluation!
        "theta_frob_err_seq": np.asarray(theta_frob_err_seq, dtype=float),
        "ne_residual_seq": np.asarray(ne_residual_seq, dtype=float),
    }