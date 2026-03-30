import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.optimize import root
from env import mu_logistic, mu_linear

#def bilinear_solver_KLreg(Phi: np.ndarray, Theta: np.ndarray, *, mu=mu_logistic):

def compute_rho_E(Phi: np.ndarray, solver: str = "SCS", max_iters: int = 50_000, verbose: bool = False):
    """
    Compute rho_E = argmax_{rho in simplex} lambda_min( sum_a rho[a] * phi_a phi_a^T )
    where Phi has shape (K, d) and phi_a = Phi[a, :].

    Returns:
        rho (K,) numpy array on simplex
        t_opt: optimal value of the min eigenvalue (lambda_min)
        status: cvxpy status
    """
    Phi = np.asarray(Phi, dtype=float)
    K, d = Phi.shape

    rho = cp.Variable(K, nonneg=True)
    t = cp.Variable()

    # Sigma(rho) = Phi^T diag(rho) Phi
    Sigma = Phi.T @ cp.diag(rho) @ Phi
    # Numerical symmetry guard
    Sigma = 0.5 * (Sigma + Sigma.T)

    constraints = [
        cp.sum(rho) == 1,
        t >= 0,                  # optional but usually sensible
        Sigma - t * np.eye(d) >> 0
    ]

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=solver, verbose=verbose, max_iters=max_iters)

    if rho.value is None:
        raise RuntimeError(f"SDP failed: status={prob.status}")

    rho_val = np.maximum(rho.value, 0.0)
    s = rho_val.sum()
    if s <= 0:
        # should not happen, but just in case
        rho_val = np.ones(K) / K
    else:
        rho_val /= s

    return rho_val, float(t.value), prob.status


# Example usage:
# rho_E, t_opt, status = compute_rho_E(env.Phi, solver="SCS")
# print("status:", status, "t_opt:", t_opt)
# print("rho_E:", rho_E, "sum:", rho_E.sum())


def project_simplex(v):
    """
    Computes the Euclidean projection of a vector v onto the probability simplex.
    Fast exact algorithm (Condat 2016).
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def compute_best_response(q: np.ndarray, G: np.ndarray, eta: float, reg_type: str, ref_policy=None) -> np.ndarray:
    """
    Compute the regularized best response against opponent policy q.
    """
    K = G.shape[0]
    
    # Safely handle the reference policy
    if ref_policy is None:
        ref_policy = np.ones(K) / K
    else:
        ref_policy = np.asarray(ref_policy, dtype=float)

    # Calculate the expected rewards for each action against q
    expected_rewards = G @ q

    if reg_type == 'reverse_kl':
        # pi_i \propto \rho_i * exp(\eta * expected_rewards_i)
        logits = eta * expected_rewards
        logits -= np.max(logits) 
        pi = ref_policy * np.exp(logits)
        pi_sum = np.sum(pi)
        if pi_sum > 0:
            pi /= pi_sum
        else:
            pi = np.ones(K) / K
        return pi
        
    elif reg_type == 'shannon':
        # Standard Softmax (Implicitly uniform reference)
        logits = eta * expected_rewards
        logits -= np.max(logits)
        pi = np.exp(logits)
        pi /= np.sum(pi)
        return pi

    elif reg_type == 'tsallis':
        # Tsallis (q=2) is a Euclidean projection
        # pi = Proj_simplex(rho + eta * expected_rewards)
        target_vec = ref_policy + eta * expected_rewards
        return project_simplex(target_vec)

    elif reg_type == 'chi_squared':
        # Chi-Squared projection requires finding lambda via bisection
        # pi_i = max(0, rho_i * (1 + eta * expected_rewards_i - lambda))
        v = eta * expected_rewards
        
        # Mask out near-zero rho to prevent division by zero in bounds
        valid_rho = ref_policy[ref_policy > 1e-12]
        if len(valid_rho) == 0:
            return np.ones(K) / K
            
        # Initialize bisection bounds for lambda
        low = np.min(v) - 1.0 / np.min(valid_rho)
        high = np.max(v) + 1.0
        
        # 50 iterations of bisection guarantees machine-level float64 precision
        for _ in range(50):
            mid = (low + high) / 2.0
            pi = np.maximum(0.0, ref_policy * (1.0 + v - mid))
            if np.sum(pi) > 1.0:
                low = mid
            else:
                high = mid
                
        # Final calculation with the converged lambda (high)
        pi = np.maximum(0.0, ref_policy * (1.0 + v - high))
        
        # Clean up floating point errors to guarantee exact sum to 1.0
        pi_sum = np.sum(pi)
        if pi_sum > 0:
            pi /= pi_sum
        else:
            pi = np.ones(K) / K
            
        return pi
        
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")

def _solve_br_fixed_point_symmetric(
    G_tilde: np.ndarray,
    *,
    eta: float,
    reg_type: str,
    ref_policy: np.ndarray,
    tol: float = 1e-8,
    max_iters_per_stage: int = 6000,
    verbose: bool = False,
) -> tuple[np.ndarray, float, bool]:
    """
    Robustly solve p = BR_eta(p) for symmetric regularized game.

    Uses eta-continuation + adaptive damping to avoid high-eta stalling.
    """
    K = G_tilde.shape[0]
    p = np.asarray(ref_policy, dtype=float).reshape(K)
    p = np.clip(p, 0.0, None)
    p = p / p.sum() if p.sum() > 0 else (np.ones(K) / K)

    eta = float(max(1e-12, eta))
    if eta <= 10.0:
        eta_stages = [eta]
    else:
        n = int(np.ceil(np.log2(eta / 10.0))) + 2
        eta_stages = [10.0] + list(np.geomspace(10.0, eta, n))
        eta_stages = [float(x) for x in eta_stages]

    final_res = np.inf
    converged = False
    for stage_idx, eta_stage in enumerate(eta_stages):
        stage_tol = tol if stage_idx == (len(eta_stages) - 1) else max(1e-6, 10.0 * tol)
        alpha = 1.0

        def br_fn(v: np.ndarray) -> np.ndarray:
            return compute_best_response(v, G_tilde, eta_stage, reg_type, ref_policy=ref_policy)

        for _ in range(int(max_iters_per_stage)):
            br = br_fn(p)
            res = float(np.max(np.abs(br - p)))
            if res <= stage_tol:
                final_res = res
                converged = (stage_idx == len(eta_stages) - 1)
                break

            accepted = False
            alpha_try = alpha
            while alpha_try >= 1e-6:
                cand = (1.0 - alpha_try) * p + alpha_try * br
                cand = np.clip(cand, 0.0, None)
                cand = cand / cand.sum() if cand.sum() > 0 else (np.ones(K) / K)
                br_cand = br_fn(cand)
                res_cand = float(np.max(np.abs(br_cand - cand)))
                # Accept if residual decreases sufficiently.
                if res_cand <= (1.0 - 1e-3 * alpha_try) * res:
                    p = cand
                    final_res = res_cand
                    alpha = min(1.0, alpha_try * 1.25)
                    accepted = True
                    break
                alpha_try *= 0.5

            if not accepted:
                # Fallback to conservative Mann step.
                p = 0.5 * p + 0.5 * br
                p = np.clip(p, 0.0, None)
                p = p / p.sum() if p.sum() > 0 else (np.ones(K) / K)
                alpha = 0.5
                final_res = float(np.max(np.abs(br_fn(p) - p)))

        if verbose:
            print(f"[bilinear_solver_reg] stage eta={eta_stage:.3g}, resid={final_res:.3e}")

    return p, float(final_res), bool(converged and final_res <= tol)


def _solve_reverse_kl_symmetric_root(
    G_tilde: np.ndarray,
    *,
    eta: float,
    ref_policy: np.ndarray,
    tol: float = 1e-10,
    maxfev: int = 5000,
) -> tuple[np.ndarray, float, bool]:
    """
    Solve symmetric reverse-KL fixed point exactly in log-ratio coordinates.

    Fixed point p = BR(p) is equivalent to:
      u_i = eta * ((G p)_i - (G p)_K),  i=1..K-1
    with p_i proportional to rho_i * exp(u_i), and u_K = 0.
    """
    K = G_tilde.shape[0]
    if K == 1:
        return np.array([1.0]), 0.0, True

    rho = np.asarray(ref_policy, dtype=float).reshape(K)
    rho = np.clip(rho, 1e-15, None)
    rho = rho / rho.sum()

    log_rho = np.log(rho)

    def p_from_u(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(K - 1)
        u_full = np.concatenate([u, np.array([0.0])])
        logits = log_rho + u_full
        logits -= np.max(logits)
        w = np.exp(logits)
        return w / w.sum()

    def F(u: np.ndarray) -> np.ndarray:
        p = p_from_u(u)
        g = G_tilde @ p
        return u - eta * (g[:-1] - g[-1])

    sol = root(F, np.zeros(K - 1), method="hybr", options={"xtol": float(tol), "maxfev": int(maxfev)})
    p = p_from_u(sol.x)
    br = compute_best_response(p, G_tilde, eta, "reverse_kl", ref_policy=rho)
    resid = float(np.max(np.abs(br - p)))
    success = bool(sol.success and np.isfinite(resid))
    return p, resid, success


def bilinear_solver_reg(
    Phi: np.ndarray,
    Theta: np.ndarray,
    *,
    mu=mu_logistic,
    eta=1.0,
    reg_type='reverse_kl',
    ref_policy=None,
    tol: float = 1e-8,
    max_iters: int = 6000,
    verbose: bool = False,
):
    """
    Solve the REGULARIZED symmetric zero-sum game using Damped Best Response iteration.
    """
    Phi = np.asarray(Phi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    K = Phi.shape[0]

    # Handle the reference policy
    if ref_policy is None:
        ref_policy = np.ones(K) / K
    else:
        ref_policy = np.asarray(ref_policy, dtype=float)

    # Build payoff matrix G
    Z = Phi @ Theta @ Phi.T
    G = mu(Z)
    
    # Center the matrix to make it strictly skew-symmetric (for symmetric game properties)
    G_tilde = G - 0.5 

    # Robust solvers for symmetric fixed-point p = BR_eta(p).
    if reg_type in {"reverse_kl", "shannon"}:
        rho_eff = ref_policy if reg_type == "reverse_kl" else (np.ones(K) / K)
        p, fp_resid, converged = _solve_reverse_kl_symmetric_root(
            G_tilde,
            eta=float(eta),
            ref_policy=rho_eff,
            tol=min(float(tol), 1e-10),
            maxfev=max(2000, int(max_iters)),
        )
        # Safety fallback if root solver fails.
        if (not converged) or (fp_resid > 10.0 * float(tol)):
            p, fp_resid, converged = _solve_br_fixed_point_symmetric(
                G_tilde,
                eta=float(eta),
                reg_type=reg_type,
                ref_policy=ref_policy,
                tol=float(tol),
                max_iters_per_stage=int(max_iters),
                verbose=bool(verbose),
            )
    else:
        p, fp_resid, converged = _solve_br_fixed_point_symmetric(
            G_tilde,
            eta=float(eta),
            reg_type=reg_type,
            ref_policy=ref_policy,
            tol=float(tol),
            max_iters_per_stage=int(max_iters),
            verbose=bool(verbose),
        )

    if verbose:
        msg = "converged" if converged else "not_converged"
        print(f"[bilinear_solver_reg] {msg}, fixed_point_resid={fp_resid:.3e}, eta={eta}, reg={reg_type}")
            
    # Clean up final policy to ensure absolute numerical strictness
    pi_star = np.clip(p, 0.0, None)
    pi_sum = pi_star.sum()
    if pi_sum > 0:
        pi_star /= pi_sum
    else:
        pi_star = np.ones(K) / K

    # The game is symmetric, so pi1 = pi2 = pi_star. Value is strictly 0.5.
    return pi_star, pi_star, 0.5, G


def fixed_point_residual(
    p: np.ndarray,
    G: np.ndarray,
    *,
    eta: float,
    reg_type: str,
    ref_policy: np.ndarray | None = None,
) -> float:
    """
    Compute ||BR(p) - p||_inf for the centered symmetric game.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    G = np.asarray(G, dtype=float)
    br = compute_best_response(p, G - 0.5, float(eta), reg_type, ref_policy=ref_policy)
    return float(np.max(np.abs(br - p)))


def _project_skew_frob(theta: np.ndarray, frob_bound: float | None = None) -> np.ndarray:
    """Project to skew-symmetric matrices and optional Frobenius ball."""
    theta = 0.5 * (theta - theta.T)
    if frob_bound is not None:
        bound = float(frob_bound)
        if bound <= 0:
            return np.zeros_like(theta)
        nrm = float(np.linalg.norm(theta, ord="fro"))
        if nrm > bound:
            theta = theta * (bound / (nrm + 1e-12))
    return theta


def estimate_theta_logistic_projected(
    phi1_list,
    phi2_list,
    r_list,
    *,
    l2: float = 0.0,
    frob_bound: float | None = None,
    max_iters: int = 2000,
    tol: float = 1e-7,
    step_init: float = 1.0,
    line_search_beta: float = 0.5,
    line_search_c: float = 1e-4,
    min_step: float = 1e-10,
    warm_start: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Efficient projected-gradient solver for logistic MLE with skew constraint.

    Objective:
      (1/T) * sum_t [log(1+exp(z_t)) - r_t z_t] + 0.5*l2*||Theta||_F^2
      where z_t = phi1_t^T Theta phi2_t
    subject to:
      Theta + Theta^T = 0, and optionally ||Theta||_F <= frob_bound.
    """
    T = len(r_list)
    if T == 0:
        raise ValueError("Need at least one sample to estimate Theta.")

    phi1 = np.asarray(phi1_list, dtype=float)
    phi2 = np.asarray(phi2_list, dtype=float)
    r = np.asarray(r_list, dtype=float).reshape(-1)
    if phi1.ndim != 2 or phi2.ndim != 2:
        raise ValueError("phi1_list and phi2_list must be arrays of vectors.")
    if phi1.shape != phi2.shape:
        raise ValueError(f"phi1/phi2 shape mismatch: {phi1.shape} vs {phi2.shape}")
    if phi1.shape[0] != T:
        raise ValueError("Number of feature rows must match len(r_list).")

    d = phi1.shape[1]
    if warm_start is None:
        theta = np.zeros((d, d), dtype=float)
    else:
        theta = np.asarray(warm_start, dtype=float).reshape(d, d)
    theta = _project_skew_frob(theta, frob_bound)

    l2 = float(max(0.0, l2))
    T_inv = 1.0 / float(T)

    def obj_grad(theta_curr: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.einsum("ti,ij,tj->t", phi1, theta_curr, phi2, optimize=True)
        z_clip = np.clip(z, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-z_clip))
        loss = float(np.mean(np.log1p(np.exp(z_clip)) - r * z_clip))
        if l2 > 0:
            loss += 0.5 * l2 * float(np.sum(theta_curr * theta_curr))

        w = (sig - r) * T_inv
        grad = phi1.T @ (w[:, None] * phi2)
        if l2 > 0:
            grad = grad + l2 * theta_curr
        grad = 0.5 * (grad - grad.T)
        return loss, grad

    f_curr, g_curr = obj_grad(theta)
    for it in range(int(max_iters)):
        g_norm = float(np.linalg.norm(g_curr, ord="fro"))
        if g_norm <= tol:
            break

        step = float(step_init)
        accepted = False
        while step >= min_step:
            theta_try = _project_skew_frob(theta - step * g_curr, frob_bound)
            f_try, _ = obj_grad(theta_try)
            if f_try <= f_curr - line_search_c * step * (g_norm ** 2):
                theta = theta_try
                f_curr = f_try
                accepted = True
                break
            step *= line_search_beta

        if not accepted:
            break

        _, g_curr = obj_grad(theta)
        if verbose and (it + 1) % 100 == 0:
            print(f"[logistic-pg] iter={it+1}, obj={f_curr:.6e}, grad_fro={g_norm:.3e}, step={step:.2e}")

    return theta


def init_theta_logistic_onepass_ons(
    d: int,
    *,
    a0: float = 1.0,
    step_size: float = 1.0,
    frob_bound: float | None = None,
    hess_floor: float = 1e-6,
    hess_cap: float = 0.25,
    warm_start: np.ndarray | None = None,
) -> dict:
    """
    Initialize one-pass ONS state for logistic bilinear model in vec(theta) space.
    """
    d = int(d)
    if d <= 0:
        raise ValueError(f"d must be positive. Got d={d}")
    a0 = float(a0)
    if a0 <= 0:
        raise ValueError(f"a0 must be positive. Got a0={a0}")

    dim = d * d
    if warm_start is None:
        theta_mat = np.zeros((d, d), dtype=float)
    else:
        theta_mat = np.asarray(warm_start, dtype=float).reshape(d, d)
    theta_mat = _project_skew_frob(theta_mat, frob_bound)
    theta_vec = theta_mat.reshape(-1, order="F")

    return {
        "d": d,
        "theta_vec": theta_vec,
        "A_inv": np.eye(dim, dtype=float) / a0,
        "step_size": float(step_size),
        "frob_bound": frob_bound,
        "hess_floor": float(max(0.0, hess_floor)),
        "hess_cap": float(max(0.0, hess_cap)),
        "t": 0,
    }


def update_theta_logistic_onepass_ons(
    state: dict,
    phi1_t: np.ndarray,
    phi2_t: np.ndarray,
    r_t: float,
) -> tuple[np.ndarray, dict]:
    """
    One-pass ONS update:
      A_t = A_{t-1} + h_t x_t x_t^T,  h_t ~= sigma(z_t)(1-sigma(z_t))
      theta_t = Proj(theta_{t-1} - eta * A_t^{-1} g_t)
    where g_t = (sigma(z_t) - r_t) x_t.
    """
    d = int(state["d"])
    theta_vec = np.asarray(state["theta_vec"], dtype=float).reshape(-1)
    A_inv = np.asarray(state["A_inv"], dtype=float)

    phi1_t = np.asarray(phi1_t, dtype=float).reshape(d)
    phi2_t = np.asarray(phi2_t, dtype=float).reshape(d)
    r_t = float(r_t)

    x_t = np.outer(phi1_t, phi2_t).reshape(-1, order="F")
    z_t = float(theta_vec @ x_t)
    z_clip = float(np.clip(z_t, -50.0, 50.0))
    p_t = 1.0 / (1.0 + np.exp(-z_clip))
    grad = (p_t - r_t) * x_t

    h_floor = float(state.get("hess_floor", 1e-6))
    h_cap = float(state.get("hess_cap", 0.25))
    h_t = float(np.clip(p_t * (1.0 - p_t), h_floor, h_cap))
    u = np.sqrt(h_t) * x_t

    # Sherman-Morrison update for A_inv with rank-1 outer product u u^T.
    Au = A_inv @ u
    denom = 1.0 + float(u @ Au)
    if denom > 1e-12:
        A_inv = A_inv - np.outer(Au, Au) / denom

    eta = float(state.get("step_size", 1.0))
    theta_vec = theta_vec - eta * (A_inv @ grad)

    theta_mat = theta_vec.reshape((d, d), order="F")
    theta_mat = _project_skew_frob(theta_mat, state.get("frob_bound", None))
    theta_vec = theta_mat.reshape(-1, order="F")

    state["theta_vec"] = theta_vec
    state["A_inv"] = A_inv
    state["t"] = int(state.get("t", 0)) + 1

    stats = {
        "p_t": float(p_t),
        "z_t": float(z_t),
        "h_t": float(h_t),
        "grad_norm": float(np.linalg.norm(grad)),
    }
    return theta_mat, stats

def estimate_theta_ridge_cvxpy(
    phi1_list,
    phi2_list,
    r_list,
    *,
    lam: float = 1.0,
    solver: str = "SCS"
):
    """
    Skew-Symmetric Ridge Regression for the Linear Link function.
    Because mu(z) = 0.5 + 0.125*z, E[r_t] = 0.5 + 0.125*<Theta, X_t>.
    Therefore, the regression target for <Theta, X_t> is 8 * (r_t - 0.5).
    """
    T_curr = len(r_list)
    if T_curr == 0:
        raise ValueError("Need at least one sample.")

    d = phi1_list[0].shape[0]
    Theta = cp.Variable((d, d))

    # Skew-symmetry constraint
    constraints = [Theta + Theta.T == 0]

    # Vectorize inputs for CVXPY speed
    X_stack = np.array([np.outer(phi1_list[t], phi2_list[t]).flatten('F') for t in range(T_curr)])
    targets = 8.0 * (np.array(r_list) - 0.5)

    # Loss: || X * vec(Theta) - targets ||^2 + lam * ||Theta||_F^2
    z = X_stack @ cp.vec(Theta)
    loss = cp.sum_squares(z - targets) + lam * cp.sum_squares(Theta)

    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(solver=solver)

    if Theta.value is None:
        raise RuntimeError(f"CVXPY Ridge failed. status={prob.status}")

    return np.asarray(Theta.value, dtype=float)


def bilinear_solver_unreg(Phi: np.ndarray, Theta: np.ndarray, *, mu=mu_logistic):
    """
    Solve the zero-sum game:
        max_{pi1 in simplex} min_{pi2 in simplex}  E_{i~pi1, j~pi2} mu(phi_i^T Theta phi_j)

    Inputs:
      Phi:   (K, d) array, Phi[i] = phi_i feature vector for arm i
      Theta: (d, d) array

    Output:
      pi1: (K,) equilibrium distribution for max-player
      pi2: (K,) equilibrium distribution for min-player
      v:   game value (scalar)
      G:   (K, K) payoff matrix with G[i,j] = mu(phi_i^T Theta phi_j)
    """
    Phi = np.asarray(Phi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)

    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2D (K,d). Got shape {Phi.shape}")
    K, d = Phi.shape
    if Theta.shape != (d, d):
        raise ValueError(f"Theta must have shape (d,d)=({d},{d}). Got {Theta.shape}")

    # Build payoff matrix G_{ij} = mu(phi_i^T Theta phi_j)
    Z = Phi @ Theta @ Phi.T          # (K,K), where Z[i,j] = phi_i^T Theta phi_j
    G = mu(Z)                        # (K,K) in (0,1)

    # ---- LP for max-player: maximize v s.t. G^T pi1 >= v*1, sum pi1=1, pi1>=0 ----
    # Variables: x = [pi1 (K entries), v]
    c = np.zeros(K + 1)
    c[-1] = -1.0                     # minimize -v == maximize v

    A_ub = np.hstack([-G.T, np.ones((K, 1))])  # -G^T pi1 + v <= 0
    b_ub = np.zeros(K)

    A_eq = np.zeros((1, K + 1))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Payoffs in [0,1] -> value v in [0,1] is safe
    bounds = [(0.0, 1.0)] * K + [(0.0, 1.0)]

    res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res1.success:
        raise RuntimeError(f"linprog (max-player) failed: {res1.message}")

    pi1 = res1.x[:K]
    v1 = res1.x[-1]

    # ---- LP for min-player: minimize w s.t. G pi2 <= w*1, sum pi2=1, pi2>=0 ----
    # Variables: y = [pi2 (K entries), w]
    c2 = np.zeros(K + 1)
    c2[-1] = 1.0                     # minimize w

    A_ub2 = np.hstack([G, -np.ones((K, 1))])   # G pi2 - w <= 0
    b_ub2 = np.zeros(K)

    A_eq2 = np.zeros((1, K + 1))
    A_eq2[0, :K] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0.0, 1.0)] * K + [(0.0, 1.0)]

    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res2.success:
        raise RuntimeError(f"linprog (min-player) failed: {res2.message}")

    pi2 = res2.x[:K]
    v2 = res2.x[-1]

    # Numerically, v1 and v2 should match (up to tolerance)
    v = 0.5 * (v1 + v2)

    # Clean tiny negatives and renormalize (sometimes linprog gives 1e-12 artifacts)
    pi1 = np.clip(pi1, 0.0, None)
    pi2 = np.clip(pi2, 0.0, None)
    pi1 = pi1 / pi1.sum() if pi1.sum() > 0 else np.ones(K) / K
    pi2 = pi2 / pi2.sum() if pi2.sum() > 0 else np.ones(K) / K

    return pi1, pi2, v, G
