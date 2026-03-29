import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.optimize import root
from env import mu_logistic, mu_linear

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

def bilinear_solver_reg(Phi: np.ndarray, Theta: np.ndarray, *, mu=mu_logistic, eta=1.0, reg_type='reverse_kl', ref_policy=None):
    """
    Solve the REGULARIZED symmetric zero-sum game.
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

    # We want to find a policy p such that p = BestResponse(p)
    # Define the root-finding objective: F(p) = p - BestResponse(p) = 0
    def fixed_point_objective(p):
        # Ensure p stays a valid probability distribution during the solver's search
        p_clean = np.clip(p, 0.0, None)
        if p_clean.sum() > 0:
            p_clean /= p_clean.sum()
        else:
            p_clean = np.ones(K) / K
            
        # Pass the ref_policy down to the best response calculator
        br = compute_best_response(p_clean, G_tilde, eta, reg_type, ref_policy)
        return p_clean - br

    # Initial guess is the uniform distribution
    p0 = np.ones(K) / K
    
    # Use scipy's root finder (Powell's hybrid method is very robust for this)
    res = root(fixed_point_objective, p0, method='hybr')
    
    if not res.success:
        # Fallback to a simple iterative fixed-point loop if the root finder fails
        p = p0
        for _ in range(500):
            p = 0.5 * p + 0.5 * compute_best_response(p, G_tilde, eta, reg_type, ref_policy)
            
    # Clean up final policy
    pi_star = np.clip(res.x, 0.0, None)
    pi_star /= pi_star.sum()

    # The game is symmetric, so pi1 = pi2 = pi_star. Value is strictly 0.5.
    return pi_star, pi_star, 0.5, G

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
    Because mu(z) = 0.5 + 0.25*z, E[r_t] = 0.5 + 0.25*<Theta, X_t>.
    Therefore, the regression target for <Theta, X_t> is 4 * (r_t - 0.5).
    """
    T_curr = len(r_list)
    if T_curr == 0:
        raise ValueError("Need at least one sample.")

    d = phi1_list[0].shape[0]
    Theta = cp.Variable((d, d))

    # Skew-symmetry constraint
    constraints = [Theta + Theta.T == 0]

    # Vectorize inputs for CVXPY speed
    X_stack = np.array([np.outer(phi1_list[t], phi2_list[t]).flatten() for t in range(T_curr)])
    targets = 4.0 * (np.array(r_list) - 0.5)

    # Loss: || X * vec(Theta) - targets ||^2 + lam * ||Theta||_F^2
    z = X_stack @ cp.vec(Theta)
    loss = cp.sum_squares(z - targets) + lam * cp.sum_squares(Theta)

    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(solver=solver)

    if Theta.value is None:
        raise RuntimeError(f"CVXPY Ridge failed. status={prob.status}")

    return np.asarray(Theta.value, dtype=float)