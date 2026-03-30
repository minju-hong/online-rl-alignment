# env.py (minimal, UNCONTEXTUAL |X|=1)
# - X: single context (index 0)
# - A: actions are indices {0,...,K-1}
# - phi(a): pre-generated feature table Phi[a] in unit ball (fixed for whole run)
# - Theta_star: skew-symmetric low-rank-ish matrix
# - mu: logistic sigmoid or linear
# - reset_episode(T, episode_seed): pre-generates uniform noise u_t and uses episode RNG for actions

from __future__ import annotations
import numpy as np

def mu_logistic(z):
    """Logistic link mu(z) = sigmoid(z). Works on scalars or numpy arrays."""
    z = np.clip(z, -50.0, 50.0)
    out = 1.0 / (1.0 + np.exp(-z))
    # If scalar in -> return Python float; else return ndarray
    return float(out) if np.isscalar(out) else out

def mu_linear(z):
    """Linear link mu(z) = 0.5 + 0.125 * z, safely clipped to [0, 1]."""
    out = 0.5 + 0.125 * z
    out = np.clip(out, 0.0, 1.0)
    return float(out) if np.isscalar(out) else out

def random_policy_dirichlet(K: int) -> np.ndarray:
    """
    Draws a perfectly uniform random probability distribution.
    Requires an alpha array (we use ones for a uniform prior).
    """
    return np.random.dirichlet(np.ones(K))

def random_policy_uniform(K: int) -> np.ndarray:
    """
    Quickly normalizes standard uniform noise.
    """
    p = np.random.rand(K)
    return p / np.sum(p)

class GBPMEnv:
    def __init__(
            self, 
            K: int = 50, 
            d: int = 10, 
            r: int = 1, 
            S: float = 4.0, 
            instance_seed: int = 0,
            mu=mu_logistic,
    ):
      
        self.N = 1               # fixed: uncontextual
        self.K, self.d = int(K), int(d)
        self.r = int(r)
        self.S = float(S)
        self.mu = mu

        self.rng_instance = np.random.default_rng(int(instance_seed))

        # Select the first K standard basis vectors
        if self.K > self.d:
            self.K = self.d
            # raise ValueError(f"Cannot select K={self.K} standard basis vectors in d={self.d} dimensions.")
        self.Phi = np.eye(self.d)[:self.K]

        # # Known feature table Phi[a] (this plays the role of phi(a), since |X|=1)
        # Phi = self.rng_instance.normal(size=(self.K, self.d))

        # # Normalize each row to unit norm (handle the measure-zero all-zero row safely)
        # norms = np.linalg.norm(Phi, axis=1, keepdims=True)
        # norms = np.where(norms == 0.0, 1.0, norms)

        # Phi = 1.0 * Phi / norms   # now every row has norm exactly 1 (up to floating error)
        # self.Phi = Phi


        # Ground-truth parameter Theta_star
        self.Theta_star = self._make_theta_star()

        # --- THE PLANTING TRICK ---
        # Extract the principal directions of the true matrix
        U, _, Vh = np.linalg.svd(self.Theta_star)
        
        # # Plant the top singular vectors into the first two arms
        # if self.K >= 2:
        #     self.Phi[0] = U[:, 0]  # The direction of maximum variance
        #     self.Phi[1] = Vh[0, :] # The corresponding orthogonal direction
            
        #     # Ensure they maintain the same feature norm as the rest of the arms
        #     self.Phi[0] /= np.linalg.norm(self.Phi[0])
        #     self.Phi[1] /= np.linalg.norm(self.Phi[1])

        # Episode state
        self.u_seq = None
        self.t = 0
        self.rng_episode = None

    def _make_theta_star(self) -> np.ndarray:
        """Make a skew-symmetric, rank<=2r-ish Theta_star and scale its size."""
        d, r = self.d, self.r
        if 2 * r > d:
            raise ValueError(f"Need 2r <= d for this construction. Got r={r}, d={d}.")
        rng = self.rng_instance

        # Orthonormal Q in R^{d x 2r}
        G = rng.normal(size=(d, 2 * r))
        Q, _ = np.linalg.qr(G)

        # 2r x 2r skew block diagonal (r blocks)
        B = np.zeros((2 * r, 2 * r))
        s = rng.uniform(0.1, 1.0, size=r)
        for i, si in enumerate(s):
            B[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array([[0.0, si], [-si, 0.0]])

        Theta = Q @ B @ Q.T
        Theta = 0.5 * (Theta - Theta.T)  # enforce skew

        # scale size (simple; good enough for experiments)
        fro = np.linalg.norm(Theta, ord="fro") + 1e-12
        Theta = Theta * (self.S / fro)
        return Theta

    def reset_episode(self, T: int, episode_seed: int):
        """
        Pre-generate uniform noise u_t for t=1..T, and set an episode RNG
        (so action sampling can be reproducible/fair across algorithms).
        """
        rng = np.random.default_rng(int(episode_seed))
        self.u_seq = rng.random(size=int(T))  # Bernoulli noise
        self.rng_episode = rng                # also used for action sampling
        self.t = 0

    def step(self, pi1, pi2):
        """
        One protocol step at time t (uncontextual, so x is always 0):
        - actions sampled from pi1 over K arms, pi2 over K arms
        - reward r = 1[u_t < mu(phi(a1)^T Theta phi(a2))]

        pi1/pi2 can be:
          - vector shape (K,)
          - function returning (K,) probs: pi() or pi(x) (x ignored / always 0)

        Returns: (x_idx=0, a1, a2, r, p)
        """
        if self.u_seq is None or self.rng_episode is None:
            raise RuntimeError("Call reset_episode(T, episode_seed) before step().")
        if self.t >= len(self.u_seq):
            raise RuntimeError("Episode finished: t reached T. Call reset_episode again.")

        x = 0  # only one context

        def get_probs(pi):
            if callable(pi):
                # allow either pi() or pi(x) for convenience
                try:
                    p = np.asarray(pi(x), dtype=float)
                except TypeError:
                    p = np.asarray(pi(), dtype=float)
            else:
                p = np.asarray(pi, dtype=float)

            if p.shape != (self.K,):
                raise ValueError(f"Policy must produce shape (K,) = ({self.K},), got {p.shape}")

            p = np.clip(p, 0.0, None)
            s = float(p.sum())
            return (np.ones(self.K) / self.K) if s <= 0 else (p / s)

        p1 = get_probs(pi1)
        p2 = get_probs(pi2)

        # sample actions (episode RNG for reproducibility)
        a1 = int(self.rng_episode.choice(self.K, p=p1))
        a2 = int(self.rng_episode.choice(self.K, p=p2))

        # compute preference probability using fixed arm features
        phi1 = self.Phi[a1]
        phi2 = self.Phi[a2]
        z = float(phi1 @ self.Theta_star @ phi2)
        p = float(self.mu(z))

        # bandit feedback with shared uniform u_t
        u = float(self.u_seq[self.t])
        r = 1.0 if u < p else 0.0

        self.t += 1
        return x, a1, a2, r, p