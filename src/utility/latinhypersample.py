import numpy as np
from scipy.stats import norm  # for inverse CDF

def lhs(n: int, d: int, *, criterion="random", seed=None) -> np.ndarray:
    """
    Latin Hypercube Samples in [0,1]^d.
    n: number of samples, d: dimensions.
    criterion: "random" (within-stratum uniform) or "center" (stratum centers).
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, n + 1)
    H = np.empty((n, d))

    for j in range(d):
        if criterion == "center":
            pts = (edges[:-1] + edges[1:]) / 2.0
        else:
            pts = edges[:-1] + (edges[1:] - edges[:-1]) * rng.random(n)
        rng.shuffle(pts)            # independent permutation per dimension
        H[:, j] = pts

    # avoid exactly 0 or 1 before using ppf
    eps = np.finfo(float).eps
    return np.clip(H, eps, 1 - eps)

if __name__ == "__main__":
    # Example: 2D (P, Q) with Normal marginals
    n, d = 10, 2
    U = lhs(n, d, criterion="random", seed=42)          # in (0,1)

    # map to target marginals
    P = norm.ppf(U[:, 0], loc=50.0, scale=5.0)          # MW
    Q = norm.ppf(U[:, 1], loc=10.0, scale=2.0)          # Mvar

    print(U.shape)
    print(U)
