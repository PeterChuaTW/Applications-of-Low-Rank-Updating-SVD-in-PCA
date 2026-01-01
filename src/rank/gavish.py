"""
Gavish-Donoho optimal hard threshold for singular values.
"""
import numpy as np
import warnings


def gavish_donoho_threshold(X, sigma=None):
    """
    Gavish-Donoho optimal hard threshold (Random Matrix Theory).
    Fixed implementation following Gavish & Donoho (2014).

    Parameters:
    -----------
    X : array, shape (m, n)
        Data matrix (should be mean-centered)
    sigma : float, optional
        Noise standard deviation. If None, estimated from data.

    Returns:
    --------
    threshold : float
        Optimal hard threshold value τ
    n_components : int
        Number of singular values exceeding τ
    singular_values : array
        All singular values from SVD
    omega : float
        The ω(β) coefficient used
    """
    X = np.asarray(X)
    m, n = X.shape

    if m == 0 or n == 0:
        raise ValueError(f"Invalid dimensions: {m} × {n}")

    # Ensure m ≤ n
    if m > n:
        X = X.T
        m, n = n, m

    beta = m / n

    # Compute SVD
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    # Noise estimation
    if sigma is None:
        n_noise = max(5, int(0.25 * len(s)))
        tail_values = s[-n_noise:]
        sigma_median = np.median(tail_values)

        mad = np.median(np.abs(tail_values - sigma_median))
        sigma = sigma_median if mad < sigma_median * 0.5 else mad * 1.4826

        if sigma < (1e-10 * s[0] if s[0] > 0 else 1e-10):
            sigma = 1e-10 * s[0] if s[0] > 0 else 1e-10

    # omega(beta)
    if beta < 1:
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    else:
        omega = 2.858

    # τ = ω(β) × σ × √n
    threshold = omega * sigma * np.sqrt(n)

    n_components = np.sum(s > threshold)

    # Safety fallback
    if n_components == 0:
        if s[0] > 3 * sigma * np.sqrt(n):
            n_components = 1
            warnings.warn(
                f"Gavish-Donoho selected k=0, but s[0]={s[0]:.2f} >> τ={threshold:.2f}. "
                f"Setting k=1. Consider using energy method instead."
            )
        else:
            n_components = 1
            warnings.warn(
                f"Gavish-Donoho threshold τ={threshold:.2f} exceeds all singular values. "
                f"Setting k=1 by default. Use energy method instead."
            )

    return threshold, n_components, s, omega
