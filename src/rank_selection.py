"""
Rank Selection Methods for PCA.

This module provides various methods to determine the optimal number of
principal components (rank) to retain in PCA analysis.

Methods included:
1. Cumulative Energy Method (Traditional)
2. Gavish-Donoho Optimal Hard Threshold (Modern, based on Random Matrix Theory)

References:
-----------
[1] Gavish, M., & Donoho, D. L. (2014). "The optimal hard threshold for 
    singular values is 4/sqrt(3)". IEEE Transactions on Information Theory,
    60(8), 5040-5053.
[2] Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). 
    Springer Series in Statistics.
"""
import numpy as np


def determine_n_components_by_energy(singular_values, threshold=0.95):
    """
    Determine number of components using cumulative energy (variance) method.
    
    This is the traditional PCA method: retain components that cumulatively
    explain a specified percentage of total variance. Common thresholds are
    90%, 95%, or 99%.
    
    Mathematical Formula:
    ---------------------
    Given singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ from SVD: X = UΣVᵀ
    
    Total variance: var_total = Σᵢ σᵢ²
    Cumulative variance ratio: r_k = (Σᵢ₌₁ᵏ σᵢ²) / var_total
    
    Select k such that: r_k ≥ threshold
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD, assumed to be sorted in descending order
    threshold : float, default=0.95
        Cumulative energy threshold (e.g., 0.90 for 90%, 0.95 for 95%, 0.99 for 99%)
        Must be in range (0, 1]
        
    Returns:
    --------
    n_components : int
        Number of components to retain
    cumulative_variance_ratio : array-like, shape (n,)
        Cumulative variance ratio for each component
    explained_variance : float
        Actual cumulative variance ratio at selected n_components
        
    Examples:
    ---------
    >>> import numpy as np
    >>> from batch_pca import BatchPCA
    >>> X = np.random.randn(100, 50)
    >>> pca = BatchPCA(n_components=None)
    >>> pca.fit(X)
    >>> n_95 = determine_n_components_by_energy(pca.singular_values_, threshold=0.95)
    >>> print(f"Components for 95% variance: {n_95}")
    
    Notes:
    ------
    - This method is fast: O(n) where n is the number of singular values
    - It's interpretable: directly tied to explained variance
    - Conservative: tends to retain more components than necessary
    - Common in practice: 95% is a widely accepted default
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"Threshold must be in (0, 1], got {threshold}")
    
    singular_values = np.asarray(singular_values)
    
    if len(singular_values) == 0:
        raise ValueError("singular_values must not be empty")
    
    # Compute variance (squared singular values)
    variance = singular_values ** 2
    total_variance = np.sum(variance)
    
    if total_variance == 0:
        raise ValueError("Total variance is zero, cannot determine components")
    
    # Compute cumulative variance ratio
    cumulative_variance = np.cumsum(variance)
    cumulative_variance_ratio = cumulative_variance / total_variance
    
    # Find first index where cumulative ratio >= threshold
    # argmax returns first True index
    idx = np.argmax(cumulative_variance_ratio >= threshold)
    
    # Handle edge case: if threshold > max ratio, return all components
    if cumulative_variance_ratio[idx] < threshold:
        n_components = len(singular_values)
    else:
        n_components = idx + 1  # +1 because indices are 0-based
    
    explained_variance = cumulative_variance_ratio[n_components - 1]
    
    return n_components, cumulative_variance_ratio, explained_variance


def gavish_donoho_threshold(X, sigma=None):
    """
    Compute Gavish-Donoho optimal hard threshold for singular values.
    
    This method is based on Random Matrix Theory and provides a mathematically
    optimal threshold for separating signal from noise in the presence of
    additive Gaussian white noise.
    
    Mathematical Background:
    ------------------------
    For a data matrix X = Signal + Noise (with iid Gaussian noise),
    the optimal hard threshold is:
    
    τ_opt = ω(β) · σ_noise · √(max(m, n))
    
    where:
    - β = min(m, n) / max(m, n) is the aspect ratio
    - ω(β) is the optimal coefficient depending on β
    - σ_noise is the noise level
    - m, n are matrix dimensions
    
    For square matrices (β ≈ 1): ω(1) ≈ 2.858 ≈ 4/√3
    
    Singular values σᵢ > τ_opt are considered signal, σᵢ ≤ τ_opt are noise.
    
    Reference:
    ----------
    Gavish, M., & Donoho, D. L. (2014). "The optimal hard threshold for 
    singular values is 4/sqrt(3)". IEEE Transactions on Information Theory,
    60(8), 5040-5053.
    DOI: 10.1109/TIT.2014.2323359
    
    Parameters:
    -----------
    X : array-like, shape (m, n)
        Data matrix (can be centered or uncentered)
    sigma : float, optional
        Known noise level. If None, will be estimated from the data
        using the median of the smallest 25% singular values.
        
    Returns:
    --------
    threshold : float
        Optimal hard threshold value τ_opt
    n_components : int
        Number of singular values above threshold (recommended rank)
    singular_values : array-like
        All singular values (sorted descending)
    omega : float
        The computed ω(β) coefficient for reference
        
    Examples:
    ---------
    >>> import numpy as np
    >>> # Generate noisy low-rank matrix
    >>> np.random.seed(42)
    >>> U = np.random.randn(100, 10)
    >>> S = np.diag(np.arange(10, 0, -1))
    >>> V = np.random.randn(80, 10).T
    >>> X_clean = U @ S @ V
    >>> X_noisy = X_clean + 0.5 * np.random.randn(100, 80)
    >>> tau, k, s, omega = gavish_donoho_threshold(X_noisy)
    >>> print(f"Optimal threshold: {tau:.3f}")
    >>> print(f"Recommended rank: {k}")
    >>> print(f"Omega coefficient: {omega:.3f}")
    
    Notes:
    ------
    - This method assumes additive Gaussian white noise
    - It's data-driven: automatically determines threshold from data
    - It's aggressive: tends to retain fewer components than energy methods
    - Excellent for denoising applications
    - Fast: O(min(m,n)²·max(m,n)) for SVD computation
    
    Comparison with Energy Method:
    ------------------------------
    - Gavish-Donoho: Based on statistical theory, optimal for denoising
    - Cumulative Energy: Based on variance explained, interpretable but empirical
    - Gavish-Donoho typically selects fewer components than 95% energy method
    """
    X = np.asarray(X)
    m, n = X.shape
    
    if m == 0 or n == 0:
        raise ValueError(f"Invalid matrix dimensions: {m} x {n}")
    
    # Compute aspect ratio β = min(m,n) / max(m,n)
    beta = min(m, n) / max(m, n)
    
    # Compute SVD (thin/reduced form for efficiency)
    # This is O(min(m,n)² · max(m,n))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Estimate noise level if not provided
    if sigma is None:
        # Use median of smallest 25% singular values as noise estimate
        # This is a robust estimator assuming most small singular values are noise
        n_noise = max(1, int(0.25 * len(s)))
        sigma = np.median(s[-n_noise:])
        
        # Prevent division by zero
        if sigma == 0:
            # If estimated noise is zero, use a small fraction of max singular value
            sigma = 1e-10 * s[0] if s[0] > 0 else 1e-10
    
    # Compute optimal coefficient ω(β)
    # This is based on the asymptotic formula from Gavish-Donoho (2014)
    # Polynomial approximation for ω(β) derived from their paper
    # For β → 1 (square matrix): ω(1) ≈ 2.858 ≈ 4/√3
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    
    # Compute optimal hard threshold
    # τ_opt = ω(β) · σ · √(max(m, n))
    threshold = omega * sigma * np.sqrt(max(m, n))
    
    # Count singular values above threshold
    n_components = np.sum(s > threshold)
    
    # Handle edge case: if no components pass threshold, keep at least 1
    if n_components == 0:
        n_components = 1
    
    return threshold, n_components, s, omega


def compare_rank_selection_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    """
    Compare multiple rank selection methods on the same dataset.
    
    This utility function runs both cumulative energy and Gavish-Donoho
    methods and returns a comparison summary.
    
    Parameters:
    -----------
    X : array-like, shape (m, n)
        Data matrix (typically mean-centered for PCA)
    thresholds : tuple of float, default=(0.90, 0.95, 0.99)
        Energy thresholds to test
    sigma : float, optional
        Noise level for Gavish-Donoho (if None, will be estimated)
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'energy_methods': dict mapping threshold -> (n_components, explained_var)
        - 'gavish_donoho': dict with 'n_components', 'threshold', 'omega'
        - 'singular_values': array of all singular values
        
    Examples:
    ---------
    >>> import numpy as np
    >>> from data_loader import load_orl_faces, normalize_faces
    >>> faces, labels, _ = load_orl_faces()
    >>> centered_faces, _ = normalize_faces(faces)
    >>> results = compare_rank_selection_methods(centered_faces)
    >>> print("Comparison of Rank Selection Methods:")
    >>> for thresh, (k, var) in results['energy_methods'].items():
    ...     print(f"{int(thresh*100)}% Energy: k={k} (explains {var:.2%})")
    >>> gd = results['gavish_donoho']
    >>> print(f"Gavish-Donoho: k={gd['n_components']} (threshold={gd['threshold']:.2f})")
    """
    X = np.asarray(X)
    
    # Compute SVD once
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    results = {
        'singular_values': s,
        'energy_methods': {},
        'gavish_donoho': {}
    }
    
    # Test cumulative energy methods
    for threshold in thresholds:
        n_comp, cum_var_ratio, explained_var = determine_n_components_by_energy(
            s, threshold=threshold
        )
        results['energy_methods'][threshold] = {
            'n_components': n_comp,
            'explained_variance': explained_var
        }
    
    # Test Gavish-Donoho method
    tau, n_comp_gd, _, omega = gavish_donoho_threshold(X, sigma=sigma)
    
    # Calculate explained variance for Gavish-Donoho selection
    variance = s ** 2
    total_variance = np.sum(variance)
    explained_var_gd = np.sum(variance[:n_comp_gd]) / total_variance
    
    results['gavish_donoho'] = {
        'n_components': n_comp_gd,
        'threshold': tau,
        'omega': omega,
        'explained_variance': explained_var_gd
    }
    
    return results


def print_rank_selection_comparison(results):
    """
    Pretty print the results from compare_rank_selection_methods().
    
    Parameters:
    -----------
    results : dict
        Output from compare_rank_selection_methods()
    """
    print("\n" + "="*70)
    print("RANK SELECTION METHODS COMPARISON")
    print("="*70)
    
    print("\nMethod 1: Cumulative Energy (Traditional)")
    print("-" * 70)
    print(f"{'Threshold':<15} {'k (components)':<20} {'Explained Variance':<25}")
    print("-" * 70)
    
    for threshold, data in sorted(results['energy_methods'].items()):
        k = data['n_components']
        var = data['explained_variance']
        print(f"{int(threshold*100):>3}% Energy{'':<6} {k:<20} {var:>6.2%}")
    
    print("\nMethod 2: Gavish-Donoho Optimal Hard Threshold (Modern)")
    print("-" * 70)
    gd = results['gavish_donoho']
    print(f"Threshold (τ):        {gd['threshold']:.4f}")
    print(f"Omega coefficient:    {gd['omega']:.4f}")
    print(f"Components (k):       {gd['n_components']}")
    print(f"Explained Variance:   {gd['explained_variance']:.2%}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    k_95 = results['energy_methods'][0.95]['n_components']
    k_gd = results['gavish_donoho']['n_components']
    
    print(f"\n• For INTERPRETABILITY: Use 95% Energy → k = {k_95}")
    print(f"• For DENOISING:        Use Gavish-Donoho → k = {k_gd}")
    
    if k_gd < k_95:
        reduction = (1 - k_gd/k_95) * 100
        print(f"\n⚡ Gavish-Donoho is {reduction:.1f}% more compact than 95% Energy")
    elif k_gd > k_95:
        increase = (k_gd/k_95 - 1) * 100
        print(f"\n📊 Gavish-Donoho retains {increase:.1f}% more components than 95% Energy")
    else:
        print(f"\n✅ Both methods agree: k = {k_95}")
    
    print("="*70 + "\n")
