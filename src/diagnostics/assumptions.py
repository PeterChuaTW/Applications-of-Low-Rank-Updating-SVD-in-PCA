"""
Noise Analysis for Validating Gavish-Donoho Assumptions.

This module provides tools to verify whether a dataset's noise structure
matches the assumptions required by the Gavish-Donoho optimal threshold method.

Key Assumptions of Gavish-Donoho (2014):
1. Additive noise model: X = Signal + Noise
2. Noise is Gaussian (normally distributed)
3. Noise is white (independent across pixels/features)
4. Noise is homoscedastic (constant variance)

References:
-----------
[1] Gavish, M., & Donoho, D. L. (2014). "The optimal hard threshold for 
    singular values is 4/sqrt(3)". IEEE Transactions on Information Theory.
[2] Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test 
    for normality (complete samples)". Biometrika, 52(3-4), 591-611.
"""
import numpy as np
from scipy import stats
import warnings


def estimate_noise_from_residuals(X, n_components):
    """
    Estimate noise by computing residuals after low-rank approximation.
    
    The idea: X ≈ U_k Σ_k V_k^T + Noise
    Residuals = X - U_k Σ_k V_k^T should primarily contain noise.
    
    Parameters:
    -----------
    X : array, shape (m, n)
        Data matrix
    n_components : int
        Number of components to use for low-rank approximation
        
    Returns:
    --------
    residuals : array, shape (m, n)
        Residual matrix (estimated noise)
    residuals_flat : array, shape (m*n,)
        Flattened residuals for statistical testing
    """
    # Compute SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Truncate to k components
    k = min(n_components, len(s))
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct low-rank approximation
    X_approx = U_k @ np.diag(s_k) @ Vt_k
    
    # Compute residuals
    residuals = X - X_approx
    residuals_flat = residuals.flatten()
    
    return residuals, residuals_flat


def check_gaussian_noise(residuals_flat, alpha=0.05):
    """
    Test if residuals follow a Gaussian (normal) distribution.
    
    Uses multiple statistical tests:
    1. Shapiro-Wilk test (most powerful for small/medium samples)
    2. Anderson-Darling test (good for detecting tail deviations)
    3. Kolmogorov-Smirnov test (general distribution test)
    
    Parameters:
    -----------
    residuals_flat : array, shape (n,)
        Flattened residual values
    alpha : float, default=0.05
        Significance level for hypothesis tests
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'is_gaussian': bool, True if passes all tests
        - 'shapiro': dict with statistic and p-value
        - 'anderson': dict with statistic and critical values
        - 'ks': dict with statistic and p-value
        - 'skewness': float, measure of asymmetry
        - 'kurtosis': float, measure of tail heaviness
        
    Notes:
    ------
    Null Hypothesis (H0): Data comes from a normal distribution
    - If p-value > alpha: Cannot reject H0 → data is consistent with Gaussian
    - If p-value < alpha: Reject H0 → data is NOT Gaussian
    
    For Anderson-Darling:
    - If statistic < critical_value[2] (5% level): data is Gaussian
    """
    # Remove NaN/Inf values
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    if len(residuals_clean) < 3:
        raise ValueError("Need at least 3 data points for normality testing")
    
    # Sample if too large (Shapiro-Wilk has limit of 5000)
    if len(residuals_clean) > 5000:
        warnings.warn(
            f"Sampling 5000 points from {len(residuals_clean)} for Shapiro-Wilk test"
        )
        residuals_sample = np.random.choice(residuals_clean, 5000, replace=False)
    else:
        residuals_sample = residuals_clean
    
    results = {}
    
    # 1. Shapiro-Wilk test (most powerful for n < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(residuals_sample)
    results['shapiro'] = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p > alpha
    }
    
    # 2. Anderson-Darling test (good for detecting deviations in tails)
    anderson_result = stats.anderson(residuals_clean, dist='norm')
    # Critical value at 5% significance level (index 2)
    results['anderson'] = {
        'statistic': anderson_result.statistic,
        'critical_values': anderson_result.critical_values,
        'significance_levels': anderson_result.significance_level,
        'is_normal': anderson_result.statistic < anderson_result.critical_values[2]
    }
    
    # 3. Kolmogorov-Smirnov test
    # Standardize data (mean=0, std=1) for comparison with standard normal
    residuals_standardized = (residuals_clean - np.mean(residuals_clean)) / np.std(residuals_clean)
    ks_stat, ks_p = stats.kstest(residuals_standardized, 'norm')
    results['ks'] = {
        'statistic': ks_stat,
        'p_value': ks_p,
        'is_normal': ks_p > alpha
    }
    
    # 4. Compute skewness and kurtosis
    results['skewness'] = stats.skew(residuals_clean)
    results['kurtosis'] = stats.kurtosis(residuals_clean)  # Excess kurtosis (0 for normal)
    
    # Overall decision: pass if majority of tests pass
    tests_passed = sum([
        results['shapiro']['is_normal'],
        results['anderson']['is_normal'],
        results['ks']['is_normal']
    ])
    
    results['is_gaussian'] = tests_passed >= 2
    results['tests_passed'] = f"{tests_passed}/3"
    
    return results


def check_white_noise(residuals, max_lags=50):
    """
    Test if residuals exhibit white noise characteristics (no autocorrelation).
    
    White noise means:
    - No correlation between different time points (or spatial locations)
    - Autocorrelation function (ACF) ≈ 0 for all lags > 0
    
    Parameters:
    -----------
    residuals : array, shape (m, n)
        Residual matrix
    max_lags : int, default=50
        Maximum number of lags to test for autocorrelation
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'is_white': bool, True if residuals are white noise
        - 'ljung_box_p': float, p-value from Ljung-Box test
        - 'acf_values': array, autocorrelation function values
        - 'significant_lags': int, number of lags with significant correlation
        
    Notes:
    ------
    Ljung-Box Test:
    - H0: Residuals are independently distributed (white noise)
    - If p-value > 0.05: Cannot reject H0 → consistent with white noise
    - If p-value < 0.05: Reject H0 → significant autocorrelation exists
    """
    residuals_flat = residuals.flatten()
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    if len(residuals_clean) < max_lags + 1:
        max_lags = len(residuals_clean) - 1
    
    results = {}
    
    # Compute autocorrelation function (ACF)
    # For large data, sample to speed up computation
    if len(residuals_clean) > 10000:
        sample_size = 10000
        residuals_sample = np.random.choice(residuals_clean, sample_size, replace=False)
    else:
        residuals_sample = residuals_clean
    
    # Manual ACF computation for simplicity
    mean = np.mean(residuals_sample)
    var = np.var(residuals_sample)
    
    acf = np.zeros(max_lags + 1)
    acf[0] = 1.0  # ACF at lag 0 is always 1
    
    for lag in range(1, max_lags + 1):
        if len(residuals_sample) > lag:
            c = np.mean((residuals_sample[:-lag] - mean) * (residuals_sample[lag:] - mean))
            acf[lag] = c / var if var > 0 else 0
    
    results['acf_values'] = acf
    
    # Ljung-Box test for white noise
    # Tests if any of a group of autocorrelations are different from zero
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals_sample, lags=min(20, max_lags), return_df=True)
        
        # lb_result is a DataFrame with columns: 'lb_stat' and 'lb_pvalue'
        # Take minimum p-value (most conservative)
        if hasattr(lb_result, 'lb_pvalue'):
            # New API (statsmodels >= 0.12)
            lb_p = np.min(lb_result['lb_pvalue'].values)
        elif 'lb_pvalue' in lb_result.columns:
            # Alternative column name
            lb_p = np.min(lb_result['lb_pvalue'].values)
        else:
            # Fallback: try to get p-values from any column
            lb_p = np.min(lb_result.iloc[:, 1].values)  # Second column usually contains p-values
        
        results['ljung_box_p'] = lb_p
        results['ljung_box_available'] = True
    except ImportError:
        # Fallback: use simple criterion if statsmodels not available
        # Count lags with |ACF| > 1.96/sqrt(n) (95% confidence bound)
        n = len(residuals_sample)
        conf_bound = 1.96 / np.sqrt(n)
        significant_lags = np.sum(np.abs(acf[1:]) > conf_bound)
        results['ljung_box_p'] = 1.0 if significant_lags < 3 else 0.01
        results['ljung_box_available'] = False
        warnings.warn("statsmodels not available, using simplified white noise test")
    except Exception as e:
        # Generic error fallback
        warnings.warn(f"Ljung-Box test failed: {e}. Using simplified test.")
        n = len(residuals_sample)
        conf_bound = 1.96 / np.sqrt(n)
        significant_lags = np.sum(np.abs(acf[1:]) > conf_bound)
        results['ljung_box_p'] = 1.0 if significant_lags < 3 else 0.01
        results['ljung_box_available'] = False
    
    # Count significant autocorrelations
    n = len(residuals_sample)
    conf_bound = 1.96 / np.sqrt(n)  # 95% confidence interval
    results['significant_lags'] = np.sum(np.abs(acf[1:]) > conf_bound)
    results['confidence_bound'] = conf_bound
    
    # Decision: white noise if p > 0.05 and few significant lags
    results['is_white'] = (
        results['ljung_box_p'] > 0.05 and 
        results['significant_lags'] < 0.1 * max_lags  # Allow up to 10% significant lags
    )
    
    return results


def validate_gavish_donoho_assumptions(X, n_components=None, alpha=0.05, max_lags=50):
    """
    Comprehensive validation of Gavish-Donoho assumptions.
    
    This is the main function you should use to check if your data is suitable
    for the Gavish-Donoho optimal threshold method.
    
    Parameters:
    -----------
    X : array, shape (m, n)
        Data matrix (typically mean-centered)
    n_components : int, optional
        Number of components for low-rank approximation to estimate noise.
        If None, uses min(m, n) // 4 (rule of thumb)
    alpha : float, default=0.05
        Significance level for statistical tests
    max_lags : int, default=50
        Maximum lags for autocorrelation analysis
        
    Returns:
    --------
    validation_results : dict
        Comprehensive results including:
        - 'assumptions_met': bool, overall assessment
        - 'gaussian_test': dict, results from check_gaussian_noise()
        - 'white_noise_test': dict, results from check_white_noise()
        - 'recommendation': str, text recommendation
        
    Examples:
    ---------
    >>> from data_loader import load_orl_faces, normalize_faces
    >>> faces, labels, _ = load_orl_faces()
    >>> centered_faces, _ = normalize_faces(faces)
    >>> results = validate_gavish_donoho_assumptions(centered_faces)
    >>> print(results['recommendation'])
    """
    X = np.asarray(X)
    m, n = X.shape
    
    # Default n_components: use ~25% of max dimension
    if n_components is None:
        n_components = min(m, n) // 4
        n_components = max(1, n_components)  # At least 1
    
    print("\n" + "="*70)
    print("VALIDATING GAVISH-DONOHO ASSUMPTIONS")
    print("="*70)
    print(f"\nData dimensions: {m} × {n}")
    print(f"Components for noise estimation: {n_components}")
    
    # Step 1: Estimate noise from residuals
    print("\nStep 1: Estimating noise from residuals...")
    residuals, residuals_flat = estimate_noise_from_residuals(X, n_components)
    
    noise_std = np.std(residuals_flat)
    noise_mean = np.mean(residuals_flat)
    print(f"  Residual mean: {noise_mean:.6f} (should be ≈ 0)")
    print(f"  Residual std:  {noise_std:.6f}")
    
    # Step 2: Test for Gaussian distribution
    print("\nStep 2: Testing for Gaussian (normal) distribution...")
    gaussian_results = check_gaussian_noise(residuals_flat, alpha=alpha)
    
    print(f"  Shapiro-Wilk test:  p = {gaussian_results['shapiro']['p_value']:.4f} ", end="")
    print("✓" if gaussian_results['shapiro']['is_normal'] else "✗")
    
    print(f"  Anderson-Darling:   statistic = {gaussian_results['anderson']['statistic']:.4f} ", end="")
    print("✓" if gaussian_results['anderson']['is_normal'] else "✗")
    
    print(f"  Kolmogorov-Smirnov: p = {gaussian_results['ks']['p_value']:.4f} ", end="")
    print("✓" if gaussian_results['ks']['is_normal'] else "✗")
    
    print(f"\n  Skewness: {gaussian_results['skewness']:.4f} (should be ≈ 0)")
    print(f"  Kurtosis: {gaussian_results['kurtosis']:.4f} (should be ≈ 0)")
    print(f"\n  Overall: {gaussian_results['tests_passed']} tests passed")
    
    # Step 3: Test for white noise
    print("\nStep 3: Testing for white noise (independence)...")
    white_results = check_white_noise(residuals, max_lags=max_lags)
    
    print(f"  Ljung-Box test: p = {white_results['ljung_box_p']:.4f} ", end="")
    print("✓" if white_results['ljung_box_p'] > alpha else "✗")
    
    print(f"  Significant autocorrelations: {white_results['significant_lags']}/{max_lags}")
    print(f"  95% confidence bound: ±{white_results['confidence_bound']:.4f}")
    
    # Overall assessment
    assumptions_met = gaussian_results['is_gaussian'] and white_results['is_white']
    
    validation_results = {
        'assumptions_met': assumptions_met,
        'gaussian_test': gaussian_results,
        'white_noise_test': white_results,
        'residual_stats': {
            'mean': noise_mean,
            'std': noise_std
        }
    }
    
    # Generate recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if assumptions_met:
        recommendation = (
            "✅ PASSED: Your data appears to satisfy Gavish-Donoho assumptions.\n"
            "   Residuals are approximately Gaussian and white.\n"
            "   ➜ Gavish-Donoho optimal threshold is RECOMMENDED for your dataset."
        )
    elif gaussian_results['is_gaussian'] and not white_results['is_white']:
        recommendation = (
            "⚠️  PARTIAL: Residuals are Gaussian but show autocorrelation.\n"
            "   This is common in image data (spatial correlation).\n"
            "   ➜ Gavish-Donoho may still work, but results should be validated.\n"
            "   ➜ Compare with 95% energy method as sanity check."
        )
    elif not gaussian_results['is_gaussian'] and white_results['is_white']:
        recommendation = (
            "⚠️  PARTIAL: Residuals are independent but not Gaussian.\n"
            "   This suggests non-Gaussian noise or heavy-tailed distribution.\n"
            "   ➜ Gavish-Donoho may be suboptimal.\n"
            "   ➜ Prefer cumulative energy method (95%) for safety."
        )
    else:
        recommendation = (
            "❌ FAILED: Residuals are neither Gaussian nor white.\n"
            "   Gavish-Donoho assumptions are violated.\n"
            "   ➜ DO NOT USE Gavish-Donoho for automatic rank selection.\n"
            "   ➜ USE cumulative energy method (95%) instead."
        )
    
    validation_results['recommendation'] = recommendation
    print("\n" + recommendation)
    print("\n" + "="*70 + "\n")
    
    return validation_results
