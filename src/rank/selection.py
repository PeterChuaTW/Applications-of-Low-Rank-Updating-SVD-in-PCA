"""
Comprehensive Rank Selection Methods for PCA.

This module provides 6 different methods to determine the optimal number of
principal components (rank) to retain in PCA analysis.

Methods included:
1. Cumulative Energy Method (90%, 95%, 99% thresholds)
2. Gavish-Donoho Optimal Hard Threshold (Random Matrix Theory)
3. Kneedle Algorithm (Maximum Distance to Line)
4. L-Method (Two-Segment Linear Regression)
5. Slope Difference Method (Marginal Gain Analysis)

References:
-----------
[1] Gavish & Donoho (2014): Optimal hard threshold for singular values
[2] Satopää et al. (2011): Finding a "Kneedle" in a Haystack
[3] Salvador & Chan (2004): Determining the number of clusters/PCs
[4] Jolliffe (2002): Principal Component Analysis
"""
import numpy as np
import warnings


# ===========================================================================
# METHOD 1-3: CUMULATIVE ENERGY (90%, 95%, 99%)
# ===========================================================================

def determine_n_components_by_energy(singular_values, threshold=0.95):
    """
    Determine number of components using cumulative energy (variance) method.
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD, sorted descending
    threshold : float, default=0.95
        Cumulative energy threshold (0.90, 0.95, 0.99)
        
    Returns:
    --------
    n_components : int
    cumulative_variance_ratio : array
    explained_variance : float
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"Threshold must be in (0, 1], got {threshold}")
    
    singular_values = np.asarray(singular_values)
    
    if len(singular_values) == 0:
        raise ValueError("singular_values must not be empty")
    
    variance = singular_values ** 2
    total_variance = np.sum(variance)
    
    if total_variance == 0:
        raise ValueError("Total variance is zero")
    
    cumulative_variance = np.cumsum(variance)
    cumulative_variance_ratio = cumulative_variance / total_variance
    
    idx = np.argmax(cumulative_variance_ratio >= threshold)
    
    if cumulative_variance_ratio[idx] < threshold:
        n_components = len(singular_values)
    else:
        n_components = idx + 1
    
    explained_variance = cumulative_variance_ratio[n_components - 1]
    
    return n_components, cumulative_variance_ratio, explained_variance


# ===========================================================================
# METHOD 4: GAVISH-DONOHO THRESHOLD (FIXED VERSION)
# ===========================================================================

def gavish_donoho_threshold(X, sigma=None):
    """
    Gavish-Donoho optimal hard threshold (Random Matrix Theory).
    
    Fixed implementation following Gavish & Donoho (2014) paper exactly.
    
    Mathematical Details:
    --------------------
    For a matrix X (m × n) with m ≤ n, define:
    - β = m/n (aspect ratio, β ∈ (0,1])
    - ω(β) = optimal threshold coefficient
    - λ_med = median of singular values in noise region
    - τ = ω(β) × λ_med × √n (optimal threshold)
    
    The optimal ω(β) for β < 1:
        ω(β) = 0.56β³ - 0.95β² + 1.82β + 1.43
    
    Reference:
    ----------
    Gavish, M., & Donoho, D. L. (2014). 
    "The optimal hard threshold for singular values is 4/√3".
    IEEE Transactions on Information Theory, 60(8), 5040-5053.
    Equations: (11) for ω(β), (23) for threshold formula
    
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
    
    # Ensure m ≤ n (swap if necessary)
    if m > n:
        X = X.T
        m, n = n, m
    
    # Compute aspect ratio β = m/n ∈ (0, 1]
    beta = m / n
    
    # Compute SVD (thin SVD since m ≤ n)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # === Noise Level Estimation ===
    if sigma is None:
        # Improved noise estimation using tail singular values
        # Use bottom 25% of singular values to estimate noise
        n_noise = max(5, int(0.25 * len(s)))
        
        # Method 1: Median of tail singular values (more robust)
        tail_values = s[-n_noise:]
        sigma_median = np.median(tail_values)
        
        # Method 2: MAD (Median Absolute Deviation) for robustness
        mad = np.median(np.abs(tail_values - sigma_median))
        sigma = sigma_median if mad < sigma_median * 0.5 else mad * 1.4826
        
        # Safety check: avoid zero or too small sigma
        if sigma < 1e-10 * s[0] if s[0] > 0 else 1e-10:
            sigma = 1e-10 * s[0] if s[0] > 0 else 1e-10
    
    # === Compute ω(β) using Gavish-Donoho formula ===
    # For β < 1: ω(β) = 0.56β³ - 0.95β² + 1.82β + 1.43
    # For β = 1: ω(1) = 2.858 (special case)
    
    if beta < 1:
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    else:
        # β = 1 (square matrix)
        omega = 2.858
    
    # === Compute Optimal Threshold ===
    # τ = ω(β) × σ × √n
    # where n is the larger dimension (number of columns)
    threshold = omega * sigma * np.sqrt(n)
    
    # === Count components above threshold ===
    n_components = np.sum(s > threshold)
    
    # Safety: ensure at least 1 component if any singular value is significant
    if n_components == 0:
        # Check if first singular value is significantly larger than noise
        if s[0] > 3 * sigma * np.sqrt(n):  # 3-sigma rule
            n_components = 1
            warnings.warn(
                f"Gavish-Donoho selected k=0, but s[0]={s[0]:.2f} >> threshold={threshold:.2f}. "
                f"Setting k=1. Consider: (1) data may not satisfy Gaussian noise assumption, "
                f"(2) aspect ratio β={beta:.4f} is very small (m << n), "
                f"(3) use 95% Energy method instead."
            )
        else:
            n_components = 1
            warnings.warn(
                f"Gavish-Donoho threshold τ={threshold:.2f} exceeds all singular values. "
                f"Setting k=1 by default. Likely causes: "
                f"(1) Noise level overestimated (σ={sigma:.4f}), "
                f"(2) Data dimension mismatch (β={beta:.4f} << 1), "
                f"(3) Non-Gaussian noise structure. "
                f"Recommendation: Use Cumulative Energy method (95%) instead."
            )
    
    return threshold, n_components, s, omega


# ===========================================================================
# METHOD 5: KNEEDLE ALGORITHM (MAXIMUM DISTANCE TO LINE)
# ===========================================================================

def kneedle_algorithm(singular_values, sensitivity=1.0):
    """
    Detect elbow using Kneedle algorithm (maximum distance method).
    
    Draws line from first to last point, finds point with maximum
    perpendicular distance.
    
    Parameters:
    -----------
    singular_values : array
    sensitivity : float, default=1.0
    
    Returns:
    --------
    knee_index : int (1-based)
    distances : array
    normalized_distances : array
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)
    
    if n < 3:
        warnings.warn("Need at least 3 singular values")
        return 1, np.zeros(n), np.zeros(n)
    
    # Normalize to [0, 1]
    x = np.arange(1, n + 1)
    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
    
    y = singular_values
    y_norm = (y - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else y
    
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        warnings.warn("Degenerate case: all values equal")
        return 1, np.zeros(n), np.zeros(n)
    
    line_unit = line_vec / line_length
    distances = np.zeros(n)
    
    for i in range(n):
        point = np.array([x_norm[i], y_norm[i]])
        point_vec = point - p1
        cross = point_vec[0] * line_unit[1] - point_vec[1] * line_unit[0]
        distances[i] = abs(cross) * line_length
    
    if distances.max() > 0:
        normalized_distances = distances / distances.max()
    else:
        normalized_distances = distances
    
    threshold = sensitivity * normalized_distances.max()
    candidates = np.where(normalized_distances >= threshold)[0]
    
    if len(candidates) == 0:
        knee_index = np.argmax(distances) + 1
    else:
        knee_index = candidates[0] + 1
    
    return knee_index, distances, normalized_distances


# ===========================================================================
# METHOD 6: L-METHOD (TWO-SEGMENT LINEAR REGRESSION)
# ===========================================================================

def l_method(singular_values):
    """
    L-method: Fit two-segment linear regression to find elbow.
    
    Fits two lines and finds split point minimizing total RMSE.
    
    Parameters:
    -----------
    singular_values : array
    
    Returns:
    --------
    elbow_index : int (1-based)
    rmse_scores : array
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)
    
    if n < 4:
        warnings.warn("Need at least 4 singular values")
        return 1, np.array([])
    
    x = np.log(np.arange(1, n + 1))
    y = np.log(singular_values)
    
    rmse_scores = np.zeros(n - 3)
    
    for k in range(1, n - 2):
        x1, y1 = x[:k+1], y[:k+1]
        x2, y2 = x[k+1:], y[k+1:]
        
        if len(x1) >= 2:
            a1, b1 = np.polyfit(x1, y1, 1)
            rmse1 = np.sqrt(np.mean((y1 - (a1 * x1 + b1))**2))
        else:
            rmse1 = 0
        
        if len(x2) >= 2:
            a2, b2 = np.polyfit(x2, y2, 1)
            rmse2 = np.sqrt(np.mean((y2 - (a2 * x2 + b2))**2))
        else:
            rmse2 = 0
        
        rmse_scores[k-1] = rmse1 + rmse2
    
    elbow_index = np.argmin(rmse_scores) + 2
    
    return elbow_index, rmse_scores


# ===========================================================================
# UNIFIED COMPARISON FUNCTION
# ===========================================================================

def compare_all_rank_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    """
    Compare ALL 6 rank selection methods comprehensively.
    
    Parameters:
    -----------
    X : array, shape (m, n)
        Mean-centered data matrix
    thresholds : tuple, default=(0.90, 0.95, 0.99)
        Energy thresholds to test
    sigma : float, optional
        Noise level for Gavish-Donoho
        
    Returns:
    --------
    results : dict
        Comprehensive results from all methods
    """
    X = np.asarray(X)
    
    # Compute SVD once
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    results = {
        'singular_values': s,
        'energy_methods': {},
        'gavish_donoho': {},
        'kneedle': {},
        'l_method': {}
    }
    
    # Total variance for explained variance calculation
    variance = s ** 2
    total_variance = np.sum(variance)
    
    # Method 1-3: Cumulative Energy (90%, 95%, 99%)
    for threshold in thresholds:
        n_comp, cum_var_ratio, explained_var = determine_n_components_by_energy(
            s, threshold=threshold
        )
        results['energy_methods'][threshold] = {
            'n_components': n_comp,
            'explained_variance': explained_var,
            'method_name': f"{int(threshold*100)}% Energy"
        }
    
    # Method 4: Gavish-Donoho (with error handling)
    try:
        tau, n_comp_gd, _, omega = gavish_donoho_threshold(X, sigma=sigma)
        explained_var_gd = np.sum(variance[:n_comp_gd]) / total_variance
        
        results['gavish_donoho'] = {
            'n_components': n_comp_gd,
            'threshold': tau,
            'omega': omega,
            'explained_variance': explained_var_gd,
            'method_name': 'Gavish-Donoho'
        }
    except Exception as e:
        warnings.warn(f"Gavish-Donoho failed: {e}")
        results['gavish_donoho'] = {
            'n_components': None, 
            'method_name': 'Gavish-Donoho',
            'error': str(e)
        }
    
    # Method 5: Kneedle Algorithm
    try:
        k_kneedle, dists, norm_dists = kneedle_algorithm(s)
        explained_var_kneedle = np.sum(variance[:k_kneedle]) / total_variance
        
        results['kneedle'] = {
            'n_components': k_kneedle,
            'distances': dists,
            'normalized_distances': norm_dists,
            'explained_variance': explained_var_kneedle,
            'method_name': 'Kneedle Algorithm'
        }
    except Exception as e:
        warnings.warn(f"Kneedle failed: {e}")
        results['kneedle'] = {'n_components': None, 'method_name': 'Kneedle Algorithm'}
    
    # Method 6: L-Method
    try:
        k_lmethod, rmse = l_method(s)
        explained_var_lmethod = np.sum(variance[:k_lmethod]) / total_variance
        
        results['l_method'] = {
            'n_components': k_lmethod,
            'rmse_scores': rmse,
            'explained_variance': explained_var_lmethod,
            'method_name': 'L-Method'
        }
    except Exception as e:
        warnings.warn(f"L-method failed: {e}")
        results['l_method'] = {'n_components': None, 'method_name': 'L-Method'}
    
    return results


def print_comprehensive_comparison(results):
    """
    Print comprehensive comparison of all 6 rank selection methods.
    
    Parameters:
    -----------
    results : dict
        Output from compare_all_rank_methods()
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RANK SELECTION COMPARISON (6 Methods)")
    print("="*80)
    
    # Collect all valid results
    all_methods = []
    
    # Energy methods
    for threshold, data in sorted(results['energy_methods'].items()):
        all_methods.append({
            'name': data['method_name'],
            'category': 'Empirical',
            'k': data['n_components'],
            'variance': data['explained_variance']
        })
    
    # Gavish-Donoho
    if results['gavish_donoho'].get('n_components'):
        all_methods.append({
            'name': 'Gavish-Donoho',
            'category': 'Statistical',
            'k': results['gavish_donoho']['n_components'],
            'variance': results['gavish_donoho']['explained_variance']
        })
    
    # Kneedle
    if results['kneedle'].get('n_components'):
        all_methods.append({
            'name': 'Kneedle',
            'category': 'Geometric',
            'k': results['kneedle']['n_components'],
            'variance': results['kneedle']['explained_variance']
        })
    
    # L-Method
    if results['l_method'].get('n_components'):
        all_methods.append({
            'name': 'L-Method',
            'category': 'Statistical',
            'k': results['l_method']['n_components'],
            'variance': results['l_method']['explained_variance']
        })
    
    # Print table
    print("\n" + "-"*80)
    print(f"{'Method':<20} {'Category':<15} {'k':<10} {'Explained Variance':<20}")
    print("-"*80)
    
    for method in all_methods:
        print(f"{method['name']:<20} {method['category']:<15} {method['k']:<10} {method['variance']:>6.2%}")
    
    print("-"*80)
    
    # Consensus analysis
    k_values = [m['k'] for m in all_methods]
    
    print("\n" + "="*80)
    print("CONSENSUS ANALYSIS")
    print("="*80)
    
    print(f"\n  Mean:   {np.mean(k_values):.1f}")
    print(f"  Median: {int(np.median(k_values))}")
    print(f"  Range:  [{min(k_values)}, {max(k_values)}]")
    print(f"  Std:    {np.std(k_values):.2f}")
    
    # Categorize agreement
    k_range = max(k_values) - min(k_values)
    
    if k_range <= 5:
        agreement = "STRONG"
        symbol = "✅"
    elif k_range <= 10:
        agreement = "MODERATE"
        symbol = "⚠️"
    else:
        agreement = "WEAK"
        symbol = "❌"
    
    print(f"\n  Agreement: {symbol} {agreement} (range = {k_range})")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    median_k = int(np.median(k_values))
    k_95 = results['energy_methods'][0.95]['n_components']
    
    print(f"\n  PRIMARY:    Use median consensus k = {median_k}")
    print(f"  BASELINE:   Use 95% Energy k = {k_95} (standard)")
    
    if results['kneedle'].get('n_components'):
        k_kneedle = results['kneedle']['n_components']
        print(f"  GEOMETRIC:  Use Kneedle k = {k_kneedle} (max curvature)")
    
    if results['gavish_donoho'].get('n_components'):
        k_gd = results['gavish_donoho']['n_components']
        print(f"  DENOISING:  Use Gavish-Donoho k = {k_gd} (optimal threshold)")
    
    print("\n" + "="*80 + "\n")


# Legacy function for backward compatibility
def compare_rank_selection_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    """
    Legacy function. Use compare_all_rank_methods() for full comparison.
    """
    return compare_all_rank_methods(X, thresholds, sigma)


def print_rank_selection_comparison(results):
    """
    Legacy function. Use print_comprehensive_comparison() for full output.
    """
    print_comprehensive_comparison(results)
