"""
Elbow Point Detection Methods for Rank Selection.

This module provides geometric and data-driven methods to automatically
detect the "elbow" (knee point) in the singular value curve, which
indicates the optimal number of principal components to retain.

Methods included:
1. Kneedle Algorithm (Maximum Distance to Line)
2. Slope Difference Method (Marginal Gain Analysis)
3. L-Method (Two-Segment Linear Regression)

References:
-----------
[1] Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
    "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior".
    31st International Conference on Distributed Computing Systems Workshops.
[2] Salvador, S., & Chan, P. (2004). "Determining the number of clusters/
    segments in hierarchical clustering/segmentation algorithms".
    16th IEEE International Conference on Tools with AI.
"""
import numpy as np
from scipy.spatial import distance
import warnings


def kneedle_algorithm(singular_values, sensitivity=1.0):
    """
    Detect elbow point using Kneedle algorithm (maximum distance method).
    
    This is the most geometrically rigorous definition of "elbow":
    1. Draw a straight line from first to last point in scree plot
    2. Find the point on the curve with maximum perpendicular distance to this line
    3. That point is the elbow (maximum curvature)
    
    Mathematical Formulation:
    -------------------------
    Given points (1, σ₁), (2, σ₂), ..., (n, σₙ)
    
    Line equation from (1, σ₁) to (n, σₙ):
    L: ax + by + c = 0
    
    Distance from point (k, σₖ) to line L:
    d_k = |a·k + b·σₖ + c| / √(a² + b²)
    
    Elbow point: k* = argmax_k d_k
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD (sorted descending)
    sensitivity : float, default=1.0
        Sensitivity parameter S ∈ [0, ∞)
        - S = 0: No detection (returns last index)
        - S = 1: Standard sensitivity (recommended)
        - S > 1: More conservative (detects earlier knees)
        
    Returns:
    --------
    knee_index : int
        Index of the elbow point (1-based component number)
    distances : array-like
        Perpendicular distances for all points
    normalized_distances : array-like
        Normalized distances used for detection
        
    Examples:
    ---------
    >>> from batch_pca import BatchPCA
    >>> pca = BatchPCA(n_components=None)
    >>> pca.fit(X)
    >>> k, dists, norm_dists = kneedle_algorithm(pca.singular_values_)
    >>> print(f"Elbow at component {k}")
    
    Notes:
    ------
    - Works best when there's a clear elbow in the scree plot
    - Robust to noise compared to derivative-based methods
    - No arbitrary thresholds needed
    - May fail if curve is monotonic or multi-modal
    
    Reference:
    ----------
    Satopää, V., et al. (2011). "Finding a 'Kneedle' in a Haystack"
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)
    
    if n < 3:
        warnings.warn("Need at least 3 singular values for elbow detection")
        return 1, np.zeros(n), np.zeros(n)
    
    # Normalize data to [0, 1] range for both axes
    # This ensures equal weighting of x and y dimensions
    x = np.arange(1, n + 1)
    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
    
    y = singular_values
    y_norm = (y - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else y
    
    # Define line from first to last point
    # Line equation: ax + by + c = 0
    # Using normalized coordinates
    p1 = np.array([x_norm[0], y_norm[0]])  # First point (0, 1)
    p2 = np.array([x_norm[-1], y_norm[-1]])  # Last point (1, 0)
    
    # Vector from p1 to p2
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        warnings.warn("Degenerate case: all singular values are equal")
        return 1, np.zeros(n), np.zeros(n)
    
    # Unit vector along the line
    line_unit = line_vec / line_length
    
    # Calculate perpendicular distance from each point to the line
    distances = np.zeros(n)
    
    for i in range(n):
        # Point on curve
        point = np.array([x_norm[i], y_norm[i]])
        
        # Vector from p1 to point
        point_vec = point - p1
        
        # Perpendicular distance = |cross product| / |line length|
        # In 2D: cross product gives scalar
        cross = point_vec[0] * line_unit[1] - point_vec[1] * line_unit[0]
        distances[i] = abs(cross) * line_length
    
    # Normalize distances to [0, 1]
    if distances.max() > 0:
        normalized_distances = distances / distances.max()
    else:
        normalized_distances = distances
    
    # Apply sensitivity threshold
    # Points above sensitivity * max_distance are candidates
    threshold = sensitivity * normalized_distances.max()
    candidates = np.where(normalized_distances >= threshold)[0]
    
    if len(candidates) == 0:
        # Fallback: use point with max distance
        knee_index = np.argmax(distances) + 1  # Convert to 1-based
    else:
        # Among candidates, choose the first one (leftmost)
        # This is more conservative
        knee_index = candidates[0] + 1  # Convert to 1-based
    
    return knee_index, distances, normalized_distances


def slope_difference_method(singular_values, threshold=0.1):
    """
    Detect elbow by analyzing marginal gain (slope differences).
    
    This method looks for where adding one more component gives
    diminishing returns - the point where Δσᵢ becomes small.
    
    Mathematical Formulation:
    -------------------------
    Marginal gain: Δᵢ = σᵢ - σᵢ₊₁
    
    Second derivative (acceleration): Δ²ᵢ = Δᵢ - Δᵢ₊₁
    
    Elbow point: where Δ²ᵢ changes sign or becomes small
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD (sorted descending)
    threshold : float, default=0.1
        Relative threshold for "small" changes
        (fraction of first derivative)
        
    Returns:
    --------
    elbow_index : int
        Index of the elbow point (1-based)
    first_diffs : array-like, shape (n-1,)
        First differences (marginal gains)
    second_diffs : array-like, shape (n-2,)
        Second differences (acceleration)
        
    Examples:
    ---------
    >>> k, diffs1, diffs2 = slope_difference_method(singular_values)
    >>> print(f"Elbow at component {k}")
    
    Notes:
    ------
    - Sensitive to noise in singular values
    - May detect multiple candidate points
    - Threshold is somewhat arbitrary
    - Works well when there's a sharp transition
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)
    
    if n < 3:
        warnings.warn("Need at least 3 singular values")
        return 1, np.array([]), np.array([])
    
    # First differences (marginal gains)
    first_diffs = np.diff(singular_values)  # Negative values (decreasing)
    first_diffs_abs = np.abs(first_diffs)
    
    # Second differences (acceleration)
    second_diffs = np.diff(first_diffs_abs)
    
    # Normalize by maximum first difference
    if first_diffs_abs.max() > 0:
        normalized_diffs = first_diffs_abs / first_diffs_abs.max()
    else:
        normalized_diffs = first_diffs_abs
    
    # Method 1: Find where first derivative drops below threshold
    candidates = np.where(normalized_diffs < threshold)[0]
    
    if len(candidates) > 0:
        # Take the first point where it drops
        elbow_index = candidates[0] + 1  # +1 for 1-based indexing
    else:
        # Method 2: Find maximum second derivative (maximum acceleration)
        # This is where the curve bends most sharply
        elbow_index = np.argmax(np.abs(second_diffs)) + 1
    
    return elbow_index, first_diffs, second_diffs


def l_method(singular_values):
    """
    L-method: Fit two-segment linear regression to find elbow.
    
    This method fits two lines to the log-log plot of singular values
    and finds the intersection point (the elbow).
    
    Mathematical Formulation:
    -------------------------
    For each candidate point k:
    1. Fit line to points [1, k]: y = a₁x + b₁
    2. Fit line to points [k+1, n]: y = a₂x + b₂
    3. Compute RMSE for both segments
    
    Elbow point: k* = argmin_k (RMSE₁ + RMSE₂)
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD (sorted descending)
        
    Returns:
    --------
    elbow_index : int
        Index of the elbow point (1-based)
    rmse_scores : array-like
        RMSE scores for each candidate split point
        
    Examples:
    ---------
    >>> k, rmse = l_method(singular_values)
    >>> print(f"Elbow at component {k}")
    
    Notes:
    ------
    - Uses log-log scale for better linearity
    - More stable than derivative methods
    - Assumes two distinct regimes (signal vs noise)
    - Computationally more expensive
    
    Reference:
    ----------
    Salvador, S., & Chan, P. (2004). "Determining the number of clusters"
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)
    
    if n < 4:
        warnings.warn("Need at least 4 singular values for L-method")
        return 1, np.array([])
    
    # Use log scale (common for scree plots)
    x = np.log(np.arange(1, n + 1))
    y = np.log(singular_values)
    
    # Try all possible split points (from 2 to n-2)
    rmse_scores = np.zeros(n - 3)
    
    for k in range(1, n - 2):  # k is the split point (0-based)
        # Segment 1: [0, k]
        x1, y1 = x[:k+1], y[:k+1]
        
        # Segment 2: [k+1, n-1]
        x2, y2 = x[k+1:], y[k+1:]
        
        # Fit lines using least squares
        # y = ax + b
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
        
        # Combined RMSE
        rmse_scores[k-1] = rmse1 + rmse2
    
    # Find split with minimum RMSE
    elbow_index = np.argmin(rmse_scores) + 2  # +2 for offset and 1-based
    
    return elbow_index, rmse_scores


def compare_elbow_methods(singular_values):
    """
    Compare all three elbow detection methods.
    
    Parameters:
    -----------
    singular_values : array-like, shape (n,)
        Singular values from SVD (sorted descending)
        
    Returns:
    --------
    results : dict
        Dictionary containing results from all methods:
        - 'kneedle': (elbow_index, distances, normalized_distances)
        - 'slope': (elbow_index, first_diffs, second_diffs)
        - 'l_method': (elbow_index, rmse_scores)
        
    Examples:
    ---------
    >>> results = compare_elbow_methods(singular_values)
    >>> print(f"Kneedle: k={results['kneedle'][0]}")
    >>> print(f"Slope: k={results['slope'][0]}")
    >>> print(f"L-method: k={results['l_method'][0]}")
    """
    singular_values = np.asarray(singular_values)
    
    results = {}
    
    # Method 1: Kneedle
    try:
        k1, dists, norm_dists = kneedle_algorithm(singular_values)
        results['kneedle'] = {
            'elbow_index': k1,
            'distances': dists,
            'normalized_distances': norm_dists
        }
    except Exception as e:
        warnings.warn(f"Kneedle failed: {e}")
        results['kneedle'] = {'elbow_index': None}
    
    # Method 2: Slope Difference
    try:
        k2, diff1, diff2 = slope_difference_method(singular_values)
        results['slope'] = {
            'elbow_index': k2,
            'first_diffs': diff1,
            'second_diffs': diff2
        }
    except Exception as e:
        warnings.warn(f"Slope method failed: {e}")
        results['slope'] = {'elbow_index': None}
    
    # Method 3: L-method
    try:
        k3, rmse = l_method(singular_values)
        results['l_method'] = {
            'elbow_index': k3,
            'rmse_scores': rmse
        }
    except Exception as e:
        warnings.warn(f"L-method failed: {e}")
        results['l_method'] = {'elbow_index': None}
    
    return results


def print_elbow_comparison(results, singular_values):
    """
    Pretty print comparison of elbow detection methods.
    
    Parameters:
    -----------
    results : dict
        Output from compare_elbow_methods()
    singular_values : array-like
        Original singular values for variance calculation
    """
    print("\n" + "="*70)
    print("ELBOW POINT DETECTION METHODS COMPARISON")
    print("="*70)
    
    # Calculate explained variance for each method
    total_var = np.sum(singular_values ** 2)
    
    print("\n" + "-"*70)
    print(f"{'Method':<25} {'k (elbow)':<15} {'Explained Variance':<20}")
    print("-"*70)
    
    for method_name, data in results.items():
        k = data['elbow_index']
        if k is not None and k > 0:
            var = np.sum(singular_values[:k] ** 2) / total_var
            display_name = {
                'kneedle': 'Kneedle Algorithm',
                'slope': 'Slope Difference',
                'l_method': 'L-Method'
            }.get(method_name, method_name)
            
            print(f"{display_name:<25} {k:<15} {var:>6.2%}")
        else:
            print(f"{method_name:<25} {'FAILED':<15} {'-':<20}")
    
    print("-"*70)
    
    # Consensus
    elbows = [data['elbow_index'] for data in results.values() 
              if data['elbow_index'] is not None]
    
    if len(elbows) > 0:
        print(f"\nConsensus: Mean elbow = {np.mean(elbows):.1f}")
        print(f"           Median elbow = {int(np.median(elbows))}")
        print(f"           Range: [{min(elbows)}, {max(elbows)}]")
    
    print("\n" + "="*70 + "\n")
