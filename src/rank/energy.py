"""
Energy-based rank selection methods.

Includes cumulative energy / explained variance threshold selection.
"""
import numpy as np


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
