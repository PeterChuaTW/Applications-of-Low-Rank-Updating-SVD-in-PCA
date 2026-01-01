"""
Unified Auto Rank Selection Module.

This module provides a comprehensive comparison and consensus-based
rank selection framework for PCA and Incremental PCA.

Included methods:
- Energy-based explained variance threshold
- Gavish–Donoho optimal hard threshold
- Kneedle (geometric elbow detection)
- L-method (piecewise linear regression)

This module is designed for both offline analysis and streaming settings.
"""

import numpy as np
import warnings

from .energy import determine_n_components_by_energy
from .gavish import gavish_donoho_threshold
from .elbow import kneedle_algorithm, l_method


# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------

def compare_all_rank_methods(
    X,
    thresholds=(0.90, 0.95, 0.99),
    sigma=None
):
    """
    Compare multiple rank selection methods on a given dataset.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.
    thresholds : tuple of float
        Energy thresholds for explained variance methods.
    sigma : float or None
        Noise level estimate (optional, for Gavish-Donoho).

    Returns
    -------
    results : dict
        Dictionary containing rank selection results from all methods.
    """
    X = np.asarray(X)

    # Compute SVD once for fairness
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    variance = s ** 2
    total_variance = np.sum(variance)

    results = {
        "singular_values": s,
        "energy_methods": {},
        "gavish_donoho": None,
        "kneedle": None,
        "l_method": None,
    }

    # -----------------------------------------------------------------
    # Energy-based methods
    # -----------------------------------------------------------------
    for threshold in thresholds:
        k, cum_ratio, explained_var = determine_n_components_by_energy(
            s, threshold=threshold
        )
        results["energy_methods"][threshold] = {
            "n_components": k,
            "explained_variance": explained_var,
            "cumulative_ratio": cum_ratio,
            "method_name": f"{int(threshold * 100)}% Energy",
            "category": "Empirical",
        }

    # -----------------------------------------------------------------
    # Gavish–Donoho
    # -----------------------------------------------------------------
    try:
        tau, k_gd, _, omega = gavish_donoho_threshold(X, sigma=sigma)
        results["gavish_donoho"] = {
            "n_components": k_gd,
            "threshold": tau,
            "omega": omega,
            "explained_variance": np.sum(variance[:k_gd]) / total_variance,
            "method_name": "Gavish–Donoho",
            "category": "Statistical",
        }
    except Exception as e:
        warnings.warn(f"Gavish–Donoho failed: {e}")
        results["gavish_donoho"] = {
            "n_components": None,
            "method_name": "Gavish–Donoho",
            "category": "Statistical",
            "error": str(e),
        }

    # -----------------------------------------------------------------
    # Kneedle
    # -----------------------------------------------------------------
    try:
        k_kneedle, dists, norm_dists = kneedle_algorithm(s)
        results["kneedle"] = {
            "n_components": k_kneedle,
            "distances": dists,
            "normalized_distances": norm_dists,
            "explained_variance": np.sum(variance[:k_kneedle]) / total_variance,
            "method_name": "Kneedle",
            "category": "Geometric",
        }
    except Exception as e:
        warnings.warn(f"Kneedle failed: {e}")
        results["kneedle"] = {
            "n_components": None,
            "method_name": "Kneedle",
            "category": "Geometric",
            "error": str(e),
        }

    # -----------------------------------------------------------------
    # L-method
    # -----------------------------------------------------------------
    try:
        k_l, rmse = l_method(s)
        results["l_method"] = {
            "n_components": k_l,
            "rmse_scores": rmse,
            "explained_variance": np.sum(variance[:k_l]) / total_variance,
            "method_name": "L-Method",
            "category": "Statistical",
        }
    except Exception as e:
        warnings.warn(f"L-method failed: {e}")
        results["l_method"] = {
            "n_components": None,
            "method_name": "L-Method",
            "category": "Statistical",
            "error": str(e),
        }

    return results


# ---------------------------------------------------------------------
# Reporting Utilities
# ---------------------------------------------------------------------

def print_comprehensive_comparison(results):
    """
    Print a comprehensive comparison and consensus analysis
    of all rank selection methods.
    """
    print("\n" + "=" * 80)
    print("UNIFIED AUTO-RANK SELECTION COMPARISON")
    print("=" * 80)

    all_methods = []

    for _, data in sorted(results["energy_methods"].items()):
        all_methods.append(data)

    for key in ["gavish_donoho", "kneedle", "l_method"]:
        method = results.get(key)
        if method and method.get("n_components") is not None:
            all_methods.append(method)

    print("\n" + "-" * 80)
    print(f"{'Method':<20} {'Category':<15} {'k':<8} {'Explained Var':<15}")
    print("-" * 80)

    for m in all_methods:
        print(
            f"{m['method_name']:<20} "
            f"{m['category']:<15} "
            f"{m['n_components']:<8} "
            f"{m['explained_variance']:.2%}"
        )

    print("-" * 80)

    k_values = [m["n_components"] for m in all_methods]
    k_range = max(k_values) - min(k_values)

    print("\n" + "=" * 80)
    print("CONSENSUS ANALYSIS")
    print("=" * 80)

    print(f"  Mean:   {np.mean(k_values):.2f}")
    print(f"  Median: {int(np.median(k_values))}")
    print(f"  Range:  [{min(k_values)}, {max(k_values)}]")
    print(f"  Std:    {np.std(k_values):.2f}")

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

    print("\n" + "=" * 80)
    print("RECOMMENDED USAGE")
    print("=" * 80)

    median_k = int(np.median(k_values))
    k_95 = results["energy_methods"][0.95]["n_components"]

    print(f"  PRIMARY (Consensus): k = {median_k}")
    print(f"  BASELINE (95% Energy): k = {k_95}")

    if results["gavish_donoho"].get("n_components"):
        print(f"  DENOISING (Gavish–Donoho): k = {results['gavish_donoho']['n_components']}")

    if results["kneedle"].get("n_components"):
        print(f"  GEOMETRIC (Kneedle): k = {results['kneedle']['n_components']}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------

def compare_rank_selection_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    """Legacy alias."""
    return compare_all_rank_methods(X, thresholds, sigma)


def print_rank_selection_comparison(results):
    """Legacy alias."""
    print_comprehensive_comparison(results)
