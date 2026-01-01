"""
Unified rank selection comparison module.

Combines:
- Energy threshold methods
- Gavish-Donoho threshold
- Kneedle algorithm
- L-method
"""
import numpy as np
import warnings

from .energy import determine_n_components_by_energy
from .gavish import gavish_donoho_threshold
from .elbow import kneedle_algorithm, l_method


def compare_all_rank_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    """
    Compare ALL rank selection methods.
    """
    X = np.asarray(X)

    # Compute SVD once
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    results = {
        'singular_values': s,
        'energy_methods': {},
        'gavish_donoho': {},
        'kneedle': {},
        'l_method': {}
    }

    variance = s ** 2
    total_variance = np.sum(variance)

    # Energy methods
    for threshold in thresholds:
        n_comp, cum_var_ratio, explained_var = determine_n_components_by_energy(s, threshold=threshold)
        results['energy_methods'][threshold] = {
            'n_components': n_comp,
            'explained_variance': explained_var,
            'method_name': f"{int(threshold*100)}% Energy"
        }

    # Gavish-Donoho
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

    # Kneedle
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

    # L-method
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
    Print comprehensive comparison of rank selection methods.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RANK SELECTION COMPARISON")
    print("="*80)

    all_methods = []

    for threshold, data in sorted(results['energy_methods'].items()):
        all_methods.append({
            'name': data['method_name'],
            'category': 'Empirical',
            'k': data['n_components'],
            'variance': data['explained_variance']
        })

    if results['gavish_donoho'].get('n_components'):
        all_methods.append({
            'name': 'Gavish-Donoho',
            'category': 'Statistical',
            'k': results['gavish_donoho']['n_components'],
            'variance': results['gavish_donoho']['explained_variance']
        })

    if results['kneedle'].get('n_components'):
        all_methods.append({
            'name': 'Kneedle',
            'category': 'Geometric',
            'k': results['kneedle']['n_components'],
            'variance': results['kneedle']['explained_variance']
        })

    if results['l_method'].get('n_components'):
        all_methods.append({
            'name': 'L-Method',
            'category': 'Statistical',
            'k': results['l_method']['n_components'],
            'variance': results['l_method']['explained_variance']
        })

    print("\n" + "-"*80)
    print(f"{'Method':<20} {'Category':<15} {'k':<10} {'Explained Variance':<20}")
    print("-"*80)

    for method in all_methods:
        print(f"{method['name']:<20} {method['category']:<15} {method['k']:<10} {method['variance']:>6.2%}")

    print("-"*80)

    k_values = [m['k'] for m in all_methods]

    print("\n" + "="*80)
    print("CONSENSUS ANALYSIS")
    print("="*80)

    print(f"\n  Mean:   {np.mean(k_values):.1f}")
    print(f"  Median: {int(np.median(k_values))}")
    print(f"  Range:  [{min(k_values)}, {max(k_values)}]")
    print(f"  Std:    {np.std(k_values):.2f}")

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

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    median_k = int(np.median(k_values))
    k_95 = results['energy_methods'][0.95]['n_components']

    print(f"\n  PRIMARY:    Use median consensus k = {median_k}")
    print(f"  BASELINE:   Use 95% Energy k = {k_95} (standard)")

    if results['kneedle'].get('n_components'):
        print(f"  GEOMETRIC:  Use Kneedle k = {results['kneedle']['n_components']}")

    if results['gavish_donoho'].get('n_components'):
        print(f"  DENOISING:  Use Gavish-Donoho k = {results['gavish_donoho']['n_components']}")

    print("\n" + "="*80 + "\n")


# Legacy function for backward compatibility
def compare_rank_selection_methods(X, thresholds=(0.90, 0.95, 0.99), sigma=None):
    return compare_all_rank_methods(X, thresholds, sigma)


def print_rank_selection_comparison(results):
    print_comprehensive_comparison(results)
