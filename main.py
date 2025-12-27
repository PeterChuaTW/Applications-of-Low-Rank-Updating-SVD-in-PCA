"""Main script demonstrating Incremental PCA using Brand's algorithm.

This script:
1. Loads the ORL Face Database (or generates synthetic data)
2. Validates assumptions for advanced rank selection methods
3. Compares 6 different rank selection strategies comprehensively
4. Compares Incremental PCA vs Batch PCA
5. Evaluates reconstruction error and performance
6. Generates diagnostic visualizations
"""
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from incremental_pca import IncrementalPCA
from batch_pca import BatchPCA
from data_loader import load_orl_faces, normalize_faces
from utils import (
    compare_pca_methods, 
    print_comparison_results,
    reconstruction_error,
    normalized_reconstruction_error
)
from rank_selection import (
    compare_all_rank_methods,
    print_comprehensive_comparison
)
from noise_analysis import (
    validate_gavish_donoho_assumptions,
    estimate_noise_from_residuals
)
from visualize_assumptions import (
    plot_residual_diagnostics
)
from visualize_rank_selection import (
    plot_all_rank_visualizations
)


def subspace_distance(U1, U2):
    """Calculate principal angle-based subspace distance."""
    # Ensure same number of components
    k = min(U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    
    # Compute singular values of U1^T @ U2
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    
    # Clamp to [0, 1] to avoid numerical errors
    s = np.clip(s, 0, 1)
    
    # Compute principal angles and distance
    angles = np.arccos(s)
    return np.linalg.norm(angles)


def main():
    """Main execution function."""
    
    print("="*80)
    print("Incremental PCA using Brand's Algorithm")
    print("ORL Face Database Analysis with Comprehensive Rank Selection")
    print("="*80)
    
    # Create output directory for plots
    os.makedirs('output', exist_ok=True)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    faces, labels, is_real = load_orl_faces('data/ORL_Faces')
    
    if not is_real:
        print("\n   ⚠️  WARNING: Using synthetic data for demonstration")
        print("   Please download ORL Database manually if needed")
    else:
        print("\n   ✅ Using REAL ORL Face Database")
    
    print(f"\n   Dataset shape: {faces.shape}")
    print(f"   Number of samples: {faces.shape[0]}")
    print(f"   Features per sample: {faces.shape[1]} (92×112 pixels)")
    
    # ========================================================================
    # STEP 2: PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING (Mean Centering)")
    print("="*80)
    
    centered_faces, mean_face = normalize_faces(faces)
    print("\n   ✅ Data centered (zero mean)")
    print(f"   Mean face min: {mean_face.min():.4f}")
    print(f"   Mean face max: {mean_face.max():.4f}")
    print(f"   Centered data mean: {np.mean(centered_faces):.2e}")
    
    # ========================================================================
    # STEP 3: VALIDATE GAVISH-DONOHO ASSUMPTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: VALIDATING GAVISH-DONOHO ASSUMPTIONS")
    print("="*80)
    print("\n   Testing if data satisfies Gaussian white noise assumption...")
    
    # Use a reasonable number of components for noise estimation
    n_components_for_noise = min(centered_faces.shape) // 4
    
    validation_results = validate_gavish_donoho_assumptions(
        centered_faces,
        n_components=n_components_for_noise,
        alpha=0.05,
        max_lags=50
    )
    
    # Generate diagnostic visualizations
    print("\n   Generating diagnostic plots...")
    residuals, residuals_flat = estimate_noise_from_residuals(
        centered_faces, 
        n_components_for_noise
    )
    
    plot_residual_diagnostics(
        residuals=residuals,
        residuals_flat=residuals_flat,
        max_lags=50,
        save_dir='output'
    )
    
    # ========================================================================
    # STEP 4: COMPREHENSIVE RANK SELECTION (6 METHODS)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: COMPREHENSIVE RANK SELECTION (6 Methods)")
    print("="*80)
    print("\n   Comparing:")
    print("   1. 90% Cumulative Energy")
    print("   2. 95% Cumulative Energy")
    print("   3. 99% Cumulative Energy")
    print("   4. Gavish-Donoho Optimal Threshold")
    print("   5. Kneedle Algorithm (Maximum Curvature)")
    print("   6. L-Method (Two-Segment Regression)")
    
    rank_results = compare_all_rank_methods(
        centered_faces,
        thresholds=(0.90, 0.95, 0.99)
    )
    
    print_comprehensive_comparison(rank_results)
    
    # Generate comprehensive visualizations for all 6 methods
    print("\n   Generating comprehensive visualizations...")
    plot_all_rank_visualizations(
        rank_results['singular_values'],
        rank_results,
        save_dir='output'
    )
    
    # ========================================================================
    # STEP 5: SELECT OPTIMAL RANK (95% ENERGY FOR IMAGE RECONSTRUCTION)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: SELECTING OPTIMAL RANK")
    print("="*80)
    
    # For IMAGE RECONSTRUCTION, use 95% Energy (not median consensus)
    k_95 = rank_results['energy_methods'][0.95]['n_components']
    
    # Collect all valid k values for analysis
    k_values = []
    for threshold, data in rank_results['energy_methods'].items():
        k_values.append(data['n_components'])
    
    if rank_results['gavish_donoho'].get('n_components'):
        k_gavish = rank_results['gavish_donoho']['n_components']
        if k_gavish > 1:  # Only include if reasonable
            k_values.append(k_gavish)
    
    if rank_results['kneedle'].get('n_components'):
        k_values.append(rank_results['kneedle']['n_components'])
    
    if rank_results['l_method'].get('n_components'):
        k_l = rank_results['l_method']['n_components']
        if k_l < centered_faces.shape[0]:  # Only include if not trivial
            k_values.append(k_l)
    
    median_k = int(np.median(k_values))
    
    # DECISION: Use 95% Energy for image reconstruction
    n_components = k_95
    selection_rationale = "95% Energy (Image Reconstruction Standard)"
    
    print(f"\n   Rank Selection Analysis:")
    print(f"   - Median consensus: k = {median_k}")
    print(f"   - 95% Energy:       k = {k_95}")
    print(f"   - Range: [{min(k_values)}, {max(k_values)}]")
    
    print(f"\n   🎯 SELECTED: k = {n_components} (95% Energy)")
    print(f"      Rationale: Image reconstruction requires ≥95% variance")
    print(f"      Median consensus (k={median_k}) optimizes denoising, not visual quality")
    
    if not validation_results['assumptions_met']:
        print(f"      Note: Gavish-Donoho assumptions violated (expected for images)")
    
    # ========================================================================
    # STEP 6: PCA COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: INCREMENTAL PCA vs BATCH PCA COMPARISON")
    print("="*80)
    
    batch_size = 10
    
    print(f"\n   Configuration:")
    print(f"   - Number of components: {n_components}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Selection method: {selection_rationale}")
    
    # Initialize PCA models
    print("\n   Initializing PCA models...")
    inc_pca = IncrementalPCA(n_components=n_components)
    batch_pca = BatchPCA(n_components=n_components)
    
    # Compare methods
    print("\n   Running PCA comparison...")
    results = compare_pca_methods(
        X=centered_faces,
        n_components=n_components,
        batch_size=batch_size,
        incremental_pca=inc_pca,
        batch_pca=batch_pca
    )
    
    # Calculate subspace distance
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    subspace_dist = subspace_distance(inc_components.T, batch_components.T)
    
    # Print results
    print_comparison_results(results)
    
    # ========================================================================
    # STEP 7: ADDITIONAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Explained variance
    inc_variance_ratio = inc_pca.get_explained_variance_ratio()
    batch_variance_ratio = batch_pca.explained_variance_ratio_
    
    print(f"\n   Explained variance by first 10 components:")
    print(f"   - Incremental PCA: {np.sum(inc_variance_ratio[:10]):.4f}")
    print(f"   - Batch PCA:       {np.sum(batch_variance_ratio[:10]):.4f}")
    
    print(f"\n   Total explained variance ({n_components} components):")
    print(f"   - Incremental PCA: {np.sum(inc_variance_ratio):.4f}")
    print(f"   - Batch PCA:       {np.sum(batch_variance_ratio):.4f}")
    
    # Component similarity
    n_compare = min(5, n_components)
    print(f"\n   Component similarity (first {n_compare} components):")
    
    similarities = []
    for i in range(n_compare):
        similarity = abs(np.dot(inc_components[i], batch_components[i]))
        similarity /= (np.linalg.norm(inc_components[i]) * np.linalg.norm(batch_components[i]))
        similarities.append(similarity)
        print(f"   - Component {i+1}: {similarity:.6f}")
    
    print(f"\n   Average similarity: {np.mean(similarities):.6f}")
    print(f"   Subspace distance: {subspace_dist:.6f}")
    
    # ========================================================================
    # STEP 8: COMPREHENSIVE SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print(f"\n   Dataset:")
    print(f"   - Samples: {faces.shape[0]}")
    print(f"   - Dimensions: {faces.shape[1]}")
    
    print(f"\n   Rank Selection Results (6 Methods):")
    print(f"   - 90% Energy:     k = {rank_results['energy_methods'][0.90]['n_components']}")
    print(f"   - 95% Energy:     k = {k_95} ⭐ SELECTED")
    print(f"   - 99% Energy:     k = {rank_results['energy_methods'][0.99]['n_components']}")
    
    if rank_results['gavish_donoho'].get('n_components'):
        print(f"   - Gavish-Donoho:  k = {rank_results['gavish_donoho']['n_components']}")
    
    if rank_results['kneedle'].get('n_components'):
        print(f"   - Kneedle:        k = {rank_results['kneedle']['n_components']}")
    
    if rank_results['l_method'].get('n_components'):
        print(f"   - L-Method:       k = {rank_results['l_method']['n_components']}")
    
    print(f"\n   Gavish-Donoho Validation:")
    print(f"   - Gaussian: {'✓ PASS' if validation_results['gaussian_test']['is_gaussian'] else '✗ FAIL'}")
    print(f"   - White Noise: {'✓ PASS' if validation_results['white_noise_test']['is_white'] else '✗ FAIL'}")
    print(f"   - Overall: {'✓ PASS' if validation_results['assumptions_met'] else '✗ FAIL'}")
    
    print(f"\n   Performance (Incremental vs Batch):")
    print(f"   - Incremental time: {results['incremental']['fit_time']:.4f}s")
    print(f"   - Batch time:       {results['batch']['fit_time']:.4f}s")
    print(f"   - Speedup:          {results['speedup']:.2f}x")
    
    print(f"\n   Accuracy:")
    print(f"   - Reconstruction error: {results['incremental']['reconstruction_error']:.4f}")
    print(f"   - Subspace distance:    {subspace_dist:.6f}")
    
    print(f"\n   Output Files:")
    print(f"   📄 output/residual_diagnostics.png")
    print(f"      (Q-Q plot, histogram, ACF, residual heatmap)")
    print(f"   📄 output/scree_plot_with_elbows.png")
    print(f"      (Scree plot with all 6 elbow points marked)")
    print(f"   📄 output/rank_method_comparison.png")
    print(f"      (Bar charts comparing k and variance)")
    print(f"   📄 output/rank_consensus.png")
    print(f"      (Consensus analysis visualization)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\n   ✅ All 6 rank selection methods compared")
    print("   ✅ Results saved to 'output/' directory")
    print("   ✅ Ready for report and presentation\n")
    
    return {
        'pca_results': results,
        'rank_results': rank_results,
        'validation_results': validation_results,
        'selected_rank': n_components,
        'selection_method': selection_rationale,
        'all_k_values': k_values,
        'subspace_distance': subspace_dist
    }


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
