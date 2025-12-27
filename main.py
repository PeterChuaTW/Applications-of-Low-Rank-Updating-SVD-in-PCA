"""Main script demonstrating Incremental PCA using Brand's algorithm.

This script:
1. Loads the ORL Face Database (or generates synthetic data)
2. Validates assumptions for advanced rank selection methods
3. Compares different rank selection strategies
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
    compare_rank_selection_methods,
    print_rank_selection_comparison
)
from noise_analysis import (
    validate_gavish_donoho_assumptions,
    estimate_noise_from_residuals
)
from visualize_assumptions import (
    plot_residual_diagnostics,
    plot_rank_comparison
)


def main():
    """Main execution function."""
    
    print("="*70)
    print("Incremental PCA using Brand's Algorithm")
    print("ORL Face Database Analysis with Advanced Rank Selection")
    print("="*70)
    
    # Create output directory for plots
    os.makedirs('output', exist_ok=True)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING (Mean Centering)")
    print("="*70)
    
    centered_faces, mean_face = normalize_faces(faces)
    print("\n   ✅ Data centered (zero mean)")
    print(f"   Mean face min: {mean_face.min():.4f}")
    print(f"   Mean face max: {mean_face.max():.4f}")
    print(f"   Centered data mean: {np.mean(centered_faces):.2e}")
    
    # ========================================================================
    # STEP 3: VALIDATE GAVISH-DONOHO ASSUMPTIONS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: VALIDATING GAVISH-DONOHO ASSUMPTIONS")
    print("="*70)
    print("\n   Testing if data satisfies Gaussian white noise assumption...")
    
    # Use a reasonable number of components for noise estimation
    # Rule of thumb: 25% of min dimension
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
    # STEP 4: RANK SELECTION COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: RANK SELECTION METHOD COMPARISON")
    print("="*70)
    
    rank_results = compare_rank_selection_methods(
        centered_faces,
        thresholds=(0.90, 0.95, 0.99)
    )
    
    print_rank_selection_comparison(rank_results)
    
    # Visualize rank comparison
    print("   Generating rank comparison plot...")
    plot_rank_comparison(rank_results, save_dir='output')
    
    # ========================================================================
    # STEP 5: SELECT OPTIMAL RANK
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: SELECTING OPTIMAL RANK")
    print("="*70)
    
    # Decision logic based on validation results
    if validation_results['assumptions_met']:
        # Use Gavish-Donoho if assumptions are met
        n_components = rank_results['gavish_donoho']['n_components']
        selection_method = "Gavish-Donoho (assumptions satisfied)"
        print(f"\n   ✅ Using Gavish-Donoho: k = {n_components}")
    else:
        # Fallback to 95% energy if assumptions violated
        n_components = rank_results['energy_methods'][0.95]['n_components']
        selection_method = "95% Cumulative Energy (fallback)"
        print(f"\n   ⚠️  Using 95% Energy (Gavish-Donoho assumptions not fully met): k = {n_components}")
    
    print(f"   Selection method: {selection_method}")
    print(f"   Final rank (k): {n_components}")
    
    # For comparison, also try with both methods
    k_energy_95 = rank_results['energy_methods'][0.95]['n_components']
    k_gavish = rank_results['gavish_donoho']['n_components']
    
    # ========================================================================
    # STEP 6: PCA COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: INCREMENTAL PCA vs BATCH PCA COMPARISON")
    print("="*70)
    
    batch_size = 10  # Batch size for incremental PCA
    
    print(f"\n   Configuration:")
    print(f"   - Number of components: {n_components}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Selection method: {selection_method}")
    
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
    
    # Print results
    print_comparison_results(results)
    
    # ========================================================================
    # STEP 7: ADDITIONAL ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: ADDITIONAL ANALYSIS")
    print("="*70)
    
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
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    
    n_compare = min(5, n_components)
    print(f"\n   Component similarity (first {n_compare} components):")
    
    similarities = []
    for i in range(n_compare):
        # Cosine similarity (components might differ in sign)
        similarity = abs(np.dot(inc_components[i], batch_components[i]))
        similarity /= (np.linalg.norm(inc_components[i]) * np.linalg.norm(batch_components[i]))
        similarities.append(similarity)
        print(f"   - Component {i+1}: {similarity:.6f}")
    
    print(f"\n   Average similarity: {np.mean(similarities):.6f}")
    
    # ========================================================================
    # STEP 8: SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n   Dataset:")
    print(f"   - Samples: {faces.shape[0]}")
    print(f"   - Dimensions: {faces.shape[1]}")
    
    print(f"\n   Rank Selection:")
    print(f"   - 90% Energy: k = {rank_results['energy_methods'][0.90]['n_components']}")
    print(f"   - 95% Energy: k = {k_energy_95}")
    print(f"   - 99% Energy: k = {rank_results['energy_methods'][0.99]['n_components']}")
    print(f"   - Gavish-Donoho: k = {k_gavish}")
    print(f"   - Selected: k = {n_components} ({selection_method})")
    
    print(f"\n   Gavish-Donoho Validation:")
    print(f"   - Gaussian test: {'✓ PASS' if validation_results['gaussian_test']['is_gaussian'] else '✗ FAIL'}")
    print(f"   - White noise test: {'✓ PASS' if validation_results['white_noise_test']['is_white'] else '✗ FAIL'}")
    print(f"   - Overall: {'✓ PASS' if validation_results['assumptions_met'] else '✗ FAIL'}")
    
    print(f"\n   Performance:")
    print(f"   - Incremental PCA time: {results['incremental_time']:.4f}s")
    print(f"   - Batch PCA time: {results['batch_time']:.4f}s")
    print(f"   - Speedup: {results['batch_time']/results['incremental_time']:.2f}x")
    
    print(f"\n   Accuracy:")
    print(f"   - Reconstruction error: {results['reconstruction_error']:.6f}")
    print(f"   - Subspace distance: {results['subspace_distance']:.6f}")
    
    print(f"\n   Output Files:")
    print(f"   - output/residual_diagnostics.png (Q-Q, histogram, ACF, heatmap)")
    print(f"   - output/rank_comparison.png (rank selection comparison)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\n   ✅ All results saved to 'output/' directory")
    print("   ✅ Ready for report and presentation\n")
    
    return {
        'pca_results': results,
        'rank_results': rank_results,
        'validation_results': validation_results,
        'selected_rank': n_components,
        'selection_method': selection_method
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
