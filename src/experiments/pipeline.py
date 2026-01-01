import os
import numpy as np

from src.data.loader import load_orl_faces
from src.data.preprocess import normalize_faces

from src.pca.incremental import IncrementalPCA
from src.pca.batch import BatchPCA

from src.rank.selection import compare_all_rank_methods, print_comprehensive_comparison

from src.diagnostics.assumptions import (
    validate_gavish_donoho_assumptions,
    estimate_noise_from_residuals
)

from src.visualization.assumptions import plot_residual_diagnostics
from src.visualization.rank import plot_all_rank_visualizations

from src.experiments.compare import compare_pca_methods, print_comparison_results
from src.pca.metrics import subspace_distance


# -----------------------------
# Helper Steps
# -----------------------------

def step_load_data(config):
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    faces, labels, is_real = load_orl_faces(config.data_dir)

    if not is_real:
        print("\n   ⚠️  WARNING: Using synthetic data for demonstration")
    else:
        print("\n   ✅ Using REAL ORL Face Database")

    print(f"\n   Dataset shape: {faces.shape}")
    print(f"   Number of samples: {faces.shape[0]}")
    print(f"   Features per sample: {faces.shape[1]} (92×112 pixels)")

    return faces, labels, is_real


def step_preprocess(faces):
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING (Mean Centering)")
    print("="*80)

    centered_faces, mean_face = normalize_faces(faces)

    print("\n   ✅ Data centered (zero mean)")
    print(f"   Mean face min: {mean_face.min():.4f}")
    print(f"   Mean face max: {mean_face.max():.4f}")
    print(f"   Centered data mean: {np.mean(centered_faces):.2e}")

    return centered_faces, mean_face


def step_validate_assumptions(centered_faces, config):
    print("\n" + "="*80)
    print("STEP 3: VALIDATING GAVISH-DONOHO ASSUMPTIONS")
    print("="*80)

    print("\n   Testing if data satisfies Gaussian white noise assumption...")

    n_components_for_noise = min(centered_faces.shape) // 4

    validation_results = validate_gavish_donoho_assumptions(
        centered_faces,
        n_components=n_components_for_noise,
        alpha=config.alpha,
        max_lags=config.max_lags
    )

    print("\n   Generating diagnostic plots...")
    residuals, residuals_flat = estimate_noise_from_residuals(
        centered_faces,
        n_components_for_noise
    )

    plot_residual_diagnostics(
        residuals=residuals,
        residuals_flat=residuals_flat,
        max_lags=config.max_lags,
        save_dir=config.output_dir
    )

    return validation_results


def step_rank_selection(centered_faces, config):
    print("\n" + "="*80)
    print("STEP 4: COMPREHENSIVE RANK SELECTION (6 Methods)")
    print("="*80)

    rank_results = compare_all_rank_methods(
        centered_faces,
        thresholds=config.thresholds
    )

    print_comprehensive_comparison(rank_results)

    print("\n   Generating comprehensive visualizations...")
    plot_all_rank_visualizations(
        rank_results["singular_values"],
        rank_results,
        save_dir=config.output_dir
    )

    # Select k based on config.energy_threshold
    k_selected = rank_results["energy_methods"][config.energy_threshold]["n_components"]

    print("\n" + "="*80)
    print("STEP 5: SELECTING OPTIMAL RANK")
    print("="*80)

    print(f"\n   🎯 SELECTED: k = {k_selected} ({int(config.energy_threshold*100)}% Energy)")
    return rank_results, k_selected


def step_compare_pca(centered_faces, n_components, config):
    print("\n" + "="*80)
    print("STEP 6: INCREMENTAL PCA vs BATCH PCA COMPARISON")
    print("="*80)

    print(f"\n   Configuration:")
    print(f"   - Number of components: {n_components}")
    print(f"   - Batch size: {config.batch_size}")

    inc_pca = IncrementalPCA(n_components=n_components)
    batch_pca = BatchPCA(n_components=n_components)

    print("\n   Running PCA comparison...")
    results = compare_pca_methods(
        X=centered_faces,
        batch_size=config.batch_size,
        incremental_pca=inc_pca,
        batch_pca=batch_pca
    )

    # Subspace distance
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    subspace_dist = subspace_distance(inc_components.T, batch_components.T)

    print_comparison_results(results)

    return results, subspace_dist, inc_pca, batch_pca


def step_summary(faces, rank_results, validation_results, results, selected_rank, subspace_dist, config):
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)

    print(f"\n   Dataset:")
    print(f"   - Samples: {faces.shape[0]}")
    print(f"   - Dimensions: {faces.shape[1]}")

    print(f"\n   Rank Selection Results (6 Methods):")
    for threshold in config.thresholds:
        print(f"   - {int(threshold*100)}% Energy: k = {rank_results['energy_methods'][threshold]['n_components']}")
    print(f"   ⭐ Selected: {selected_rank} ({int(config.energy_threshold*100)}% Energy)")

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

    print(f"\n   Output Files in '{config.output_dir}/':")
    print("   - residual_diagnostics.png")
    print("   - scree_plot_with_elbows.png")
    print("   - rank_method_comparison.png")
    print("   - rank_consensus.png")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\n   ✅ Ready for report and presentation\n")


# -----------------------------
# Main Pipeline
# -----------------------------

def run_full_analysis(config):
    """
    Run the full analysis pipeline.
    Returns a dict of results for notebooks/tests.
    """
    print("="*80)
    print("Incremental PCA using Brand's Algorithm")
    print("ORL Face Database Analysis with Comprehensive Rank Selection")
    print("="*80)

    os.makedirs(config.output_dir, exist_ok=True)

    faces, labels, is_real = step_load_data(config)
    centered_faces, mean_face = step_preprocess(faces)

    validation_results = step_validate_assumptions(centered_faces, config)
    rank_results, selected_rank = step_rank_selection(centered_faces, config)

    pca_results, subspace_dist, inc_pca, batch_pca = step_compare_pca(
        centered_faces,
        selected_rank,
        config
    )

    step_summary(
        faces,
        rank_results,
        validation_results,
        pca_results,
        selected_rank,
        subspace_dist,
        config
    )

    return {
        "faces": faces,
        "labels": labels,
        "is_real": is_real,
        "centered_faces": centered_faces,
        "mean_face": mean_face,
        "validation_results": validation_results,
        "rank_results": rank_results,
        "selected_rank": selected_rank,
        "pca_results": pca_results,
        "subspace_distance": subspace_dist
    }
