"""Main script demonstrating Incremental PCA using Brand's algorithm.

This script:
1. Loads the ORL Face Database (or generates synthetic data)
2. Compares Incremental PCA vs Batch PCA
3. Evaluates reconstruction error
4. Benchmarks performance
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


def main():
    """Main execution function."""
    
    print("="*60)
    print("Incremental PCA using Brand's Algorithm")
    print("ORL Face Database Analysis")
    print("="*60)
    
    # Load data - FIXED: handle 3 return values
    print("\n1. Loading ORL Face Database...")
    faces, labels, is_real = load_orl_faces('data/ORL_Faces')
    
    if not is_real:
        print("   ⚠️  WARNING: Using synthetic data for demonstration")
        print("   Please download ORL Database manually if needed")
    else:
        print("   ✅ Using REAL ORL Face Database")
    
    print(f"   Dataset shape: {faces.shape}")
    print(f"   Number of samples: {faces.shape[0]}")
    print(f"   Features per sample: {faces.shape[1]} (92x112 pixels)")
    
    # Normalize data (Mean Centering - REQUIRED for PCA)
    print("\n2. Preprocessing: Mean Centering...")
    centered_faces, mean_face = normalize_faces(faces)
    
    # Set parameters
    n_components = 50  # Number of principal components
    batch_size = 10    # Batch size for incremental PCA
    
    print(f"\n3. Configuration:")
    print(f"   Number of components: {n_components}")
    print(f"   Batch size for incremental PCA: {batch_size}")
    
    # Initialize PCA models
    print("\n4. Initializing PCA models...")
    inc_pca = IncrementalPCA(n_components=n_components)
    batch_pca = BatchPCA(n_components=n_components)
    
    # Compare methods - FIXED: use centered_faces
    print("\n5. Comparing PCA methods...")
    results = compare_pca_methods(
        X=centered_faces,
        n_components=n_components,
        batch_size=batch_size,
        incremental_pca=inc_pca,
        batch_pca=batch_pca
    )
    
    # Print results
    print_comparison_results(results)
    
    # Additional analysis
    print("\n6. Additional Analysis:")
    
    # Explained variance
    inc_variance_ratio = inc_pca.get_explained_variance_ratio()
    batch_variance_ratio = batch_pca.explained_variance_ratio_
    
    print(f"\n   Explained variance by first 10 components:")
    print(f"   Incremental PCA: {np.sum(inc_variance_ratio[:10]):.4f}")
    print(f"   Batch PCA: {np.sum(batch_variance_ratio[:10]):.4f}")
    
    print(f"\n   Total explained variance ({n_components} components):")
    print(f"   Incremental PCA: {np.sum(inc_variance_ratio):.4f}")
    print(f"   Batch PCA: {np.sum(batch_variance_ratio):.4f}")
    
    # Component similarity
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    
    # Calculate similarity of first few components
    n_compare = min(5, n_components)
    print(f"\n   Component similarity (first {n_compare} components):")
    for i in range(n_compare):
        # Cosine similarity (components might differ in sign)
        similarity = abs(np.dot(inc_components[i], batch_components[i]))
        similarity /= (np.linalg.norm(inc_components[i]) * np.linalg.norm(batch_components[i]))
        print(f"   Component {i+1}: {similarity:.6f}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
