"""Visualization script for Incremental PCA results.

Generates comparison visualizations:
1. Explained variance comparison
2. Principal components (Eigenfaces)
3. Reconstruction quality comparison

Now integrates with rank selection module to automatically choose k.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, 'src')

from incremental_pca import IncrementalPCA
from batch_pca import BatchPCA
from data_loader import load_orl_faces, normalize_faces
from rank_selection import determine_n_components_by_energy


def visualize_explained_variance(inc_pca, batch_pca, n_components_used, save_path='explained_variance.png'):
    """
    Plot explained variance comparison.
    
    Parameters:
    -----------
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    n_components_used : int
        Number of components used in training
    save_path : str
        Path to save the figure
    """
    inc_var = inc_pca.get_explained_variance_ratio()
    batch_var = batch_pca.explained_variance_ratio_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual explained variance
    x = np.arange(1, len(inc_var) + 1)
    ax1.plot(x, inc_var, 'b-o', label='Incremental PCA', markersize=4, linewidth=1.5)
    ax1.plot(x, batch_var, 'r--s', label='Batch PCA', markersize=4, linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Component Index', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title(f'Explained Variance per Component (k={n_components_used})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    inc_cumsum = np.cumsum(inc_var)
    batch_cumsum = np.cumsum(batch_var)
    
    ax2.plot(x, inc_cumsum, 'b-o', label='Incremental PCA', markersize=4, linewidth=1.5)
    ax2.plot(x, batch_cumsum, 'r--s', label='Batch PCA', markersize=4, linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0.95, color='green', linestyle=':', linewidth=2, label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_principal_components(inc_pca, batch_pca, save_path='principal_components.png'):
    """
    Visualize first few principal components (Eigenfaces).
    
    Parameters:
    -----------
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    save_path : str
        Path to save the figure
    """
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    
    n_show = min(10, inc_components.shape[0])
    
    fig, axes = plt.subplots(2, n_show, figsize=(2*n_show, 4))
    
    for i in range(n_show):
        # Reshape to 92x112 (ORL face dimensions)
        inc_eigenface = inc_components[i].reshape(112, 92)
        batch_eigenface = batch_components[i].reshape(112, 92)
        
        # Incremental PCA
        axes[0, i].imshow(inc_eigenface, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f'Inc PC {i+1}', fontsize=10)
        else:
            axes[0, i].set_title(f'PC {i+1}', fontsize=10)
        
        # Batch PCA
        axes[1, i].imshow(batch_eigenface, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f'Batch PC {i+1}', fontsize=10)
        else:
            axes[1, i].set_title(f'PC {i+1}', fontsize=10)
    
    # Row labels
    axes[0, 0].text(-0.3, 0.5, 'Incremental\nPCA', fontsize=12, 
                    transform=axes[0, 0].transAxes, 
                    rotation=0, va='center', ha='right', fontweight='bold')
    axes[1, 0].text(-0.3, 0.5, 'Batch\nPCA', fontsize=12, 
                    transform=axes[1, 0].transAxes, 
                    rotation=0, va='center', ha='right', fontweight='bold')
    
    plt.suptitle('Principal Components (Eigenfaces)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_reconstructions(inc_pca, batch_pca, X_centered, mean_face, n_samples=5, save_path='reconstructions.png'):
    """
    Visualize original vs reconstructed images.
    
    Parameters:
    -----------
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    X_centered : array
        Mean-centered data
    mean_face : array
        Mean face for reconstruction
    n_samples : int
        Number of samples to show
    save_path : str
        Path to save the figure
    """
    # Select samples
    indices = np.linspace(0, X_centered.shape[0]-1, n_samples, dtype=int)
    
    # Reconstruct
    X_inc_transformed = inc_pca.transform(X_centered)
    X_inc_reconstructed = inc_pca.inverse_transform(X_inc_transformed)
    
    X_batch_transformed = batch_pca.transform(X_centered)
    X_batch_reconstructed = batch_pca.inverse_transform(X_batch_transformed)
    
    # Plot
    fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
    
    for idx, sample_idx in enumerate(indices):
        # Original (add back mean)
        original = (X_centered[sample_idx] + mean_face).reshape(112, 92)
        inc_recon = (X_inc_reconstructed[sample_idx] + mean_face).reshape(112, 92)
        batch_recon = (X_batch_reconstructed[sample_idx] + mean_face).reshape(112, 92)
        
        axes[0, idx].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Original {idx+1}', fontsize=10)
        
        axes[1, idx].imshow(inc_recon, cmap='gray', vmin=0, vmax=255)
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Inc Recon {idx+1}', fontsize=10)
        
        axes[2, idx].imshow(batch_recon, cmap='gray', vmin=0, vmax=255)
        axes[2, idx].axis('off')
        axes[2, idx].set_title(f'Batch Recon {idx+1}', fontsize=10)
    
    # Row labels
    axes[0, 0].text(-0.2, 0.5, 'Original', fontsize=12, 
                    transform=axes[0, 0].transAxes, 
                    rotation=0, va='center', ha='right', fontweight='bold')
    axes[1, 0].text(-0.2, 0.5, 'Incremental\nPCA', fontsize=12, 
                    transform=axes[1, 0].transAxes, 
                    rotation=0, va='center', ha='right', fontweight='bold')
    axes[2, 0].text(-0.2, 0.5, 'Batch\nPCA', fontsize=12, 
                    transform=axes[2, 0].transAxes, 
                    rotation=0, va='center', ha='right', fontweight='bold')
    
    plt.suptitle('Image Reconstruction Comparison', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main visualization function with automatic rank selection."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize Incremental PCA with automatic rank selection')
    parser.add_argument('--components', type=int, default=None,
                       help='Number of components (if not specified, uses 95%% Energy method)')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Variance threshold for automatic selection (default: 0.95)')
    args = parser.parse_args()
    
    print("="*60)
    print("Incremental PCA Visualization")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    faces, labels, is_real = load_orl_faces('data/ORL_Faces')
    
    if is_real:
        print("   ✅ Using REAL ORL Face Database")
    else:
        print("   ⚠️  Using synthetic data (download ORL manually for real data)")
    
    print(f"   Loaded {faces.shape[0]} face images")
    
    # Preprocess
    print("\n2. Preprocessing: Mean Centering...")
    centered_faces, mean_face = normalize_faces(faces)
    
    # Automatic rank selection
    if args.components is None:
        print(f"\n3. Automatic Rank Selection (Target: {args.threshold*100:.0f}% variance)...")
        
        # Compute SVD for rank selection
        _, s, _ = np.linalg.svd(centered_faces, full_matrices=False)
        
        # Use energy method - correct function call
        n_components, _, variance = determine_n_components_by_energy(s, threshold=args.threshold)
        
        print(f"   ✅ Selected k = {n_components} ({variance*100:.2f}% variance)")
        print(f"      Method: {args.threshold*100:.0f}% Cumulative Energy")
    else:
        n_components = args.components
        print(f"\n3. Using specified k = {n_components}...")
    
    # Train PCA models
    print("\n4. Training PCA models...")
    print(f"   Components: {n_components}")
    
    batch_size = 10
    print(f"   Batch size: {batch_size}")
    
    inc_pca = IncrementalPCA(n_components=n_components)
    batch_pca = BatchPCA(n_components=n_components)
    
    # Incremental fit
    for i in range(0, centered_faces.shape[0], batch_size):
        batch = centered_faces[i:i+batch_size]
        inc_pca.partial_fit(batch)
    
    # Batch fit
    batch_pca.fit(centered_faces)
    
    print("   ✅ Incremental PCA trained")
    print("   ✅ Batch PCA trained")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    visualize_explained_variance(inc_pca, batch_pca, n_components)
    visualize_principal_components(inc_pca, batch_pca)
    visualize_reconstructions(inc_pca, batch_pca, centered_faces, mean_face)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("Generated files:")
    print("  - explained_variance.png")
    print("  - principal_components.png")
    print("  - reconstructions.png")
    print("="*60)


if __name__ == "__main__":
    main()
