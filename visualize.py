"""
Visualization examples for Incremental PCA.

This script creates visualizations comparing incremental and batch PCA:
1. Explained variance comparison
2. First few principal components
3. Reconstruction quality
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from incremental_pca import IncrementalPCA
from batch_pca import BatchPCA
from data_loader import load_orl_faces, reshape_to_image, normalize_faces


def visualize_explained_variance(inc_pca, batch_pca, n_components=50):
    """
    Visualize explained variance for both PCA methods.
    
    Parameters:
    -----------
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    n_components : int
        Number of components to visualize
    """
    inc_variance_ratio = inc_pca.get_explained_variance_ratio()
    batch_variance_ratio = batch_pca.explained_variance_ratio_
    
    n = min(n_components, len(inc_variance_ratio), len(batch_variance_ratio))
    components = np.arange(1, n + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Individual explained variance
    plt.subplot(1, 2, 1)
    plt.plot(components, inc_variance_ratio[:n], 'b-o', label='Incremental PCA', markersize=4)
    plt.plot(components, batch_variance_ratio[:n], 'r-s', label='Batch PCA', markersize=4)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(components, np.cumsum(inc_variance_ratio[:n]), 'b-o', label='Incremental PCA', markersize=4)
    plt.plot(components, np.cumsum(batch_variance_ratio[:n]), 'r-s', label='Batch PCA', markersize=4)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('explained_variance.png', dpi=150, bbox_inches='tight')
    print("Saved: explained_variance.png")
    plt.close()


def visualize_principal_components(inc_pca, batch_pca, n_display=5):
    """
    Visualize the first few principal components as images.
    
    Parameters:
    -----------
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    n_display : int
        Number of components to display
    """
    inc_components = inc_pca.get_components()
    batch_components = batch_pca.components_
    
    n = min(n_display, inc_components.shape[0], batch_components.shape[0])
    
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    
    for i in range(n):
        # Incremental PCA component
        component_img = reshape_to_image(inc_components[i])
        axes[0, i].imshow(component_img, cmap='gray')
        axes[0, i].set_title(f'Inc PC {i+1}')
        axes[0, i].axis('off')
        
        # Batch PCA component
        component_img = reshape_to_image(batch_components[i])
        axes[1, i].imshow(component_img, cmap='gray')
        axes[1, i].set_title(f'Batch PC {i+1}')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Incremental', fontsize=12, rotation=0, ha='right')
    axes[1, 0].set_ylabel('Batch', fontsize=12, rotation=0, ha='right')
    
    plt.suptitle('Principal Components (Eigenfaces)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('principal_components.png', dpi=150, bbox_inches='tight')
    print("Saved: principal_components.png")
    plt.close()


def visualize_reconstructions(X, inc_pca, batch_pca, n_display=5):
    """
    Visualize original images and their reconstructions.
    
    Parameters:
    -----------
    X : array-like
        Original data
    inc_pca : IncrementalPCA
        Fitted incremental PCA model
    batch_pca : BatchPCA
        Fitted batch PCA model
    n_display : int
        Number of samples to display
    """
    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(X), n_display, replace=False)
    
    # Get reconstructions
    X_inc_transformed = inc_pca.transform(X[indices])
    X_inc_reconstructed = inc_pca.inverse_transform(X_inc_transformed)
    
    X_batch_transformed = batch_pca.transform(X[indices])
    X_batch_reconstructed = batch_pca.inverse_transform(X_batch_transformed)
    
    fig, axes = plt.subplots(3, n_display, figsize=(3*n_display, 9))
    
    for i in range(n_display):
        # Original
        img = reshape_to_image(X[indices[i]])
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Incremental reconstruction
        img = reshape_to_image(X_inc_reconstructed[i])
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Inc Recon {i+1}')
        axes[1, i].axis('off')
        
        # Batch reconstruction
        img = reshape_to_image(X_batch_reconstructed[i])
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f'Batch Recon {i+1}')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12, rotation=0, ha='right')
    axes[1, 0].set_ylabel('Incremental', fontsize=12, rotation=0, ha='right')
    axes[2, 0].set_ylabel('Batch', fontsize=12, rotation=0, ha='right')
    
    plt.suptitle('Image Reconstruction Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=150, bbox_inches='tight')
    print("Saved: reconstructions.png")
    plt.close()


def main():
    """Main visualization function."""
    print("="*60)
    print("Incremental PCA Visualization")
    print("="*60)
    
    # Load data - FIXED: handle 3 return values
    print("\n1. Loading data...")
    faces, labels, is_real = load_orl_faces('data/ORL_Faces')
    
    if not is_real:
        print("   ⚠️  WARNING: Using synthetic data for demonstration")
    else:
        print("   ✅ Using REAL ORL Face Database")
    
    print(f"   Loaded {len(faces)} face images")
    
    # Mean Centering (REQUIRED for PCA)
    print("\n2. Preprocessing: Mean Centering...")
    centered_faces, mean_face = normalize_faces(faces)
    
    # Set parameters
    n_components = 50
    batch_size = 10
    
    print(f"\n3. Training PCA models...")
    print(f"   Components: {n_components}")
    print(f"   Batch size: {batch_size}")
    
    # Train incremental PCA - FIXED: use centered_faces
    inc_pca = IncrementalPCA(n_components=n_components)
    for i in range(0, len(centered_faces), batch_size):
        inc_pca.partial_fit(centered_faces[i:i+batch_size])
    print("   ✓ Incremental PCA trained")
    
    # Train batch PCA - FIXED: use centered_faces
    batch_pca = BatchPCA(n_components=n_components)
    batch_pca.fit(centered_faces)
    print("   ✓ Batch PCA trained")
    
    # Create visualizations - FIXED: use centered_faces
    print("\n4. Creating visualizations...")
    
    visualize_explained_variance(inc_pca, batch_pca, n_components)
    visualize_principal_components(inc_pca, batch_pca, n_display=5)
    visualize_reconstructions(centered_faces, inc_pca, batch_pca, n_display=5)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("Generated files:")
    print("  - explained_variance.png")
    print("  - principal_components.png")
    print("  - reconstructions.png")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
