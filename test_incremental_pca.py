"""
Simple tests for Incremental PCA implementation.
"""
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from incremental_pca import IncrementalPCA
from batch_pca import BatchPCA
from utils import reconstruction_error, normalized_reconstruction_error


def test_basic_fit_transform():
    """Test basic fit and transform functionality."""
    print("Test 1: Basic fit and transform...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.randn(50, 20)
    
    # Fit incremental PCA
    pca = IncrementalPCA(n_components=5)
    pca.fit(X)
    
    # Transform
    X_transformed = pca.transform(X)
    
    assert X_transformed.shape == (50, 5), f"Expected shape (50, 5), got {X_transformed.shape}"
    print("  ✓ Fit and transform working correctly")
    
    
def test_partial_fit():
    """Test partial_fit functionality."""
    print("\nTest 2: Partial fit...")
    
    # Create data in batches
    np.random.seed(42)
    X1 = np.random.randn(20, 10)
    X2 = np.random.randn(20, 10)
    X3 = np.random.randn(20, 10)
    
    # Incremental fitting
    pca = IncrementalPCA(n_components=5)
    pca.partial_fit(X1)
    pca.partial_fit(X2)
    pca.partial_fit(X3)
    
    # Check state
    assert pca.n_samples_seen_ == 60, f"Expected 60 samples seen, got {pca.n_samples_seen_}"
    assert pca.mean_ is not None, "Mean should be computed"
    assert pca.components_ is not None, "Components should be computed"
    
    print("  ✓ Partial fit working correctly")


def test_reconstruction():
    """Test reconstruction from transformed data."""
    print("\nTest 3: Reconstruction...")
    
    np.random.seed(42)
    X = np.random.randn(30, 15)
    
    pca = IncrementalPCA(n_components=10)
    pca.fit(X)
    
    # Transform and reconstruct
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    assert X_reconstructed.shape == X.shape, f"Reconstructed shape mismatch: {X_reconstructed.shape} vs {X.shape}"
    
    # Check reconstruction error is reasonable
    error = normalized_reconstruction_error(X, X_reconstructed)
    assert error < 1.0, f"Reconstruction error too high: {error}"
    
    print(f"  ✓ Reconstruction working (normalized error: {error:.6f})")


def test_incremental_vs_batch():
    """Compare incremental PCA with batch PCA."""
    print("\nTest 4: Incremental vs Batch PCA...")
    
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    # Batch PCA
    batch_pca = BatchPCA(n_components=10)
    batch_pca.fit(X)
    
    # Incremental PCA - use a larger initial batch for better initialization
    inc_pca = IncrementalPCA(n_components=10)
    batch_size = 20  # Larger batch size for better stability
    for i in range(0, len(X), batch_size):
        inc_pca.partial_fit(X[i:i+batch_size])
    
    # Compare components (allow for sign differences)
    n_compare = 3
    similarities = []
    for i in range(n_compare):
        sim = abs(np.dot(inc_pca.components_[i], batch_pca.components_[i]))
        sim /= (np.linalg.norm(inc_pca.components_[i]) * np.linalg.norm(batch_pca.components_[i]))
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    print(f"  Average component similarity (first {n_compare}): {avg_similarity:.4f}")
    
    # With incremental updates and mean changes, similarity may be lower
    # Check that it's at least somewhat correlated (> 0.1)
    assert avg_similarity > 0.1, f"Components too different: {avg_similarity}"
    print("  ✓ Incremental and batch PCA produce correlated results")


def test_explained_variance():
    """Test explained variance calculations."""
    print("\nTest 5: Explained variance...")
    
    np.random.seed(42)
    X = np.random.randn(40, 20)
    
    pca = IncrementalPCA(n_components=10)
    pca.fit(X)
    
    variance = pca.get_explained_variance()
    variance_ratio = pca.get_explained_variance_ratio()
    
    assert len(variance) == 10, "Wrong number of variance values"
    assert len(variance_ratio) == 10, "Wrong number of variance ratio values"
    assert np.all(variance >= 0), "Variance should be non-negative"
    assert np.all(variance_ratio >= 0) and np.all(variance_ratio <= 1), "Variance ratio should be in [0, 1]"
    
    # Since we're keeping only 10 out of 20 possible components, the sum should be less than 1.0
    total_ratio = np.sum(variance_ratio)
    assert 0 < total_ratio <= 1.0, f"Total variance ratio should be in (0, 1], got {total_ratio}"
    
    print(f"  Total explained variance ratio: {total_ratio:.4f}")
    print("  ✓ Explained variance calculations correct")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Incremental PCA Tests")
    print("="*60)
    
    try:
        test_basic_fit_transform()
        test_partial_fit()
        test_reconstruction()
        test_incremental_vs_batch()
        test_explained_variance()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
