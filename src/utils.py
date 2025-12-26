"""
Utility functions for PCA evaluation and benchmarking.
"""
import numpy as np
import time


def reconstruction_error(X_original, X_reconstructed):
    """
    Calculate the reconstruction error between original and reconstructed data.
    
    Uses Frobenius norm (sum of squared differences).
    
    Parameters:
    -----------
    X_original : array-like, shape (n_samples, n_features)
        Original data
    X_reconstructed : array-like, shape (n_samples, n_features)
        Reconstructed data
        
    Returns:
    --------
    error : float
        Frobenius norm of the difference
    """
    diff = X_original - X_reconstructed
    return np.linalg.norm(diff, 'fro')


def normalized_reconstruction_error(X_original, X_reconstructed):
    """
    Calculate normalized reconstruction error.
    
    Normalizes by the Frobenius norm of the original data.
    
    Parameters:
    -----------
    X_original : array-like, shape (n_samples, n_features)
        Original data
    X_reconstructed : array-like, shape (n_samples, n_features)
        Reconstructed data
        
    Returns:
    --------
    normalized_error : float
        Normalized reconstruction error
    """
    error = reconstruction_error(X_original, X_reconstructed)
    norm_original = np.linalg.norm(X_original, 'fro')
    return error / norm_original if norm_original > 0 else 0


def mean_squared_error(X_original, X_reconstructed):
    """
    Calculate mean squared error per element.
    
    Parameters:
    -----------
    X_original : array-like, shape (n_samples, n_features)
        Original data
    X_reconstructed : array-like, shape (n_samples, n_features)
        Reconstructed data
        
    Returns:
    --------
    mse : float
        Mean squared error
    """
    return np.mean((X_original - X_reconstructed) ** 2)


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        
    def get_elapsed(self):
        """Get elapsed time in seconds."""
        return self.elapsed


def benchmark_pca(pca_func, X, batch_size=None, **kwargs):
    """
    Benchmark a PCA function/method.
    
    Parameters:
    -----------
    pca_func : callable
        Function that performs PCA fitting
    X : array-like, shape (n_samples, n_features)
        Data to fit
    batch_size : int, optional
        If provided, data is processed in batches (for incremental PCA)
    **kwargs : dict
        Additional arguments to pass to pca_func
        
    Returns:
    --------
    result : dict
        Dictionary containing timing and performance metrics
    """
    result = {
        'fit_time': 0,
        'transform_time': 0,
        'total_time': 0
    }
    
    with Timer() as fit_timer:
        if batch_size is not None:
            # Incremental fitting
            n_samples = X.shape[0]
            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                pca_func(batch)
        else:
            # Batch fitting
            pca_func(X)
    
    result['fit_time'] = fit_timer.get_elapsed()
    result['total_time'] = fit_timer.get_elapsed()
    
    return result


def compare_pca_methods(X, n_components, batch_size, incremental_pca, batch_pca):
    """
    Compare incremental and batch PCA methods.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to process
    n_components : int
        Number of principal components
    batch_size : int
        Batch size for incremental PCA
    incremental_pca : IncrementalPCA
        Incremental PCA instance
    batch_pca : BatchPCA
        Batch PCA instance
        
    Returns:
    --------
    results : dict
        Dictionary containing comparison metrics
    """
    results = {
        'incremental': {},
        'batch': {}
    }
    
    # Benchmark incremental PCA
    print("Running Incremental PCA...")
    with Timer() as inc_timer:
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            incremental_pca.partial_fit(batch)
    
    results['incremental']['fit_time'] = inc_timer.get_elapsed()
    
    # Transform and reconstruct with incremental PCA
    with Timer() as inc_transform_timer:
        X_inc_transformed = incremental_pca.transform(X)
        X_inc_reconstructed = incremental_pca.inverse_transform(X_inc_transformed)
    
    results['incremental']['transform_time'] = inc_transform_timer.get_elapsed()
    results['incremental']['reconstruction_error'] = reconstruction_error(X, X_inc_reconstructed)
    results['incremental']['normalized_error'] = normalized_reconstruction_error(X, X_inc_reconstructed)
    results['incremental']['mse'] = mean_squared_error(X, X_inc_reconstructed)
    
    # Benchmark batch PCA
    print("Running Batch PCA...")
    with Timer() as batch_timer:
        batch_pca.fit(X)
    
    results['batch']['fit_time'] = batch_timer.get_elapsed()
    
    # Transform and reconstruct with batch PCA
    with Timer() as batch_transform_timer:
        X_batch_transformed = batch_pca.transform(X)
        X_batch_reconstructed = batch_pca.inverse_transform(X_batch_transformed)
    
    results['batch']['transform_time'] = batch_transform_timer.get_elapsed()
    results['batch']['reconstruction_error'] = reconstruction_error(X, X_batch_reconstructed)
    results['batch']['normalized_error'] = normalized_reconstruction_error(X, X_batch_reconstructed)
    results['batch']['mse'] = mean_squared_error(X, X_batch_reconstructed)
    
    # Calculate speedup
    results['speedup'] = results['batch']['fit_time'] / results['incremental']['fit_time']
    
    return results


def print_comparison_results(results):
    """
    Pretty print comparison results.
    
    Parameters:
    -----------
    results : dict
        Results from compare_pca_methods
    """
    print("\n" + "="*60)
    print("PCA METHOD COMPARISON RESULTS")
    print("="*60)
    
    print("\nIncremental PCA:")
    print(f"  Fit time: {results['incremental']['fit_time']:.4f} seconds")
    print(f"  Transform time: {results['incremental']['transform_time']:.4f} seconds")
    print(f"  Reconstruction error: {results['incremental']['reconstruction_error']:.4f}")
    print(f"  Normalized error: {results['incremental']['normalized_error']:.6f}")
    print(f"  Mean squared error: {results['incremental']['mse']:.6f}")
    
    print("\nBatch PCA:")
    print(f"  Fit time: {results['batch']['fit_time']:.4f} seconds")
    print(f"  Transform time: {results['batch']['transform_time']:.4f} seconds")
    print(f"  Reconstruction error: {results['batch']['reconstruction_error']:.4f}")
    print(f"  Normalized error: {results['batch']['normalized_error']:.6f}")
    print(f"  Mean squared error: {results['batch']['mse']:.6f}")
    
    print("\nPerformance Comparison:")
    print(f"  Speedup (Batch/Incremental): {results['speedup']:.2f}x")
    
    if results['speedup'] > 1:
        print(f"  Incremental PCA is {results['speedup']:.2f}x faster!")
    else:
        print(f"  Batch PCA is {1/results['speedup']:.2f}x faster!")
    
    print("="*60)
