"""
Experiment utilities for comparing PCA methods.
"""
import time
from src.pca.metrics import (
    reconstruction_error,
    normalized_reconstruction_error,
    mean_squared_error
)


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
        return self.elapsed


def compare_pca_methods(X, batch_size, incremental_pca, batch_pca):
    """
    Compare incremental and batch PCA methods.

    Note:
        incremental_pca and batch_pca are already initialized with n_components.
    """
    results = {
        "incremental": {},
        "batch": {}
    }

    print("Running Incremental PCA...")
    with Timer() as inc_timer:
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            incremental_pca.partial_fit(batch)
    results["incremental"]["fit_time"] = inc_timer.get_elapsed()

    with Timer() as inc_transform_timer:
        X_inc_transformed = incremental_pca.transform(X)
        X_inc_reconstructed = incremental_pca.inverse_transform(X_inc_transformed)
    results["incremental"]["transform_time"] = inc_transform_timer.get_elapsed()

    results["incremental"]["reconstruction_error"] = reconstruction_error(X, X_inc_reconstructed)
    results["incremental"]["normalized_error"] = normalized_reconstruction_error(X, X_inc_reconstructed)
    results["incremental"]["mse"] = mean_squared_error(X, X_inc_reconstructed)

    print("Running Batch PCA...")
    with Timer() as batch_timer:
        batch_pca.fit(X)
    results["batch"]["fit_time"] = batch_timer.get_elapsed()

    with Timer() as batch_transform_timer:
        X_batch_transformed = batch_pca.transform(X)
        X_batch_reconstructed = batch_pca.inverse_transform(X_batch_transformed)
    results["batch"]["transform_time"] = batch_transform_timer.get_elapsed()

    results["batch"]["reconstruction_error"] = reconstruction_error(X, X_batch_reconstructed)
    results["batch"]["normalized_error"] = normalized_reconstruction_error(X, X_batch_reconstructed)
    results["batch"]["mse"] = mean_squared_error(X, X_batch_reconstructed)

    results["speedup"] = results["batch"]["fit_time"] / results["incremental"]["fit_time"]

    return results


def print_comparison_results(results):
    """
    Pretty print comparison results.
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

    if results["speedup"] > 1:
        print(f"  Incremental PCA is {results['speedup']:.2f}x faster!")
    else:
        print(f"  Batch PCA is {1/results['speedup']:.2f}x faster!")

    print("="*60)
