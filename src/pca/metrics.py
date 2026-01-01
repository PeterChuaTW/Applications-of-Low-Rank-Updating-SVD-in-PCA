"""
Metrics for PCA evaluation.
"""
import numpy as np


def reconstruction_error(X_original, X_reconstructed):
    diff = X_original - X_reconstructed
    return np.linalg.norm(diff, 'fro')


def normalized_reconstruction_error(X_original, X_reconstructed):
    error = reconstruction_error(X_original, X_reconstructed)
    norm_original = np.linalg.norm(X_original, 'fro')
    return error / norm_original if norm_original > 0 else 0


def mean_squared_error(X_original, X_reconstructed):
    return np.mean((X_original - X_reconstructed) ** 2)


def subspace_distance(U1, U2):
    """
    Principal angle-based subspace distance between two orthonormal bases.
    """
    k = min(U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]

    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return np.linalg.norm(angles)
