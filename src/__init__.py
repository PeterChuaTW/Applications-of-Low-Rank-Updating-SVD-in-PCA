"""
Applications of Low-Rank Updating SVD in PCA

A Python implementation of Incremental PCA using Brand's low-rank SVD updating algorithm.
"""

from .incremental_pca import IncrementalPCA
from .batch_pca import BatchPCA
from .utils import (
    reconstruction_error,
    normalized_reconstruction_error,
    mean_squared_error,
    compare_pca_methods,
    print_comparison_results
)
from .data_loader import (
    load_orl_faces,
    generate_synthetic_faces,
    normalize_faces,
    split_data
)

__all__ = [
    'IncrementalPCA',
    'BatchPCA',
    'reconstruction_error',
    'normalized_reconstruction_error',
    'mean_squared_error',
    'compare_pca_methods',
    'print_comparison_results',
    'load_orl_faces',
    'generate_synthetic_faces',
    'normalize_faces',
    'split_data'
]
