"""
Batch PCA implementation for comparison with Incremental PCA.
"""
import numpy as np


class BatchPCA:
    """
    Standard batch PCA using full SVD decomposition.
    
    This implementation computes PCA on the entire dataset at once,
    serving as a baseline for comparison with Incremental PCA.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize the Batch PCA model.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep. If None, all components are kept.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_samples_seen_ = 0
        
    def fit(self, X):
        """
        Fit the PCA model with data X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Compute mean
        self.mean_ = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Determine number of components to keep
        if self.n_components is not None:
            k = min(self.n_components, len(S))
        else:
            k = len(S)
            
        # Store components
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        
        # Compute explained variance
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)
        total_var = np.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        self.n_samples_seen_ = n_samples
        
        return self
        
    def transform(self, X):
        """
        Transform data to principal component space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if self.mean_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        
        # Project onto principal components
        return X_centered @ self.components_.T
        
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Data in original space
        """
        if self.mean_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        X_transformed = np.array(X_transformed, dtype=np.float64)
        
        # Reconstruct
        X_centered = X_transformed @ self.components_
        
        # Add back the mean
        return X_centered + self.mean_
        
    def fit_transform(self, X):
        """
        Fit the model and transform the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X).transform(X)
