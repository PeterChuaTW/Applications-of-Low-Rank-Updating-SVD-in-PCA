"""
Incremental PCA using Brand's low-rank SVD updating algorithm.
"""
import numpy as np


class IncrementalPCA:
    """
    Incremental PCA using Brand's low-rank SVD updating algorithm.
    
    This implementation allows for incremental updates to the PCA model
    as new data samples arrive, without recomputing the entire SVD.
    
    Note: For PCA, we want principal components in feature space.
    We store the singular vectors of X^T @ X (covariance matrix).
    """
    
    def __init__(self, n_components=None):
        """
        Initialize the Incremental PCA model.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep. If None, all components are kept.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # Principal components (k, n_features)
        self.singular_values_ = None  # Singular values
        self.n_samples_seen_ = 0
        
    def fit(self, X):
        """
        Fit the model with initial data X.
        
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
        
        # Compute SVD: X_centered = U @ diag(S) @ Vt
        # U: (n_samples, min(n_samples, n_features))
        # S: (min(n_samples, n_features),)
        # Vt: (min(n_samples, n_features), n_features)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Principal components are rows of Vt (or columns of V)
        # Keep only n_components
        if self.n_components is not None:
            k = min(self.n_components, len(S))
        else:
            k = len(S)
            
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.n_samples_seen_ = n_samples
        
        return self
        
    def partial_fit(self, X):
        """
        Incrementally fit the model with new data using Brand's algorithm.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to incrementally fit
        """
        X = np.array(X, dtype=np.float64)
        
        if self.mean_ is None:
            # First time seeing data
            return self.fit(X)
            
        n_new_samples, n_features = X.shape
        
        # Update mean incrementally
        n_total = self.n_samples_seen_ + n_new_samples
        new_mean = (self.n_samples_seen_ * self.mean_ + np.sum(X, axis=0)) / n_total
        
        # Center new data with current mean (before updating)
        X_centered = X - self.mean_
        
        # Apply Brand's algorithm for SVD updating
        self._brand_update(X_centered)
        
        # Update mean and sample count
        self.mean_ = new_mean
        self.n_samples_seen_ = n_total
        
        return self
        
    def _brand_update(self, X_new):
        """
        Brand's algorithm for low-rank SVD updating.
        
        We maintain: A^T @ A ≈ V @ diag(S^2) @ V^T
        where V = components_^T (n_features x k)
        
        Parameters:
        -----------
        X_new : array-like, shape (m, n_features)
            New centered data to add
        """
        if X_new.shape[0] == 0:
            return
            
        m = X_new.shape[0]
        k = len(self.singular_values_)
        
        # Current SVD gives us the principal directions
        # V = components_.T is (n_features, k)
        V = self.components_.T
        S = self.singular_values_
        
        # Project new data onto current principal components
        # P = X_new @ V  (m x k)
        P = X_new @ V
        
        # Compute residual
        # R = X_new - P @ V^T  (m x n_features)
        R = X_new - P @ V.T
        
        # QR decomposition of R^T
        # Q: (n_features, j), RR: (j, m)
        Q, RR = np.linalg.qr(R.T, mode='reduced')
        j = Q.shape[1]
        
        # Construct augmented matrix K
        # K = [diag(S)   P^T  ]
        #     [  0       RR   ]
        # K: ((k+j) x (k+m))
        K = np.zeros((k + j, k + m))
        K[:k, :k] = np.diag(S)
        K[:k, k:] = P.T
        K[k:, k:] = RR
        
        # SVD of K
        _, S_new, Vt_K = np.linalg.svd(K, full_matrices=False)
        
        # Update principal components
        # V_new = [V Q] @ U_K[:,:k_new]
        # Where U_K are the left singular vectors of K (not returned, so use V)
        # Actually we need: V_new^T = Vt_K[:k_new, :] @ [V^T; Q^T]
        
        # Build augmented V matrix
        V_aug = np.vstack([V.T, Q.T])  # (k+j, n_features)
        
        # New components
        k_new = min(self.n_components, len(S_new)) if self.n_components else len(S_new)
        components_new = Vt_K[:k_new, :k+j] @ V_aug  # (k_new, n_features)
        
        self.components_ = components_new
        self.singular_values_ = S_new[:k_new]
        
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
        
    def get_components(self):
        """
        Get the principal components (eigenvectors).
        
        Returns:
        --------
        components : array-like, shape (n_components, n_features)
            Principal components
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.components_
        
    def get_explained_variance(self):
        """
        Get the explained variance for each component.
        
        Returns:
        --------
        explained_variance : array-like, shape (n_components,)
            Explained variance for each component
        """
        if self.singular_values_ is None:
            raise ValueError("Model has not been fitted yet.")
        return (self.singular_values_ ** 2) / (self.n_samples_seen_ - 1)
        
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.
        
        Returns:
        --------
        explained_variance_ratio : array-like, shape (n_components,)
            Proportion of variance explained by each component
        """
        explained_variance = self.get_explained_variance()
        total_variance = np.sum(explained_variance)
        return explained_variance / total_variance if total_variance > 0 else explained_variance

