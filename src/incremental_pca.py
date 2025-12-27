"""
Incremental PCA using Brand's low-rank SVD updating algorithm.
"""
import numpy as np


class IncrementalPCA:
    """
    Incremental PCA using Brand's low-rank SVD updating algorithm.
    
    This implementation allows for incremental updates to the PCA model
    as new data samples arrive, without recomputing the entire SVD.
    
    Reference:
    Brand, M. (2006). "Fast low-rank modifications of the thin singular 
    value decomposition". ACM Transactions on Graphics, 25(2), 349-357.
    
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
        self.total_variance_ = None  # Total variance (for explained_variance_ratio)
        
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
        # Note: full_matrices=False gives Thin SVD (required for memory efficiency)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Principal components are rows of Vt (or columns of V)
        # Keep only n_components
        if self.n_components is not None:
            k = min(self.n_components, len(S))
        else:
            k = len(S)
        
        # Store total variance (from all singular values, not just kept ones)
        self.total_variance_ = np.sum(S ** 2) / (n_samples - 1)
            
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
        n_total = self.n_samples_seen_ + n_new_samples
        
        # CRITICAL FIX: Compute new global mean FIRST
        # All data must be centered with the same mean for SVD math to work
        new_mean = (self.n_samples_seen_ * self.mean_ + np.sum(X, axis=0)) / n_total
        
        # Center new data with NEW global mean (not old mean!)
        X_centered = X - new_mean
        
        # Apply Brand's algorithm for SVD updating
        self._brand_update(X_centered)
        
        # Update mean and sample count
        self.mean_ = new_mean
        self.n_samples_seen_ = n_total
        
        return self
        
    def _brand_update(self, X_new):
        """
        Brand's algorithm for low-rank SVD updating.
        
        Given current SVD approximation and new centered data X_new,
        update the SVD without full recomputation.
        
        Algorithm (Brand 2006):
        1. Project new data onto current basis: w = V^T @ X_new^T
        2. Compute residual: p = X_new^T - V @ w
        3. Orthonormalize residual: Q, R = qr(p)
        4. Form augmented matrix K and compute its SVD
        5. Update V using rotation matrix U_K (left singular vectors of K)
        
        Parameters:
        -----------
        X_new : array-like, shape (m, n_features)
            New centered data to add
        """
        if X_new.shape[0] == 0:
            return
            
        m = X_new.shape[0]  # Number of new samples
        k = len(self.singular_values_)  # Current rank
        
        # Update total variance estimate
        if self.total_variance_ is not None:
            old_total_var = self.total_variance_ * (self.n_samples_seen_ - 1)
            new_var = np.sum(X_new ** 2)
            self.total_variance_ = (old_total_var + new_var) / (self.n_samples_seen_ + m - 1)
        
        # Current components: V is (n_features, k), stored as Vt which is (k, n_features)
        # Brand's notation: V = components_.T
        V = self.components_.T  # Shape: (n_features, k)
        S = self.singular_values_  # Shape: (k,)
        
        # Step 1: Project new data onto current principal components
        # w = projection coefficients of X_new onto current basis V
        w = X_new @ V  # Shape: (m, k)
        
        # Step 2: Compute residual (part of X_new not captured by current PCs)
        p = X_new - w @ V.T  # Shape: (m, n_features)
        
        # Step 3: QR decomposition of residual's transpose
        # QR gives us an orthonormal basis (Q) and the norms (R)
        # Note: We transpose p because QR expects (n_features, m)
        Q, R = np.linalg.qr(p.T, mode='reduced')  # Q: (n_features, j), R: (j, m)
        j = Q.shape[1]  # Number of new basis vectors
        
        # Step 4: Construct augmented core matrix K
        # K represents the data in the augmented space [V, Q]
        # K = [diag(S)   w^T  ]
        #     [  0        R   ]
        # Shape: (k+j, k+m)
        K = np.zeros((k + j, k + m))
        K[:k, :k] = np.diag(S)  # Old singular values
        K[:k, k:] = w.T         # Projection of new data
        K[k:, k:] = R           # Residual norms
        
        # Step 5: SVD of core matrix K (much smaller than original data!)
        # This is the key efficiency gain: K is (k+j) x (k+m), not n x N
        # CRITICAL: We need U_K (left singular vectors) for feature space rotation
        U_K, S_new, _ = np.linalg.svd(K, full_matrices=False)
        
        # Step 6: Update principal components
        # Mathematical formula: V_new = [V, Q] @ U_K
        # Since self.components_ stores V^T, we compute: V_new^T = U_K^T @ [V^T; Q^T]
        
        # Build augmented V matrix: [V, Q]^T
        V_aug = np.vstack([V.T, Q.T])  # Shape: (k+j, n_features)
        
        # Rotate the augmented basis using U_K (NOT Vt_K!)
        # This is where the previous bug was: using Vt_K caused random noise
        k_new = min(self.n_components, len(S_new)) if self.n_components else len(S_new)
        components_new = U_K[:, :k_new].T @ V_aug  # Shape: (k_new, n_features)
        
        # Update stored values
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
        # components_ is (k, n_features), so we need transpose for projection
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
        
        # Reconstruct: X_centered = X_transformed @ components_
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
            Proportion of variance explained by each component relative to total variance
        """
        explained_variance = self.get_explained_variance()
        
        if self.total_variance_ is not None and self.total_variance_ > 0:
            return explained_variance / self.total_variance_
        else:
            # Fallback: normalize by sum of kept components
            total_variance = np.sum(explained_variance)
            return explained_variance / total_variance if total_variance > 0 else explained_variance
