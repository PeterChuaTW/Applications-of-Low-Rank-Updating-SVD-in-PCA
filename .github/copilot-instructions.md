# Copilot Instructions for Incremental PCA Project

## Core Rules (MUST FOLLOW)

### 1. SVD Usage
- **ALWAYS** use `np.linalg.svd(A, full_matrices=False)` (Thin SVD)
- **NEVER** use `full_matrices=True` (causes memory overflow with 10304×10304 matrices)

### 2. Transpose Handling
- NumPy's `svd()` returns `U, s, vh` where `vh` is **already V^T**
- Formula: `A = U @ np.diag(s) @ vh` (NOT `vh.T`)
- When storing principal components, keep them as rows (V^T format)

### 3. Variable Naming (Brand's Notation)
Follow Brand (2006) paper exactly:
- `w`: Projection coefficients (U^T @ c)
- `p`: Residual vector (c - U @ w)
- `m`: Normalized residual (p / ||p||)
- `K`: Core matrix for SVD updating

### 4. Mean Centering Protocol
**Order matters:**
1. Compute new global mean: `new_mean = (n_old * old_mean + sum(X_new)) / n_total`
2. Center new data: `X_centered = X_new - new_mean`
3. Apply Brand's update to SVD
4. Update stored mean: `self.mean_ = new_mean`

### 5. Data Preprocessing
- All PCA operations require mean-centered data
- Original data shape: (n_samples, 10304) for ORL faces
- Data type: `np.float64`

### 6. Performance Measurement
- Compare timing using loops (20+ iterations)
- Report average ± std deviation
- Complexity: O(mk²) per batch (m=batch_size, k=n_components)

## Testing Requirements
- Test 1: Single batch == standard PCA
- Test 2: Two-batch incremental ≈ batch PCA (error < 1e-10)
- Test 3: Reconstruction error decreases with k

## Forbidden Practices
- ❌ No `full_matrices=True`
- ❌ No double-transpose of `vh`
- ❌ No storing full data matrix in memory
- ❌ No using "Bi-Cross Validation" or "PGD" methods
