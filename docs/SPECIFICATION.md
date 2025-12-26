# Technical Specification: Incremental PCA Implementation

## Project Overview
**Title:** Applications of Low-Rank Updating Singular Value Decomposition in Incremental Principal Component Analysis (PCA)

**Team:** 蔡宇德, 陳柏諭, 鄭丞佑, 陳柏任

**Institution:** 國立成功大學 - 線性代數與數值方法

**Duration:** 5 weeks

---

## 1. Mathematical Foundation

### 1.1 Core Paper
Brand, M. (2006). "Fast low-rank modifications of the thin singular value decomposition". *ACM Transactions on Graphics*, 25(2), 349-357.

### 1.2 Problem Statement
Given existing SVD \( A = U \Sigma V^T \) and new data \( B \), efficiently compute updated SVD without full recomputation:

\[
\begin{bmatrix} A \\ B \end{bmatrix} = \tilde{U} \tilde{\Sigma} \tilde{V}^T
\]

### 1.3 Brand's Algorithm (Simplified)
1. **Project new data:** \( w = U^T c \) (projection coefficients)
2. **Compute residual:** \( p = c - U w \)
3. **Normalize residual:** \( m = p / \|p\| \), \( \rho = \|p\| \)
4. **Build core matrix:**
   \[
   K = \begin{bmatrix} \Sigma & w \\ 0 & \rho \end{bmatrix}
   \]
5. **SVD of K:** \( K = \hat{U} \hat{\Sigma} \hat{V}^T \)
6. **Update:**
   \[
   \tilde{U} = \begin{bmatrix} U & m \end{bmatrix} \hat{U}, \quad
   \tilde{V} = \begin{bmatrix} V & 0 \end{bmatrix} \hat{V}
   \]

---

## 2. Implementation Specification

### 2.1 Class: `IncrementalPCA`

#### Constructor
```python
def __init__(self, n_components: int = None)
```
**Parameters:**
- `n_components` (int, optional): Number of principal components to retain. If None, keep all.

**Attributes:**
- `mean_` (ndarray): Global mean vector (n_features,)
- `components_` (ndarray): Principal components as rows (n_components, n_features)
- `singular_values_` (ndarray): Singular values (n_components,)
- `n_samples_seen_` (int): Total number of samples processed

---

#### Method: `fit(X)`
```python
def fit(self, X: np.ndarray) -> self
```
**Purpose:** Initialize PCA with first batch of data.

**Algorithm:**
1. Compute mean: \( \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i \)
2. Center data: \( X_c = X - \mathbf{1} \bar{x}^T \)
3. SVD: \( X_c = U \Sigma V^T \) (Thin SVD: `full_matrices=False`)
4. Store: `components_ = V^T[:k]` (first k rows)

**Complexity:** \( O(n m^2) \) where \( n = \) features, \( m = \) samples

---

#### Method: `partial_fit(X)`
```python
def partial_fit(self, X: np.ndarray) -> self
```
**Purpose:** Incrementally update PCA with new data.

**Algorithm:**
1. **Update mean:**
   \[
   \bar{x}_{\text{new}} = \frac{n_{\text{old}} \bar{x}_{\text{old}} + \sum_{i} x_i^{\text{new}}}{n_{\text{old}} + m}
   \]
2. **Center new data:** \( X_c = X - \bar{x}_{\text{new}} \)
3. **Apply Brand's update:**
   - Project: \( P = X_c V \) (where \( V = \) `components_.T`)
   - Residual: \( R = X_c - P V^T \)
   - QR decompose: \( R^T = Q \cdot RR \)
   - Build core matrix \( K \)
   - SVD of \( K \)
   - Update components
4. **Store new mean:** `mean_ = new_mean`

**Complexity:** \( O(m k^2) \) where \( m = \) new samples, \( k = \) components

---

### 2.2 Critical Implementation Details

#### 2.2.1 SVD Configuration
```python
# ✅ CORRECT
U, s, vh = np.linalg.svd(X, full_matrices=False)
A_reconstructed = U @ np.diag(s) @ vh

# ❌ WRONG - Memory overflow
U, s, vh = np.linalg.svd(X, full_matrices=True)

# ❌ WRONG - Double transpose
A_reconstructed = U @ np.diag(s) @ vh.T
```

#### 2.2.2 Mean Centering Order
```python
# ✅ CORRECT
new_mean = (n_old * old_mean + X_new.sum(axis=0)) / (n_old + m)
X_centered = X_new - new_mean  # Use NEW mean
# ... apply Brand's update ...
self.mean_ = new_mean

# ❌ WRONG
X_centered = X_new - self.mean_  # Old mean!
self.mean_ = new_mean
```

---

## 3. Testing Requirements

### 3.1 Unit Tests

#### Test 1: Batch Equivalence
```python
# Single batch incremental should equal batch PCA
ipca = IncrementalPCA(n_components=50)
ipca.fit(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X)

assert np.allclose(ipca.components_, pca.components_, atol=1e-10)
```

#### Test 2: Incremental Accuracy
```python
# Two-batch incremental ≈ one-batch PCA
ipca = IncrementalPCA(n_components=50)
ipca.fit(X[:200])
ipca.partial_fit(X[200:])

pca = PCA(n_components=50)
pca.fit(X)

# Subspace distance should be small
error = subspace_distance(ipca.components_, pca.components_)
assert error < 1e-6
```

#### Test 3: Reconstruction Error
```python
# Error should decrease with more components
errors = []
for k in [10, 20, 30, 40, 50]:
    ipca = IncrementalPCA(n_components=k)
    ipca.fit(X)
    X_recon = ipca.inverse_transform(ipca.transform(X))
    errors.append(np.linalg.norm(X - X_recon, 'fro'))

assert all(errors[i] > errors[i+1] for i in range(len(errors)-1))
```

---

## 4. Performance Benchmarks

### 4.1 Time Complexity
| Method | Complexity | Notes |
|--------|-----------|-------|
| Batch PCA | \( O(n m^2) \) | Full SVD every time |
| Incremental PCA (per batch) | \( O(m k^2) \) | \( k \ll n \) |
| **Speedup** | \( \sim \frac{n}{k} \) | For \( k = 50 \), speedup \( \approx 200\times \) |

### 4.2 Measurement Protocol
```python
import time

def benchmark(method, X, n_runs=20):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        method(X)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)
```

---

## 5. Dataset Specification

**Name:** ORL Face Database

**Size:** 400 images (40 persons × 10 images/person)

**Dimensions:** 92 × 112 pixels → flattened to 10,304 features

**Format:** Grayscale PGM files

**Structure:**
```
orl_faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...
├── s2/
└── ...
```

---

## 6. Deliverables Checklist

- [ ] `src/incremental_pca.py` - Core implementation
- [ ] `src/batch_pca.py` - Baseline for comparison
- [ ] `src/utils.py` - Reconstruction error metrics
- [ ] `src/data_loader.py` - ORL database loader
- [ ] `experiments/benchmark.py` - Performance comparison
- [ ] `tests/test_incremental_pca.py` - Unit tests
- [ ] `docs/IMPLEMENTATION_NOTES.md` - Debugging notes
- [ ] Final report (LaTeX/PDF)

---

## References

1. Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. *ACM Transactions on Graphics (TOG)*, 25(2), 349-357.

2. Ross, D. A., Lim, J., Lin, R. S., & Yang, M. H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77(1), 125-141.

3. ORL Face Database: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
