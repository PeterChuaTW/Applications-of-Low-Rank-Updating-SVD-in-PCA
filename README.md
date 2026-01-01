# Applications of Low-Rank Updating SVD in PCA

A Python implementation of **Incremental PCA** using **Brand’s low-rank SVD updating algorithm**, with comprehensive comparison against Batch PCA and an **automatic rank selection framework** based on **multi-method consensus** (energy thresholds, elbow detection, and statistical thresholding).


---

## Overview

This project implements and compares two PCA approaches:

1. **Incremental PCA (Brand, 2006)**  
   Efficiently updates the PCA model as new data arrives via low-rank SVD updates.

2. **Batch PCA**  
   Computes PCA on the full dataset in a single pass as a baseline reference.

A central contribution of this project is **automatic rank selection (auto-rank)** —  
designed to eliminate the need for manual tuning of the PCA dimensionality in practice.

Experiments are conducted on the **ORL Face Database**
(400 grayscale images, 92×112 pixels), and we report:

- **Performance benchmarking** (fit / transform runtime)
- **Reconstruction quality metrics** (error, normalized error, MSE)
- **Auto-rank selection outcomes** across multiple criteria
- **Cross-method consensus analysis** (agreement vs disagreement)
- **Diagnostic visualizations** for statistical assumption validation

---

## Auto Rank Selection (Consensus-Based)

Choosing the number of principal components (**rank k**) is a critical step in PCA,
yet it is often selected manually using ad hoc heuristics.
To address this, we implement a **consensus-based automatic rank selection framework**
that estimates an appropriate rank directly from data.

We evaluate **six complementary rank selection criteria**, including:

- **Cumulative energy thresholds** (e.g., 90%, 95%, 99%)
- **Gavish–Donoho optimal hard threshold** (random matrix theory)
- **Elbow-based detection methods** (Kneedle, L-method)
- **Noise-aware diagnostics and validation**

Rather than relying on a single heuristic, we perform **cross-method comparison and
consensus analysis** to study:

- **Stability** of selected ranks across criteria
- **Agreement / disagreement** between empirical, geometric, and statistical methods
- **Sensitivity to noise-model assumptions**, especially for Gavish–Donoho thresholding

This framework reflects realistic scenarios in which the intrinsic data dimension is unknown
and must be inferred automatically from observations — especially in online or streaming settings.


---


## Features

- **Brand’s low-rank SVD updating algorithm** for efficient Incremental PCA
- **Batch PCA baseline** for accuracy and performance comparison
- **Consensus-based automatic rank selection (auto-rank)** using six criteria:
  - energy thresholds (90% / 95% / 99%)
  - Gavish–Donoho optimal hard threshold (RMT)
  - elbow-based methods (Kneedle, L-method)
- **Noise assumption diagnostics** for validating Gavish–Donoho applicability
- **Automatic ORL dataset download** with robust fallbacks (official + Google Drive + synthetic)
- **Visualization suite**, including:
  - rank comparison plots and consensus analysis  
  - residual diagnostic plots (histogram / Q–Q / ACF)  
  - eigenfaces and reconstruction results
- Modern dependency management using **uv + pyproject.toml**


---

## Installation (uv)

> This project uses **uv** instead of `requirements.txt`.

### 1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or macOS (Homebrew):

```bash
brew install uv
```

### 2) Create environment + install dependencies

```bash
uv sync
```

---

## Quick Start

Run the full pipeline:

```bash
uv run python main.py
```

This will:

1. **Download** the ORL dataset (or fall back to synthetic data)
2. **Preprocess** the images (mean centering)
3. **Validate** Gavish–Donoho noise assumptions (diagnostics + plots)
4. **Perform auto-rank selection** using six criteria and run **consensus analysis**
5. **Compare** Incremental PCA versus Batch PCA (runtime + reconstruction error)
6. **Save** all diagnostic plots and rank-selection figures into `output/`


---

## Verify Dataset

```bash
uv run python verify_data.py
```

---

## Outputs

After running `main.py`, outputs will be saved under:

```
output/
├── residual_diagnostics.png
├── scree_plot_with_elbows.png
├── rank_method_comparison.png
└── rank_consensus.png
```

> Note: `output/` and `data/ORL_Faces/` are excluded from Git tracking via `.gitignore`.

---

## Using the API

```python
from src.pca.incremental import IncrementalPCA
from src.pca.batch import BatchPCA
from src.data.loader import load_orl_faces
from src.data.preprocess import normalize_faces

faces, labels, is_real = load_orl_faces("data/ORL_Faces")
X, mean_face = normalize_faces(faces)

inc = IncrementalPCA(n_components=50)
for i in range(0, len(X), 10):
    inc.partial_fit(X[i:i+10])

Z = inc.transform(X)
X_rec = inc.inverse_transform(Z)

batch = BatchPCA(n_components=50)
batch.fit(X)
```

---

## Project Structure

```
.
├── src/
│   ├── pca/                 # IncrementalPCA + BatchPCA implementations
│   ├── data/                # Dataset loader + preprocessing
│   ├── rank/                # Rank selection methods (energy, elbow, GD, etc.)
│   ├── diagnostics/         # Noise / assumption validation diagnostics
│   ├── visualization/       # Plotting utilities
│   ├── experiments/         # Main pipeline + experiments
│   └── __init__.py
├── docs/                    # References + ORL usage guide
├── main.py                  # Lightweight entrypoint
├── verify_data.py
├── test_incremental_pca.py
├── pyproject.toml
└── README.md
```

---

## References

1. Brand, M. (2006).  
   _Fast low-rank modifications of the thin singular value decomposition._  
   Linear Algebra and its Applications, 415(1), 20–30.

2. Gavish, M., & Donoho, D. (2014).  
   _The optimal hard threshold for singular values is 4/√3._  
   IEEE Transactions on Information Theory, 60(8), 5040–5053.

3. ORL Face Database  
   https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

---

## License

MIT License - see LICENSE file.

---

## Authors

**Numerical Analysis Final Project**  
National Cheng Kung University (NCKU)

- 蔡宇德 (Chua Yee Teck)
- 陳柏諭 (Chen Po-Yu)
- 鄭丞佑 (Cheng Cheng-Yu)
- 陳柏任 (Chen Po-Jen)
