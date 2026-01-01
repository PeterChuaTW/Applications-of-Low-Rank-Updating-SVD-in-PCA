# Applications of Low-Rank Updating SVD in PCA

A Python implementation of **Incremental PCA** using **Brand's low-rank SVD updating algorithm**, with comprehensive comparison against Batch PCA and multiple rank selection strategies.

---

## Overview

This project implements and compares:

1. **Incremental PCA** (Brand, 2006)  
   Efficiently updates the PCA model when new data arrives using low-rank SVD updates.

2. **Batch PCA**  
   Standard PCA computed on the entire dataset at once.

We use the **ORL Face Database** (400 grayscale face images, 92×112 pixels) and report:

- performance benchmark (fit/transform speed)
- reconstruction quality metrics
- rank selection comparison (6 methods)
- diagnostic plots and consensus analysis

---

## Features

- ✅ Brand’s low-rank SVD update algorithm
- ✅ Batch PCA baseline implementation
- ✅ 6 different rank selection methods (Energy / Gavish-Donoho / Kneedle / L-method)
- ✅ Noise assumption diagnostics for Gavish-Donoho
- ✅ Automatic ORL dataset download with fallback
- ✅ Visualization tools (rank plots, residual diagnostics, eigenfaces, reconstructions)
- ✅ Modern dependency management with **uv + pyproject.toml**

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

1. Download ORL dataset (or fall back to synthetic data)
2. Preprocess (mean centering)
3. Validate Gavish-Donoho assumptions
4. Compare **6 rank selection methods**
5. Run **Incremental PCA vs Batch PCA**
6. Save diagnostic plots into `output/`

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
