Applications of Low-Rank Updating SVD in PCA
================================================

A Python implementation of Incremental PCA using Brand's low-rank SVD updating algorithm,
with comprehensive comparison against Batch PCA and a consensus-based automatic rank
selection framework.

------------------------------------------------
Overview
------------------------------------------------

This project implements and compares:

1. Incremental PCA (Brand, 2006)
   An efficient PCA method that updates the model as new data arrives using low-rank SVD updates.

2. Batch PCA
   The classical PCA approach computed on the entire dataset at once.

A major focus of this project is **automatic rank selection (auto rank)**,
designed to avoid manual tuning of the PCA dimension.

Experiments are conducted on the ORL Face Database (400 grayscale images, 92×112 pixels).
We report:

- Performance benchmarks (fit / transform speed)
- Reconstruction quality metrics
- Automatic rank selection behavior
- Cross-method rank consensus analysis
- Diagnostic plots for model validation

------------------------------------------------
Auto Rank Selection (Consensus-Based)
------------------------------------------------

Determining the appropriate PCA rank is a critical yet often manual step in practice.
To address this, this project implements a **consensus-based automatic rank selection framework**.

We evaluate **six rank selection criteria**, including:

- Energy threshold methods
- Gavish–Donoho optimal hard threshold
- Elbow-based approaches (Kneedle, L-method)
- Noise diagnostics–aware strategies

Rather than relying on a single heuristic, we perform **cross-method comparison and
consensus analysis** to examine:

- Stability of selected ranks
- Agreement and disagreement between criteria
- Sensitivity to noise assumptions

This design reflects realistic scenarios where the intrinsic data dimension is unknown
and must be inferred automatically from observations.

------------------------------------------------
Features
------------------------------------------------

- Brand’s low-rank SVD update algorithm
- Batch PCA baseline implementation
- Consensus-based automatic rank selection using six criteria
- Noise assumption diagnostics for Gavish–Donoho
- Automatic ORL dataset download with synthetic fallback
- Visualization tools:
  rank comparison plots, rank consensus analysis,
  residual diagnostics, eigenfaces, and reconstructions
- Modern dependency management using uv and pyproject.toml

------------------------------------------------
Installation (uv)
------------------------------------------------

This project uses uv instead of requirements.txt.

1) Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

or (macOS with Homebrew):

brew install uv

2) Create environment and install dependencies

uv sync

------------------------------------------------
Quick Start
------------------------------------------------

Run the full experimental pipeline:

uv run python main.py

This will:

1. Download the ORL dataset (or fall back to synthetic data)
2. Preprocess images (mean centering and normalization)
3. Validate Gavish–Donoho noise assumptions
4. Perform automatic rank selection and consensus analysis
5. Compare Incremental PCA versus Batch PCA
6. Save all diagnostic plots into the output/ directory

------------------------------------------------
Verify Dataset
------------------------------------------------

uv run python verify_data.py

------------------------------------------------
Outputs
------------------------------------------------

After running main.py, results are saved under:

output/
├── residual_diagnostics.png
├── scree_plot_with_elbows.png
├── rank_method_comparison.png
└── rank_consensus.png

Note:
The output/ directory and data/ORL_Faces/ are excluded from Git tracking via .gitignore.

------------------------------------------------
Using the API
------------------------------------------------

Example usage:

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

------------------------------------------------
Project Structure
------------------------------------------------

.
├── src/
│   ├── pca/                 IncrementalPCA and BatchPCA implementations
│   ├── data/                Dataset loader and preprocessing
│   ├── rank/                Rank selection methods and consensus logic
│   ├── diagnostics/         Noise and assumption diagnostics
│   ├── visualization/       Plotting utilities
│   ├── experiments/         Main pipeline and experiments
│   └── __init__.py
├── docs/                    References and ORL usage guide
├── main.py                  Entry point
├── verify_data.py
├── test_incremental_pca.py
├── pyproject.toml
└── README.md

------------------------------------------------
References
------------------------------------------------

1. Brand, M. (2006).
   Fast low-rank modifications of the thin singular value decomposition.
   Linear Algebra and its Applications, 415(1), 20–30.

2. Gavish, M., and Donoho, D. (2014).
   The optimal hard threshold for singular values is 4/sqrt(3).
   IEEE Transactions on Information Theory, 60(8), 5040–5053.

3. ORL Face Database
   https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

------------------------------------------------
License
------------------------------------------------

MIT License.
See the LICENSE file for details.

------------------------------------------------
Authors
------------------------------------------------

**Numerical Analysis Final Project**  
National Cheng Kung University (NCKU)

- 蔡宇德 (Chua Yee Teck)
- 陳柏諭 (Chen Po-Yu)
- 鄭丞佑 (Cheng Cheng-Yu)
- 陳柏任 (Chen Po-Jen)
