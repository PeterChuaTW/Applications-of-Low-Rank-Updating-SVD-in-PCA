import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_orl_faces
from src.data.preprocess import normalize_faces
from src.pca.incremental import IncrementalPCA


# =========================
# Utility: model-based reconstruction error
# =========================
def model_reconstruction_error(pca_model, X):
    Z = pca_model.transform(X)
    X_hat = pca_model.inverse_transform(Z)
    return np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')



# =========================
# Models
# =========================
ipca_fixed = IncrementalPCA(
    n_components=190,          # baseline fixed rank
    rank_strategy="fixed"
)

ipca_auto = IncrementalPCA(
    rank_strategy="energy",    # streaming auto-rank
    energy_threshold=0.95,
    rank_update_interval=1
)


# =========================
# Load & preprocess data
# =========================
X, y, is_real = load_orl_faces("data/ORL_Faces")
X_centered, mean_face = normalize_faces(X)


# =========================
# Create streaming batches
# =========================
batch_size = 10
batches = [
    X_centered[i:i + batch_size]
    for i in range(0, len(X_centered), batch_size)
]


# =========================
# Experiment A: Fixed-rank Incremental PCA
# =========================
fixed_errors = []

for batch in batches:
    ipca_fixed.partial_fit(batch)
    fixed_errors.append(
        model_reconstruction_error(ipca_fixed, X_centered)
    )


# =========================
# Experiment B: Streaming Auto-rank Incremental PCA
# =========================
auto_errors = []
auto_ranks = []
auto_explained_variance = []

for batch in batches:
    ipca_auto.partial_fit(batch)

    # reconstruction error
    auto_errors.append(
        model_reconstruction_error(ipca_auto, X_centered)
    )

    # record current rank
    auto_ranks.append(ipca_auto.n_components)

    # explained variance ratio (based on current singular values)
    s = ipca_auto.singular_values_
    if s is not None:
        var_ratio = np.sum(s[:ipca_auto.n_components]**2) / np.sum(s**2)
        auto_explained_variance.append(var_ratio)
    else:
        auto_explained_variance.append(np.nan)



# =========================
# Plot 1: Rank evolution (核心證據)
# =========================
plt.figure()
plt.plot(auto_ranks, marker='o')
plt.xlabel("Batch index")
plt.ylabel("Selected rank k")
plt.title("Streaming Auto-rank: Rank Evolution")
plt.grid(True)
plt.savefig("plot_1.png")


# =========================
# Plot 2: Early-stage reconstruction error
# =========================
plt.figure()
plt.plot(fixed_errors[:20], label="Fixed rank", marker='o')
plt.plot(auto_errors[:20], label="Streaming auto-rank", marker='o')
plt.xlabel("Batch index")
plt.ylabel("Reconstruction error")
plt.title("Early-stage Reconstruction Error Comparison")
plt.legend()
plt.grid(True)
plt.savefig("plot_2.png")


# =========================
# Plot 3: Explained variance during streaming (加分用)
# =========================
plt.figure()
plt.plot(auto_explained_variance, marker='o')
plt.xlabel("Batch index")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance During Streaming Auto-rank")
plt.ylim(0, 1.0)
plt.grid(True)
plt.savefig("plot_3.png")

