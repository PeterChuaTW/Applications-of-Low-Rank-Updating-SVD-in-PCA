# Applications of Low-Rank Updating SVD in PCA

A Python implementation of Incremental PCA using Brand's low-rank SVD updating algorithm.

## Overview

This project implements and compares two PCA methods:
1. **Incremental PCA** - Uses Brand's algorithm for efficient low-rank SVD updating
2. **Batch PCA** - Standard PCA computed on the entire dataset at once

The implementation processes the ORL Face Database (400 images, 92×112 pixels) and provides comprehensive performance benchmarking and reconstruction error analysis.

## Features

- ✅ **Brand's Algorithm**: Efficient low-rank SVD updating for incremental PCA
- ✅ **NumPy-based**: All matrix operations use NumPy for performance
- ✅ **Reconstruction Error**: Calculate and compare reconstruction quality
- ✅ **Performance Benchmarking**: Compare timing and efficiency
- ✅ **ORL Face Database Support**: Load and process face images
- ✅ **Synthetic Data Generation**: Automatic fallback if database not available

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main demonstration script:

```bash
python main.py
```

This will:
1. Load the ORL Face Database (or generate synthetic data)
2. Perform Incremental PCA with batch updates
3. Perform Batch PCA for comparison
4. Display timing and reconstruction error metrics

### Using the API

```python
from src.incremental_pca import IncrementalPCA
from src.batch_pca import BatchPCA
from src.data_loader import load_orl_faces

# Load data
faces, labels = load_orl_faces('data/ORL_Faces')

# Incremental PCA
inc_pca = IncrementalPCA(n_components=50)
for i in range(0, len(faces), 10):
    batch = faces[i:i+10]
    inc_pca.partial_fit(batch)

# Transform and reconstruct
transformed = inc_pca.transform(faces)
reconstructed = inc_pca.inverse_transform(transformed)

# Batch PCA
batch_pca = BatchPCA(n_components=50)
batch_pca.fit(faces)
```

## Project Structure

```
.
├── src/
│   ├── __init__.py           # Package initialization
│   ├── incremental_pca.py    # Incremental PCA with Brand's algorithm
│   ├── batch_pca.py          # Standard batch PCA
│   ├── utils.py              # Utilities for benchmarking and error calculation
│   └── data_loader.py        # ORL Face Database loading utilities
├── data/
│   └── ORL_Faces/            # ORL Face Database (place images here)
├── main.py                   # Main demonstration script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Algorithm Details

### Brand's Low-Rank SVD Updating

Given an existing SVD decomposition:
```
A = U * S * V^T
```

When new data `X_new` arrives, Brand's algorithm efficiently updates the SVD to:
```
[A; X_new] = U_new * S_new * V_new^T
```

The algorithm:
1. Projects new data onto the current basis: `L = X_new * V`
2. Computes the orthogonal component: `H = X_new - L * V^T`
3. Performs QR decomposition on `H`: `H = Q * R`
4. Constructs augmented matrix `K` and computes its SVD
5. Updates `U`, `S`, and `V^T` using the results

This avoids recomputing the full SVD on the combined dataset, making it much more efficient for streaming data.

## ORL Face Database

The ORL (Olivetti Research Laboratory) Face Database contains:
- 400 images total
- 40 subjects
- 10 images per subject
- 92×112 pixels per image (grayscale)

### Directory Structure

Place the ORL Face Database in `data/ORL_Faces/` with the following structure:
```
data/ORL_Faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...
├── s2/
│   └── ...
└── s40/
    └── ...
```

If the database is not available, the program will automatically generate synthetic face-like data for demonstration.

## Performance Metrics

The implementation compares methods using:
- **Fit Time**: Time to train the PCA model
- **Transform Time**: Time to project data to PC space
- **Reconstruction Error**: Frobenius norm of the difference
- **Normalized Error**: Error normalized by original data norm
- **Mean Squared Error**: Average squared error per element
- **Speedup**: Relative performance comparison

## Example Output

```
==============================================================
PCA METHOD COMPARISON RESULTS
==============================================================

Incremental PCA:
  Fit time: 0.1234 seconds
  Transform time: 0.0056 seconds
  Reconstruction error: 1234.5678
  Normalized error: 0.123456
  Mean squared error: 0.001234

Batch PCA:
  Fit time: 0.2345 seconds
  Transform time: 0.0067 seconds
  Reconstruction error: 1234.5680
  Normalized error: 0.123456
  Mean squared error: 0.001234

Performance Comparison:
  Speedup (Batch/Incremental): 1.90x
  Incremental PCA is 1.90x faster!
==============================================================
```

## Dependencies

- Python 3.6+
- NumPy >= 1.21.0
- Matplotlib >= 3.3.0 (for visualization)
- Pillow >= 8.0.0 (for image loading)

## References

1. Brand, M. (2006). "Fast low-rank modifications of the thin singular value decomposition." Linear Algebra and its Applications, 415(1), 20-30.
2. ORL Face Database: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

## License

MIT License - see LICENSE file for details

## Author

Numerical Analysis Final Project
