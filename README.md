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
- ✅ **Automatic Download**: ORL Face Database downloaded automatically if not present
- ✅ **Data Verification**: Built-in verification to ensure data quality
- ✅ **Reconstruction Error**: Calculate and compare reconstruction quality
- ✅ **Performance Benchmarking**: Compare timing and efficiency
- ✅ **ORL Face Database Support**: Load and process face images
- ✅ **Synthetic Data Fallback**: Automatic fallback if database not available

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
1. **Automatically download** the ORL Face Database (if not present)
2. Perform Incremental PCA with batch updates
3. Perform Batch PCA for comparison
4. Display timing and reconstruction error metrics

### Verify Database

Before running experiments, verify that the ORL database is correctly loaded:

```bash
python verify_data.py
```

This will:
- Download the database if not present
- Verify data structure and content
- Report whether real or synthetic data is being used
- Provide detailed diagnostic information

### Visualization

Generate visualizations comparing incremental and batch PCA:

```bash
python visualize.py
```

This creates three PNG files:
- `explained_variance.png` - Variance explained by each component
- `principal_components.png` - First few principal components (eigenfaces)
- `reconstructions.png` - Original vs reconstructed images

### Using the API

```python
from src.incremental_pca import IncrementalPCA
from src.batch_pca import BatchPCA
from src.data_loader import load_orl_faces, normalize_faces

# Load data (automatically downloads if needed)
faces, labels, is_real = load_orl_faces('data/ORL_Faces')

if not is_real:
    print("WARNING: Using synthetic data for demonstration only")

# Preprocess: Mean Centering (PCA standard)
centered_faces, mean_face = normalize_faces(faces)

# Incremental PCA
inc_pca = IncrementalPCA(n_components=50)
for i in range(0, len(centered_faces), 10):
    batch = centered_faces[i:i+10]
    inc_pca.partial_fit(batch)

# Transform and reconstruct
transformed = inc_pca.transform(centered_faces)
reconstructed = inc_pca.inverse_transform(transformed)

# Batch PCA
batch_pca = BatchPCA(n_components=50)
batch_pca.fit(centered_faces)
```

## Project Structure

```
.
├── src/
│   ├── __init__.py           # Package initialization
│   ├── incremental_pca.py    # Incremental PCA with Brand's algorithm
│   ├── batch_pca.py          # Standard batch PCA
│   ├── utils.py              # Utilities for benchmarking and error calculation
│   └── data_loader.py        # ORL Face Database loading utilities with auto-download
├── data/
│   ├── README.md             # Data directory documentation
│   └── ORL_Faces/            # ORL Face Database (auto-downloaded)
├── docs/
│   └── ORL_DATABASE_GUIDE.md # Comprehensive database usage guide
├── main.py                   # Main demonstration script
├── visualize.py              # Visualization script
├── verify_data.py            # Database verification script
├── test_incremental_pca.py   # Unit tests
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

### Automatic Download

**New Feature**: The database is now downloaded automatically!

When you run the code for the first time, it will:
1. Check if `data/ORL_Faces/` exists
2. If not, download from official AT&T archive
3. Extract and verify the database
4. Clean up temporary files

No manual download needed! 🎉

### Manual Download (Optional)

If automatic download fails, you can manually download:

1. Visit: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
2. Download `att_faces.zip`
3. Extract to `data/ORL_Faces/`

### Expected Directory Structure

```
data/ORL_Faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ... (10 images)
├── s2/
│   └── ...
└── s40/
    └── ...
```

### Data Verification

Use `verify_data.py` to ensure data quality:

```bash
python verify_data.py
```

Expected output:
```
==============================================================
ORL FACE DATABASE VERIFICATION
==============================================================

✓ Check 1/7: Total samples = 400 ✓
✓ Check 2/7: Feature dimension = 10304 (92×112 pixels) ✓
✓ Check 3/7: Label range = [0, 39] (40 subjects) ✓
✓ Check 4/7: Subjects = 40, Images per subject = 10 ✓
✓ Check 5/7: Pixel range = [0.00, 255.00] ✓
✓ Check 6/7: Data type = float64 ✓
✓ Check 7/7: Using REAL ORL Face Database ✓

🎉 All checks passed! Database is ready for experiments.
```

For detailed usage instructions, see [ORL Database Guide](docs/ORL_DATABASE_GUIDE.md).

## Performance Metrics

The implementation compares methods using:
- **Fit Time**: Time to train the PCA model
- **Transform Time**: Time to project data to PC space
- **Reconstruction Error**: Frobenius norm of the difference
- **Normalized Error**: Error normalized by original data norm
- **Mean Squared Error**: Average squared error per element
- **Speedup**: Relative performance comparison

### Important Notes on Accuracy

The incremental PCA implementation using Brand's algorithm provides approximate results that may differ from batch PCA due to:

1. **Incremental Mean Updates**: The mean is updated incrementally, which means early batches are centered with a different mean than later batches.
2. **Numerical Accumulation**: With many incremental updates, small numerical errors can accumulate.
3. **Initial Batch Size**: The first batch establishes the initial SVD basis, so using a very small initial batch can affect final accuracy.

**Trade-off**: Incremental PCA prioritizes:
- ✅ **Speed**: 2-3x faster than batch PCA
- ✅ **Memory efficiency**: Processes data in small batches
- ✅ **Online learning**: Can update model as new data arrives
- ⚠️ **Approximate results**: Components may differ from batch PCA but still capture variance structure

For applications requiring exact PCA results, use batch PCA. For large-scale or streaming applications where speed and memory efficiency are critical, incremental PCA is preferred.

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

## New in Version 2.0

- ✅ **Automatic ORL database download**
- ✅ **Data verification script** (`verify_data.py`)
- ✅ **Real vs synthetic data detection** (3rd return value in `load_orl_faces()`)
- ✅ **Mean Centering preprocessing** (PCA standard in `normalize_faces()`)
- ✅ **Comprehensive usage guide** (`docs/ORL_DATABASE_GUIDE.md`)
- ✅ **Improved error handling and logging**

## References

1. Brand, M. (2006). "Fast low-rank modifications of the thin singular value decomposition." Linear Algebra and its Applications, 415(1), 20-30.
2. ORL Face Database: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
3. AT&T Laboratories Cambridge

## License

MIT License - see LICENSE file for details

## Authors

**Numerical Analysis Final Project**

National Cheng Kung University (NCKU)
- 蔡宇德 (Chua Yee Teck)
- 陳柏諾 (Chen Po-Yu)
- 鄭丞佑 (Cheng Cheng-Yu)
- 陳柏任 (Chen Po-Jen)
