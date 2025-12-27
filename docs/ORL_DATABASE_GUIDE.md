# ORL Face Database - Usage Guide

## Overview

This guide explains how to use the automatic download and verification features for the ORL (Olivetti Research Laboratory) Face Database in this project.

## Quick Start

### Option 1: Automatic Download (Recommended)

Simply run your code - the database will be downloaded automatically:

```python
from src.data_loader import load_orl_faces

# Automatically downloads if not present
faces, labels, is_real = load_orl_faces()

if is_real:
    print("✓ Using real ORL Face Database")
else:
    print("⚠ Using synthetic data")
```

### Option 2: Verify Before Experiments

Run the verification script to ensure data quality:

```bash
python verify_data.py
```

Expected output:
```
==============================================================
Downloading ORL Face Database...
Source: https://www.cl.cam.ac.uk/Research/DTG/...
==============================================================
Downloading... (this may take a minute)
✓ Downloaded to data/att_faces.zip
✓ Extracted successfully
✓ Moved to data/ORL_Faces
==============================================================
✓ Successfully loaded 400 real face images
  - Subjects: 40
  - Images per subject: ~10
==============================================================
🎉 All checks passed! Database is ready for experiments.
```

## New Features

### 1. Automatic Download Function

**`download_orl_database(data_dir='data/ORL_Faces')`**

- Downloads from official AT&T Face Database archive
- Automatically extracts to correct directory structure
- Validates downloaded data
- Cleans up temporary files
- Returns `True` if successful, `False` otherwise

### 2. Database Verification

**`verify_database_structure(data_dir='data/ORL_Faces')`**

- Checks if directory structure is correct
- Ensures at least 30 subject directories exist (s1-s40)
- Returns `True` if structure is valid

### 3. Enhanced Load Function

**`load_orl_faces(data_dir='data/ORL_Faces', auto_download=True)`**

- **New parameter**: `auto_download` (default: `True`)
- **New return value**: `is_real` (3rd return value)
  - `True`: Using real ORL database
  - `False`: Using synthetic data
- Detailed progress logging
- Better error messages

### 4. PCA-Compliant Preprocessing

**`normalize_faces(faces)`**

- Changed to **Mean Centering** (PCA standard)
- Returns `(centered_faces, mean_face)`
- Essential preprocessing step before PCA

## Usage Examples

### Example 1: Basic Usage

```python
from src.data_loader import load_orl_faces, normalize_faces

# Load data (auto-downloads if needed)
faces, labels, is_real = load_orl_faces()

if not is_real:
    print("WARNING: Using synthetic data for demonstration only!")

# Mean Centering (required for PCA)
centered_faces, mean_face = normalize_faces(faces)

print(f"Data shape: {faces.shape}")
print(f"Mean face shape: {mean_face.shape}")
```

### Example 2: Manual Download Control

```python
from src.data_loader import load_orl_faces, download_orl_database

# Explicitly download first
success = download_orl_database('data/ORL_Faces')

if success:
    # Load with auto_download disabled
    faces, labels, is_real = load_orl_faces(auto_download=False)
else:
    print("Download failed - using synthetic data")
    faces, labels, is_real = load_orl_faces(auto_download=False)
```

### Example 3: Integration with Incremental PCA

```python
from src.data_loader import load_orl_faces, normalize_faces
from src.incremental_pca import IncrementalPCA

# Load and verify data
faces, labels, is_real = load_orl_faces()

if not is_real:
    raise RuntimeError("Experiments require real ORL database!")

# Preprocess: Mean Centering
centered_faces, mean_face = normalize_faces(faces)

# Run Incremental PCA
inc_pca = IncrementalPCA(n_components=50)

# Process in batches of 10
for i in range(0, len(centered_faces), 10):
    batch = centered_faces[i:i+10]
    inc_pca.partial_fit(batch)

print("Incremental PCA completed!")
```

### Example 4: Verification in Unit Tests

```python
import unittest
from src.data_loader import load_orl_faces, verify_database_structure

class TestDataLoader(unittest.TestCase):
    def test_database_exists(self):
        """Ensure ORL database is available"""
        self.assertTrue(verify_database_structure('data/ORL_Faces'))
    
    def test_load_real_data(self):
        """Ensure we're using real data, not synthetic"""
        faces, labels, is_real = load_orl_faces()
        self.assertTrue(is_real, "Must use real ORL database for tests")
        self.assertEqual(faces.shape[0], 400)
        self.assertEqual(faces.shape[1], 10304)
```

## Troubleshooting

### Problem 1: Download Fails

**Symptoms**: "Failed to download ORL database" message

**Solutions**:

1. **Check internet connection**
   ```bash
   ping www.cl.cam.ac.uk
   ```

2. **Manual download**:
   - Visit: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
   - Download `att_faces.zip`
   - Extract to `data/ORL_Faces/`

3. **Alternative source** (Kaggle):
   ```bash
   # If you have kaggle CLI
   kaggle datasets download -d kasikrit/att-database-of-faces
   unzip att-database-of-faces.zip -d data/ORL_Faces/
   ```

### Problem 2: Using Synthetic Data Unintentionally

**Symptoms**: `is_real = False` but you expected real data

**Check**:
```python
from src.data_loader import verify_database_structure

if not verify_database_structure('data/ORL_Faces'):
    print("Database structure is invalid or missing")
    # Re-download
    from src.data_loader import download_orl_database
    download_orl_database('data/ORL_Faces')
```

### Problem 3: Permission Errors

**Symptoms**: "Permission denied" when downloading

**Solution**:
```bash
# Ensure data directory is writable
chmod -R u+w data/

# Or create directory manually
mkdir -p data/ORL_Faces
```

### Problem 4: Incomplete Download

**Symptoms**: Verification fails, partial files in directory

**Solution**: The code automatically cleans up failed downloads, but you can manually reset:
```bash
rm -rf data/ORL_Faces data/att_faces.zip data/att_faces_temp
python verify_data.py
```

## Database Structure

### Expected Directory Layout

```
data/ORL_Faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ... (10 images total)
├── s2/
│   └── ...
└── s40/
    └── ...
```

### File Format Details

- **Format**: PGM (Portable Gray Map) or JPG/PNG
- **Dimensions**: 92×112 pixels
- **Color**: Grayscale (1 channel)
- **Total files**: 400 images (40 subjects × 10 images)

## Integration with Existing Code

### Updating main.py

**Before**:
```python
from src.data_loader import load_orl_faces
faces, labels = load_orl_faces()
```

**After**:
```python
from src.data_loader import load_orl_faces, normalize_faces

# Load with verification
faces, labels, is_real = load_orl_faces()

if not is_real:
    print("WARNING: Experiments use synthetic data for demo only")

# PCA preprocessing
centered_faces, mean_face = normalize_faces(faces)
```

### Updating Test Scripts

**Add at the beginning of test scripts**:
```python
import sys
from src.data_loader import load_orl_faces

# Verify real data is used
_, _, is_real = load_orl_faces()
if not is_real:
    print("ERROR: Tests require real ORL database")
    sys.exit(1)
```

## For Final Report

### Recommended Section Text

> **Dataset and Preprocessing**
> 
> This research utilizes the ORL (Olivetti Research Laboratory) Face Database, also known as the AT&T Face Database, containing 400 grayscale images (92×112 pixels) from 40 subjects with 10 images each. We implemented an automatic download system to ensure reproducibility.
> 
> All images undergo **Mean Centering** preprocessing before PCA computation, as per the standard practice in numerical linear algebra. The centered data matrix $\mathbf{X}_{centered} = \mathbf{X} - \bar{\mathbf{x}} \mathbf{1}^T$, where $\bar{\mathbf{x}}$ is the mean face vector (average across all 400 samples).
> 
> Data verification was performed using the included `verify_data.py` script to confirm:
> - Sample count: 400 images
> - Feature dimension: 10,304 (92×112)
> - Label distribution: 40 subjects, 10 images each
> - Data type: float64 for numerical stability

### Include Verification Output

Attach the output of `python verify_data.py` as evidence:

```
🎉 All checks passed! Database is ready for experiments.
```

## Technical Details

### Download Process Flow

1. Check if `data/ORL_Faces/` exists
2. If not, download `att_faces.zip` from official URL
3. Extract to temporary directory
4. Move to target location
5. Verify structure (30+ subject directories)
6. Clean up temporary files
7. Return success status

### Error Handling

- **Network errors**: Automatic fallback to synthetic data
- **Extraction errors**: Cleanup of partial files
- **Permission errors**: Clear error messages
- **Invalid structure**: Re-download attempt

### File Format Compatibility

| Format | Support | Auto-resize | Notes |
|--------|---------|-------------|-------|
| `.pgm` | ✓ Primary | Yes | Original format |
| `.jpg` | ✓ Fallback | Yes | Common alternative |
| `.png` | ✓ Fallback | Yes | Lossless alternative |
| `.gif` | ✓ Fallback | Yes | Rare but supported |
| `.bmp` | ✓ Fallback | Yes | Windows format |

## Performance Notes

- **Download time**: ~30-60 seconds (depends on network)
- **Extraction time**: ~5-10 seconds
- **First load time**: ~2-3 seconds (400 images)
- **Subsequent loads**: Same (no caching)

## Best Practices

1. **Always check `is_real` flag** in experiments
2. **Run `verify_data.py` before final experiments**
3. **Use Mean Centering before PCA** (not standardization)
4. **Document data source** in reports
5. **Include verification output** as evidence

## References

- **ORL Database**: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
- **AT&T Laboratories Cambridge**
- **Paper**: F. Samaria and A. Harter (1994). "Parameterisation of a stochastic model for human face identification"

## Changelog

### Version 2.0 (Current)
- ✓ Automatic download functionality
- ✓ Database structure verification
- ✓ Real vs synthetic data detection
- ✓ Mean Centering preprocessing
- ✓ Comprehensive verification script

### Version 1.0 (Original)
- Basic loading from local directory
- Synthetic data fallback
- Standard normalization
