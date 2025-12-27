# ORL Face Database

This directory is intended for the ORL (Olivetti Research Laboratory) Face Database.

## Expected Structure

```
ORL_Faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── ...
│   └── 10.pgm
├── s2/
│   └── ...
...
└── s40/
    └── ...
```

## Database Information

- **Total images**: 400 (40 subjects × 10 images per subject)
- **Image size**: 92×112 pixels (grayscale)
- **Format**: PGM (Portable GrayMap) or can be JPG/PNG

## Download

The ORL Face Database can be downloaded from:
- Original source: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
- Alternative sources: Various academic repositories

## Automatic Fallback

If the database is not available, the program will automatically generate synthetic face-like data for demonstration purposes.

## Image Format Support

The data loader supports multiple image formats:
- `.pgm` (original format)
- `.jpg` / `.jpeg`
- `.png`

Images will be automatically:
- Converted to grayscale if needed
- Resized to 92×112 pixels if needed
- Flattened to vectors for PCA processing
