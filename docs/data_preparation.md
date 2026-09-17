# Data Preparation Guide for RCOD (Real-World Image Super-Resolution)

This document provides instructions for preparing datasets used in **RCOD** (AAAI-26).

---

## 1. Benchmark Datasets for Evaluation

### 1.1 Real-World Datasets
1. **DRealSR**:
   - Download the real test dataset pairs from [DRealSR Official Repository](https://github.com/xiezw5/DRealSR).
   - Expected structure:
     ```text
     DrealSRVal_crop128/
     ├── test_LR/
     └── test_HR/
     ```
2. **RealSR**:
   - Download real paired images from [RealSR Dataset](https://github.com/cszn/BSRGAN).
   - Expected structure:
     ```text
     RealSRVal_crop128/
     ├── test_LR/
     └── test_HR/
     ```

### 1.2 Synthetic Validation Dataset
- **DIV2K-Val**:
  - Download DIV2K validation high-resolution images from [DIV2K Official Site](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
  - Degrade via the Real-ESRGAN degradation pipeline to produce the synthetic validation paired set `DIV2K_V2_val`.

---

## 2. Training Datasets

> **Note on Copyright**: LSDIR and FFHQ datasets require users to agree to their respective licenses. Do not distribute raw images directly.

1. **LSDIR (Large-Scale Diverse Image Restoration Dataset)**:
   - Download HR images from [LSDIR repository](https://github.com/YingGuan/LSDIR).
   - Generate image paths list `LSDIR/file_paths.txt`.

2. **FFHQ (Flickr-Faces-HQ)**:
   - Download the first 10K images from [FFHQ repository](https://github.com/NVlabs/ffhq-dataset).
   - Generate image paths list `ffhq/file_paths.txt`.

---

## 3. Real-ESRGAN Degradation Pipeline

The training pipeline generates synthetic LR images online from HR images. The degradation configuration is specified in `params_realesrgan.yml`.
Degradation stages include:
- Blur (Gaussian / Sinc / Generalized Gaussian)
- Downsampling (Area / Bilinear / Bicubic)
- Noise injection (Gaussian / Poisson)
- JPEG compression
