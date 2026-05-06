# TBM Data Reconstruction using LSTM Seq2Seq Model

## Overview
This repository contains the preprocessing pipeline and deep learning model developed for reconstructing missing segments in Tunnel Boring Machine (TBM) operational data.  

The workflow combines signal processing, statistical resampling, interpolation, and a multivariate sequence-to-sequence (Seq2Seq) Long Short-Term Memory (LSTM) model for data reconstruction.

The implementation was developed as part of a Master's thesis at TU Graz.

---

## Repository Structure

The repository consists of three scripts:

### 1. Preprocessing Pipeline
**File:** `data_preprocessing.py`

This script prepares raw TBM data for modeling. The key steps include:

- Reading and merging multiple CSV files
- Filtering invalid or missing entries
- Removing standstill conditions
- Wavelet-based signal denoising
- Resampling data to uniform spatial intervals (1 m)
- Hybrid interpolation of missing data:
- Exporting processed datasets as `.parquet` files

Due to potential differences in data structure and preprocessing requirements, minor adjustments may be necessary to reproduce results on external datasets.

---

### 2. Linear Interpolation Errors
**File:** `linear_interp_errors.py`

This script implements the linear interpolation method for reconstructing artificial gaps in TBM data used for future comparison. The script takes the preprocessed dataset in parquet format and creates user-defined artificial gaps in the data. It then fills the gaps using linear interpolation and computes the RMSE by comparing it with the true values from the preprocessed dataset.

---

### 3. LSTM Reconstruction Model
**File:** `lstm_evaluation.py`

This script implements a multivariate Seq2Seq LSTM model for reconstructing artificial gaps in TBM data.

#### Key Features:
- Bidirectional reconstruction:
  - Forward model
  - Backward model (time-reversed data)
  - Weighted averaging of predictions
- Smoothness-regularized loss function
- Flexible gap definition and training configuration
- Automatic scaling using MinMax normalization

#### Model Workflow:
1. Artificial gaps are defined in the dataset
2. Training sequences are generated excluding gap regions
3. Forward and backward LSTM models are trained
4. Missing segments are reconstructed chunk-wise
5. Final prediction is obtained via weighted averaging

#### Outputs:
- Reconstruction plots (per parameter)
- Error metrics (RMSE, nRMSE)
- Results summary CSV: results/results_summary.csv

---

## Requirements

Install the required Python packages before running the scripts:

```bash
pip install numpy pandas matplotlib scikit-learn pyarrow pywt torch scipy
```

## Disclaimer

- This repository is intended for research and educational purposes.
- The dataset used in this study is not included due to size and confidentiality constraints.

## Author

- Vaibhav Shringi, BEng
- Master’s Program in Geotechnical and Hydraulic Engineering
- TU Graz, Austria
