# MAINFRAME 2026 Tutorial Collection

This folder contains tutorial notebooks for the **MAINFRAME 2026** workshop, demonstrating molecular hit prediction using state-of-the-art multi-modal AI models for drug discovery.

## Overview

These tutorials showcase two advanced molecular AI models applied to real-world screening datasets:

- **MAMMAL** (Molecular Aligned Multi-Modal Architecture and Language) - [Paper](https://arxiv.org/abs/2410.22367v2) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-alignment/)
- **MMELON** (Multi-view Molecular Embedding with Late Fusion) - [Paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202517840) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-view)

## Datasets

The tutorials use two screening datasets:
- **PGK2 DEL**: Derived from PGK2 DNA-encoded library screening data
- **WDR91 ASMS**: WDR91 affinity selection mass spectrometry data


## Data Availability

All datasets required for these tutorials are publicly available and can be downloaded from:

**Source**: [https://www.aircheck.ai/datasets](https://www.aircheck.ai/datasets)
**Location**: Navigate to the **"Datasets for Hands-on"** tab

### WDR91 Dataset Files
- **Training data**: `DREAM_Challenge_1_TrainSet.parquet`
  - Contains molecular structures and binding labels for model development
- **Evaluation data**: `DREAM_Target2035_Challenge_test_data.csv`
  - Independent test set for model performance assessment

### PGK2 Dataset Files
- **Training data**: `PGK2_CDD.parquet`
  - DNA-encoded library screening data from Baylor College of Medicine 
- **Evaluation data**: `PGK2_Creative.parquet`
  - DNA-encoded library screening data from Creative Biolabs


## Data Processing

Before running the tutorials, the downloaded datasets must be preprocessed into the required format. All processed files should contain two columns: `smiles` (molecular structure) and `label` (binary classification target).

### WDR91 Processing Steps

1. **Split training data**: Randomly partition `DREAM_Challenge_1_TrainSet.parquet` into:
   - Training set: 70%
   - Validation set: 20%
   - Test set: 10%
   
   Save each split as CSV files with columns `smiles` and `label`:
   - `wdr91/train.csv`
   - `wdr91/val.csv`
   - `wdr91/test.csv`

2. **Prepare evaluation data**: Rename `DREAM_Target2035_Challenge_test_data.csv` to `wdr91_eval.csv` for consistency

### PGK2 Processing Steps

1. **Split training data**: Randomly partition `PGK2_CDD.parquet` into:
   - Training set: 70%
   - Validation set: 20%
   - Test set: 10%
   
   Save each split as CSV files with columns `smiles` and `label`:
   - `pgk2/train.csv`
   - `pgk2/val.csv`
   - `pgk2/test.csv`

2. **Prepare evaluation data**: Filter `PGK2_Creative.parquet` to remove any compounds present in the training set (to prevent data leakage), then save as `pgk2_creative.csv` with columns `smiles` and `label`

**Note**: Ensure random seed is set for reproducibility when performing data splits.

## Notebooks

### 1. Fine-Tuning Notebooks

#### `MAMMAL_finetune.ipynb`
Fine-tune a pretrained MAMMAL model on binary ligand-target classification tasks.

**Key Features:**
- Generic prompt format for molecular encoding
- Pre-trained on large-scale molecular datasets
- Binary classification for hit prediction
- PyTorch Lightning training pipeline


---

#### `MMELON_finetuning.ipynb`
Fine-tune the MMELON multi-modal molecular encoder for hit prediction.

**Key Features:**
- Multi-modal molecular representations (graphs, fingerprints, etc.)
- Pre-trained on large-scale molecular datasets
- Binary classification for screening data
- PyTorch Lightning integration


---

### 2. Inference Notebooks

Run inference with fine-tuned MAMMAL and MMELON models on WDR91-ASMS and PGK2-DEL datasets.

1. `MAMMAL_inference.ipynb`
2.  `MMELON_inference.ipynb`

---

### 3. Analysis Notebook

#### `data_and_predictions_analysis.ipynb`
Unified workflow for comprehensive analysis and comparison of MMELON and MAMMAL predictions.

**Key Features:**
- Hit prediction metrics (ROC-AUC, PR-AUC, Enrichment@K)
- Molecular clustering and diversity analysis
- Side-by-side model comparison
