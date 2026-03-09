# MAINFRAME 2026 Tutorial Collection

This folder contains tutorial notebooks for the **MAINFRAME 2026** workshop, demonstrating molecular hit prediction using state-of-the-art multi-modal AI models for drug discovery.

## Overview

These tutorials showcase two advanced molecular AI models applied to real-world screening datasets:

- **MAMMAL** (Molecular Aligned Multi-Modal Architecture and Language) - [Paper](https://arxiv.org/abs/2410.22367v2) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-alignment/)
- **MMELON** (Multi-view Molecular Embedding with Late Fusion) - [Paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202517840) | [GitHub](https://github.com/BiomedSciAI/biomed-multi-view)

## Datasets

The tutorials use two screening datasets:
- **PGK2 DEL**: PGK2 DNA-encoded library screening data
- **WDR91 ASMS**: WDR91 affinity selection mass spectrometry data

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
