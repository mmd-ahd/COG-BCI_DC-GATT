# DC-GATT: Dynamic Connectivity Graph Attention Transformer for Multi-Task EEG Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)]()

> Official PyTorch implementation of the paper **"DC-GATT: Dynamic Connectivity Graph Attention Transformer for Multi-Task EEG Classification"** by Mohammad Ahadzadeh, Pouya Taghipour Langrodi, Golnaz Baghdadi, and Arian Afshar (Amirkabir University of Technology).

## Authors

*   **Mohammad Ahadzadeh** - Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran | `mmd.ahd@aut.ac.ir`
*   **Arian Afshar** - Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran | `Arafsh1382@aut.ac.ir`
*   **Pouya Taghipour Langrodi** - Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran | `Pouya_Taghipour@aut.ac.ir`
*   **Golnaz Baghdadi** - Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran | `Golnaz_Baghdadi@aut.ac.ir`

## Overview

Decoding cognitive states across multiple tasks and sessions is a critical challenge in Brain-Computer Interface (BCI) research due to the non-stationarity of neural functional interactions. This repository contains the source code for **DC-GATT**, an end-to-end deep learning framework that models fine-scale temporal evolutions of brain networks.

By jointly modeling session-level dynamic functional connectivity (using weighted Phase-Lag Index, wPLI) and leveraging a transformer-based temporal attention mechanism, DC-GATT effectively classifies complex cognitive tasks (PVT, N-back, Flanker) with high precision.

### Key Contributions
* **Dynamic Graph Construction:** Extracts sequential graph snapshots from continuous EEG utilizing time-resolved wPLI and $k$-NN sparsification.
* **Spatiotemporal Encoder:** Fuses Temporal 1D CNNs with Graph Attention Networks (GATv2) to capture distinct morphological and spatial topological features.
* **Temporal Transformer:** Employs multi-head self-attention to model long-range temporal dependencies across 26 discrete 200-ms windows.
* **Hierarchical Attention Pooling:** Introduces a novel dual-attention pooling scheme (spatial followed by temporal) to intelligently aggregate high-dimensional EEG dynamics.

---

## Results on COG-BCI Dataset

DC-GATT demonstrates state-of-the-art performance on the multi-task COG-BCI dataset, significantly outperforming static graph baselines:

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **DC-GATT (Ours)** | **85.3 ± 5.3** | **85.4** | **92.7** | **85.5** |
| Static GAT Baseline | 70.9 ± 5.3 | 70.9 | 85.4 | 71.0 |

---

## Repository Structure

```text
├── CB_preprocessing.py                 # Pipeline for FIR filtering, ICA, and artifact rejection
├── CB_epoching.py                      # Universal event-related epoching script
├── CB_equalize_epochs.py               # Balances class distribution across tasks
├── CB_connectivity.py                  # Computes session-level time-resolved wPLI
├── CB_Dataset.py                       # PyTorch Geometric Dataset for dynamic graph construction
├── CB_DCGATT.py                        # Main DC-GATT model architecture definition
├── CB_SpatiotemporalEncoder.py         # Sub-module: 1D CNN + GATv2 Layers
├── CB_TemporalTransformerEncoder.py    # Sub-module: Transformer Encoder with Positional Encoding
├── CB_PositionalEncoding.py            # Sub-module: Sinusoidal Positional Embeddings
├── CB_HierarchicalAttentionPooling.py  # Sub-module: Spatial & Temporal Attentional Aggregation
├── CB_MLPClassifier.py                 # Sub-module: Final classification head
├── CB_Train.py                         # 5-fold cross-validation training and evaluation script
├── CB_inspection.py                    # Data visualization and sanity-check utilities
└── LICENSE                             # MIT License

```

## Dependencies & Installation

Ensure you have Python 3.8+ installed. The required packages include `torch`, `torch-geometric`, `mne`, `mne-connectivity`, `mne-icalabel`, and `scikit-learn`.

```bash
# Clone the repository
git clone [https://github.com/mmd-ahd/COG-BCI_DC-GATT.git](https://github.com/mmd-ahd/COG-BCI_DC-GATT.git)
cd COG-BCI_DC-GATT

# Install dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install torch_geometric
pip install mne mne-connectivity mne-icalabel
pip install scikit-learn tqdm numpy

```

---

## Usage Guide

### 1. Data Preparation

Download the [COG-BCI dataset](https://www.google.com/search?q=https://doi.org/10.1038/s41597-022-01898-w) and place it in the `data/` directory.

### 2. Preprocessing

Run the preprocessing pipeline. This step applies band-pass filtering (1-100Hz), performs ICA using the infomax-extended algorithm, automatically rejects artifacts via `ICLabel`, and applies a secondary CAR reference.

```bash
python CB_preprocessing.py

```

### 3. Epoching & Balancing

Extract epochs for the N-back, Flanker, and PVT tasks (-0.2s to 1s relative to stimulus) and equalize the number of epochs to prevent class imbalance.

```bash
python CB_epoching.py
python CB_equalize_epochs.py

```

### 4. Dynamic Connectivity Extraction

Compute the weighted Phase-Lag Index (wPLI) using Continuous Wavelet Transform (Morlet wavelets, 4-45 Hz) to generate time-resolved functional connectivity matrices.

```bash
python CB_connectivity.py

```

### 5. Training the Model

Train the DC-GATT model using 5-Fold Cross-Validation. The training script employs a cosine annealing learning rate scheduler, automated mixed precision (AMP), and session-level voting evaluation.

```bash
python CB_Train.py

```

---

## Acknowledgments

This research has been supervised and supported by the AUT Neural and Cognitive Systems Engineering Lab (NCSELab) at Amirkabir University of Technology (Tehran Polytechnic).
