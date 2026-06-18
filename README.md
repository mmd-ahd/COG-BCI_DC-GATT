# DC-GATT: Dynamic Connectivity Graph Attention Transformer for Multi-Task EEG Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MNE-EEG](https://img.shields.io/badge/MNE--EEG-1.4%2B-7E4A9E.svg)](https://mne.tools/)

[cite_start]Official PyTorch implementation for **DC-GATT** (Dynamic Connectivity Graph Attention Transformer), an end-to-end deep learning framework designed for multi-task cognitive state decoding from electroencephalography (EEG) signals across multiple sessions[cite: 4, 15, 18]. 

[cite_start]DC-GATT captures fine-scale temporal network reconfigurations by modeling EEG epochs as sequential, overlapping graph snapshots[cite: 19, 128]. [cite_start]By utilizing a specialized hierarchical dual-attention pooling scheme alongside transformer-based self-attention, the framework filters transient noise and explicitly weights task-relevant spatial and temporal brain network reconfigurations[cite: 20, 59, 213].

---

## Architecture Blueprint

[cite_start]The DC-GATT model architecture spans across four core sequential stages[cite: 133, 137, 140, 169]:

1. **Data Preprocessing & Artifact Scrubbing (`CB_preprocessing.py`, `CB_epoching.py`, `CB_equalize_epochs.py`)**: 
   [cite_start]Continuous data is band-pass filtered (1–100 Hz)[cite: 73]. [cite_start]Task data is concatenated per subject to form robust data matrices for Independent Component Analysis (ICA) via the Extended Infomax algorithm[cite: 74, 77]. [cite_start]Automated component classification is performed with `ICLabel`, dropping channels not labeled as "brain" or "other"[cite: 78, 79]. [cite_start]Clean data is re-referenced via a two-step Common Average Reference (CAR), low-pass filtered at 45 Hz, notch-filtered at 50 Hz, downsampled to 250 Hz, and equalized via Global Field Power (GFP) standard deviation sorting to mitigate outlier epochs[cite: 4, 80, 81, 82, 4].
2. **Dynamic Brain Topology (`CB_connectivity.py`)**: 
   [cite_start]Time-resolved Spectral Connectivity is estimated via the **Weighted Phase-Lag Index (wPLI)** derived from a Continuous Wavelet Transform (CWT) using Morlet wavelets (4–45 Hz)[cite: 83, 90]. [cite_start]Wavelet cycles scale linearly with frequency ($n_{\text{cycles}} = 0.75 \times f$) to standardize temporal bounds across frequency spectra[cite: 88, 89].
3. **Spatiotemporal Node Encoding (`CB_SpatiotemporalEncoder.py`, `CB_TemporalCNN.py`)**: 
   [cite_start]Each node's raw 200-ms sliding window signal is processed by a 1D Temporal CNN to compress dimensionality and filter noise[cite: 129, 133, 134]. [cite_start]Localized morphological features are then contextually distributed across a 2-hop neighborhood structure via a two-layer **GATv2** neural network operating on a sparsified $k$-NN topological graph snapshot ($k=22$)[cite: 132, 135, 136].
4. **Temporal Transformer (`CB_TemporalTransformerEncoder.py`, `CB_PositionalEncoding.py`)**: 
   [cite_start]Models multi-scale structural dynamics and long-range variations across the window-embedding sequence ($T = 26$ snapshots) using multi-head self-attention mechanisms joined with sinusoidal positional encodings[cite: 128, 137, 138, 139].
5. **Hierarchical Attention Pooling (`CB_HierarchicalAttentionPooling.py`, `CB_TemporalAttentionPooling.py`)**: 
   [cite_start]Sequentially pools representations across space (nodes) and time (windows) adaptively, steering the classifier toward discriminative windows while suppressing noise[cite: 20, 141, 142, 213, 214].
6. **MLP Classifier (`CB_MLPClassifier.py`)**: 
   [cite_start]Projects the regularized, layer-normalized whole-epoch representation through a small feed-forward system initialized with Xavier uniform constraints to output three target classification logits[cite: 4, 169, 170].

---

## Dataset & Classification Problem

The codebase targets validation over the public **COG-BCI dataset**, an extensive multi-task and multi-session resource profiling passive Brain-Computer Interfaces. The framework targets classification across 62 active cephalic channels across three tasks:

* 
**`twoBACK` (Class 0)**: N-back working memory assignment.


* 
**`Flanker` (Class 1)**: Cognitive conflict management and executive task control.


* 
**`PVT` (Class 2)**: Psychomotor Vigilance Task tracing continuous visual sustained attention.



Graphs are synthesized using a 1.2-second post-stimulus window windowed dynamically into 200-ms slices sliding every 40-ms, providing a static length sequence of exactly **26 graph snapshots** per trial epoch.

---

## Hyperparameter Setup

Optimal parameters contained within `CB_Train.py` are listed below:

| Structural Component | Hyperparameter Variable | Production Specification |
| --- | --- | --- |
| **Window Parameters** | Graph Sparsification ($k$-NN) | 12 (Training default) / 22 (Paper standard) 

 |
|  | Graph Snapshot Window / Step Size | 200 ms / 40 ms ($T = 26$ frames) 

 |
| **Temporal 1D CNN** | In/Out Channel Count | Kernel Width | 1 / 48 | 5 

 |
| **Spatial GATv2** | Hidden Dimensionality | Output Vector Dim | 64 | 96 

 |
|  | Head Number | Internal Layer Dropout | 8 | 0.235 

 |
| **Transformer Block** | Transformer Layers | Head Number | 1 | 8 

 |
|  | Feedforward Dimension | Dropout | 128 | 0.391 

 |
| **Pooling & Output** | Attention Tanh Gate Dimension | 96 

 |
|  | Predictor Hidden Neurons | Dropout | 64 | 0.381 

 |
| **Training Controls** | Data Batch Size | Base Learning Rate | 16 | $$3.86 \times 10^{-4$ 

 |
|  | L2 Weight Decay | Label Smoothing | $1.4 \times 10^{-5}$ | 0.1 

 |
|  | Learning Strategy Scheduler | Cosine Decay with 3 Warmup Epochs 

 |

---

## Empirical Verification

DC-GATT exhibits clear performance advantages over standard static architectures by evaluating temporal graph trajectories:

* 
**Mean Cross-Validation Accuracy**: **$85.3\% \pm 5.3\%$** (vs Static Baseline GAT yielding $70.9\% \pm 5.3\%$).


* 
**Macro Diagnostics**: Macro Sensitivity of **85.4%**, Specificity of **92.7%**, and an F1-score of **85.5%**.



### Session-Level Soft-Voting Performance (Confusion Matrix)

```plaintext
                   Predicted Task Logits
               N-back     Flanker      PVT
    N-back       74         11          2
True Flanker     12         71          4
    PVT           2          7         78

```

---

## Quick Start Pipeline

### 1. Requirements Installation

Clone the environment repository and establish structural deep learning components:

```bash
git clone [https://github.com/mmd-ahd/COG-BCI_DC-GATT.git](https://github.com/mmd-ahd/COG-BCI_DC-GATT.git)
cd COG-BCI_DC-GATT
pip install torch torch-geometric mne mne-connectivity mne-icalabel scikit-learn numpy tqdm

```

### 2. Operational Routine

Execute the complete dataset pipeline in order:

```bash
# Step 1: Preprocess raw .set matrices via artifact removal (Filter + ICA + ICLabel)
python CB_preprocessing.py

# Step 2: Epoch stimulus intervals and balance the trials count with GFP metrics
python CB_epoching.py
python CB_equalize_epochs.py

# Step 3: Estimate time-resolved dense wPLI spectral arrays over Morlet CWT ranges
python CB_connectivity.py

# Step 4: Run the subject-stratified cross-validation training cycle 
python CB_Train.py

```

---

## Citation

If you incorporate this system or code architecture within your research pipelines, please attribute the work using the citation below:

```bibtex
@article{ahadzadeh2026dcgatt,
  title={DC-GATT: Dynamic Connectivity Graph Attention Transformer for Multi-Task EEG Classification},
  author={Ahadzadeh, Mohammad and Langrodi, Pouya Taghipour and Afshar, Arian and Baghdadi, Golnaz},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={28},
  number={12},
  pages={7032--7039},
  year={2026},
  doi={10.1109/JBHI.2024.3452410}
}

```

```

```
