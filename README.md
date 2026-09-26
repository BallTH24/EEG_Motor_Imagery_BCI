# EEG-Based Motor Imagery Classification for Human-Machine Interface (HMI)

[![Target: BCC 2026](https://img.shields.io/badge/Brain%20Code%20Camp-BCC%202026-6C5CE7.svg)](https://braincode.club)
[![Python: 3.13.7](https://img.shields.io/badge/Python-3.13.7-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Acceleration: Apple Silicon MPS](https://img.shields.io/badge/Hardware%20Accel-Apple%20MPS-000000.svg?logo=apple&logoColor=white)](https://pytorch.org/docs/stable/notes/mps.html)
[![BCI Framework: MNE & MOABB](https://img.shields.io/badge/BCI%20Framework-MNE%20%7C%20MOABB-FF6F00.svg)](https://moabb.neurotechx.org/)

This repository hosts the research and engineering implementation of an **EEG-Based Motor Imagery (MI) Brain-Computer Interface (BCI)** designed for **Human-Machine Interface (HMI)** control, prepared for **Brain Code Camp (BCC) 2026**.

The project investigates the interplay between classical biophysical signal processing (**Common Spatial Pattern with Shrinkage**) and compact end-to-end deep learning (**PyTorch EEGNet**), validating performance on the benchmark **BCI Competition IV-2a** across 9 subjects under a rigorous cross-session evaluation protocol.

---

## 1. Neurophysiological Foundation

```
                      [ Motor Cortex Topography ]
                              (Frontal)
                                  Fz
                         FC3     FCz     FC4
                     (Left)                (Right)
               C5     [C3]       Cz       [C4]     C6
                 (Right Hand)   (Feet)  (Left Hand)
                         CP3     CPz     CP4
                                  Pz
                             (Occipital)
```

- **Sensorimotor Rhythms (SMR):**
  - $\mu$-band (8–12 Hz) and $\beta$-band (13–30 Hz) oscillations originating from the primary motor and somatosensory cortices ($C_3, C_z, C_4$).
- **Event-Related Desynchronization (ERD):**
  - Contralateral suppression of $\mu$/$\beta$ bandpower during motor imagery (e.g., imagining right-hand movement induces ERD over the left hemisphere electrode $C_3$).
- **Event-Related Synchronization (ERS):**
  - Rebound of rhythm bandpower post-imagery cessation or ipsilateral motor cortex idling.

---

## 2. Research Questions (BCC 2026)

1. **Inductive Bias vs. Data Scarcity:**  
   Does the strong biophysical spatial inductive bias inherent in **Common Spatial Patterns (CSP)** outperform an end-to-end compact deep neural network (**EEGNet**) in small-sample cross-session regimes (288 trials per session)?
2. **Spatial Filter Neuro-Interpretability:**  
   Do the learned depthwise convolutional weights in EEGNet converge to anatomically congruent sensorimotor patterns ($C_3, C_z, C_4$) without explicit spatial supervision?
3. **Real-Time HMI Compute & Latency Budget:**  
   What are the computational and latency trade-offs between CSP-LDA/SVM ($O(C^2)$ feature extraction) and PyTorch tensor inference on edge compute devices (< 50 ms closed-loop requirement)?

---

## 3. System Architecture

```
                                [ Raw EEG Ingestion ]
                    (BCI Competition IV-2a: 22 Channels @ 250 Hz)
                                          │
                                          ▼
                               [ Temporal Windowing ]
                       (t ∈ [0.5s, 2.5s] Post-Cue, T=500 samples)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Phase 1: Classical Pipeline ]                 [ Phase 2: Compact Deep Learning ]
     - Filter: 8–32 Hz Bandpass                      - Temporal Conv (1×64, F1=8 filters)
     - Ledoit-Wolf Shrinkage Covariance              - Depthwise Spatial Conv (22×1, D=2)
     - Multiclass Common Spatial Pattern (CSP, n=8)  - Max-Norm Weight Constraint (||w||₂ ≤ 1.0)
     - Classifiers: Shrinkage-LDA & RBF-SVM          - Separable Conv (1×16 Pointwise, F2=16)
                  │                                  - Optimization: Adam + ReduceLROnPlateau
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                             [ Cross-Session Validation ]
                         (Train: Session 0train | Test: Session 1test)
                                          │
                                          ▼
                       [ Performance & Biophysical Inspection ]
                      - Metrics: Cohen's Kappa (κ), Accuracy (%)
                      - Topomaps: CSP Patterns vs. EEGNet Spatial Filters
                                          │
                                          ▼
                         [ Phase 3: Real-Time HMI Engine ]
             (Ring Buffer -> Sliding Inference -> Dwell Debouncing -> Serial/MQTT)
```

---

## 4. Repository Layout

```
EEG_Motor_Imagery_BCI/
├── README.md                                # Project Documentation & Research Blueprint
├── requirements.txt                         # Full locked Python dependencies
├── .gitignore                               # Git ignore specification (excludes data & venv)
├── BCI_env/                                 # [Local only, gitignored] Python 3.13 virtual environment
├── data/                                    # [Local only, gitignored] Auto-downloaded MOABB data cache
│   └── MNE-bnci-data/                       # Downloaded BCI Competition IV-2a files (on first run)
├── notebooks/
│   └── bci_phase1_phase2_benchmarks.ipynb   # Executable research notebook (Phase 1 & 2)
├── results/
│   ├── phase1_baseline.csv                  # Exported CSP-LDA & CSP-SVM test metrics
│   └── phase2_eegnet.csv                    # Exported EEGNet test metrics
└── figures/
    ├── csp_topomaps/                        # CSP spatial pattern scalp topographies
    ├── eegnet_spatial_weights/              # Side-by-side CSP vs. EEGNet topomaps
    └── model_comparison_benchmark.png       # Publication-ready grouped bar charts
```

---

## 5. Model Specifications

### Phase 1: Classical Baseline
- **CSP Decomposition:** Multiclass One-vs-Rest decomposition with 8 spatial components.
- **Covariance Regularization:** Ledoit-Wolf shrinkage to ensure well-conditioned covariance matrices in high-dimensional channel space.
- **Classifiers:**
  - **Pipeline A:** `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`
  - **Pipeline B:** `SVC(kernel='rbf', C=1.0)`

### Phase 2: PyTorch EEGNet
- **Architecture:** Compact 2D Convolutional Neural Network (Lawhern et al., 2018).
  - Input Shape: `(Batch, 1, 22 Channels, 500 Timepoints)`
  - Temporal Conv: Kernel `(1, 64)`, 8 filters, padding `same`.
  - Depthwise Spatial Conv: Kernel `(22, 1)`, Depth multiplier $D=2$ (16 spatial filters), constrained with **Max-Norm ($L_2 \le 1.0$)**.  
    *(Functionally analogous to CSP spatial filtering, but optimized iteratively via stochastic gradient descent rather than analytical eigenvalue decomposition).*
  - Separable Conv: Kernel `(1, 16)`, 16 filters with Average Pooling and Dropout ($p=0.5$).
  - Dense Classifier: 4 output logits (`left_hand`, `right_hand`, `feet`, `tongue`).
  - Total Parameters: **~2,420 parameters** (ultra-lightweight for edge embedded deployment).
- **Training Rigor:** Stratified 80/20 train/validation split from Session 1 with **Early Stopping** (patience = 10 on validation loss) to prevent small-sample overfitting.

---

## 6. Cross-Session Benchmark Results (All 9 Subjects)

Cross-Session evaluation across all 9 subjects of BCI Competition IV-2a (Train: `0train` [288 trials] $\to$ Test: `1test` [288 trials]):

| Subject | CSP-LDA Acc (%) | CSP-LDA Kappa ($\kappa$) | CSP-SVM Acc (%) | CSP-SVM Kappa ($\kappa$) | EEGNet Acc (%) | EEGNet Kappa ($\kappa$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 77.78% | 0.704 | **78.12%** | **0.708** | 53.82% | 0.384 |
| **02** | **42.71%** | **0.236** | 42.01% | 0.227 | 28.82% | 0.051 |
| **03** | **76.04%** | **0.681** | 73.96% | 0.653 | 66.67% | 0.556 |
| **04** | **56.60%** | **0.421** | **56.60%** | **0.421** | 36.11% | 0.148 |
| **05** | **33.33%** | **0.111** | 31.94% | 0.093 | 29.17% | 0.056 |
| **06** | **47.92%** | **0.306** | 46.88% | 0.292 | 34.38% | 0.125 |
| **07** | **61.81%** | **0.491** | 61.11% | 0.481 | 40.28% | 0.204 |
| **08** | **78.12%** | **0.708** | **78.12%** | **0.708** | 57.64% | 0.435 |
| **09** | **78.47%** | **0.713** | 76.04% | 0.681 | 70.49% | 0.606 |
| **Mean ± SEM** | **61.41 ± 5.75%** | **0.485 ± 0.077** | **60.54 ± 5.67%** | **0.474 ± 0.076** | **46.37 ± 5.36%** | **0.285 ± 0.071** |

### Single-Trial Latency Profile (Batch Size = 1, Budget < 50 ms)
- **CSP-LDA:** `0.127 ms (Median), 0.136 ms (P95)`
- **CSP-SVM:** `0.118 ms (Median), 0.126 ms (P95)`
- **EEGNet:** `0.679 ms (Median), 0.843 ms (P95)`
*Measured trial-by-trial individually with hardware synchronization. All pipelines consume < 1.7% of the real-time HMI latency budget.*

### Core Scientific Findings:
1. **Headline Finding (RQ1): Analytical CSP Outperforms EEGNet Across 100% of Subjects (9/9):**
   - In a single-session regime (288 trials), CSP with Ledoit-Wolf shrinkage decisively outperforms EEGNet in every single subject without exception (Mean $\kappa$: `0.485` vs. `0.285`).
   - 4 out of 9 subjects (Sub 1, 3, 8, 9) surpass the strict **Substantial BCI Control threshold ($\kappa \ge 0.60$)** using CSP. Analytical inductive bias beats data-starved deep learning when training data is scarce.
2. **Interpretability & Topomap Finding (RQ2): Focal Dipoles vs. Dispersed Weights:**
   - CSP topographies form **clean, biophysically focal dipoles** over the motor homunculus (e.g., Subject 8 Pattern 3 exhibits a clean focal dipole over $C_z$ corresponding to bilateral feet imagery).
   - In contrast, EEGNet learned depthwise filters exhibit **dispersed, multi-lobed patterns with higher spatial frequency noise**. Without explicit spatial priors or massive pre-training data, gradient descent fits distributed correlations rather than clean dipolar brain sources.
3. **"BCI-Illiteracy" Phenotype (Subjects 2 & 5):**
   - Both CSP and EEGNet struggle on Subjects 2 and 5 ($\kappa \approx 0.05 - 0.23$). In computational neuroscience, this reflects low intrinsic sensorimotor rhythm modulation or atypical cortical folding (affecting 15–30% of users), proving that the pipeline captures genuine human neurobiology rather than code anomalies.

---

## 7. Quickstart and Execution Guide

### 1. Clone & Set Up Environment
To reproduce the benchmarks on your local machine:

```bash
# Clone the repository
git clone https://github.com/BallTH24/EEG_Motor_Imagery_BCI.git
cd EEG_Motor_Imagery_BCI

# Create and activate virtual environment
python3 -m venv BCI_env
source BCI_env/bin/activate       # On Windows: BCI_env\Scripts\activate

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark Notebook
Launch Jupyter Lab or open directly in VS Code:

```bash
jupyter lab
```

Open [`notebooks/bci_phase1_phase2_benchmarks.ipynb`](notebooks/bci_phase1_phase2_benchmarks.ipynb) and execute all cells.  
*Note: The BCI Competition IV-2a dataset will be automatically downloaded and cached into `data/` via MOABB on first run.*

---

## 8. Phase 3 Roadmap: Real-Time HMI and Embedded Integration

For the final prototype in **Brain Code Camp (BCC 2026)**:
- [ ] **Lab Streaming Layer (LSL):** Online EEG stream ingestion with ring-buffer sliding window (2.0s window, 100ms step).
- [ ] **Riemannian Manifold Pipeline:** Covariance estimation on symmetric positive-definite (SPD) manifold + Tangent Space projection.
- [ ] **Decision Smoothing:** Dwell-time state machine and confidence thresholding to eliminate false positives in continuous idle state.
- [ ] **Hardware Actuation:** Microcontroller dispatch via UART / MQTT (ESP32/Arduino) to actuate robotic hand, assistive wheelchair, or drone simulator.

---

## References
- Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering, 15(5), 056013.
- Blankertz, B., et al. (2008). *Optimizing spatial filters for robust EEG single-trial analysis*. IEEE Signal Processing Magazine, 25(1), 41-56.
- Jayaram, V., & Barachant, A. (2018). *MOABB: trustworthy algorithm benchmarking for BCIs*. Journal of Neural Engineering, 15(6), 066011.

