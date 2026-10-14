# EEG-Based Motor Imagery Classification for Human-Machine Interface (HMI)

[![Target: BCC 2026](https://img.shields.io/badge/Brain%20Code%20Camp-BCC%202026-6C5CE7.svg)](https://course2026-braincodecamp.web.app/intro.html)
[![Community: BrainCode 101](https://img.shields.io/badge/Community-BrainCode%20101-00B894.svg)](https://braincode101.com)
[![Package Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Acceleration: Apple Silicon MPS](https://img.shields.io/badge/Hardware%20Accel-Apple%20MPS-000000.svg?logo=apple&logoColor=white)](https://pytorch.org/docs/stable/notes/mps.html)
[![BCI Framework: MNE & MOABB](https://img.shields.io/badge/BCI%20Framework-MNE%20%7C%20MOABB-FF6F00.svg)](https://moabb.neurotechx.com/docs/index.html)

This repository hosts the research and engineering implementation of an **EEG-Based Motor Imagery (MI) Brain-Computer Interface (BCI)** designed for **Human-Machine Interface (HMI)** control, prepared for [**Brain Code Camp (BCC) 2026**](https://course2026-braincodecamp.web.app/intro.html) by [**BrainCode 101**](https://braincode101.com).

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
├── pyproject.toml                           # Declarative project & dependency specification (PEP 621)
├── uv.lock                                  # Fully deterministic cross-platform lockfile (Astral uv)
├── .python-version                          # Pinned Python version (3.13)
├── requirements.txt                         # Fallback locked requirements for standard pip
├── README.md                                # Project Documentation & Research Blueprint
├── .gitignore                               # Git ignore specification (excludes data & venvs)
├── Presentation slides/                     # BCC presentation slide deck (PDF)
├── src/                                     # Production HMI Engine Modules
│   ├── __init__.py                          # Package initialization
│   └── hmi_stream.py                        # Circular buffer, 2-stage decoder, debouncer & simulator
├── notebooks/
│   ├── bci_phase1_phase2_benchmarks.ipynb   # Executable research notebook (Phase 1 & 2)
│   └── phase3_continuous_hmi.ipynb          # Continuous asynchronous HMI stream notebook (Phase 3)
├── results/
│   ├── phase1_baseline.csv                  # Exported CSP-LDA & CSP-SVM test metrics
│   ├── phase2_eegnet.csv                    # Exported EEGNet test metrics
│   └── phase3_hmi_metrics.csv               # Exported continuous HMI streaming metrics
├── figures/
│   ├── csp_topomaps/                        # CSP spatial pattern scalp topographies
│   ├── eegnet_spatial_weights/              # Side-by-side CSP vs. EEGNet topomaps
│   ├── model_comparison_benchmark.png       # Publication-ready grouped bar charts
│   └── phase3_continuous_timeline_sub01.png # Chronological continuous streaming timeline
├── data/                                    # [Local only, gitignored] Auto-downloaded MOABB data cache
│   └── MNE-bnci-data/                       # Downloaded BCI Competition IV-2a files (on first run)
└── .venv/                                   # [Local only, gitignored] uv virtual environment
```

---

## 5. Model Specifications

### Experimental Filtering Parity Confirmation
To guarantee a strictly controlled scientific benchmark of spatial filtering capabilities, both Classical (CSP) and Deep Learning (EEGNet) models operated on the exact same temporal filtering and trial epoching via MOABB:
- **Bandpass Filter:** $8.0 - 32.0\text{ Hz}$ (5th-order zero-phase Butterworth filter covering sensorimotor $\mu$ and $\beta$ rhythms).
- **Paradigm:** `moabb.paradigms.MotorImagery(fmin=8.0, fmax=32.0, events=['left_hand', 'right_hand', 'feet', 'tongue'], tmin=0.5, tmax=2.5)`
- **Trial Length:** $2.0\text{ s}$ post-cue window ($T = 500$ timepoints at $250\text{ Hz}$).
- **Scientific Control:** By strictly freezing the frequency band and epoch boundaries across all pipelines, any observed performance discrepancy isolates inductive spatial filtering and covariance regularization mechanics, eliminating spectral confounders.

### Phase 1: Classical Baseline (CSP + Shrinkage-LDA / SVM)
- **CSP Decomposition:** Multiclass One-vs-Rest decomposition with 8 spatial components ($m=4$ component pairs per class dichotomy).
- **Covariance Regularization:** Analytical Ledoit-Wolf shrinkage to regularize empirical sample covariance matrices.
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
- **Training Rigor:** Stratified 80/20 train/validation split from Session 1 with **Exponential Moving Average (EMA, $\alpha=0.7$) Early Stopping** (`min_epochs=15`, `patience=10`) to prevent small-sample overfitting.

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
   - In a single-session calibration regime (288 trials), CSP with Ledoit-Wolf shrinkage decisively outperforms EEGNet in every single subject without exception (Mean $\kappa$: `0.485` vs. `0.285`, Accuracy: `61.41%` vs. `46.37%`).
   - 4 out of 9 subjects (Sub 1, 3, 8, 9) surpass the strict **Substantial BCI Control threshold ($\kappa \ge 0.60$)** using CSP, with Sub 9 reaching $\kappa = 0.713$. In contrast, EEGNet only achieved $\kappa > 0.60$ in a single participant (Sub 9, $\kappa = 0.606$).
2. **Comprehensive Theory: Why Ledoit-Wolf Shrinkage Decisively Outperforms EEGNet:**
   - **Strong Biophysical Inductive Bias vs. Sample Scarcity:** In small-sample regimes ($N = 288$ trials, 72 per class), CSP optimizes spatial projections via closed-form generalized eigenvalue decomposition of class covariance matrices ($\Sigma_1 w = \lambda \Sigma_2 w$). This solves an exact Rayleigh quotient maximizing bandpower variance ratio (ERD/ERS). In contrast, EEGNet (~2,420 parameters) suffers from severe sample starvation when optimizing unconstrained weights from scratch via stochastic gradient descent over non-convex cross-entropy loss.
   - **Ledoit-Wolf Regularization Against Covariance Dispersion:** Estimating empirical sample covariance $S = \frac{1}{T-1} X X^T$ across 22 channels on short time windows ($T=500$) leads to severe eigenvalue dispersion (systematic overestimation of large eigenvalues and underestimation of small eigenvalues). Ledoit-Wolf shrinkage mathematically shrinks $S$ towards a well-conditioned identity target:
     $$\Sigma_{\text{reg}} = (1 - \alpha) S + \alpha \frac{\text{Tr}(S)}{C} I$$
     where the optimal shrinkage intensity $\alpha \in [0, 1]$ is computed analytically to minimize Frobenius mean squared error. This regularized covariance prevents noise overfitting and renders the spatial filters robust against cross-session electrode impedance drift and non-stationarity.
   - **Focal Biophysical Dipoles vs. Dispersed Spatial Weights:** CSP topographies form **clean, biophysically focal dipoles** over primary motor areas ($C_3, C_4, C_z$) perfectly aligned with the sensorimotor homunculus (e.g., Sub 8 Pattern 3 exhibits a clean focal dipole over $C_z$ corresponding to bilateral feet imagery). In contrast, EEGNet's depthwise convolutional filters learn **dispersed, multi-lobed patterns with high spatial frequency noise**. In the absence of explicit spatial priors (e.g., Laplacian continuity or dipolar source constraints), unconstrained gradient descent fits distributed correlations that fail to generalize across sessions.
3. **"BCI-Illiteracy" Phenotype (Subjects 2 & 5):**
   - Both CSP and EEGNet struggle on Subjects 2 and 5 ($\kappa \approx 0.05 - 0.23$). In computational neuroscience, this reflects low intrinsic sensorimotor rhythm modulation or atypical cortical folding (affecting 15–30% of users), proving that the pipeline captures genuine human neurobiology rather than code anomalies.

---

## 7. Quickstart and Execution Guide

### Method 1: Using `uv` (Recommended — Deterministic & Zero-Config)

[Astral `uv`](https://github.com/astral-sh/uv) automatically installs the exact Python 3.13 runtime, resolves cross-platform lockfiles from `uv.lock`, and eliminates library version mismatches across macOS, Linux, and Windows:

```bash
# 1. Clone repository
git clone https://github.com/BallTH24/EEG_Motor_Imagery_BCI.git
cd EEG_Motor_Imagery_BCI

# 2. Sync exact locked environment (automatically provisions Python 3.13 & dependencies into .venv)
uv sync

# 3. Launch Jupyter Lab in the synced environment
uv run jupyter lab
```

### Method 2: Traditional `venv` + `pip` (Fallback)

If `uv` is not installed on your system:

```bash
# 1. Clone repository
git clone https://github.com/BallTH24/EEG_Motor_Imagery_BCI.git
cd EEG_Motor_Imagery_BCI

# 2. Create and activate virtual environment
python3 -m venv BCI_env
source BCI_env/bin/activate       # On Windows: BCI_env\Scripts\activate

# 3. Install locked dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Lab
jupyter lab
```

Open [`notebooks/bci_phase1_phase2_benchmarks.ipynb`](notebooks/bci_phase1_phase2_benchmarks.ipynb) and execute all cells.  
*Note: The BCI Competition IV-2a dataset (~1.6 GB) is automatically downloaded and cached into `data/` via MOABB on first run.*

---

## 8. Phase 3: Continuous Asynchronous HMI Streaming Engine

In closed-loop Human-Machine Interfaces, EEG signals stream continuously without pre-segmented trials. We mitigate the "False Positive Paradox" (out-of-distribution overconfidence on idle data) by implementing a hierarchical gating architecture.

### Architecture & Components (`src/hmi_stream.py`):
1. **Circular Ring Buffer:** Fixed window $T = 500$ samples ($2.0\text{ s}$ at $250\text{ Hz}$), sliding step $\Delta t = 25$ samples ($100\text{ ms}$).
2. **Stage 1 (True Binary Intent Gatekeeper):** A dedicated CSP-LDA binary classifier trained explicitly on inter-trial `REST` vs. motor imagery `ACTIVE` epochs. Windows scoring $P(\text{ACTIVE} \mid X) < 0.50$ are aggressively rejected as `IDLE`.
3. **Stage 2 (Directional Decoder):** High-confidence active windows are routed to the 4-class analytical Ledoit-Wolf CSP + Shrinkage-LDA pipeline for directional classification.
4. **Stage 3 (Temporal Dwell Debouncing & Refractory State Machine):** Requires $K = 3$ consecutive agreeing windows ($300\text{ ms}$ dwell time) before dispatching an actuator command, followed by a $1.0\text{ s}$ refractory lockout to prevent chatter.

### Continuous Stream Benchmark Results (Session 2 `1test`):

| Subject | Trial Detection Rate (%) | Trial Accuracy (%) | Mean Onset Latency (ms) | Median Onset Latency (ms) | Rest FPR (Triggers/min) | Step Latency Median (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 75.00% | 50.67% | 1756.8 ms | 1681.0 ms | 12.41 | 0.127 ms |
| **03** | 86.46% | 52.59% | 1918.0 ms | 1906.0 ms | 32.68 | 0.137 ms |
| **08** | 92.01% | 40.50% | 1326.7 ms | 1011.0 ms | 37.14 | 0.240 ms |

### Key Engineering Insights:
- **Restoration of Precision via Hierarchical Gating:** Injecting explicit inter-trial `REST` data into a Stage 1 binary Gatekeeper doubled Trial Accuracy (from ~25% to ~50%) and drastically cut the Rest False Positive Rate (e.g., from 53.2 down to 12.4 triggers/min for Subject 1).
- **Ultra-Low Compute Overhead:** Per-step execution (including buffer rolling and two consecutive CSP-LDA inferences) takes **$\le 0.24\text{ ms}$**, consuming less than **$0.5\%$** of the $50\text{ ms}$ real-time control budget.

### Maker & Embedded Systems Integration Roadmap:
- **UART / Serial Dispatcher:** Formatted payload packets `[HEADER, COMMAND_ID, CONFIDENCE, CRC]` sent via pySerial to ESP32 / Arduino microcontrollers.
- **Actuator Hardware:** Driving micro-servos for robotic prosthetic hands, wheelchair directional motors, or visual feedback GUI via WebSockets.

---

## References
- Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering, 15(5), 056013.
- Blankertz, B., et al. (2008). *Optimizing spatial filters for robust EEG single-trial analysis*. IEEE Signal Processing Magazine, 25(1), 41-56.
- Jayaram, V., & Barachant, A. (2018). *MOABB: trustworthy algorithm benchmarking for BCIs*. Journal of Neural Engineering, 15(6), 066011.

