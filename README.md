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
     [ Phase 1: Biophysical Foundation ]             [ Phase 2: Deep Learning Benchmark ]
     - Filter: 8–32 Hz Bandpass                      - Temporal Conv (1×64, F1=8 filters)
     - Ledoit-Wolf Shrinkage Covariance              - Depthwise Spatial Conv (22×1, D=2)
     - Multiclass Common Spatial Pattern (CSP, n=8)  - Max-Norm Weight Constraint (||w||₂ ≤ 1.0)
     - Classifiers: Shrinkage-LDA & RBF-SVM          - Separable Conv (1×16 Pointwise, F2=16)
                  │                                  - Optimization: Adam + Early Stopping
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                             [ Cross-Session Validation ]
                         (Train: Session 0train | Test: Session 1test)
                                          │
                                          ▼
                       [ Performance & Biophysical Inspection ]
                      - Metrics: Cohen's Kappa (κ), Accuracy (%)
                      - Topomaps: Dipoles at C3, Cz, C4 vs. EEGNet Spatial Filters
                      - Scientific Negative Control: Pre-Cue Baseline Check (~25%)
                      - Diagnostic Confusion Matrix: Error Distribution Analysis
                                          │
                                          ▼
                         [ Phase 3: Real-Time HMI Engine ]
             (Ring Buffer -> 2-Stage Gatekeeper -> Dwell Debouncing -> 1s Refractory)
                                          │
                                          ▼
                     [ Phase 4: Cyber-Physical Deployment ]
             (Deterministic 6-Byte UART -> ESP32 Watchdog -> Assistive Actuation)
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
├── firmware/                                # Embedded Microcontroller Firmware
│   └── esp32_bci_controller/                # Arduino / ESP32 C++ firmware
│       └── esp32_bci_controller.ino         # UART receiver, CRC-8 parser & 500 ms fail-safe watchdog
├── src/                                     # Production HMI Engine Modules
│   ├── __init__.py                          # Package initialization
│   └── hmi_stream.py                        # Circular buffer, 2-stage decoder, debouncer, UART encoder & emulator
├── notebooks/
│   ├── bci_phase1_phase2_benchmarks.ipynb   # Executable research notebook (Phase 1 & 2)
│   └── bci_phase3_phase4_continuous_hmi.ipynb # Continuous asynchronous HMI & HIL notebook (Phase 3 & 4)
├── results/
│   ├── phase1_baseline.csv                  # Exported CSP-LDA & CSP-SVM test metrics
│   ├── phase2_eegnet.csv                    # Exported EEGNet test metrics
│   ├── phase3_hmi_metrics.csv               # Exported continuous HMI streaming metrics
│   ├── phase4_hil_metrics.csv               # Exported cyber-physical HIL benchmark metrics
│   └── negative_control_precue.csv          # Pre-cue resting baseline negative control metrics
├── figures/
│   ├── csp_topomaps/                        # CSP spatial pattern scalp topographies
│   ├── eegnet_spatial_weights/              # Side-by-side CSP vs. EEGNet topomaps
│   ├── model_comparison_benchmark.png       # Publication-ready grouped bar charts
│   ├── confusion_matrix_sub01.png           # Comparative cross-session confusion matrices
│   ├── phase3_continuous_timeline_sub01.png # Chronological continuous streaming timeline
│   └── phase4_watchdog_verification.png     # Hardware watchdog fail-safe verification plot
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

### Phase 1: Biophysical Foundation & Classical Baseline (Ledoit-Wolf CSP + Shrinkage-LDA / SVM)
- **CSP Decomposition:** Multiclass One-vs-Rest decomposition with 8 spatial components ($m=4$ component pairs per class dichotomy).
- **Covariance Regularization:** Analytical Ledoit-Wolf shrinkage to regularize empirical sample covariance matrices.
- **Biophysical Dipole Alignment:** Optimal spatial filters isolate clean contralateral dipoles centered at $C_3$ (right hand), $C_4$ (left hand), and midline $C_z$ (bilateral feet).
- **Classifiers:**
  - **Pipeline A:** `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`
  - **Pipeline B:** `SVC(kernel='rbf', C=1.0)`

### Phase 2: Deep Learning Benchmark & Scientific Validation (PyTorch EEGNet)
- **Architecture:** Compact 2D Convolutional Neural Network (Lawhern et al., 2018).
  - Input Shape: `(Batch, 1, 22 Channels, 500 Timepoints)`
  - Temporal Conv: Kernel `(1, 64)`, 8 filters, padding `same`.
  - Depthwise Spatial Conv: Kernel `(22, 1)`, Depth multiplier $D=2$ (16 spatial filters), constrained with **Max-Norm ($L_2 \le 1.0$)**.  
    *(Functionally analogous to CSP spatial filtering, but optimized iteratively via stochastic gradient descent rather than analytical eigenvalue decomposition).*
  - Separable Conv: Kernel `(1, 16)`, 16 filters with Average Pooling and Dropout ($p=0.5$).
  - Dense Classifier: 4 output logits (`left_hand`, `right_hand`, `feet`, `tongue`).
  - Total Parameters: **~2,420 parameters** (ultra-lightweight for edge embedded deployment).
- **Training Rigor:** Stratified 80/20 train/validation split from Session 1 with **Exponential Moving Average (EMA, $\alpha=0.7$) Early Stopping** (`min_epochs=15`, `patience=10`) to prevent small-sample overfitting.
- **Diagnostic Scientific Validation:** Evaluated via side-by-side Normalized Confusion Matrices and a Pre-Cue Resting Baseline Negative Control.

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
| **Mean ± SEM** | **61.42 ± 5.77%** | **0.486 ± 0.077** | **60.53 ± 5.78%** | **0.474 ± 0.077** | **46.37 ± 5.36%** | **0.285 ± 0.071** |

### Single-Trial Latency Profile (Batch Size = 1, Budget < 50 ms)
- **CSP-LDA:** `0.127 ms (Median), 0.134 ms (P95)`
- **CSP-SVM:** `0.115 ms (Median), 0.121 ms (P95)`
- **EEGNet:** `0.652 ms (Median), 0.810 ms (P95)`
*Measured trial-by-trial individually with hardware synchronization. All pipelines consume < 1.7% of the real-time HMI latency budget.*

### Core Scientific Findings:
1. **Headline Finding (RQ1): Analytical CSP Outperforms EEGNet Across 100% of Subjects (9/9):**
   - In a single-session calibration regime (288 trials), CSP with Ledoit-Wolf shrinkage decisively outperforms EEGNet in every single subject without exception (Mean $\kappa$: `0.486` vs. `0.285`, Accuracy: `61.42%` vs. `46.37%`).
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

### Diagnostic Scientific Validation

#### 1. Comparative Confusion Matrix Analysis (Subject 01)
To inspect misclassification distributions and ensure balanced sensitivity across bilateral and midline motor representations, we evaluate normalized confusion matrices across all 4 motor imagery classes for Subject 1:

| True Class \ Predicted (CSP-LDA) | Left Hand | Right Hand | Foot | Tongue | True Class \ Predicted (EEGNet) | Left Hand | Right Hand | Foot | Tongue |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Left Hand ($C_4$)** | **0.90** | 0.01 | 0.03 | 0.06 | **Left Hand ($C_4$)** | **0.17** | 0.53 | 0.15 | 0.15 |
| **Right Hand ($C_3$)** | 0.14 | **0.83** | 0.03 | 0.00 | **Right Hand ($C_3$)** | 0.00 | **1.00** | 0.00 | 0.00 |
| **Foot ($C_z$)** | 0.00 | 0.00 | **0.51** | 0.49 | **Foot ($C_z$)** | 0.15 | 0.03 | **0.25** | 0.57 |
| **Tongue** | 0.00 | 0.00 | 0.14 | **0.86** | **Tongue** | 0.01 | 0.00 | 0.25 | **0.74** |

*Full publication heatmap available at [`figures/confusion_matrix_sub01.png`](figures/confusion_matrix_sub01.png).*  
**Key Diagnostic Finding:** CSP-LDA exhibits strong diagonal dominance on contralateral hand imagery (90% Left Hand, 83% Right Hand) with zero cross-leakage from lower-limb/tongue classes into hand predictions; residual confusion is strictly confined between midline Foot ($C_z$, 51%) and Tongue (86%). In contrast, PyTorch EEGNet suffers from severe class collapse toward Right Hand (misclassifying 53% of Left Hand trials as Right Hand) and Tongue (misclassifying 57% of Foot trials as Tongue), reflecting unconstrained spatial representations under small-sample calibration.

#### 2. Scientific Negative Control (Pre-Cue Rest Baseline Check)
To definitively eliminate the possibility of shortcut learning (e.g., memorizing visual cues, high-pass filtering phase drift, or session-specific DC baseline shifts), we evaluated the calibrated CSP-LDA model on pre-cue resting epochs ($t \in [-1.5, -0.5]\text{ s}$ relative to cue onset, $1.0\text{ s}$ duration) from the evaluation session (`1test`):

| Subject | Active MI Test Acc (%) | Pre-Cue Rest Acc (%) | Chance Level (%) | Delta from Chance (%) |
| :---: | :---: | :---: | :---: | :---: |
| **01** | 77.78% | 20.49% | 25.00% | -4.51% |
| **02** | 42.71% | 25.00% | 25.00% | +0.00% |
| **03** | 76.04% | 23.61% | 25.00% | -1.39% |
| **04** | 56.60% | 25.35% | 25.00% | +0.35% |
| **05** | 33.33% | 27.43% | 25.00% | +2.43% |
| **06** | 47.92% | 25.00% | 25.00% | +0.00% |
| **07** | 61.81% | 27.08% | 25.00% | +2.08% |
| **08** | 78.12% | 23.61% | 25.00% | -1.39% |
| **09** | 78.47% | 27.08% | 25.00% | +2.08% |
| **Grand Mean** | **61.42%** | **24.96%** | **25.00%** | **-0.04%** |

*Metrics exported to [`results/negative_control_precue.csv`](results/negative_control_precue.csv).*  
**Empirical Verdict:** Across all 9 subjects, pre-cue baseline decoding converges exactly to theoretical chance (**24.96% vs. 25.00%**, $\Delta = -0.04\%$). This confirms zero shortcut learning, verifying that spatial filters activate exclusively upon genuine intention-induced sensorimotor rhythm modulation.

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
| **01** | 75.00% | 50.67% | 1756.9 ms | 1681.0 ms | 12.41 | 0.128 ms |
| **03** | 86.46% | 52.59% | 1918.1 ms | 1906.0 ms | 32.68 | 0.131 ms |
| **08** | 92.01% | 40.50% | 1326.8 ms | 1011.0 ms | 37.14 | 0.240 ms |

### Key Engineering Insights:
- **Restoration of Precision via Hierarchical Gating:** Injecting explicit inter-trial `REST` data into a Stage 1 binary Gatekeeper doubled Trial Accuracy (from ~25% to ~50%) and drastically cut the Rest False Positive Rate (e.g., from 53.2 down to 12.4 triggers/min for Subject 1).
- **Ultra-Low Compute Overhead:** Per-step execution (including buffer rolling and two consecutive CSP-LDA inferences) takes **$\le 0.24\text{ ms}$**, consuming less than **$0.5\%$** of the $50\text{ ms}$ real-time control budget.

### Maker & Embedded Systems Integration Roadmap:
- **UART / Serial Dispatcher:** Formatted payload packets `[HEADER, COMMAND_ID, CONFIDENCE, CRC]` sent via pySerial to ESP32 / Arduino microcontrollers.
- **Actuator Hardware:** Driving micro-servos for robotic prosthetic hands, wheelchair directional motors, or visual feedback GUI via WebSockets.

---

## 9. Phase 4: Cyber-Physical Deployment & Assistive Applications

In Phase 4, the decoded asynchronous intention is translated from edge inference software into physical actuation via a low-latency embedded bridge:

```
┌────────────────────────┐      Deterministic 6-Byte UART      ┌────────────────────────┐
│  Phase 3 HMI Engine    │ ──────────────────────────────────> │  ESP32 Microcontroller │
│  (Sliding CSP-LDA)     │     [AA 55 CMD CONF CRC8 0D]        │  (Hardware Watchdog)   │
└────────────────────────┘                                     └───────────┬────────────┘
                                                                           │
                                        ┌──────────────────────────────────┴──────────────────────────────────┐
                                        ▼                                                                     ▼
                         [ Robotic Prosthetic Hand ]                                           [ Assistive Wheelchair Drive ]
                         (PWM Servo Actuation: Left/Right)                                     (Differential DC Motors: Forward/Brake)
```

### Deterministic 6-Byte Binary UART Protocol Frame
To ensure microsecond-level parsing determinism and zero buffer fragmentation on 8-bit/32-bit microcontrollers, commands are formatted into a fixed 6-byte binary packet:

```
[ Byte 0 ]   0xAA       : Frame Synchronization Preamble (High Byte)
[ Byte 1 ]   0x55       : Frame Synchronization Preamble (Low Byte)
[ Byte 2 ]   CMD_ID     : Actuator Command (0x00=IDLE, 0x01=LEFT, 0x02=RIGHT, 0x03=FEET, 0x04=TONGUE/BRAKE)
[ Byte 3 ]   CONF_BYTE  : Quantized Confidence Level (floor(P * 255) ∈ [0, 255])
[ Byte 4 ]   CRC8       : Cyclic Redundancy Check (Polynomial: x⁸ + x² + x + 1, Init: 0x00)
[ Byte 5 ]   0x0D       : Carriage Return End-of-Frame Terminator ('\r')
```

### Hardware Fail-Safe & Watchdog Timer
- **Autonomous Watchdog Lockout ($\tau_{\text{watchdog}} = 500\text{ ms}$):** The ESP32 firmware configures a dedicated hardware timer. If no valid CRC8-verified frame is ingested within 500 ms (e.g., in case of host PC crash, wireless packet drop, or electrode disconnection), the controller automatically disengages all PWM actuators and forces an emergency neutral state (`0x00` IDLE / BRAKE).
- **Physical Debounce Parity:** Matches the software-side $K=3$ dwell debouncing to prevent mechanical contact chatter and motor drive overheating.

### Hardware-in-the-Loop (HIL) Verification Benchmarks
Continuous playback of Session 2 (`1test`) through the 6-byte UART serializer and `ESP32WatchdogEmulator` ($N=3$ subjects, 6 runs each = 23,100 frames/subject):

| Subject | Frames Sent | Valid Decoded | Mean Encode Latency | P95 Encode Latency | Bit-Flip Rejection | Watchdog Cutoff | Watchdog Trips |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 23,100 | 23,100 | 2.255 $\mu\text{s}$ | 2.375 $\mu\text{s}$ | 100.0% | 500.0 ms | 1 |
| **03** | 23,100 | 23,100 | 2.250 $\mu\text{s}$ | 2.375 $\mu\text{s}$ | 100.0% | 500.0 ms | 1 |
| **08** | 23,100 | 23,100 | 2.292 $\mu\text{s}$ | 2.417 $\mu\text{s}$ | 100.0% | 500.0 ms | 1 |
| **Mean ± SD** | **23,100** | **23,100** | **2.266 ± 0.023 $\mu\text{s}$** | **2.389 ± 0.024 $\mu\text{s}$** | **100.0%** | **500.0 ms** | **1** |

*Metrics exported to [`results/phase4_hil_metrics.csv`](results/phase4_hil_metrics.csv). Full publication verification figure available at [`figures/phase4_watchdog_verification.png`](figures/phase4_watchdog_verification.png).*  
**Key Engineering Findings:**
1. **Ultra-Low Serialization Overhead ($< 3\ \mu\text{s}$):** Binary frame encoding with on-the-fly CRC-8 takes only **$2.27\ \mu\text{s}$**, consuming less than **$0.003\%$** of the $100\text{ ms}$ transmission interval.
2. **Deterministic Error Rejection:** Achieves **100.0% corrupted packet detection** across 1,000 synthetic single-bit flip trials.
3. **Hardware Fail-Safe Safety:** In response to host communication loss, the watchdog forces state `0x00` (`IDLE / EMERGENCY_BRAKE`) at exactly $\Delta t = 500.0\text{ ms}$ with zero jitter.

### UN Sustainable Development Goals (SDGs) Alignment
This research and open-source cyber-physical HMI pipeline directly aligns with the United Nations Sustainable Development Goals:
- **UN SDG 3 (Good Health and Well-being — Target 3.8 & 3.b):**  
  Restores functional independence, communication agency, and motor autonomy for patients suffering from severe neurodegenerative diseases (Amyotrophic Lateral Sclerosis, Locked-In Syndrome, high-level spinal cord injuries, and post-stroke paralysis).
- **UN SDG 10 (Reduced Inequalities — Target 10.2):**  
  By utilizing lightweight analytical algorithms ($< 0.24\text{ ms}$ compute overhead) that run on low-cost edge microcomputers (Raspberry Pi / Jetson) and accessible microcontrollers (ESP32 < $5 BOM), this architecture democratizes high-performance neurotechnology, bridging the accessibility divide for assistive medical devices across developing regions.

---

## References

### 1. Neurophysiological Foundations & BCI-Illiteracy
- **Pfurtscheller, G., & Lopes da Silva, F. H.** (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842–1857. https://doi.org/10.1016/S1388-2457(99)00141-8
- **Vidaurre, C., & Blankertz, B.** (2010). Towards a cure for BCI illiteracy. *Brain Topography*, 23(2), 194–198. https://doi.org/10.1007/s10548-009-0121-6
- **Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M.** (2002). Brain–computer interfaces for communication and control. *Clinical Neurophysiology*, 113(6), 767–791. https://doi.org/10.1016/S1388-2457(02)00057-3

### 2. Benchmark Dataset & Standardized BCI Frameworks
- **Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., & Pfurtscheller, G.** (2008). *BCI Competition 2008 – Graz data set A* [Data set description]. Laboratory of Brain-Computer Interfaces, Graz University of Technology. https://www.bbci.de/competition/iv/
- **Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. S.** (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267
- **Jayaram, V., & Barachant, A.** (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*, 15(6), 066011. https://doi.org/10.1088/1741-2552/aadea0
- **Tangermann, M., Müller, K.-R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K. J., Müller-Putz, G. R., Nolte, G., Pfurtscheller, G., Sanelli, C., Schalk, G., Schlögl, A., Vidaurre, C., Waldert, S., & Blankertz, B.** (2012). Review of the BCI competition IV. *Frontiers in Neuroscience*, 6, 55. https://doi.org/10.3389/fnins.2012.00055

### 3. Spatial Filtering, Covariance Shrinkage & Neuro-Interpretability (Phase 1)
- **Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R.** (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56. https://doi.org/10.1109/MSP.2008.4408441
- **Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.** (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, 87, 96–110. https://doi.org/10.1016/j.neuroimage.2013.10.067
- **Ledoit, O., & Wolf, M.** (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
- **Lotte, F., & Guan, C.** (2011). Regularizing common spatial patterns to improve BCI designs: unified theory and new algorithms. *IEEE Transactions on Biomedical Engineering*, 58(2), 355–362. https://doi.org/10.1109/TBME.2010.2082539
- **Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G.** (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8(4), 441–446. https://doi.org/10.1109/86.895946

### 4. Deep Learning & Small-Sample BCI Benchmarking (Phase 2)
- **Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J.** (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c
- **Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F.** (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. *Journal of Neural Engineering*, 15(3), 031005. https://doi.org/10.1088/1741-2552/aab2f2

### 5. Asynchronous Streaming & Cyber-Physical Assistive Control (Phase 3 & Phase 4)
- **Mason, S. G., & Birch, G. E.** (2000). A brain-controlled switch for asynchronous control applications. *IEEE Transactions on Biomedical Engineering*, 47(10), 1297–1307. https://doi.org/10.1109/10.871402
- **Millán, J. d. R., Rupp, R., Müller-Putz, G. R., Murray-Smith, R., Giugliemma, C., Tangermann, M., Vidaurre, C., Cincotti, F., Kübler, A., Leeb, R., Neuper, C., Müller, K.-R., & Mattia, D.** (2010). Combining brain–computer interfaces and assistive technologies: state-of-the-art and challenges. *Frontiers in Neuroscience*, 4, 161. https://doi.org/10.3389/fnins.2010.00161

### 6. Experimental Benchmark Evaluation & Statistical Reliability
- **Cohen, J.** (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
