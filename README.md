# Tomato Leaf Disease Classification: Few-Shot Data Augmentation Comparison

## Research Overview

This study compares the effectiveness of various data augmentation methods for tomato leaf disease classification under a few-shot learning scenario. The compared methods are:

| Abbreviation | Method | Description |
|--------------|--------|-------------|
| **Baseline** | No augmentation | Original images only (20 images/class) |
| **TDA** | Traditional Data Augmentation | Classical augmentation (flip, rotate, color jitter) |
| **SD** | Stable Diffusion img2img | Generate new images using Stable Diffusion |
| **GAN** | DCGAN | Generate new images using a self-trained GAN |
| **CDA** | Combined TDA + SD | Combination of TDA and SD |

**Classification Model:** EfficientNet-B0 (pretrained on ImageNet)

**Source Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 5 tomato leaf disease classes

**5 Classes:**
1. `Tomato___Early_blight`
2. `Tomato___healthy`
3. `Tomato___Late_blight`
4. `Tomato___Leaf_Mold`
5. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`

---

## System Requirements

- **NVIDIA GPU** with CUDA (mandatory — all scripts check `torch.cuda.is_available()`)
- **Python** 3.8+
- **VRAM** ≥ 6 GB (for Stable Diffusion); ≥ 4 GB (for other steps)
- **Disk Space** ≥ 10 GB (for dataset + model weights)

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Preparing the Source Data](#2-preparing-the-source-data)
3. [Experiment Pipelines](#3-experiment-pipelines)
4. [Script Details](#4-script-details)
5. [Directory Structure](#5-directory-structure)
6. [Output & Results](#6-output--results)
7. [Additional Documentation](#7-additional-documentation)

---

## 1. Environment Setup

### 1.1. Clone this repository

```bash
git clone https://github.com/SirKimCh/tomato_vs.git
cd tomato_vs
```

### 1.2. Clone the parent project (dependency)

Code in `tomato_vs/` depends on source code from the [leaf-disease-ai](https://github.com/junayed-hasan/leaf-disease-ai) project. The `tomato_vs/` directory must be placed inside `leaf-disease-ai/`:

```
leaf-disease-ai/               ← Parent project
├── src/                        ← Source code imported by tomato_vs
│   ├── models/
│   │   └── efficientnet_b0.py  ← EfficientNet-B0 model
│   └── configurations/
│       └── augmentation_config.py  ← TDA configuration
├── requirements.txt
└── tomato_vs/                  ← THIS RESEARCH'S CODE DIRECTORY
    ├── 01_data_setup.py
    ├── ...
```

```bash
git clone https://github.com/junayed-hasan/leaf-disease-ai.git
cd leaf-disease-ai

# Clone tomato_vs inside the parent project
git clone https://github.com/SirKimCh/tomato_vs.git tomato_vs
```

### 1.3. Create virtual environment & install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

```bash
pip install -r requirements.txt
pip install google-genai
```

> **Note:** The `google-genai` package is required by `02_2_gen_sd.py` (generates prompts via Gemini API).

### 1.4. Configure API key (for Stable Diffusion)

Create a `.env` file inside the `tomato_vs/` directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get your API key at: [Google AI Studio](https://aistudio.google.com/apikey)

For detailed installation instructions, see: [docs/INSTALLATION.md](docs/INSTALLATION.md)

---

## 2. Preparing the Source Data

Download the **PlantVillage** dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and keep only the following **5 tomato classes**:

```
tomato_vs/Data_OG/
├── Tomato___Early_blight/        (~1000 images)
├── Tomato___healthy/             (~1591 images)
├── Tomato___Late_blight/         (~1909 images)
├── Tomato___Leaf_Mold/           (~952 images)
└── Tomato___Tomato_Yellow_Leaf_Curl_Virus/  (~5357 images)
```

> **Important:** Directory names must match exactly as shown above (with the `Tomato___` prefix).

---

## 3. Experiment Pipelines

There are **3 independent pipelines**, run in order:

### Pipeline A — Quick comparison: Baseline vs TDA vs SD (automated)

```bash
cd leaf-disease-ai
python tomato_vs/05_master_run.py
```

This script automatically runs in sequence: `01 → 02_1 → 02_2 → 03 → 04`.
Details: [docs/PIPELINE_A.md](docs/PIPELINE_A.md)

### Pipeline B — Adding GAN to the comparison

```bash
# Step 1: Setup data (if Pipeline A has not been run yet)
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100

# Step 2: Generate TDA data
python tomato_vs/02_1_gen_tda.py

# Step 3: Generate GAN data
python tomato_vs/02_3_gen_gan.py

# Step 4: Run experiments for Baseline + TDA + SD
python tomato_vs/03_run_experiments.py --output_dir tomato_vs/Results/my_run

# Step 5: Run GAN experiment separately
python tomato_vs/03_1_run_gan_experiment.py

# Step 6: Combined visualization
python tomato_vs/04_1_visualize_with_gan.py --input_dir tomato_vs/Results/my_run
```

### Pipeline C — Final comparison & Transfer Learning

> **Requirement:** The `Data_ST/` directory must contain pre-generated data (baseline, tda_x5, sd_x5).

```bash
# Final comparison: Baseline vs Combined (TDA+SD)
python tomato_vs/06_final_comparison.py

# Transfer learning: Pretrained vs No-Pretrained
python tomato_vs/07_transfer_learning_comparison.py
```

Details: [docs/PIPELINE_C.md](docs/PIPELINE_C.md)

---

## 4. Script Details

| Script | Function | Input | Output |
|--------|----------|-------|--------|
| `01_data_setup.py` | Split source data into train/test | `Data_OG/` | `datasets/baseline/train/`, `datasets/test/` |
| `02_1_gen_tda.py` | Generate images via TDA (flip, rotate, color jitter) ×5 | `datasets/baseline/train/` | `datasets/tda_x5/train/` |
| `02_2_gen_sd.py` | Generate images via Stable Diffusion img2img ×5 | `datasets/baseline/train/` | `datasets/sd_x5/train/` |
| `02_3_gen_gan.py` | Train DCGAN per-class → generate 80 images/class | `datasets/baseline/train/` | `datasets/gan_x5/train/` |
| `03_run_experiments.py` | Train & evaluate: Baseline, TDA, SD | `datasets/*/train/`, `datasets/test/` | `Results/metrics_summary.csv`, confusion matrices |
| `03_1_run_gan_experiment.py` | Train & evaluate: GAN | `datasets/gan_x5/train/` | `Results/gan_metrics_summary.csv` |
| `04_visualize_results.py` | Plot comparison charts (3 methods) | `metrics_summary.csv` | `summary_comparison.png`, charts |
| `04_1_visualize_with_gan.py` | Plot comparison charts (4 methods) | `metrics_summary.csv` + `gan_metrics_summary.csv` | `summary_comparison_with_gan.png` |
| `05_master_run.py` | Automated Pipeline A runner | — | Full results |
| `06_final_comparison.py` | Compare Baseline vs Combined (TDA+SD) | `Data_ST/`, `Data_OG/` | Metrics + confusion matrices |
| `07_transfer_learning_comparison.py` | Compare Pretrained vs From-scratch | `Data_ST/`, `Data_OG/` | 2 result sets |
| `redraw_confusion_matrix.py` | Redraw confusion matrices from CSV | `redraw.csv` | `redraw/*.png` |

For detailed descriptions, see: [docs/SCRIPTS_DETAIL.md](docs/SCRIPTS_DETAIL.md)

---

## 5. Directory Structure

```
tomato_vs/
├── 01_data_setup.py              # Step 1: Split data
├── 02_1_gen_tda.py               # Step 2a: Generate TDA
├── 02_2_gen_sd.py                # Step 2b: Generate SD
├── 02_3_gen_gan.py               # Step 2c: Generate GAN
├── 03_run_experiments.py         # Step 3: Train & eval (Baseline/TDA/SD)
├── 03_1_run_gan_experiment.py    # Step 3: Train & eval (GAN)
├── 04_visualize_results.py       # Step 4: Visualize 3 methods
├── 04_1_visualize_with_gan.py    # Step 4: Visualize 4 methods
├── 05_master_run.py              # Auto-run Pipeline A
├── 06_final_comparison.py        # Final: Baseline vs Combined
├── 07_transfer_learning_comparison.py  # Transfer learning
├── redraw_confusion_matrix.py    # Utility: redraw CM from CSV
├── redraw.csv                    # Confusion matrix data
├── .env                          # (create manually) API keys
│
├── Data_OG/                      # (download manually) Original PlantVillage dataset
├── Data_ST/                      # (generated) Static datasets for Pipeline C
├── datasets/                     # (generated) Working datasets
├── Results/                      # (generated) Experiment results
├── redraw/                       # (generated) Redrawn confusion matrix images
└── docs/                         # Detailed documentation
```

---

## 6. Output & Results

After running, all results are stored in `Results/`:

- **`metrics_summary.csv`** — Aggregated metrics table (Accuracy, Precision, Recall, F1, MCC, AUC)
- **`training_curves.csv`** — Loss/Accuracy per epoch
- **`cm_*.png`** — Confusion matrix per trial
- **`cm_aggregate_*.png`** — Aggregated confusion matrix over 5 trials
- **`loss_curve_*.png`** — Loss/Accuracy curves
- **`summary_comparison.png`** — Overall comparison chart

---

## 7. Additional Documentation

| File | Content |
|------|---------|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Detailed installation guide |
| [docs/SCRIPTS_DETAIL.md](docs/SCRIPTS_DETAIL.md) | Detailed script descriptions (parameters, algorithms, output) |
| [docs/PIPELINE_A.md](docs/PIPELINE_A.md) | Pipeline A: Automated Master Run |
| [docs/PIPELINE_C.md](docs/PIPELINE_C.md) | Pipeline C: Final Comparison & Transfer Learning |
| [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) | Model architecture & training configuration |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors & solutions |
