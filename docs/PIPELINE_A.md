# Pipeline A — Master Run (Automated)

## Overview

**Script:** `05_master_run.py`

Pipeline A automates the entire experimental workflow: split data → generate TDA → generate SD → train & evaluate → visualization. Supports grid search over Stable Diffusion parameters.

## Pipeline Diagram

```
┌──────────────────────┐
│   05_master_run.py   │
│   (Orchestrator)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  01_data_setup.py    │  Split Data_OG → baseline/train + test
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  02_1_gen_tda.py     │  baseline/train → tda_x5/train (×5)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  02_2_gen_sd.py      │  baseline/train → sd_x5/train (×5)
│  (strength, guidance)│  ← parameters change each run
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  03_run_experiments  │  Train EfficientNet-B0 on 3 datasets
│  (5 trials each)    │  Evaluate on test set
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Backup generated    │  Save generated images to Results/
│  images              │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  04_visualize_results│  Plot comparison charts
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Cleanup datasets/   │  Delete datasets/ to prepare for next run
└──────────────────────┘
```

## How to Run

```bash
cd leaf-disease-ai
python tomato_vs/05_master_run.py
```

The script will prompt interactively:

```
Select mode:
  1. Random (Run 1 random combination)
  2. Full (Run all 9 combinations)

Enter your choice (1 or 2): 2
Enter train count per class (default=10): 20
Enter test count per class (default=50): 100
```

## Two Modes

### Mode 1: Random
- Randomly selects **1 combination** (strength, guidance) → runs the pipeline once
- Estimated time: ~1–2 hours (depending on GPU)

### Mode 2: Full
- Runs **all 9 combinations** sequentially
- Grid search:

| # | Strength | Guidance |
|---|----------|----------|
| 1 | 0.35 | 6.0 |
| 2 | 0.35 | 7.5 |
| 3 | 0.35 | 9.0 |
| 4 | 0.50 | 6.0 |
| 5 | 0.50 | 7.5 |
| 6 | 0.50 | 9.0 |
| 7 | 0.65 | 6.0 |
| 8 | 0.65 | 7.5 |
| 9 | 0.65 | 9.0 |

- Estimated time: ~10–18 hours

## Output

Each run creates a directory inside `Results/`:

```
Results/
├── 20260101_120000_train20_s0.35_g7.5/
│   ├── experiment_config.txt          # Configuration
│   ├── metrics_summary.csv            # Results
│   ├── training_curves.csv            # Learning curves
│   ├── used_prompts.json              # SD prompts used
│   ├── summary_comparison.png         # Comparison chart
│   ├── loss_acc_aggregated.png        # Learning curves plot
│   ├── cm_*.png                       # Confusion matrices
│   ├── loss_curve_*.png               # Loss curves
│   └── generated_images_backup/       # Backup of generated images
│       ├── baseline/
│       ├── tda_x5/
│       └── sd_x5/
├── 20260101_140000_train20_s0.5_g6.0/
│   └── ...
└── ...
```

## Important Notes

1. **Automatic cleanup:** The `datasets/` directory is deleted after each run. Images are backed up in `generated_images_backup/`.
2. **If it crashes mid-run:** Re-run from the beginning or run each step manually.
3. **GAN is not included in this pipeline.** Run `02_3_gen_gan.py` and `03_1_run_gan_experiment.py` separately.
