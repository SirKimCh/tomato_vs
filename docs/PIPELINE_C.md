# Pipeline C — Final Comparison & Transfer Learning

## Overview

Pipeline C consists of 2 independently run scripts, using pre-prepared data in `Data_ST/` (Static Data):

1. **`06_final_comparison.py`** — Compare Baseline vs Combined (TDA + SD)
2. **`07_transfer_learning_comparison.py`** — Compare the effect of Transfer Learning

## `Data_ST/` Data Requirements

The `Data_ST/` directory contains data that was previously generated from Pipeline A (or by running steps 01 → 02 manually). Structure:

```
Data_ST/
├── baseline/
│   └── train/
│       ├── Tomato___Early_blight/     (20 original images)
│       ├── Tomato___healthy/          (20 images)
│       ├── Tomato___Late_blight/      (20 images)
│       ├── Tomato___Leaf_Mold/        (20 images)
│       └── Tomato___Tomato_Yellow.../  (20 images)
├── tda_x5/
│   └── train/
│       ├── Tomato___Early_blight/     (100 images = 20 original + 80 aug)
│       └── ...
├── sd_x5/
│   └── train/
│       ├── Tomato___Early_blight/     (100 images = 20 original + 80 sd)
│       └── ...
└── test/                              (created by script, does not need to exist beforehand)
```

> **How to create `Data_ST/`:** Copy the results from Pipeline A or run manually:
> ```bash
> python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
> python tomato_vs/02_1_gen_tda.py
> python tomato_vs/02_2_gen_sd.py
> # Copy datasets/baseline, datasets/tda_x5, datasets/sd_x5 into Data_ST/
> ```

---

## 06_final_comparison.py

### Purpose
Evaluate: **Does combining all augmentation methods (TDA + SD) outperform the Baseline?**

### Workflow

```
Step 1: Create test set from Data_OG/
        (exclude images already used in baseline training)
        → Data_ST/test/ (100 images/class)

Step 2: Create combined dataset
        = original baseline + TDA augmented + SD generated
        → Data_ST/combined_tda_sd/train/

Step 3: Train & Evaluate
        - Exp 1: baseline (20 images/class)
        - Exp 2: combined_tda_sd (~180 images/class)
        × 5 trials per experiment
```

### Training Config
- **Model:** EfficientNet-B0, pretrained=True
- **Freezing:** Partial (classifier + last 3 feature blocks)
- Same configuration as `03_run_experiments.py`

### How to Run

```bash
python tomato_vs/06_final_comparison.py

# Or specify an output directory:
python tomato_vs/06_final_comparison.py --output_dir tomato_vs/Results/my_final
```

### Output

```
Results/{timestamp}_final_comparison/
├── metrics_summary.csv
├── training_curves.csv
├── model_params.csv              # Model parameter info
├── cm_baseline_trial*.png
├── cm_combined_tda_sd_trial*.png
├── cm_aggregate_baseline.png
├── cm_aggregate_combined_tda_sd.png
└── loss_curve_*.png
```

---

## 07_transfer_learning_comparison.py

### Purpose
Evaluate the impact of **pretrained weights** on classification performance.

### Two Configurations Compared

| Config | Pretrained | Freezing | Meaning |
|--------|-----------|----------|---------|
| **Config 1**: `noptr_nofre` | ❌ No | ❌ No | Train from scratch, all layers |
| **Config 2**: `ptr_nofre` | ✅ Yes (ImageNet) | ❌ No | Fine-tune all layers |

> **Note:** Both configs have **no layer freezing** → all model layers are trained. The only difference is the initial weights (random vs ImageNet pretrained).

### Each Config Runs 2 Experiments

1. **baseline** (20 images/class)
2. **combined_tda_sd** (~180 images/class)

→ Total: 2 configs × 2 experiments × 5 trials = **20 training runs**

### How to Run

```bash
python tomato_vs/07_transfer_learning_comparison.py
```

### Output

```
Results/
├── {timestamp}_noptr_nofre/       # Config 1: No Pretrained
│   ├── metrics_summary.csv
│   ├── model_params.csv
│   ├── training_curves.csv
│   ├── cm_*.png
│   └── loss_curve_*.png
└── {timestamp}_ptr_nofre/         # Config 2: Pretrained
    ├── metrics_summary.csv
    ├── model_params.csv
    ├── training_curves.csv
    ├── cm_*.png
    └── loss_curve_*.png
```

### Comparing Results

Read the 2 `metrics_summary.csv` files from the 2 output directories:

```python
import pandas as pd

df1 = pd.read_csv('Results/{timestamp}_noptr_nofre/metrics_summary.csv')
df2 = pd.read_csv('Results/{timestamp}_ptr_nofre/metrics_summary.csv')

# Filter AVG rows
print("No Pretrained:")
print(df1[df1['Trial'] == 'AVG'][['Exp', 'Acc', 'F1', 'MCC', 'AUC']])

print("\nPretrained:")
print(df2[df2['Trial'] == 'AVG'][['Exp', 'Acc', 'F1', 'MCC', 'AUC']])
```
