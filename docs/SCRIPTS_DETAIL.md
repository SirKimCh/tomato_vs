# Detailed Script Descriptions

## 01_data_setup.py — Split Data into Train/Test

### Function
Randomly sample images from `Data_OG/` to create the training (baseline) and test sets. Ensures no overlap between train and test.

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_count` | 20 | Number of training images per class |
| `--test_count` | 100 | Number of test images per class |

### How to Run

```bash
# Run from the leaf-disease-ai/ directory
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
```

### Algorithm
1. Iterate through each class in `Data_OG/`
2. Randomly shuffle all images (seed=42)
3. Take the first `train_count` images → copy to `datasets/baseline/train/{class}/`
4. Take the next `test_count` images → copy to `datasets/test/{class}/`
5. Assert no overlap between train and test sets
6. Save metadata to `datasets/dataset_info.json`

### Output

```
datasets/
├── baseline/train/
│   ├── Tomato___Early_blight/     (20 images)
│   ├── Tomato___healthy/          (20 images)
│   ├── Tomato___Late_blight/      (20 images)
│   ├── Tomato___Leaf_Mold/        (20 images)
│   └── Tomato___Tomato_Yellow.../  (20 images)
├── test/
│   ├── Tomato___Early_blight/     (100 images)
│   └── ...
└── dataset_info.json
```

> **Note:** The script will **delete the entire** `datasets/` directory before creating new data.

---

## 02_1_gen_tda.py — Generate Images via Traditional Data Augmentation

### Function
Classical data augmentation: each original image produces 4 additional augmented images (total 5×).

### Augmentation Techniques Used
- Horizontal Flip
- Random Rotation
- Color Jitter (brightness, contrast, saturation, hue)
- Resize + Normalize (ImageNet mean/std)

Augmentation configuration is loaded from `src/configurations/augmentation_config.py`.

### How to Run

```bash
python tomato_vs/02_1_gen_tda.py
```

### Output

```
datasets/tda_x5/train/
├── Tomato___Early_blight/
│   ├── original_img.jpg           (20 original images copied)
│   ├── original_img_aug0.jpg      (augmented)
│   ├── original_img_aug1.jpg
│   ├── original_img_aug2.jpg
│   └── original_img_aug3.jpg
│   → Total: 100 images/class (20 original + 80 augmented)
└── ...
```

---

## 02_2_gen_sd.py — Generate Images via Stable Diffusion

### Function
Uses **Stable Diffusion v1.5** (img2img pipeline) to generate new images from original images. Disease description prompts are automatically generated using the **Gemini API**.

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--strength` | 0.35 | Image transformation strength (0.0=keep original, 1.0=completely new) |
| `--guidance` | 7.5 | Guidance scale (how closely the output follows the prompt) |
| `--output_log_dir` | `tomato_vs` | Directory to save prompt logs |

### How to Run

```bash
python tomato_vs/02_2_gen_sd.py --strength 0.35 --guidance 7.5
```

### Algorithm
1. Initialize Gemini client → generate disease description prompt for each class
2. Load model `runwayml/stable-diffusion-v1-5` (float16, CPU offload)
3. For each original image, generate 4 new images via img2img:
   - Resize original image → 512×512
   - Run 50 inference steps
   - Use negative prompt to avoid low-quality outputs
4. Save used prompts to `used_prompts.json`

### Special Requirements
- **Gemini API key** in the `.env` file
- **VRAM ≥ 6 GB** (uses `enable_model_cpu_offload` to save VRAM)
- **First run** will download the model (~4 GB) from HuggingFace

### Output

```
datasets/sd_x5/train/
├── Tomato___Early_blight/
│   ├── original_img.jpg           (20 original images copied)
│   ├── original_img_sd0.jpg       (generated)
│   ├── original_img_sd1.jpg
│   ├── original_img_sd2.jpg
│   └── original_img_sd3.jpg
│   → Total: 100 images/class
└── ...
```

---

## 02_3_gen_gan.py — Generate Images via DCGAN

### Function
Train a **DCGAN (Deep Convolutional GAN)** separately for each class, then generate 80 images per class.

### Hyperparameters (hardcoded)

| Parameter | Value |
|-----------|-------|
| Image size (training) | 64×64 |
| Output size | 224×224 (upscaled via BICUBIC) |
| Latent dim | 100 |
| Epochs | 500 |
| Batch size | 16 |
| Learning rate | 0.0002 |
| Generator filters | 64 → 512 |
| Discriminator filters | 64 → 512 |

### How to Run

```bash
python tomato_vs/02_3_gen_gan.py
```

### Algorithm
1. For each class:
   - Train DCGAN for 500 epochs on 20 original images
   - Generator: `ConvTranspose2d` layers (latent → 64×64 RGB)
   - Discriminator: `Conv2d` layers (64×64 RGB → real/fake)
   - Label smoothing: real_label = 0.9
2. After training, generate 80 new images → upscale to 224×224
3. Copy 20 original images + 80 GAN images = 100 images/class

### Output

```
datasets/gan_x5/train/
├── Tomato___Early_blight/
│   ├── original_img.jpg              (20 original images)
│   ├── gan_generated_0000.jpg        (80 GAN images)
│   └── ...
│   → Total: 100 images/class
└── ...
```

---

## 03_run_experiments.py — Train & Evaluate (Baseline, TDA, SD)

### Function
Train EfficientNet-B0 on 3 datasets (baseline, tda_x5, sd_x5), each with 5 trials using different seeds. Evaluate on the test set.

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | (required) | Directory to save results |
| `--train_count` | 10 | Saved to metadata |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 (pretrained) |
| Freezing | Partial (only classifier + last 3 feature blocks unfrozen) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Epochs | 50 (early stopping patience=10) |
| Batch size | 8 |
| Trials | 5 (seeds: 43, 44, 45, 46, 47) |
| Validation split | 20% |

### Metrics
- Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)
- MCC (Matthews Correlation Coefficient)
- AUC-ROC (weighted, one-vs-rest)

### How to Run

```bash
python tomato_vs/03_run_experiments.py --output_dir tomato_vs/Results/my_experiment
```

### Output

```
Results/my_experiment/
├── metrics_summary.csv            # Metrics per trial + AVG + STD
├── training_curves.csv            # Loss/Acc per epoch
├── cm_baseline_trial1.png         # Confusion matrix per trial
├── cm_tda_x5_trial1.png
├── cm_sd_x5_trial1.png
├── cm_aggregate_baseline.png      # Aggregated confusion matrix (5 trials)
├── cm_aggregate_tda_x5.png
├── cm_aggregate_sd_x5.png
├── loss_curve_baseline_trial1.png # Loss curves per trial
└── ...
```

---

## 03_1_run_gan_experiment.py — Train & Evaluate (GAN)

### Function
Same as `03_run_experiments.py` but only runs the `gan_x5` dataset. Results are saved separately in `Results/`.

### How to Run

```bash
python tomato_vs/03_1_run_gan_experiment.py
```

### Output

```
Results/
├── gan_metrics_summary.csv
├── gan_training_curves.csv
├── cm_gan_x5_trial1.png
├── cm_aggregate_gan_x5.png
└── loss_curve_gan_x5_trial1.png
```

---

## 04_visualize_results.py — Visualization (3 Methods)

### Function
Reads results from `metrics_summary.csv` and `training_curves.csv`, plots comparison charts for 3 methods: Baseline, TDA, SD.

### Arguments

| Argument | Description |
|----------|-------------|
| `--input_dir` | Directory containing `metrics_summary.csv` |

### How to Run

```bash
python tomato_vs/04_visualize_results.py --input_dir tomato_vs/Results/my_experiment
```

### Output
- `summary_comparison.png` — Bar chart of 6 metrics
- `loss_acc_aggregated.png` — Learning curves (mean ± std)
- `confusion_matrix_aggregated.png` — 3 confusion matrices side by side

---

## 04_1_visualize_with_gan.py — Visualization (4 Methods)

### Function
Combines results from `03_run_experiments.py` (Baseline/TDA/SD) and `03_1_run_gan_experiment.py` (GAN) to plot a comparison of all 4 methods.

### How to Run

```bash
python tomato_vs/04_1_visualize_with_gan.py --input_dir tomato_vs/Results/my_experiment
```

### Output
- `Results/summary_comparison_with_gan.png` — Bar chart of 4 metrics × 4 methods

---

## 05_master_run.py — Automated Pipeline

### Function
Automatically runs the full pipeline: `01 → 02_1 → 02_2 → 03 → 04`. Supports 2 modes:

1. **Random**: Run 1 randomly selected SD parameter combination
2. **Full**: Run all 9 combinations (3 strength × 3 guidance)

### SD Parameter Grid Search

| Strength | Guidance |
|----------|----------|
| 0.35 | 6.0, 7.5, 9.0 |
| 0.50 | 6.0, 7.5, 9.0 |
| 0.65 | 6.0, 7.5, 9.0 |

### How to Run

```bash
python tomato_vs/05_master_run.py
# Enter mode selection (1=Random, 2=Full)
# Enter train_count (default=10)
# Enter test_count (default=50)
```

> **Note:** Full mode runs 9 complete pipelines → very long runtime (can exceed 12 hours).

---

## 06_final_comparison.py — Final Comparison

### Function
Compare **Baseline** vs **Combined (TDA + SD)** using pre-prepared data in `Data_ST/`.

### Requirements
- `Data_ST/` directory must exist with the following structure:
  ```
  Data_ST/
  ├── baseline/train/
  ├── tda_x5/train/
  ├── sd_x5/train/
  └── combined_tda_sd/train/   (created automatically by script)
  ```
- `Data_OG/` directory is used to create the test set

### How to Run

```bash
python tomato_vs/06_final_comparison.py
# or specify an output directory:
python tomato_vs/06_final_comparison.py --output_dir tomato_vs/Results/final
```

### Special Algorithm
- **Creates the test set** from `Data_OG/`, excluding images already used in baseline training
- **Creates the combined dataset** = original baseline + TDA augmented (`_aug`) + SD generated (`_sd`)

---

## 07_transfer_learning_comparison.py — Transfer Learning

### Function
Compare 2 training configurations:

| Config | Pretrained | Freezing |
|--------|-----------|----------|
| Config 1 | ❌ No (random weights) | ❌ No (train all layers) |
| Config 2 | ✅ Yes (ImageNet) | ❌ No (train all layers) |

Each config runs 2 experiments: Baseline and Combined (TDA+SD).

### How to Run

```bash
python tomato_vs/07_transfer_learning_comparison.py
```

### Output

```
Results/
├── {timestamp}_noptr_nofre/       # No Pretrained
│   ├── metrics_summary.csv
│   └── ...
└── {timestamp}_ptr_nofre/         # Pretrained
    ├── metrics_summary.csv
    └── ...
```

---

## redraw_confusion_matrix.py — Redraw Confusion Matrices

### Function
Reads data from `redraw.csv` (precomputed 5×5 matrices) and draws high-resolution heatmap images.

### How to Run

```bash
python tomato_vs/redraw_confusion_matrix.py
```

### Output
- `redraw/*.png` — Confusion matrix images (17 images)
