# EXPERIMENTAL_DESIGN.md
# Detailed Methodology — Q1 Research Standards

---

## 1. Problem Formulation

**Task**: Multi-class image classification of tomato leaf diseases  
**Setting**: Few-shot (20 labeled images per class)  
**Goal**: Determine whether Stable Diffusion img2img augmentation improves
classification accuracy compared to traditional methods under this constraint.

**Research question**: *Does SD-based generative augmentation outperform
classical augmentation and online mixing strategies for few-shot tomato disease
classification, and at what diffusion strength does label noise degrade results?*

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | PlantVillage (Kaggle) |
| Classes | 5 tomato disease/health states |
| Train per class | 20 (few-shot constraint) |
| Test per class | 100 (fixed for all Phase-1 and Phase-2 runs, matches submitted paper) |
| Split strategy | `random.seed(42)` stratified, no overlap between train and test |
| Train–test exclusion | Asserted by set intersection check in `01_data_setup.py` |

**Classes:**
1. `Tomato___Early_blight`
2. `Tomato___healthy`
3. `Tomato___Late_blight`
4. `Tomato___Leaf_Mold`
5. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`

---

## 3. Augmentation Methods

### 3.1 Baseline (no augmentation)
Training with 20 original images per class.
Training-time transforms: RandomHorizontalFlip, RandomRotation(15°), ColorJitter.

### 3.2 TDA ×5 (Traditional Data Augmentation)
Pre-generates 4 augmented copies per original (total 5×).  
Library: `torchvision.transforms` (via `src/configurations/augmentation_config.py`).

Transform pipeline applied to each copy:
1. `Resize((224, 224))`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomRotation(degrees=20)`
4. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
5. `ToTensor()`
6. `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

Implemented in `02_1_gen_tda.py`.  
Parameter source: `AugmentationConfig.GEOMETRIC_TRANSFORMS["rotation"]["ranges"]["degrees"][1]` = 20°;
`AugmentationConfig.PHOTOMETRIC_TRANSFORMS["color_combined"]` = brightness/contrast/saturation=0.2, hue=0.1.

### 3.3 SD ×5 (Stable Diffusion img2img)
- Model: `runwayml/stable-diffusion-v1-5` (fp16, CPU offload for 6 GB VRAM)
- Input: 512×512 resized original image
- Prompt: generated via **Gemini 2.5 Flash** API (disease-specific expert description)
  > **Note**: Gemini 2.0 Flash was deprecated in June 2026. `gemini-2.5-flash` is the current stable model used in `02_2_gen_sd.py`.
- Negative prompt: suppresses cartoon, non-leaf content
- Grid search: Strength ∈ {0.35, 0.50, 0.65}, Guidance ∈ {6.0, 7.5, 9.0}
- Output: 4 generated images per original, stored as `*_sdN.jpg`
- Implemented in `02_2_gen_sd.py`

### 3.4 MixUp
Online per-batch: λ ~ Beta(0.4, 0.4), mixed loss = λ·CE(y_a) + (1-λ)·CE(y_b).
Applied to the 20 original training images.

### 3.5 CutMix
Online per-batch: random rectangular region of size proportional to (1-λ).
lam_adjusted = 1 - box_area / image_area. Mixed loss same as MixUp.
Applied to the 20 original training images.

### 3.6 RandAugment ×5
Pre-generated dataset: `torchvision.transforms.RandAugment(num_ops=2, magnitude=9)`.
4 augmented copies per original (5× total), matching TDA/SD scale.
Implemented in `02_6_gen_baselines.py`.

### 3.7 AutoAugment [R3.7 extra]
AutoAugment with IMAGENET policy (`torchvision.transforms.AutoAugment(AutoAugmentPolicy.IMAGENET)`).
Applied **online** during training to the 20 baseline images (same as MixUp/CutMix).
Uses a policy learned on ImageNet; not re-optimised for plant-disease data.
**Note**: Some AutoAugment ops (Invert, Posterize) may alter disease colour cues.
Results serve as upper bound for automated policy-based augmentation.

### 3.8 AugMix [R3.7 extra]
AugMix (`torchvision.transforms.AugMix()`, requires torchvision ≥ 0.13).
Applies augmentation at multiple severity levels and mixes them stochastically.
Applied **online** during training to the 20 baseline images.

### 3.9 SD ×5 (Label-only) [Ablation]
Same SD pipeline as 3.3 but prompt = `"tomato leaf {class_name}, disease symptoms, macro photography"`.
No Gemini API required.
Implemented in `02_2b_gen_sd_labelonly.py`.

### 3.10 CDA ×9 (Combined Data Augmentation) [Re-added from original paper]
**Combines TDA ×5 and SD ×5** into a single training dataset:
- **Source**: `combined_tda_sd/train/` = merge of `tda_x5/train/` (originals + _augN) + `sd_x5/train/` (_sdN only)
- **Size**: 20 originals + 80 TDA augmented + 80 SD augmented = **180 images/class = 9× dataset**
- **Purpose**: Tests whether combining two augmentation strategies is additive or synergistic
- **Creation**: Automated in `07_master_run.py` Phase 1 after each SD generation (step 1-B2)
- **K-fold handling**: `get_fold_aug_samples(..., per_type=True)` counts TDA and SD variants
  SEPARATELY — each type up to `aug_limit` per source stem, preventing one type from
  crowding out the other in the sorted file list
- **Sensitivity**: At aug_limit=4, CDA gets 4 TDA + 4 SD = 8 aug per original (9× total);
  at aug_limit=1, CDA gets 1 TDA + 1 SD = 2 aug per original (3× total)

---

## 4. Model Architecture

**EfficientNet-B0** (torchvision, ImageNet weights)

| Component | Configuration |
|-----------|--------------|
| Backbone | EfficientNet-B0 (5.3M parameters) |
| Pre-trained | ImageNet (ILSVRC 2012) |
| Frozen layers | First N-3 feature blocks |
| Trainable layers | Last 3 feature blocks + classifier |
| Classifier | Linear(1280 → num_classes) |
| Trainable params | ~2.2M (out of 5.3M) |

**Training hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (AdamW, standard for partial fine-tuning on small datasets; Howard & Ruder 2018) |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, η_min=1e-7) |
| Batch size | 8 |
| Max epochs | 50 |
| Early stopping | patience=10 on validation loss |
| Gradient clipping | max_norm=1.0 |

### 4.1 Training Configuration Comparison [Justification for Config 1]

Three training configurations are compared in `06_transfer_learning_comparison.py`
to empirically justify the choice of Config 1 as the primary strategy.

> **All main augmentation experiments use Config 1.** Every method in the main
> comparison (`03_run_experiments.py`: Baseline, TDA, GDA/SD, CDA, MixUp, CutMix,
> RandAugment, AutoAugment, AugMix, SD label-only) is trained under **Config 1
> (Transfer Learning + Partial Freezing)** — the model setup is identical across
> methods; only the data/augmentation differs. The 3-config study then extends
> **Baseline and CDA** to Config 2 and Config 3. Because Config 1 here is byte-for-byte
> the same model setup as `_train_eval` in `03_run_experiments.py` (same `pretrained=True`,
> same freeze of all but `features[-3:]` + classifier, same 15-fold seed), its
> Baseline/CDA numbers reproduce those of the main comparison.
>
> ⚠️ **Partial Freezing freezes the FIRST six blocks (`features[0]`–`features[5]`) and
> fine-tunes the LAST three (`features[6]`–`features[8]`) + classifier** — it does NOT
> "freeze the last 3 layers". Use this precise wording in the paper.

| Config | Pretrained | Frozen Layers | Trainable | Script Output |
|--------|-----------|--------------|-----------|---------------|
| **Config 1** | ✓ ImageNet | All except last 3 blocks + classifier | ~2.2M / 5.3M | `Config1_PartialFreezing/` |
| Config 2 | ✗ Random init | None | 5.3M / 5.3M | `Config2_FromScratch/` |
| Config 3 | ✓ ImageNet | None | 5.3M / 5.3M | `Config3_FineTuneAll/` |

**Rationale for Config 1** (Transfer Learning + Partial Freezing):
- In few-shot regime (20 images/class), fine-tuning ALL layers risks catastrophic
  forgetting and overfitting to the small training set
- Freezing early feature layers preserves generic edge/texture features from ImageNet
- Partial freezing is the standard approach for small-data fine-tuning
  (Howard & Ruder 2018; Kornblith et al. 2019)
- Config 2 (from scratch) typically underperforms with <100 images/class
- Config 3 (fine-tune all) can overfit rapidly with 20 images/class and lr=1e-4

**Datasets compared**: each of the 3 configs is trained/evaluated on the **baseline**
dataset (always) **and** the best combo's **CDA** (`combined_tda_sd`, when available).

**Cross-validation**: RepeatedStratifiedKFold (n_splits=5, n_repeats=3 → **15 folds**),
identical to the main experiments (R3.1/R3.6). Augmented images derived from held-out
originals are excluded (no leakage); validation uses originals only.

**Execution**: Phase **1-D** of `07_master_run.py` runs `06_transfer_learning_comparison.py`
once **after** SD/CDA generation (so the CDA dataset exists). The best combo's
`combined_tda_sd` is restored from its Phase-1 backup for this comparison.
Results in `Results/training_config_comparison/` (`all_configs_comparison.csv` has a
`Fold` column: folds 1–15 + AVG + STD, for both `baseline` and `combined_tda_sd`).

---

## 5. Cross-Validation Protocol

**Mode**: RepeatedStratifiedKFold (default, primary) or 5 fixed independent trials (matches submitted paper)

| Parameter | K-Fold mode | Fixed-trial mode |
|-----------|-------------|-----------------|
| n_splits | 5 | N/A |
| n_repeats | 3 | N/A |
| random_state | 42 | seeds 43–47 |
| Total evaluations | **15 per method** | **5 per method** [matches submitted paper] |

K-Fold mode satisfies reviewer R3.1/R3.6 with 15 independent evaluations (5×3 folds).
Fixed-trial mode matches the original 5-seed setup from the submitted manuscript.

**Splitting procedure (K-Fold):**
1. The 20 original baseline images per class form the splitting pool.
2. Each fold holds out 4 images per class (20% stratified) as validation.
3. The remaining 16 originals per class are used for training.
4. For augmented datasets (TDA, SD, RandAugment): augmented images are included
   **only if their source original is in the training fold**.
   Source is identified by filename suffix convention: `img_augN.jpg`, `img_sdN.jpg`, etc.
5. Validation set uses **original images only** (no augmented versions).
6. Test set is **fixed** (never changes across folds).

**Data leakage prevention:**
- No augmented image derived from a held-out original reaches the training set.
- Validation uses only original images to avoid over-optimistic validation loss.
- Test images were excluded from augmentation source pool at setup time
  (`01_data_setup.py` copies non-overlapping subsets).

---

## 6. Evaluation Metrics

| Metric | Formula / Library | Rationale |
|--------|------------------|-----------|
| Accuracy | `accuracy_score` | Overall correctness |
| Precision | `precision_score(weighted)` | Per-class correctness, handles imbalance |
| Recall | `recall_score(weighted)` | Per-class sensitivity |
| F1-Score | `f1_score(weighted)` | Harmonic mean, primary metric |
| MCC | `matthews_corrcoef` | Robust to class imbalance |
| AUC-ROC | `roc_auc_score(OvR, weighted)` | Discrimination ability |
| FID | `torchmetrics FrechetInceptionDistance` | Image quality vs real distribution (lower = better) |
| IS | `torchmetrics InceptionScore` | Generated image quality + diversity (higher = better). ⚠️ Unreliable for domain-specific images (Inception v3 trained on ImageNet); reported with caveat. FID/LPIPS are primary quality metrics. |
| LPIPS | `lpips.LPIPS(net='alex')` | Perceptual similarity to source |
| Label-noise rate | NN cosine-similarity in EfficientNet-B0 feature space | % off-distribution generated images |
| LPIPS intra-class | Mean pairwise LPIPS within class (equal-subsampled) | Diversity (higher = more diverse) |
| Feature dispersion | Mean std of EfficientNet avgpool embeddings | Diversity in feature space |

---

## 7. Statistical Analysis

### 7.1 Significance testing
- **Primary test**: Wilcoxon signed-rank test (non-parametric, paired across folds)
- **Rationale**: Non-normality cannot be assumed for n=10–15 evaluations;
  paired design controls for fold-level variance; Wilcoxon makes no normality assumption
- **Global test**: Friedman test (non-parametric repeated-measures ANOVA) across all
  methods simultaneously; performed before pairwise tests  [R9]
- **Correction**: Bonferroni correction when multiple pairs are compared
- **Threshold**: α=0.05 (adjusted for multiple comparisons)

### 7.2 Normality testing [R9]
- **Test**: Shapiro-Wilk per method × metric (n range 10–15, suitable for Shapiro-Wilk)
- Results saved to `normality_tests.csv`
- If any method fails (p ≤ 0.05): Wilcoxon is justified (already used)
- If all pass: Wilcoxon remains a valid conservative test

### 7.3 Effect size
- **Cohen's d** (pooled): (μ_A - μ_B) / s_pooled
- Magnitude: negligible |d|<0.2, small 0.2≤|d|<0.5, medium 0.5≤|d|<0.8, large |d|≥0.8
- **Caution with n=10–15**: large effect sizes can be unreliable; reported with this caveat

### 7.4 Per-class analysis [R10]
- Per-class Precision, Recall, F1 computed via `sklearn.metrics.classification_report`
- Mean ± std reported across all folds per (method, class) pair
- **Early Blight ↔ Late Blight confusion rate** specifically tracked as
  `cm[EB, LB] / total_EB_predictions` (and vice versa)
- Saved to `per_class_metrics.csv`, `per_class_summary.csv`, `per_class_analysis.csv`
- Visualised in `per_class_comparison.png`

### 7.5 Reporting
All comparisons vs Baseline in `statistical_comparison.png` (with significance stars).
Pairwise table in `statistical_tests.csv`.
Friedman global test in `friedman_tests.csv`.
Normality per method in `normality_tests.csv`.
Per-class F1 bar chart in `per_class_comparison.png`.

---

## 8. Hyperparameter Search (SD)

A 3×3 grid of (Strength, Guidance) combinations is evaluated:

| | Guidance 6.0 | Guidance 7.5 | Guidance 9.0 |
|-|-------------|-------------|-------------|
| **Strength 0.35** | GDA_1 | GDA_2 | GDA_3 |
| **Strength 0.50** | GDA_4 | GDA_5 | GDA_6 |
| **Strength 0.65** | GDA_7 | GDA_8 | GDA_9 |

The **best combination** is determined by highest sd_x5 mean accuracy across
k-fold folds, then used for the sensitivity analysis.

---

## 9. Sensitivity Analysis: Augmentation Ratio  [R3.8]

**Research question**: *Is the 20-80-80 training ratio empirically justified,
or does performance saturate at a lower augmentation count?*

| Variant | `aug_limit` | Train / class | Ratio |
|---------|-------------|--------------|-------|
| 2× | 1 | 20 orig + 20 TDA + 20 SD = 60 | 20-20-20 |
| 3× | 2 | 20 + 40 + 40 = 100 | 20-40-40 |
| 4× | 3 | 20 + 60 + 60 = 140 | 20-60-60 |
| **5× (Phase-1 main)** | **4** | **20 + 80 + 80 = 180** | **20-80-80** |

**Protocol:**
- Test set is **unchanged** (same split as Phase 1; no `01_data_setup.py` re-run).
  This ensures Phase-2 results are directly comparable to Phase-1 results.
- SD generation is performed **once** (best combo) and **shared** across all variants.
  The `aug_limit` only controls how many augmented images per source are selected
  from the already-generated dataset during k-fold training.
- k-fold CV is **required**: `aug_limit` is only respected by `get_fold_aug_samples()`
  in k-fold mode. Fixed-trial mode loads the full directory regardless.
- Results stored in `Results/sensitivity_aL{1,2,3}/`

**Note on 20-100-100**: Would require 5 augmented images per original (>4× generated).
The 2×/3×/4× progression together with the 5× Phase-1 main result gives a complete
picture of the augmentation quantity–performance curve.

---

## 10. Ablation Studies

### 10.1 Prompt Type
| Condition | Prompt Source | Expected Outcome |
|-----------|--------------|-----------------|
| `sd_x5` | **Gemini 2.5 Flash** expert description | More semantically accurate images |
| `sd_labelonly_x5` | Template `"tomato leaf {class}"` | Less targeted images |

### 10.2 Augmentation Quantity

| `--aug_limit` | Multiplier | images/class |
|--------------|------------|-------------|
| 1 | 2× | ~40 |
| 2 | 3× | ~60 |
| 3 | 4× | ~80 |
| 4 | 5× (default) | ~100 |

The quantity ablation tests whether classification accuracy monotonically increases
with more synthetic data, or whether there is a saturation point.

---

## 11. Limitations

1. **FID with small sample size**: 100–400 images are used; FID was designed for ≥2048.
   Results should be interpreted as indicative, not definitive.
2. **MixUp/CutMix/AutoAugment/AugMix dataset size**: These are applied as **online
   transforms to 20 original images per class** (not pre-generated 5×) because:
   - MixUp/CutMix require pairs within a batch
   - AutoAugment/AugMix apply a random policy every epoch, so diversity accumulates
     over 50 epochs rather than being stored on disk
   A fair 5× pre-generated AutoAugment comparison could be created by modifying
   `02_6_gen_baselines.py`, but this is not done here (acknowledged limitation).
   The online vs pre-generated distinction is noted in all paper tables.
3. **Single model architecture**: Only EfficientNet-B0 is tested; results may not
   generalise to other architectures.
4. **Effect size with n=10–15**: Large effect sizes (|d| ≥ 0.8) should be interpreted
   cautiously at n=10–15 evaluations. They are reported with magnitude labels and
   should not be over-interpreted without replication. [R9]
5. **Expert validation**: Per reviewer suggestion, expert visual evaluation of
   generated images is recommended but requires domain specialists.
6. **EB/LB confusion mitigation training**: Confusion-rate analysis is fully automated
   and reported (R10). Targeted mitigation training (focal loss) is described in
   Discussion as future work but is NOT run by default — it is not required by the
   reviewer and is not claimed in any paper result.
   **Note**: Class-weighted CE is NOT applicable here because the training set is
   perfectly balanced (20 images × 5 classes). Class weights would all equal 1.0.

---

## 12. Per-Class Analysis Protocol [R10]

### Motivation
Early Blight (EB) and Late Blight (LB) are visually similar diseases.
Confusion between them can lead to incorrect treatment decisions in practice.

### Metrics
| Metric | Formula |
|--------|---------|
| Per-class F1 | `f1_score(y_true, y_pred, average=None)` |
| EB→LB rate | `cm[EB_idx, LB_idx] / cm[EB_idx].sum()` |
| LB→EB rate | `cm[LB_idx, EB_idx] / cm[LB_idx].sum()` |

### Interpretation
- **Target**: EB→LB and LB→EB confusion rates < 0.10 (10%)
- **High flag**: Rate > **0.10** is flagged as `← HIGH` in the analysis output
  (consistent with the target; `03_3_analyze_results.py` line ~335)
- **SD hypothesis**: Gemini-guided SD augmentation generates more visually distinct
  disease representations, potentially reducing EB/LB confusion compared to baseline

### Outputs
- `per_class_metrics.csv` — raw per-trial, per-class metrics
- `per_class_summary.csv` — mean ± std summary per (method, class)
- `per_class_analysis.csv` — computed by `03_3_analyze_results.py`
- `per_class_comparison.png` — bar chart of F1 by class and method

---

## 13. GPU Optimisation Notes

### 13.1 Why RTX 5060 Ti showed only 15% GPU utilisation

The root cause is `pipe.enable_model_cpu_offload()` in `02_2_gen_sd.py`.
This is a **memory-saving feature designed for 4–8 GB VRAM cards** (e.g. RTX 3050 Ti).
It moves each pipeline component (CLIP, VAE, UNet) between CPU and GPU one at a time,
causing continuous PCIe transfer overhead that keeps the GPU idle most of the time.

On a high-VRAM card (≥ 10 GB, e.g. RTX 5060 Ti 16 GB), all SD components fit on GPU
simultaneously. The fix: skip CPU offload and use `pipe.to("cuda")` instead.

**Speedup** from removing CPU offload on RTX 5060 Ti: ~8–12× faster SD generation.

### 13.2 What was changed

| Component | Before | After |
|-----------|--------|-------|
| `02_2_gen_sd.py` | Always `enable_model_cpu_offload()` | Smart VRAM detection: offload only if VRAM < 10 GB |
| `02_2_gen_sd.py` | — | xformers memory-efficient attention if installed |
| `03_run_experiments.py` | `num_workers=0` | `0` on Windows, `min(4, CPU_count)` on Linux |
| `03_run_experiments.py` | No AMP | Mixed-precision (fp16 forward, fp32 grad) via GradScaler |
| `03_run_experiments.py` | — | `persistent_workers=True` when num_workers > 0 |

### 13.3 Why training GPU utilisation is inherently low

Even with all optimisations, GPU utilisation during **training** will be low because:
- Total dataset: 100 images (20/class × 5 classes) in training set
- Each epoch: ~12 forward+backward passes (batch=8) → ~50 ms on any modern GPU
- Bottleneck: Python overhead + DataLoader spawn, not GPU compute

This is **expected and correct** for few-shot training. The GPU's heavy work is during
**SD generation** (50 denoising steps per image × 400 images = 20,000 UNet passes).

### 13.4 Running on both machines

| Machine | VRAM | Config |
|---------|------|--------|
| RTX 3050 Ti | 4 GB | CPU offload ENABLED (auto-detected); `num_workers=0` on Windows |
| RTX 5060 Ti | 16 GB | Full GPU mode (auto-detected); `num_workers=4` on Linux |

No manual configuration needed — `02_2_gen_sd.py` detects VRAM at runtime.
