# Tomato Leaf Disease Classification — Few-Shot Generative Data Augmentation
## Revised for Reviewer Requirements (R1–R10)

> **Entry point:** `python tomato_vs/00_check_requirements.py --install`  
> **Full experiment:** `python tomato_vs/07_master_run.py`

---

## Research Overview

This study benchmarks **Stable Diffusion img2img** as a data augmentation technique for tomato leaf disease classification under a **few-shot scenario (20 images/class)**. A five-class PlantVillage subset is used with EfficientNet-B0 (ImageNet pre-trained, partial fine-tuning).

### Methods compared

| # | Method | Key Idea | Train Size |
|---|--------|----------|-----------|
| 1 | **Baseline** | No augmentation | 20 orig/class |
| 2 | **TDA ×5** | Flip + Rotate + ColorJitter (pre-generated) | 100/class |
| 3 | **SD ×5 (LLM)** | Stable Diffusion img2img + Gemini 2.5 Flash prompts | 100/class |
| 4 | **CDA ×9** | **Combined TDA+SD** (20 orig + 80 TDA + 80 SD) | **180/class** |
| 5 | **MixUp** | Online convex combination of image pairs | 20 orig/class |
| 6 | **CutMix** | Online rectangular region swapping | 20 orig/class |
| 7 | **RandAugment ×5** | Pre-generated RandAugment policy | 100/class |
| 8 | **AutoAugment** | Online AutoAugment (ImageNet policy) \[R3.7\] | 20 orig/class |
| 9 | **AugMix** | Online AugMix stochastic mixing \[R3.7\] | 20 orig/class |
| 10 | **SD ×5 (Label)** | SD with simple label-name prompts \[ablation R6\] | 100/class |

> **CDA ×9** (Combined Data Augmentation) = TDA×5 + SD×5 merged into one dataset.
> Tests whether combining two augmentation strategies is additive or synergistic.
> Created automatically in Phase 1 of `07_master_run.py`; uses `per_type` k-fold
> counting to ensure equal TDA/SD representation at every aug_limit.

> **Training Configs** (supporting analysis, Phase 0-D):
> `06_transfer_learning_comparison.py` compares 3 configurations on the baseline dataset
> to justify why Config 1 (Transfer Learning + Partial Freezing) is the primary choice:
> - Config 1: Transfer Learning + Partial Freezing ← **MAIN** (used for all experiments above)
> - Config 2: Training from Scratch
> - Config 3: Fine-tuning All Layers

> **Note (GAN):** A separate DCGAN pipeline exists (`02_3_gen_gan.py`, `03_1_run_gan_experiment.py`) but is **not part of the main submitted results** — it is an optional exploratory comparison only.

### Reviewer Requirements — fully addressed

| Ref | Reviewer Concern | Implementation | Script |
|----|-----------------|----------------|--------|
| R1 | No FID/LPIPS metrics for generated images | FID + LPIPS (gen vs source) per Strength/Guidance | `02_4_compute_image_quality.py` |
| R2 | No diversity quantification | LPIPS intra-class + feature dispersion (equal-sampled) | `02_5_compute_diversity.py` |
| R3 | No label-noise analysis | Cosine-NN label-noise proxy per Strength | `02_4_compute_image_quality.py` |
| R4 | No k-fold CV → sampling bias | RepeatedStratifiedKFold (k=5, n=3 → 15 folds), no leakage | `03_run_experiments.py --use_kfold` |
| R5 | Missing MixUp/CutMix/RandAugment baselines | Added with identical protocol | `03_run_experiments.py --extra_baselines` |
| R6 | No ablation on prompt type | SD Gemini LLM vs label-name prompts | `02_2b_gen_sd_labelonly.py + --ablation_prompt` |
| R7 | No ablation on augmentation quantity | `--aug_limit 1–4` = 2×/3×/4×/5× | `03_run_experiments.py` |
| R7+ | AutoAugment / AugMix baselines | Added as extra baselines (online) | `03_run_experiments.py --extra_baselines` |
| R8 | Ratio 20-80-80 lacks empirical basis | Sensitivity: **aug_limit ∈ {1,2,3}** → 2×/3×/4× aug ratio (fixed test set) | `07_master_run.py` Phase 2 |
| R9 | n=5 insufficient; add normality + Friedman | **k-fold 15 folds** (primary, R3.1/R3.6); fixed-trial=5 matches submitted paper; Shapiro-Wilk + Friedman test | `03_run_experiments.py --use_kfold` + `03_3_analyze_results.py` |
| R10 | Large EB↔LB confusion; no per-class analysis | Per-class F1 + EB/LB confusion rate per method | `03_run_experiments.py` + `03_3_analyze_results.py` |

---

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA CUDA | **RTX 3050 Ti 4 GB** / **RTX 5060 Ti 16 GB** |
| CUDA | 12.1+ (Ampere) / 12.8+ (Blackwell) | **12.1 / 12.8** |
| RAM | 8 GB | 16 GB |
| Disk | 15 GB | 30 GB |
| Python | 3.8+ | **3.13** |

---

## Quick Start (3 steps)

### Step 1 — Check environment

```bash
cd leaf-disease-ai
python tomato_vs/00_check_requirements.py --install
```

### Step 2 — Prepare data & API key

Place the 5 PlantVillage tomato classes in `tomato_vs/Data_OG/`:
```
Data_OG/Tomato___Early_blight/           (≥ 120 images)
Data_OG/Tomato___healthy/                (≥ 120 images)
Data_OG/Tomato___Late_blight/            (≥ 120 images)
Data_OG/Tomato___Leaf_Mold/              (≥ 120 images)
Data_OG/Tomato___Tomato_Yellow_Leaf_Curl_Virus/  (≥ 120 images)
```

Create `tomato_vs/.env`:
```
GEMINI_API_KEY=your_key_here
```

### Step 3 — Run full experiment

```bash
python tomato_vs/07_master_run.py
```

**Useful flags:**
```bash
# Single SD combination (fast test)
python tomato_vs/07_master_run.py --mode one --strength 0.35 --guidance 7.5

# Skip k-fold (use 5 fixed trials, matches submitted paper)
python tomato_vs/07_master_run.py --no_kfold

# Skip slower optional steps
python tomato_vs/07_master_run.py --skip_image_quality --skip_diversity --skip_sensitivity

# Skip CDA and training config comparison (run only 9-method main comparison)
python tomato_vs/07_master_run.py --skip_cda --skip_training_configs

# Run ONLY training config comparison (Phase 0-D standalone)
python tomato_vs/06_transfer_learning_comparison.py
```

---

## File Structure (canonical)

```
tomato_vs/
├── 00_check_requirements.py       ← Run FIRST: env check + auto-install
│
├── 01_data_setup.py               ← Split PlantVillage → train(20)/test(N)
│
├── 02_1_gen_tda.py                ← Generate TDA×5
├── 02_2_gen_sd.py                 ← Generate SD×5 (Gemini LLM prompts)
├── 02_2b_gen_sd_labelonly.py      ← Generate SD×5 (label-name prompts) [ablation]
├── 02_3_gen_gan.py                ← Train DCGAN + generate GAN×5
├── 02_4_compute_image_quality.py  ← FID / LPIPS / label-noise per Strength
├── 02_5_compute_diversity.py      ← LPIPS intra-class + feature dispersion
├── 02_6_gen_baselines.py          ← Generate RandAugment×5
│
├── 03_run_experiments.py          ← Train EfficientNet-B0 (k-fold or 5-trial) [R9]
│                                     Experiments: baseline, tda_x5, sd_x5, CDA×9,
│                                     mixup, cutmix, randaugment, autoaugment, augmix
├── 03_1_run_gan_experiment.py     ← Train EfficientNet-B0 on GAN dataset
├── 03_3_analyze_results.py        ← Wilcoxon tests / Cohen's d / ranking table
│
├── 04_visualize_results.py        ← Bar charts, learning curves, CM grids
├── 04_1_visualize_with_gan.py     ← Visualization including GAN results
│
├── 05_final_comparison.py         ← Baseline vs Combined (TDA+SD) [standalone]
├── 06_transfer_learning_comparison.py  ← 3 Training Configs comparison [Phase 0-D]
│                                          Config1=PartialFreezing, Config2=Scratch,
│                                          Config3=FineTuneAll  (justifies Config 1)
│
└── 07_master_run.py               ← MASTER RUN (fully automated, no interaction)
```

**Deprecated** (exit immediately with redirect message):
`05_master_run.py` · `06_final_comparison.py` · `07_transfer_learning_comparison.py`

---

## Experimental Protocol

- **Model**: EfficientNet-B0 (ImageNet), last 3 blocks + classifier unfrozen
- **Optimizer**: AdamW (lr=1e-4, wd=1e-4) + CosineAnnealingWarmRestarts
- **CV**: RepeatedStratifiedKFold(k=5, n=3) → 15 folds [primary]  OR  5 fixed trials [matches submitted paper]
- **Metrics**: Acc, Precision, Recall, F1 (weighted), MCC, AUC-ROC, FID, LPIPS, IS (supplementary)
- **Stats**: Friedman test (global) + Wilcoxon signed-rank + Cohen's d [R9]
- **Per-class**: F1 per class + Early Blight↔Late Blight confusion rate [R10]

---

## Documentation

| File | Content |
|------|---------|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Full installation + CUDA setup guide |
| [docs/SCRIPTS_DETAIL.md](docs/SCRIPTS_DETAIL.md) | Per-script arguments, outputs, and notes |
| [docs/EXPERIMENTAL_DESIGN.md](docs/EXPERIMENTAL_DESIGN.md) | Methodology, protocol, and limitations |
| [docs/REVIEWER_RESPONSES.md](docs/REVIEWER_RESPONSES.md) | Reviewer → code mapping |
| [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) | EfficientNet-B0 and fine-tuning details |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors and solutions |
