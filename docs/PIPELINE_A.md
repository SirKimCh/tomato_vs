# PIPELINE.md — Main Experimental Pipeline

> **The master run script handles this entire pipeline automatically.**  
> Just run: `python tomato_vs/07_master_run.py`

---

## Overview

The main pipeline runs in two phases:

### Phase 0: One-time setup
1. `01_data_setup.py` — Split 20 train + N test per class
2. `02_1_gen_tda.py` — Generate TDA×5
3. `02_6_gen_baselines.py` — Generate RandAugment×5

### Phase 1: Per-SD-combination (9 total: 3 Strengths × 3 Guidances)
For each (Strength, Guidance):
1. `02_2_gen_sd.py` — Generate SD×5 (Gemini LLM prompts)
2. `02_2b_gen_sd_labelonly.py` — Generate SD×5 (label-only, ablation)
3. `02_4_compute_image_quality.py` — FID + LPIPS + label-noise
4. `02_5_compute_diversity.py` — LPIPS intra-class + feature dispersion
5. `03_run_experiments.py --use_kfold --extra_baselines --ablation_prompt` — Train+eval all methods
6. `03_run_experiments.py --aug_limit {1,2,3}` — Quantity ablation
7. Backup generated images
8. Clean up sd_x5 and sd_labelonly_x5 (keep baseline, tda_x5, randaugment_x5)
9. `04_visualize_results.py` — Charts
10. `03_3_analyze_results.py` — Statistical significance

### Phase 2: Sensitivity analysis — augmentation ratio  [R8]
For aug_limit ∈ {1, 2, 3}  (aug_limit=4 = 5× is the Phase-1 main result):

| aug_limit | Ratio label | Train/class |
|-----------|-------------|-------------|
| 1 | 2× (20-20-20) | 60 |
| 2 | 3× (20-40-40) | 100 |
| 3 | 4× (20-60-60) | 140 |
| **4 (Phase-1 main)** | **5× (20-80-80)** | **180** |

1. Test set is **unchanged** — no `01_data_setup.py` re-run (same 80/class split as Phase 1)
2. Restore Phase-0 datasets (tda_x5, randaugment_x5, baseline) from `Results/_phase0_backup/`
3. Re-generate SD for best combo **once** (shared across all three aug_limit values)
4. `03_run_experiments.py --use_kfold --aug_limit {1,2,3}` — k-fold is **required** (`aug_limit` only takes effect in `get_fold_aug_samples()`; fixed-trial mode ignores it)
5. Visualize + analyze

> **Note**: This design was corrected from an earlier draft that mistakenly varied `test_count`.
> The correct sensitivity variable is `aug_limit` (augmentation ratio in training), NOT test set size.
> See `EXPERIMENTAL_DESIGN.md §9` and `REVIEWER_RESPONSES.md R8` for full rationale.

---

## Manual step-by-step (if not using master run)

```bash
cd leaf-disease-ai

# Phase 0
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
python tomato_vs/02_1_gen_tda.py
python tomato_vs/02_6_gen_baselines.py

# Phase 1 (single combo example: s=0.35, g=7.5)
python tomato_vs/02_2_gen_sd.py --strength 0.35 --guidance 7.5 \
    --output_log_dir tomato_vs/Results/s035_g75
python tomato_vs/02_2b_gen_sd_labelonly.py --strength 0.35 --guidance 7.5
python tomato_vs/02_4_compute_image_quality.py --strength 0.35 --guidance 7.5 \
    --run_dir tomato_vs/Results/s035_g75
python tomato_vs/02_5_compute_diversity.py --run_dir tomato_vs/Results/s035_g75
python tomato_vs/03_run_experiments.py \
    --output_dir tomato_vs/Results/s035_g75 \
    --train_count 20 --use_kfold --extra_baselines --ablation_prompt
python tomato_vs/03_3_analyze_results.py --input_dir tomato_vs/Results/s035_g75
python tomato_vs/04_visualize_results.py --input_dir tomato_vs/Results/s035_g75
```

---

## GAN comparison (separate, optional)

```bash
python tomato_vs/02_3_gen_gan.py
python tomato_vs/03_1_run_gan_experiment.py
python tomato_vs/04_1_visualize_with_gan.py --input_dir tomato_vs/Results/s035_g75
```

---

## Deprecated pipelines
- Old `05_master_run.py` → **replaced by `07_master_run.py`**
- Old `06_final_comparison.py` → **replaced by `05_final_comparison.py`**  
- Old `07_transfer_learning_comparison.py` → **replaced by `06_transfer_learning_comparison.py`**
