# SCRIPTS_DETAIL.md — Complete Script Reference (Revised)

All scripts are in `tomato_vs/` and must be run from `leaf-disease-ai/` (parent directory).

---

## 00_check_requirements.py ← **Run first**
```bash
python tomato_vs/00_check_requirements.py
python tomato_vs/00_check_requirements.py --install   # auto-install missing pkgs
```
Checks Python, CUDA 12.x, GPU (RTX 3050 Ti 4 GB / RTX 5060 Ti 16 GB), all packages, .env, Data_OG.

---

## 01_data_setup.py
```bash
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
```
Splits PlantVillage → `datasets/baseline/train/` (20/class) + `datasets/test/` (N/class).  
**Warning**: deletes and recreates `datasets/` on each run.

---

## 02_1_gen_tda.py
```bash
python tomato_vs/02_1_gen_tda.py
```
Generates `datasets/tda_x5/train/`  (×5, filenames `*_augN.jpg`).

---

## 02_2_gen_sd.py
```bash
python tomato_vs/02_2_gen_sd.py --strength 0.35 --guidance 7.5 \
    --output_log_dir tomato_vs/Results/run1
```
Generates `datasets/sd_x5/train/` (×5, filenames `*_sdN.jpg`).  
Requires `.env` with `GEMINI_API_KEY`.

---

## 02_2b_gen_sd_labelonly.py  [ablation]
```bash
python tomato_vs/02_2b_gen_sd_labelonly.py --strength 0.35 --guidance 7.5
```
Generates `datasets/sd_labelonly_x5/train/` (×5, filenames `*_sdloN.jpg`).  
No Gemini key required.

---

## 02_3_gen_gan.py
```bash
python tomato_vs/02_3_gen_gan.py
```
Trains DCGAN per-class → `datasets/gan_x5/train/` (80 GAN images + 20 originals).

---

## 02_4_compute_image_quality.py
```bash
python tomato_vs/02_4_compute_image_quality.py \
    --strength 0.35 --guidance 7.5 --run_dir Results/run1
```
Outputs `image_quality_s0.35_g7.5.csv` + `image_quality_summary_*.csv`.  
Computes FID (torchmetrics), IS/Inception Score (torchmetrics, supplementary — unreliable for plant-disease domain), LPIPS (lpips), label-noise proxy (cosine-NN).

---

## 02_5_compute_diversity.py
```bash
python tomato_vs/02_5_compute_diversity.py --run_dir Results/run1 [--n_pairs 100]
```
Outputs `diversity_metrics.csv`, `diversity_comparison.png`.  
**Key**: subsamples all datasets to baseline size for fair LPIPS comparison.

---

## 02_6_gen_baselines.py
```bash
python tomato_vs/02_6_gen_baselines.py
```
Generates `datasets/randaugment_x5/train/` (×5, filenames `*_raN.jpg`).

---

## 03_run_experiments.py
```bash
# Standard (5 fixed trials, matches submitted paper)
python tomato_vs/03_run_experiments.py --output_dir Results/run1 --train_count 20

# Q1 primary mode (k-fold + all baselines + ablation + per-class)  [R3.1/R3.6]
python tomato_vs/03_run_experiments.py --output_dir Results/run1 \
    --train_count 20 --use_kfold --extra_baselines --ablation_prompt

# Quantity ablation
python tomato_vs/03_run_experiments.py --output_dir Results/run1_2x \
    --train_count 20 --use_kfold --aug_limit 1
```

Key flags: `--use_kfold`, `--n_splits 5`, `--n_repeats 3`, `--extra_baselines`,
`--ablation_prompt`, `--aug_limit {1-4}`.  
Outputs: `metrics_summary.csv`, `training_curves.csv`, `cm_aggregate_*.png`,  
         `per_class_metrics.csv` [R10], `per_class_summary.csv` [R10].

---

## 03_1_run_gan_experiment.py
```bash
python tomato_vs/03_1_run_gan_experiment.py
```
Trains on `gan_x5`. Outputs `Results/gan_metrics_summary.csv`.

---

## 03_3_analyze_results.py
```bash
python tomato_vs/03_3_analyze_results.py --input_dir Results/run1 [--alpha 0.05]
```
Post-hoc analysis:
- Wilcoxon signed-rank tests + Cohen's d (pairwise)
- **Shapiro-Wilk normality test per method × metric**  [R9]
- **Friedman test (non-parametric ANOVA) per metric**  [R9]
- Ranking table (mean ± std, sorted by F1)
- **Per-class F1 / EB↔LB confusion rate analysis**     [R10]
- Significance bar chart + per-class bar chart

Requires `scipy`.  
Outputs: `statistical_tests.csv`, `normality_tests.csv`, `friedman_tests.csv`,
         `ranking_table.csv`, `per_class_analysis.csv` [R10],
         `statistical_comparison.png`, `per_class_comparison.png` [R10].

---

## 04_visualize_results.py
```bash
python tomato_vs/04_visualize_results.py --input_dir Results/run1
```
Auto-detects experiments from CSV. Outputs bar charts + learning curves + CM grid.

---

## 04_1_visualize_with_gan.py
```bash
python tomato_vs/04_1_visualize_with_gan.py --input_dir Results/run1
```
Adds GAN to comparison. Requires `Results/gan_metrics_summary.csv`.

---

## 05_final_comparison.py  [LEGACY — superseded]
```bash
python tomato_vs/05_final_comparison.py [--output_dir Results/final]
```
Legacy standalone: Baseline vs Combined(TDA+SD) using the old `Data_ST/` layout and
**5 fixed trials**. **Superseded** in the revision by `cda_x9` in `03_run_experiments.py`
(Baseline vs CDA, 15-fold, in the main table) and `06_transfer_learning_comparison.py`
(Baseline vs CDA × 3 configs, 15-fold). Exits with a redirect if `Data_ST/` is absent.
Use `07_master_run.py` instead.

---

## 06_transfer_learning_comparison.py
```bash
python tomato_vs/06_transfer_learning_comparison.py
```
Compares **3 training configurations** on EfficientNet-B0 with **RepeatedStratifiedKFold
(5×3 = 15 folds)** — consistent with the main experiments (R3.1/R3.6):
- Config 1 = Transfer Learning + Partial Freezing  **[MAIN]**
- Config 2 = Training from Scratch
- Config 3 = Fine-tuning All Layers

Reads `datasets/baseline/train/` + `datasets/test/` (always), and
`datasets/combined_tda_sd/train/` (CDA) **if present** → each config is evaluated on
**baseline AND CDA**.  Outputs `Results/training_config_comparison/`
(`all_configs_comparison.csv` with a `Fold` column: folds 1–15 + AVG + STD,
`config_comparison_baseline.png`, `config_comparison_combined_tda_sd.png`).

> To include CDA when running standalone, first make `datasets/combined_tda_sd/` available
> (it is created in `07_master_run.py` Phase 1; a copy is kept per combo in
> `Results/<combo>/generated_images_backup/combined_tda_sd/` — copy it into `datasets/`).
> Inside `07_master_run.py`/`08_master_run_hotfix.py` this restore is automatic (Phase 1-D).

---

## 07_master_run.py  ← **Main entry point**
```bash
python tomato_vs/07_master_run.py                                # full pipeline
python tomato_vs/07_master_run.py --mode one --strength 0.35     # single combo
python tomato_vs/07_master_run.py --no_kfold --skip_sensitivity  # faster
```
All phases are orchestrated here. See README.md for flag reference.
