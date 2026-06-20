# SCRIPTS_DETAIL.md — Complete Script Reference (Revised)

All scripts are in `tomato_vs/` and must be run from `leaf-disease-ai/` (parent directory).

---

## 00_check_requirements.py ← **Run first**
```bash
python tomato_vs/00_check_requirements.py
python tomato_vs/00_check_requirements.py --install   # auto-install missing pkgs
```
Checks Python, CUDA 12.x, GPU (GTX 3050 6 GB), all packages, .env, Data_OG.

---

## 01_data_setup.py
```bash
python tomato_vs/01_data_setup.py --train_count 20 --test_count 80
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
Computes FID (torchmetrics), LPIPS (lpips), label-noise proxy (cosine-NN).

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
# Standard (10 fixed trials)  [R9: ≥10 runs]
python tomato_vs/03_run_experiments.py --output_dir Results/run1 --train_count 20

# Q1 mode (k-fold + all baselines + ablation + per-class)
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

## 05_final_comparison.py
```bash
python tomato_vs/05_final_comparison.py [--output_dir Results/final]
```
Standalone: Baseline vs Combined(TDA+SD). Requires `Data_ST/`.

---

## 06_transfer_learning_comparison.py
```bash
python tomato_vs/06_transfer_learning_comparison.py
```
Standalone: Pretrained vs from-scratch. Requires `Data_ST/`.

---

## 07_master_run.py  ← **Main entry point**
```bash
python tomato_vs/07_master_run.py                                # full pipeline
python tomato_vs/07_master_run.py --mode one --strength 0.35     # single combo
python tomato_vs/07_master_run.py --no_kfold --skip_sensitivity  # faster
```
All phases are orchestrated here. See README.md for flag reference.
