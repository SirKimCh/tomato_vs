"""
07_master_run.py — Comprehensive Experimental Pipeline (Reviewer-Revised)
==========================================================================
Addresses ALL Q1-reviewer requirements:

  R1. Image quality  : FID + LPIPS + label-noise proxy per Strength/Guidance
  R2. Diversity      : LPIPS intra-class + feature dispersion (orig vs aug)
  R3. Label noise    : feature-space NN classifier per Strength
  R4. K-Fold CV      : RepeatedStratifiedKFold(k=5, repeats=3) replaces fixed trials
  R5. New baselines  : MixUp, CutMix, RandAugment added
  R6. Ablation       : prompt type (Gemini LLM vs label-name)
  R7. Ablation       : augmentation quantity (2× / 3× / 4× / 5×)
   R8. Sensitivity    : augmentation ratio in TRAIN (aug_limit 1→4 = 2×–5×)  [R3.8]
                        Answers: "is the 20-80-80 training ratio heuristic?"
   R9. ≥10 evaluations: k-fold gives 15 folds (5×3); fixed-trial mode uses 5 runs
                        (matches submitted paper; k-fold is the primary mode for R3.1/R3.6).
                        Shapiro-Wilk normality + Friedman test in analyze step.
 R10. Per-class      : Early Blight vs Late Blight confusion rate per method.
                       per_class_metrics.csv + per_class_comparison.png auto-saved.

No interactive input — all parameters via command-line arguments.

Current settings (đúng cho bài revision):
  test_count  : 100/class   (đảm bảo số liệu so sánh được với submitted paper)
  lr          : 1e-4        (standard for partial fine-tuning on few-shot data; Howard & Ruder 2018)
  Gemini      : gemini-2.5-flash (2.0-flash deprecated June 2026)
  Fixed trials: 5           (matches submitted paper; --no_kfold flag)
  K-Fold      : 15 folds    (PRIMARY for revision — addresses R3.1, R3.6)

⚠️  Nếu Results/ còn kết quả từ lần chạy trước với settings khác, hãy xóa trước:
      Remove-Item -Recurse -Force tomato_vs/Results/*
  (07_master_run.py tạo thư mục mới theo timestamp — kết quả cũ không bị ghi đè
   nhưng sẽ gây lẫn lộn khi phân tích tổng hợp.)

Usage:
  # Full grid search (9 SD combos × all analyses)  [DEFAULT]
  python tomato_vs/07_master_run.py

  # Single combination for testing
  python tomato_vs/07_master_run.py --mode one --strength 0.35 --guidance 7.5

  # Full run, skip k-fold to save time (use 5 fixed trials, matching submitted paper)
  python tomato_vs/07_master_run.py --no_kfold

  # Skip optional slow steps
  python tomato_vs/07_master_run.py --skip_image_quality --skip_sensitivity
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import argparse
import shutil
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

# ─────────────────────────── CLI args ────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='07_master_run: Full experimental pipeline for tomato leaf disease')
parser.add_argument('--mode',     choices=['full', 'one'], default='full',
                    help='"full" runs all 9 SD combos; "one" runs a single combo')
parser.add_argument('--train_count', type=int, default=20,
                    help='Training images per class (default 20)')
parser.add_argument('--test_count',  type=int, default=100,
                    help='Test images per class for main run (default 100, matches submitted paper)')
# one-mode overrides
parser.add_argument('--strength', type=float, default=0.50,
                    help='SD strength  (for --mode one)')
parser.add_argument('--guidance', type=float, default=7.5,
                    help='SD guidance  (for --mode one)')
# Skip flags for resuming / custom runs
parser.add_argument('--skip_data_setup',    action='store_true')
parser.add_argument('--skip_tda',           action='store_true')
parser.add_argument('--skip_randaug',       action='store_true')
parser.add_argument('--skip_sd',            action='store_true',
                    help='Skip SD generation (reuse existing sd_x5)')
parser.add_argument('--skip_label_sd',      action='store_true',
                    help='Skip label-only SD generation')
parser.add_argument('--skip_image_quality', action='store_true')
parser.add_argument('--skip_diversity',     action='store_true')
parser.add_argument('--no_kfold',           action='store_true',
                    help='Use 5 fixed trials instead of repeated k-fold (matches submitted paper)')
parser.add_argument('--skip_extra_baselines', action='store_true',
                    help='Skip MixUp/CutMix/RandAugment baselines')
parser.add_argument('--skip_ablation_prompt', action='store_true',
                    help='Skip label-only SD ablation in experiments')
parser.add_argument('--skip_quantity_ablation', action='store_true',
                    help='Skip augmentation-quantity ablation')
parser.add_argument('--skip_sensitivity',   action='store_true',
                    help='Skip sensitivity analysis (augmentation ratio: 2×/3×/4×, R3.8)')
args = parser.parse_args()

# ─────────────────────────── Paths ───────────────────────────────────────────
base_dir     = Path(__file__).parent.resolve()
results_dir  = base_dir / 'Results'
datasets_dir = base_dir / 'datasets'
results_dir.mkdir(parents=True, exist_ok=True)

python_exe   = sys.executable
SCRIPT       = lambda name: str(base_dir / name)

# Phase-0 backup location (defined here so backup_datasets can reference it)
_phase0_backup = results_dir / '_phase0_backup'
_tda_backup    = _phase0_backup / 'tda_x5'
_ra_backup     = _phase0_backup / 'randaugment_x5'
_base_backup   = _phase0_backup / 'baseline'

# ─────────────────────────── Grid ────────────────────────────────────────────
STRENGTHS  = [0.35, 0.50, 0.65]
GUIDANCES  = [6.0,  7.5,  9.0]

if args.mode == 'full':
    combos = list(product(STRENGTHS, GUIDANCES))
else:
    combos = [(args.strength, args.guidance)]

# ─────────────────────────── Helpers ─────────────────────────────────────────
def run_step(label, cmd, cwd=None, stream=True):
    """Run a subprocess.  Returns True on success, False on failure."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if stream:
        # Stream output live (important for long-running SD generation)
        ret = subprocess.run(cmd, cwd=cwd or str(base_dir.parent))
    else:
        ret = subprocess.run(cmd, cwd=cwd or str(base_dir.parent),
                             capture_output=True, text=True)
        if ret.stdout:
            print(ret.stdout)
        if ret.stderr:
            print(ret.stderr)
    if ret.returncode != 0:
        print(f"  [FAILED] {label}")
        return False
    print(f"  [OK] {label}")
    return True


def cleanup_sd_only():
    """Remove only the SD-generated datasets (keep baseline, tda_x5, randaugment_x5)."""
    for name in ['sd_x5', 'sd_labelonly_x5']:
        d = datasets_dir / name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Cleaned: {name}")


def backup_datasets(run_dir):
    """
    Backup SD-generated datasets to run_dir/generated_images_backup/.
    Only backs up SD datasets (tda_x5/baseline are already in _phase0_backup).
    """
    bk = run_dir / 'generated_images_backup'
    bk.mkdir(parents=True, exist_ok=True)
    for name in ['sd_x5', 'sd_labelonly_x5']:
        src = datasets_dir / name
        if src.exists():
            shutil.copytree(str(src), str(bk / name))
            print(f"  Backed up: {name}")
    print(f"  (tda_x5, randaugment_x5, baseline → see {_phase0_backup})")


def write_config(run_dir, strength, guidance, extras=None):
    cfg = {
        'timestamp':    datetime.now().isoformat(),
        'train_count':  args.train_count,
        'test_count':   args.test_count,
        'sd_strength':  strength,
        'sd_guidance':  guidance,
    }
    if extras:
        cfg.update(extras)
    with open(run_dir / 'experiment_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — One-time initial setup (data, TDA, RandAugment)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TOMATO-VS  ·  MASTER RUN  (Reviewer-Revised)")
print(f"  Mode        : {args.mode}  ({'all 9 combos' if args.mode=='full' else '1 combo'})")
print(f"  Train count : {args.train_count}  |  Test count : {args.test_count}")
print(f"  K-Fold      : {'RepeatedStratifiedKFold (5×3=15 folds)' if not args.no_kfold else '5 fixed trials  [matches submitted paper]'}")
print(f"  Combos      : {combos}")
print("="*70)

if not args.skip_data_setup:
    ok = run_step("PHASE 0-A  Data setup",
                  [python_exe, SCRIPT('01_data_setup.py'),
                   '--train_count', str(args.train_count),
                   '--test_count',  str(args.test_count)])
    if not ok:
        print("FATAL: data setup failed.  Aborting.")
        sys.exit(1)
else:
    print("\n[SKIP] Data setup")

if not args.skip_tda:
    ok = run_step("PHASE 0-B  TDA x5 generation",
                  [python_exe, SCRIPT('02_1_gen_tda.py')])
    if not ok:
        print("WARNING: TDA generation failed.  Continuing without tda_x5.")
else:
    print("\n[SKIP] TDA generation")

if not args.skip_randaug:
    ok = run_step("PHASE 0-C  RandAugment x5 generation",
                  [python_exe, SCRIPT('02_6_gen_baselines.py')])
    if not ok:
        print("WARNING: RandAugment generation failed.  randaugment_x5 unavailable.")
else:
    print("\n[SKIP] RandAugment generation")

# Save tda_x5, randaugment_x5 and baseline outside datasets_dir for sensitivity restore
_phase0_backup.mkdir(parents=True, exist_ok=True)

for src, dst in [(datasets_dir / 'tda_x5',          _tda_backup),
                 (datasets_dir / 'randaugment_x5',   _ra_backup),
                 (datasets_dir / 'baseline',         _base_backup)]:
    if src.exists() and not dst.exists():
        shutil.copytree(str(src), str(dst))
        print(f"  Phase-0 backed up: {src.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SD Grid Search  (9 combinations)
# ═══════════════════════════════════════════════════════════════════════════════
all_summary_rows = []          # accumulated across all combos
best_combo       = None        # (strength, guidance) of best sd_x5 acc
best_sd_acc      = -1.0

for combo_idx, (strength, guidance) in enumerate(combos, 1):
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = results_dir / f'{ts}_s{strength}_g{guidance}'
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, strength, guidance)

    print(f"\n{'#'*70}")
    print(f"# COMBO {combo_idx}/{len(combos)}  —  Strength={strength}  Guidance={guidance}")
    print(f"# Output: {run_dir}")
    print(f"{'#'*70}")

    # ── 1-A: SD generation (Gemini LLM prompts) ──────────────────────────────
    if not args.skip_sd:
        ok = run_step(
            f"1-A  SD generation  (strength={strength}, guidance={guidance})",
            [python_exe, SCRIPT('02_2_gen_sd.py'),
             '--strength',       str(strength),
             '--guidance',       str(guidance),
             '--output_log_dir', str(run_dir)]
        )
        if not ok:
            print("WARNING: SD generation failed for this combo.  Skipping.")
            continue
    else:
        print("\n[SKIP] SD generation")

    # ── 1-B: Label-only SD generation (ablation) ─────────────────────────────
    if not args.skip_label_sd and not args.skip_ablation_prompt:
        run_step(
            "1-B  SD label-only generation (ablation)",
            [python_exe, SCRIPT('02_2b_gen_sd_labelonly.py'),
             '--strength', str(strength),
             '--guidance', str(guidance)]
        )

    # ── 1-C: Image quality metrics (FID, LPIPS, Label Noise) ─────────────────
    if not args.skip_image_quality:
        run_step(
            "1-C  Image quality metrics  (FID / LPIPS / Label Noise)",
            [python_exe, SCRIPT('02_4_compute_image_quality.py'),
             '--strength', str(strength),
             '--guidance', str(guidance),
             '--run_dir',  str(run_dir)],
            stream=False
        )

    # ── 1-D: Diversity metrics ────────────────────────────────────────────────
    if not args.skip_diversity:
        run_step(
            "1-D  Diversity metrics  (LPIPS intra-class / Feature dispersion)",
            [python_exe, SCRIPT('02_5_compute_diversity.py'),
             '--run_dir', str(run_dir)],
            stream=False
        )

    # ── 1-E: Main experiments (k-fold + all baselines) ───────────────────────
    exp_cmd = [python_exe, SCRIPT('03_run_experiments.py'),
               '--output_dir',  str(run_dir),
               '--train_count', str(args.train_count)]
    if not args.no_kfold:
        exp_cmd.append('--use_kfold')
    if not args.skip_extra_baselines:
        exp_cmd.append('--extra_baselines')
    if not args.skip_ablation_prompt:
        exp_cmd.append('--ablation_prompt')

    run_step("1-E  Main experiments  (baseline / tda / sd / mixup / cutmix / randaug)",
             exp_cmd)

    # ── 1-F: Quantity ablation (aug_limit 1,2,3,4 = 2×,3×,4×,5×) ───────────
    if not args.skip_quantity_ablation:
        for lim in [1, 2, 3]:   # 4 is already covered by 1-E
            qabl_dir = run_dir / f'ablation_qty_{lim + 1}x'
            qabl_dir.mkdir(parents=True, exist_ok=True)
            qa_cmd = [python_exe, SCRIPT('03_run_experiments.py'),
                      '--output_dir',  str(qabl_dir),
                      '--train_count', str(args.train_count),
                      '--aug_limit',   str(lim)]
            if not args.no_kfold:
                qa_cmd.append('--use_kfold')
            run_step(f"1-F  Quantity ablation  (aug_limit={lim} → {lim+1}×)",
                     qa_cmd)

    # ── 1-G: Backup & partial cleanup ────────────────────────────────────────
    print("\n[Backup] Saving generated images …")
    backup_datasets(run_dir)
    cleanup_sd_only()
    print("  sd_x5 and sd_labelonly_x5 removed from datasets_dir")

    # ── 1-H: Visualise & statistical analysis ────────────────────────────────
    run_step("1-H  Visualisation",
             [python_exe, SCRIPT('04_visualize_results.py'),
              '--input_dir', str(run_dir)],
             stream=False)

    run_step("1-I  Statistical analysis  (Wilcoxon / Cohen's d)",
             [python_exe, SCRIPT('03_3_analyze_results.py'),
              '--input_dir', str(run_dir)],
             stream=False)

    # ── Collect best combo ────────────────────────────────────────────────────
    metrics_csv = run_dir / 'metrics_summary.csv'
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            sd_avg = df[(df['Exp'] == 'sd_x5') & (df['Trial'] == 'AVG')]
            if not sd_avg.empty:
                this_acc = float(sd_avg['Acc'].values[0])
                all_summary_rows.append({
                    'Strength': strength, 'Guidance': guidance,
                    'SD_Acc': this_acc
                })
                if this_acc > best_sd_acc:
                    best_sd_acc  = this_acc
                    best_combo   = (strength, guidance)
        except Exception as e:
            print(f"  Warning: could not read metrics: {e}")

    print(f"\n[DONE] Combo {combo_idx}/{len(combos)}  "
          f"Strength={strength}  Guidance={guidance}")

print(f"\n{'='*70}")
print(f"PHASE 1 COMPLETE  –  Best combo: {best_combo}  (sd_x5 Acc={best_sd_acc:.4f})")
print(f"{'='*70}")

if best_combo is None:
    best_combo = (0.35, 7.5)   # safe fallback
    print(f"  (no valid results found; using default best_combo={best_combo})")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Sensitivity Analysis: Augmentation Ratio  [R3.8]
#   "Is the 20-80-80 training ratio heuristic?"
#   Answers by varying aug_limit ∈ {1,2,3} → 2×/3×/4× augmentation.
#   aug_limit=4 (5× = 20-80-80) is the Phase-1 main experiment.
#   Test set is UNCHANGED (same split as Phase 1; no data_setup re-run needed).
#   k-fold is REQUIRED: aug_limit is only respected in get_fold_aug_samples().
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_sensitivity:
    SENSITIVITY_AUG_LIMITS = [1, 2, 3]   # aug_limit=4 (5×) already in Phase-1 main
    best_s, best_g = best_combo

    print(f"\n{'='*70}")
    print(f"PHASE 2  —  Sensitivity Analysis  (augmentation ratio, R3.8)")
    print(f"  Best combo   : strength={best_s}  guidance={best_g}")
    print(f"  Aug limits   : {SENSITIVITY_AUG_LIMITS}  (aug_limit=4 = 5× is Phase-1 main)")
    ratio_strs = [f"{lim+1}×: 20+{lim*20}+{lim*20}" for lim in SENSITIVITY_AUG_LIMITS]
    print(f"  Ratios       : {', '.join(ratio_strs)}, 5×: 20+80+80 [Phase1]")
    print(f"{'='*70}")

    # ── 2-A: Restore Phase-0 datasets (in case Phase-1 cleanup removed them) ─
    for src, dst_name in [(_tda_backup,   'tda_x5'),
                           (_ra_backup,    'randaugment_x5'),
                           (_base_backup,  'baseline')]:
        dst = datasets_dir / dst_name
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))
            print(f"  Restored {dst_name} from _phase0_backup")

    # ── 2-B: Generate SD for best combo ONCE (shared across all aug_limits) ──
    ts_s      = datetime.now().strftime('%Y%m%d_%H%M%S')
    sd_logdir = results_dir / f'{ts_s}_sensitivity_sd_log'
    sd_logdir.mkdir(parents=True, exist_ok=True)
    ok = run_step(
        f"2-B  SD generation  (strength={best_s}, guidance={best_g})",
        [python_exe, SCRIPT('02_2_gen_sd.py'),
         '--strength',       str(best_s),
         '--guidance',       str(best_g),
         '--output_log_dir', str(sd_logdir)]
    )
    if not ok:
        print(f"  Warning: SD generation failed; sd_x5 sensitivity runs may be empty.")

    # ── 2-C / 2-D / 2-E: Run for each aug_limit value ────────────────────────
    for lim in SENSITIVITY_AUG_LIMITS:
        aug_count   = lim * 20   # aug images per method per class
        ratio_label = f"20 orig + {aug_count} TDA + {aug_count} SD  ({lim + 1}×)"

        print(f"\n{'─'*60}")
        print(f"  Sensitivity aug_limit={lim}: {ratio_label}")
        print(f"{'─'*60}")

        ts2     = datetime.now().strftime('%Y%m%d_%H%M%S')
        sen_dir = results_dir / f'{ts2}_sensitivity_aL{lim}'
        sen_dir.mkdir(parents=True, exist_ok=True)
        write_config(sen_dir, best_s, best_g,
                     extras={'sensitivity_aug_limit': lim,
                             'sensitivity_ratio_label': ratio_label})

        run_step(
            f"2-C  Experiments  (aug_limit={lim}: {ratio_label})",
            [python_exe, SCRIPT('03_run_experiments.py'),
             '--output_dir',  str(sen_dir),
             '--train_count', str(args.train_count),
             '--aug_limit',   str(lim),
             '--use_kfold']   # REQUIRED: aug_limit only respected in k-fold mode
        )

        run_step(
            f"2-D  Visualisation  (aug_limit={lim})",
            [python_exe, SCRIPT('04_visualize_results.py'),
             '--input_dir', str(sen_dir)],
            stream=False
        )
        run_step(
            f"2-E  Statistical analysis  (aug_limit={lim})",
            [python_exe, SCRIPT('03_3_analyze_results.py'),
             '--input_dir', str(sen_dir)],
            stream=False
        )
        print(f"  [DONE] Sensitivity aug_limit={lim}")

    # ── 2-F: Cleanup SD after ALL aug_limit runs are done ─────────────────────
    cleanup_sd_only()
    print(f"  sd_x5 cleaned after all sensitivity runs.")

    print(f"\n{'='*70}")
    print("PHASE 2  SENSITIVITY ANALYSIS  COMPLETE")
    print(f"  Results:")
    for lim in SENSITIVITY_AUG_LIMITS:
        print(f"    sensitivity_aL{lim}/  →  {lim+1}× aug (20+{lim*20}+{lim*20}/class)")
    print(f"  Compare with Phase-1 main (aug_limit=4, 5×=20+80+80) to answer R3.8.")
    print(f"{'='*70}")
else:
    print("\n[SKIP] Sensitivity analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MASTER RUN COMPLETED")
print(f"{'='*70}")
print(f"  Mode         : {args.mode}")
print(f"  Combos run   : {len(combos)}")
print(f"  Best combo   : Strength={best_combo[0]}  Guidance={best_combo[1]}")
print(f"  SD Acc (best): {best_sd_acc:.4f}")
print(f"  Results dir  : {results_dir}")
print(f"{'='*70}")

if all_summary_rows:
    try:
        pd.DataFrame(all_summary_rows).to_csv(
            results_dir / 'all_combos_summary.csv', index=False)
        print(f"\nAll-combos summary: {results_dir / 'all_combos_summary.csv'}")
    except Exception as e:
        print(f"  Summary CSV error: {e}")

print("\nDone.  All results are in:", results_dir)




