"""
08_master_run_hotfix.py — Resume Interrupted Pipeline (Power-Loss Recovery)
============================================================================
Tự động phát hiện combo SD nào đã hoàn thành (qua Results/ folder) và tiếp tục
chạy từ chỗ bị gián đoạn. Không cần can thiệp thủ công.

⚠️  QUAN TRỌNG — TRƯỚC KHI CHẠY:
  File này CHỈ dùng để TIẾP TỤC sau khi mất điện / gián đoạn.
  Nó sẽ TỰ ĐỘNG BỎ QUA mọi combo đã có metrics_summary.csv trong Results/.
  
  Nếu bạn muốn chạy lại từ đầu với settings mới (ví dụ: test_count, lr đã sửa),
  hãy XÓA các thư mục kết quả cũ trước:
    Remove-Item -Recurse -Force tomato_vs/Results/*
  Hoặc dùng 07_master_run.py (luôn chạy lại từ đầu hoàn toàn).

Tình huống phục hồi (sau power loss trong cùng một lần chạy):
  - Phase 0 (data/TDA/RandAugment): đã xong → tự bỏ qua
  - Phase 1 combos đã có metrics_summary.csv → tự bỏ qua
  - Phase 1 combos chưa có metrics_summary.csv → chạy đầy đủ
  - Phase 2 sensitivity (aug_limit=1,2,3): tự detect → chạy phần còn lại

Nếu mất điện lần nữa: chạy lại file này — sẽ tự detect và tiếp tục.

Current settings (đúng cho bài revision):
  test_count  : 100/class
  lr          : 1e-4  (standard for partial fine-tuning on few-shot data; Howard & Ruder 2018)
  Gemini model: gemini-2.5-flash (2.0-flash deprecated June 2026)
  Fixed trials: 5 (matches submitted paper)
  K-Fold mode : 15 folds = 5×3 (PRIMARY for revision)

Usage:
    python tomato_vs/08_master_run_hotfix.py            # k-fold (15 folds) [default, primary]
    python tomato_vs/08_master_run_hotfix.py --no_kfold  # 5 fixed trials   [matches paper]
    python tomato_vs/08_master_run_hotfix.py --skip_sensitivity
    python tomato_vs/08_master_run_hotfix.py --skip_image_quality --skip_diversity
"""

import sys
import torch
if not torch.cuda.is_available():
    print("=" * 60)
    print("  ERROR: No CUDA GPU found.  Aborting.")
    print("  Check nvidia-smi and CUDA driver installation.")
    print("=" * 60)
    sys.exit(1)

import argparse
import re
import shutil
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

# ─────────────────────────── CLI args ────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='08_master_run_hotfix: Resume interrupted experimental pipeline')
parser.add_argument('--train_count', type=int, default=20,
                    help='Training images per class (default 20, must match original run)')
parser.add_argument('--test_count', type=int, default=100,
                    help='Test images per class for Phase-1 combos (default 100, matches submitted paper)')
parser.add_argument('--no_kfold', action='store_true',
                    help='Use 5 fixed trials instead of RepeatedStratifiedKFold (matches submitted paper)')
parser.add_argument('--skip_image_quality', action='store_true',
                    help='Skip FID/LPIPS/Label-noise computation (02_4)')
parser.add_argument('--skip_diversity', action='store_true',
                    help='Skip diversity metrics computation (02_5)')
parser.add_argument('--skip_extra_baselines', action='store_true',
                    help='Skip MixUp/CutMix/RandAugment extra baselines')
parser.add_argument('--skip_ablation_prompt', action='store_true',
                    help='Skip label-only SD prompt ablation')
parser.add_argument('--skip_quantity_ablation', action='store_true',
                    help='Skip augmentation-quantity ablation (2x/3x/4x)')
parser.add_argument('--skip_sensitivity', action='store_true',
                    help='Skip Phase-2 sensitivity analysis (augmentation ratio: 2×/3×/4×, R3.8)')
args = parser.parse_args()

# ─────────────────────────── Paths ───────────────────────────────────────────
base_dir     = Path(__file__).parent.resolve()
results_dir  = base_dir / 'Results'
datasets_dir = base_dir / 'datasets'
results_dir.mkdir(parents=True, exist_ok=True)

python_exe = sys.executable
SCRIPT     = lambda name: str(base_dir / name)

_phase0_backup = results_dir / '_phase0_backup'
_tda_backup    = _phase0_backup / 'tda_x5'
_ra_backup     = _phase0_backup / 'randaugment_x5'
_base_backup   = _phase0_backup / 'baseline'

# Full 9-combo grid (same as 07_master_run.py)
STRENGTHS  = [0.35, 0.50, 0.65]
GUIDANCES  = [6.0,  7.5,  9.0]
ALL_COMBOS = list(product(STRENGTHS, GUIDANCES))   # 9 combos

# ─────────────────────────── Helpers ─────────────────────────────────────────

def run_step(label: str, cmd: list, cwd=None, stream: bool = True) -> bool:
    """Run a subprocess step. Returns True on success, False on failure."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    kwargs = dict(cwd=cwd or str(base_dir.parent))
    if stream:
        ret = subprocess.run(cmd, **kwargs)
    else:
        ret = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
        if ret.stdout:
            print(ret.stdout[-4000:])   # cap long outputs
        if ret.stderr:
            print(ret.stderr[-2000:])
    ok = ret.returncode == 0
    print(f"  [{'OK' if ok else 'FAILED'}] {label}")
    return ok


def cleanup_sd_only():
    """Remove only SD-generated datasets; keep baseline/tda_x5/randaugment_x5."""
    for name in ['sd_x5', 'sd_labelonly_x5']:
        d = datasets_dir / name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Cleaned: {name}")


def backup_datasets(run_dir: Path):
    """Backup SD datasets into run_dir/generated_images_backup/."""
    bk = run_dir / 'generated_images_backup'
    bk.mkdir(parents=True, exist_ok=True)
    for name in ['sd_x5', 'sd_labelonly_x5']:
        src = datasets_dir / name
        dst = bk / name
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))
            print(f"  Backed up: {name}")
    print(f"  (tda_x5/randaugment_x5/baseline → {_phase0_backup})")


def write_config(run_dir: Path, strength: float, guidance: float, extras=None):
    cfg = {
        'timestamp':   datetime.now().isoformat(),
        'train_count': args.train_count,
        'test_count':  args.test_count,
        'sd_strength': strength,
        'sd_guidance': guidance,
        'hotfix':      True,
    }
    if extras:
        cfg.update(extras)
    (run_dir / 'experiment_config.json').write_text(
        json.dumps(cfg, indent=2), encoding='utf-8')


# ─────────────────────────── Detection helpers ───────────────────────────────

# Pattern: 20260618_041346_s0.35_g6.0
_COMBO_PAT = re.compile(r'^\d{8}_\d{6}_s(\d+\.\d+)_g(\d+\.\d+)$')
# Pattern: 20260625_123456_sensitivity_aL1
_SENS_PAT  = re.compile(r'^\d{8}_\d{6}_sensitivity_aL(\d+)$')


def detect_completed_combos() -> set:
    """Return set of (strength, guidance) tuples that have metrics_summary.csv."""
    done = set()
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        m = _COMBO_PAT.match(d.name)
        if m and (d / 'metrics_summary.csv').exists():
            done.add((float(m.group(1)), float(m.group(2))))
    return done


def detect_completed_sensitivity() -> set:
    """Return set of aug_limit integers for sensitivity runs already done."""
    done = set()
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        m = _SENS_PAT.match(d.name)
        if m and (d / 'metrics_summary.csv').exists():
            done.add(int(m.group(1)))
    return done


def collect_existing_metrics() -> list:
    """Read metrics_summary.csv from all completed Phase-1 combos."""
    rows = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _COMBO_PAT.match(d.name)
        if not m:
            continue
        csv_path = d / 'metrics_summary.csv'
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            sd_avg = df[(df['Exp'] == 'sd_x5') & (df['Trial'] == 'AVG')]
            if not sd_avg.empty:
                rows.append({
                    'Strength': float(m.group(1)),
                    'Guidance': float(m.group(2)),
                    'SD_Acc':   float(sd_avg['Acc'].values[0]),
                })
        except Exception as exc:
            print(f"  [WARN] Could not read {csv_path.name}: {exc}")
    return rows


def sd_x5_is_ready(min_images: int = 400) -> bool:
    """
    Return True if datasets/sd_x5 already holds generated images.
    Threshold = 400 (5 classes × 80 min) so a partial gen is not accepted.
    """
    sd_train = datasets_dir / 'sd_x5' / 'train'
    if not sd_train.exists():
        return False
    total = sum(
        len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
        for cls_dir in sd_train.iterdir()
        if cls_dir.is_dir()
    )
    return total >= min_images


def pick_best_combo(summary_rows: list):
    """Return (strength, guidance) with highest SD_Acc from summary_rows."""
    if not summary_rows:
        return (0.35, 7.5)   # safe fallback
    best = max(summary_rows, key=lambda r: r['SD_Acc'])
    return (best['Strength'], best['Guidance'])


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TOMATO-VS  ·  MASTER RUN HOTFIX  (Power-Loss Recovery)")
print(f"  Train count : {args.train_count}  |  Test count : {args.test_count}")
print(f"  K-Fold      : {'RepeatedStratifiedKFold (5×3=15)' if not args.no_kfold else '5 fixed trials [matches submitted paper]'}")
print("=" * 70)

# ── Detect state ──────────────────────────────────────────────────────────────
completed_combos = detect_completed_combos()
completed_sens   = detect_completed_sensitivity()
all_combos_done  = (len(completed_combos) == 9)
remaining_combos = [(s, g) for s, g in ALL_COMBOS
                    if (s, g) not in completed_combos]

print(f"\n[STATUS] Phase-1 combos:")
for s, g in ALL_COMBOS:
    status = '✓ Done' if (s, g) in completed_combos else '→ Pending'
    print(f"    s={s}  g={g}  →  {status}")

print(f"\n[STATUS] Phase-2 sensitivity (aug_limit variation, R3.8):")
for lim in [1, 2, 3]:
    status = '✓ Done' if lim in completed_sens else '→ Pending'
    print(f"    aug_limit={lim} ({lim+1}×: 20+{lim*20}+{lim*20}/class)  →  {status}")

print(f"\n[STATUS] sd_x5 in datasets: {'PRESENT ✓' if sd_x5_is_ready() else 'absent'}")
print(f"[STATUS] _phase0_backup : {'✓' if _phase0_backup.exists() else '✗ MISSING!'}")

# Sanity check on phase0 backup
if not _phase0_backup.exists():
    print("\nFATAL: _phase0_backup not found. Cannot continue.")
    print("  Please restore baseline/tda_x5/randaugment_x5 manually to:")
    print(f"  {_phase0_backup}")
    sys.exit(1)

# Collect metrics from already-done combos
all_summary_rows = collect_existing_metrics()
print(f"\n[STATUS] Metrics loaded from {len(all_summary_rows)} completed combos.")
for r in sorted(all_summary_rows, key=lambda x: (x['Strength'], x['Guidance'])):
    print(f"    s={r['Strength']}  g={r['Guidance']}  sd_acc={r['SD_Acc']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Run remaining SD combos
# ═══════════════════════════════════════════════════════════════════════════════
if not remaining_combos:
    print("\n[SKIP] All 9 Phase-1 combos already completed.")
else:
    print(f"\n{'=' * 70}")
    print(f"PHASE 1 — Resuming from combo {9 - len(remaining_combos) + 1}/9")
    print(f"  Remaining: {remaining_combos}")
    print(f"{'=' * 70}")

    sd_ready = sd_x5_is_ready()
    first_pending = remaining_combos[0]

    for combo_idx, (strength, guidance) in enumerate(remaining_combos,
                                                      start=9 - len(remaining_combos) + 1):
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = results_dir / f'{ts}_s{strength}_g{guidance}'
        run_dir.mkdir(parents=True, exist_ok=True)
        write_config(run_dir, strength, guidance)

        is_first = ((strength, guidance) == first_pending)

        print(f"\n{'#' * 70}")
        print(f"# COMBO {combo_idx}/9  —  Strength={strength}  Guidance={guidance}")
        print(f"# Output: {run_dir.name}")
        if is_first and sd_ready:
            print(f"# [HOTFIX] sd_x5 already present — skipping SD generation")
        print(f"{'#' * 70}")

        # ── 1-A: SD generation ────────────────────────────────────────────────
        if is_first and sd_ready:
            print("\n[HOTFIX] Step 1-A skipped: sd_x5 already generated (500 images).")
        else:
            ok = run_step(
                f"1-A  SD generation  (strength={strength}, guidance={guidance})",
                [python_exe, SCRIPT('02_2_gen_sd.py'),
                 '--strength',       str(strength),
                 '--guidance',       str(guidance),
                 '--output_log_dir', str(run_dir)]
            )
            if not ok:
                print(f"  WARNING: SD generation failed for s={strength} g={guidance}. Skipping combo.")
                cleanup_sd_only()
                continue

        # ── 1-B: Label-only SD (ablation) ────────────────────────────────────
        if not args.skip_ablation_prompt:
            run_step(
                "1-B  SD label-only generation (ablation, R6)",
                [python_exe, SCRIPT('02_2b_gen_sd_labelonly.py'),
                 '--strength', str(strength),
                 '--guidance', str(guidance)]
            )
        else:
            print("\n[SKIP] Label-only SD generation (--skip_ablation_prompt)")

        # ── 1-C: Image quality metrics (FID / LPIPS / Label Noise) ───────────
        if not args.skip_image_quality:
            run_step(
                "1-C  Image quality metrics  (FID / LPIPS / Label Noise, R1+R3)",
                [python_exe, SCRIPT('02_4_compute_image_quality.py'),
                 '--strength', str(strength),
                 '--guidance', str(guidance),
                 '--run_dir',  str(run_dir)],
                stream=False
            )
        else:
            print("\n[SKIP] Image quality (--skip_image_quality)")

        # ── 1-D: Diversity metrics ────────────────────────────────────────────
        if not args.skip_diversity:
            run_step(
                "1-D  Diversity metrics  (LPIPS intra-class / Feature dispersion, R2)",
                [python_exe, SCRIPT('02_5_compute_diversity.py'),
                 '--run_dir', str(run_dir)],
                stream=False
            )
        else:
            print("\n[SKIP] Diversity metrics (--skip_diversity)")

        # ── 1-E: Main experiments (k-fold + all baselines) ───────────────────
        exp_cmd = [python_exe, SCRIPT('03_run_experiments.py'),
                   '--output_dir',  str(run_dir),
                   '--train_count', str(args.train_count)]
        if not args.no_kfold:
            exp_cmd.append('--use_kfold')
        if not args.skip_extra_baselines:
            exp_cmd.append('--extra_baselines')
        if not args.skip_ablation_prompt:
            exp_cmd.append('--ablation_prompt')

        ok = run_step(
            "1-E  Main experiments  (baseline / tda / sd / mixup / cutmix / randaug)",
            exp_cmd
        )
        if not ok:
            print("  WARNING: Experiments failed for this combo. "
                  "Continuing to cleanup and next combo.")

        # ── 1-F: Quantity ablation (aug_limit 1,2,3 → 2x,3x,4x) ─────────────
        if not args.skip_quantity_ablation:
            for lim in [1, 2, 3]:   # 4 (= 5×) is the default covered in 1-E
                qabl_dir = run_dir / f'ablation_qty_{lim + 1}x'
                qabl_dir.mkdir(parents=True, exist_ok=True)
                qa_cmd = [python_exe, SCRIPT('03_run_experiments.py'),
                          '--output_dir',  str(qabl_dir),
                          '--train_count', str(args.train_count),
                          '--aug_limit',   str(lim)]
                if not args.no_kfold:
                    qa_cmd.append('--use_kfold')
                run_step(f"1-F  Quantity ablation  (aug_limit={lim} → {lim + 1}×, R7)",
                         qa_cmd)
        else:
            print("\n[SKIP] Quantity ablation (--skip_quantity_ablation)")

        # ── 1-G: Backup generated images & clean sd datasets ─────────────────
        print("\n[Backup] Saving generated images …")
        backup_datasets(run_dir)
        cleanup_sd_only()
        print("  sd_x5 and sd_labelonly_x5 removed from datasets/")

        # After first combo, sd_ready flag is no longer relevant
        sd_ready = False

        # ── 1-H: Visualisation ───────────────────────────────────────────────
        run_step(
            "1-H  Visualisation",
            [python_exe, SCRIPT('04_visualize_results.py'),
             '--input_dir', str(run_dir)],
            stream=False
        )

        # ── 1-I: Statistical analysis (Wilcoxon / Friedman / Shapiro-Wilk) ───
        run_step(
            "1-I  Statistical analysis  (Wilcoxon / Friedman / Cohen's d, R9)",
            [python_exe, SCRIPT('03_3_analyze_results.py'),
             '--input_dir', str(run_dir)],
            stream=False
        )

        # ── Collect best combo info ───────────────────────────────────────────
        metrics_csv = run_dir / 'metrics_summary.csv'
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                sd_avg = df[(df['Exp'] == 'sd_x5') & (df['Trial'] == 'AVG')]
                if not sd_avg.empty:
                    this_acc = float(sd_avg['Acc'].values[0])
                    all_summary_rows.append({
                        'Strength': strength,
                        'Guidance': guidance,
                        'SD_Acc':   this_acc,
                    })
                    print(f"  sd_x5 Acc for this combo: {this_acc:.4f}")
            except Exception as exc:
                print(f"  Warning: could not read metrics: {exc}")

        print(f"\n[DONE] Combo {combo_idx}/9  —  s={strength}  g={guidance}")

print(f"\n{'=' * 70}")
print("PHASE 1 — ALL 9 COMBOS COMPLETE")
print(f"{'=' * 70}")

# ─── Determine best_combo from ALL 9 ─────────────────────────────────────────
best_combo  = pick_best_combo(all_summary_rows)
best_sd_acc = max((r['SD_Acc'] for r in all_summary_rows), default=-1.0)

print(f"\n  Best combo (sd_x5): Strength={best_combo[0]}  Guidance={best_combo[1]}"
      f"  Acc={best_sd_acc:.4f}")

# Save combined summary
if all_summary_rows:
    try:
        df_all = pd.DataFrame(all_summary_rows).sort_values(
            ['Strength', 'Guidance']).reset_index(drop=True)
        df_all['Rank'] = df_all['SD_Acc'].rank(ascending=False).astype(int)
        df_all.to_csv(results_dir / 'all_combos_summary.csv', index=False)
        print(f"\n  All-combos summary saved: Results/all_combos_summary.csv")
        print(df_all.to_string(index=False))
    except Exception as exc:
        print(f"  Warning: could not save all_combos_summary.csv: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Sensitivity Analysis: Augmentation Ratio  [R3.8]
#   "Is the 20-80-80 training ratio heuristic?"
#   Varies aug_limit ∈ {1,2,3} → 2×/3×/4× augmentation.
#   aug_limit=4 (5× = 20-80-80) is the Phase-1 main experiment (already done).
#   Test set unchanged; SD generated once and shared across all aug_limits.
#   k-fold REQUIRED: aug_limit only respected by get_fold_aug_samples().
# ═══════════════════════════════════════════════════════════════════════════════
if args.skip_sensitivity:
    print("\n[SKIP] Sensitivity analysis (--skip_sensitivity)")
else:
    SENSITIVITY_AUG_LIMITS = [1, 2, 3]   # aug_limit=4 (5×) is Phase-1 main exp
    best_s, best_g = best_combo

    pending_aug = [lim for lim in SENSITIVITY_AUG_LIMITS
                   if lim not in completed_sens]

    print(f"\n{'=' * 70}")
    print("PHASE 2  —  Sensitivity Analysis  (augmentation ratio, R3.8)")
    print(f"  Best combo   : strength={best_s}  guidance={best_g}")
    print(f"  All aug_limits: {SENSITIVITY_AUG_LIMITS}")
    if completed_sens:
        print(f"  Already done : aug_limit={sorted(completed_sens)}  →  skipping")
    print(f"  Pending      : {pending_aug}")
    print(f"{'=' * 70}")

    if not pending_aug:
        print("  All sensitivity runs already completed.")
    else:
        # ── Restore Phase-0 datasets if Phase-1 cleanup removed them ──────────
        for src, dst_name in [(_tda_backup,   'tda_x5'),
                               (_ra_backup,    'randaugment_x5'),
                               (_base_backup,  'baseline')]:
            dst = datasets_dir / dst_name
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))
                print(f"  Restored {dst_name} from _phase0_backup")
            elif not src.exists():
                print(f"  WARNING: backup {src.name} not found. {dst_name} may be absent.")

        # Verify test/ directory exists (required for evaluation; not backed up)
        test_dir_check = datasets_dir / 'test'
        if not test_dir_check.exists():
            print(f"\nFATAL: {test_dir_check} not found and no backup available.")
            print("  Re-run 01_data_setup.py to recreate it, then run this script again.")
            sys.exit(1)

        # ── Generate SD for best combo ONCE (shared across all aug_limits) ────
        if not sd_x5_is_ready():
            ts_s      = datetime.now().strftime('%Y%m%d_%H%M%S')
            sd_logdir = results_dir / f'{ts_s}_sensitivity_sd_log'
            sd_logdir.mkdir(parents=True, exist_ok=True)
            ok = run_step(
                f"2-A  SD generation  (strength={best_s}, guidance={best_g})",
                [python_exe, SCRIPT('02_2_gen_sd.py'),
                 '--strength',       str(best_s),
                 '--guidance',       str(best_g),
                 '--output_log_dir', str(sd_logdir)]
            )
            if not ok:
                print(f"  WARNING: SD generation failed. sd_x5 sensitivity may be empty.")
        else:
            print("\n[HOTFIX] sd_x5 already present — skipping SD re-generation")

        # ── Run for each pending aug_limit ────────────────────────────────────
        for lim in pending_aug:
            aug_count   = lim * 20   # aug images per method per class
            ratio_label = f"20 orig + {aug_count} TDA + {aug_count} SD  ({lim + 1}×)"

            print(f"\n{'─' * 60}")
            print(f"  Sensitivity aug_limit={lim}: {ratio_label}")
            print(f"{'─' * 60}")

            ts2     = datetime.now().strftime('%Y%m%d_%H%M%S')
            sen_dir = results_dir / f'{ts2}_sensitivity_aL{lim}'
            sen_dir.mkdir(parents=True, exist_ok=True)
            write_config(sen_dir, best_s, best_g,
                         extras={'sensitivity_aug_limit': lim,
                                 'sensitivity_ratio_label': ratio_label,
                                 'hotfix': True})

            run_step(
                f"2-B  Experiments  (aug_limit={lim}: {ratio_label})",
                [python_exe, SCRIPT('03_run_experiments.py'),
                 '--output_dir',  str(sen_dir),
                 '--train_count', str(args.train_count),
                 '--aug_limit',   str(lim),
                 '--use_kfold']   # REQUIRED: aug_limit only respected in k-fold mode
            )

            run_step(
                f"2-C  Visualisation  (aug_limit={lim})",
                [python_exe, SCRIPT('04_visualize_results.py'),
                 '--input_dir', str(sen_dir)],
                stream=False
            )
            run_step(
                f"2-D  Statistical analysis  (aug_limit={lim})",
                [python_exe, SCRIPT('03_3_analyze_results.py'),
                 '--input_dir', str(sen_dir)],
                stream=False
            )
            print(f"  [DONE] Sensitivity aug_limit={lim}")

        # ── Cleanup SD after all pending runs done ─────────────────────────────
        cleanup_sd_only()
        print("  sd_x5 cleaned after all sensitivity runs.")

    print(f"\n{'=' * 70}")
    print("PHASE 2  —  SENSITIVITY ANALYSIS COMPLETE")
    print(f"  Results:")
    for lim in SENSITIVITY_AUG_LIMITS:
        print(f"    sensitivity_aL{lim}/  →  {lim+1}× aug (20+{lim*20}+{lim*20}/class)")
    print(f"  Compare with Phase-1 main (aug_limit=4, 5×=20+80+80) for R3.8.")
    print(f"{'=' * 70}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("HOTFIX PIPELINE — COMPLETED")
print(f"{'=' * 70}")
print(f"  Best combo   : Strength={best_combo[0]}  Guidance={best_combo[1]}")
print(f"  SD Acc (best): {best_sd_acc:.4f}")
print(f"  Results dir  : {results_dir}")
print()
print("  Phase-1 combos status:")
for s, g in ALL_COMBOS:
    done = (s, g) in detect_completed_combos()
    print(f"    s={s}  g={g}  →  {'✓' if done else '?'}")
print()
print("  Phase-2 sensitivity status:")
final_sens = detect_completed_sensitivity()
for lim in [1, 2, 3]:
    done = lim in final_sens
    print(f"    aug_limit={lim} ({lim+1}×: 20+{lim*20}+{lim*20}/class)  →  {'✓' if done else '?'}")
print()
print("  Next step (if paper revision): run 05_final_comparison.py")
print(f"{'=' * 70}")
print("Done.")

