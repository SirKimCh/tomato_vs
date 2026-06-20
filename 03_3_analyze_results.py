"""
03_3_analyze_results.py
========================
Post-hoc statistical analysis of experiment results.

Reads metrics_summary.csv (+ optional per_class_metrics.csv) from
03_run_experiments.py and performs:

  1. Wilcoxon signed-rank tests (non-parametric, paired) for every method pair
  2. Cohen's d effect-size estimation (pooled)
  3. Shapiro-Wilk normality test per method × metric          [R9]
  4. Friedman test (non-parametric ANOVA) across all methods  [R9]
  5. Ranking table (mean ± std, sorted by F1)
  6. Significance-annotated bar plots
  7. Per-class Precision/Recall/F1 analysis                   [R10]
  8. Early Blight ↔ Late Blight confusion-rate analysis       [R10]

Saves (in --input_dir):
  statistical_tests.csv      — pairwise p-values, Cohen's d, effect size
  normality_tests.csv        — Shapiro-Wilk W and p per method × metric
  friedman_tests.csv         — Friedman chi² and p per metric
  ranking_table.csv          — mean ± std table, sorted by F1
  per_class_analysis.csv     — per-class F1/Precision/Recall (if available)
  statistical_comparison.png — bar chart with sig brackets
  per_class_comparison.png   — per-class F1 bar chart (if available)

Usage:
  python tomato_vs/03_3_analyze_results.py --input_dir Results/my_run [--alpha 0.05]
"""

import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, shapiro, friedmanchisquare

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True,
                    help='Path to run directory containing metrics_summary.csv')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Significance level (default 0.05)')
args = parser.parse_args()

input_dir = Path(args.input_dir)
metrics_file = input_dir / 'metrics_summary.csv'
if not metrics_file.exists():
    print(f"Error: {metrics_file} not found.  Run 03_run_experiments.py first.")
    sys.exit(1)

df_all = pd.read_csv(metrics_file)

# Keep only per-fold/trial rows (not AVG/STD summary rows)
df = df_all[~df_all['Trial'].isin(['AVG', 'STD'])].copy()
df['Trial'] = pd.to_numeric(df['Trial'], errors='coerce')
df = df.dropna(subset=['Trial'])

methods   = sorted(df['Exp'].unique().tolist())
metrics   = ['Acc', 'F1', 'MCC', 'AUC']
ALPHA     = args.alpha

print(f"\n{'='*60}")
print(f"STATISTICAL ANALYSIS  (alpha={ALPHA})")
print(f"Methods  : {methods}")
print(f"N folds  : {df.groupby('Exp')['Trial'].count().to_dict()}")
print(f"{'='*60}")

# ─── Cohen's d ───────────────────────────────────────────────────────────────
def cohens_d(a, b):
    """Pooled Cohen's d (signed)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return float((np.mean(a) - np.mean(b)) / (sp + 1e-12))

# ─── Pairwise tests ───────────────────────────────────────────────────────────
stat_rows = []
for metric in metrics:
    for m_a, m_b in itertools.combinations(methods, 2):
        sa = df[df['Exp'] == m_a][metric].values
        sb = df[df['Exp'] == m_b][metric].values

        # Align by trial order (both must have same number of evaluations)
        n = min(len(sa), len(sb))
        if n < 3:
            p_val = np.nan
            note  = 'insufficient samples'
        else:
            sa, sb = sa[:n], sb[:n]
            try:
                diff = sa - sb
                if np.all(diff == 0):
                    p_val = 1.0
                    note  = 'identical'
                else:
                    result = wilcoxon(sa, sb, alternative='two-sided')
                    p_val  = float(result.pvalue)
                    note   = ''
            except Exception as e:
                p_val = np.nan
                note  = str(e)

        d = cohens_d(df[df['Exp'] == m_a][metric].values,
                     df[df['Exp'] == m_b][metric].values)

        sig = ('***' if not np.isnan(p_val) and p_val < 0.001 else
               '**'  if not np.isnan(p_val) and p_val < 0.01  else
               '*'   if not np.isnan(p_val) and p_val < ALPHA else
               'ns')

        mean_a = df[df['Exp'] == m_a][metric].mean()
        mean_b = df[df['Exp'] == m_b][metric].mean()
        winner = m_a if mean_a > mean_b else m_b

        d_mag    = abs(d) if not np.isnan(d) else np.nan
        d_interp = ('large'      if not np.isnan(d_mag) and d_mag >= 0.8 else
                    'medium'     if not np.isnan(d_mag) and d_mag >= 0.5 else
                    'small'      if not np.isnan(d_mag) and d_mag >= 0.2 else
                    'negligible' if not np.isnan(d_mag) else 'N/A')

        stat_rows.append({
            'Metric':      metric,
            'Method_A':    m_a,
            'Method_B':    m_b,
            'Mean_A':      round(mean_a, 4),
            'Mean_B':      round(mean_b, 4),
            'Diff(A-B)':   round(mean_a - mean_b, 4),
            'Cohen_d':     round(float(d), 4) if not np.isnan(d) else np.nan,
            'Effect_Size': d_interp,
            'p_value':     round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            'Significant': sig,
            'Winner':      winner,
            'Note':        note,
        })

df_stats = pd.DataFrame(stat_rows)
df_stats.to_csv(input_dir / 'statistical_tests.csv', index=False)
print(f"\nStatistical tests saved: {input_dir / 'statistical_tests.csv'}")

# ─── Print pairwise Acc table ─────────────────────────────────────────────────
print(f"\n{'Metric':<5} {'Method A':<22} {'Method B':<22} {'p-val':>8} {'Sig':>4} {'d':>7} {'Effect':>10}")
print('─' * 84)
for _, row in df_stats[df_stats['Metric'] == 'Acc'].iterrows():
    p_str = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else '   N/A'
    d_str = f"{row['Cohen_d']:+.3f}" if not np.isnan(row['Cohen_d']) else '    N/A'
    print(f"{'Acc':<5} {row['Method_A']:<22} {row['Method_B']:<22} "
          f"{p_str:>8} {row['Significant']:>4} {d_str:>7} {row['Effect_Size']:>10}")

# ─── Shapiro-Wilk normality tests (R9) ───────────────────────────────────────
print(f"\n{'='*60}")
print("NORMALITY TESTS — Shapiro-Wilk  [R9]")
print(f"{'='*60}")
print(f"{'Method':<22} {'Metric':<6} {'W':>8} {'p':>8} {'Normal?':>8}")
print('─' * 56)

norm_rows = []
for method in methods:
    for metric in metrics:
        vals = df[df['Exp'] == method][metric].values
        if len(vals) < 3:
            w_val, p_norm, normal = np.nan, np.nan, 'N/A (n<3)'
        elif len(vals) > 5000:
            w_val, p_norm, normal = np.nan, np.nan, 'N/A (n>5000)'
        else:
            try:
                w_val, p_norm = shapiro(vals)
                normal = 'YES' if p_norm > ALPHA else 'NO'
            except Exception as e:
                w_val, p_norm, normal = np.nan, np.nan, f'err:{e}'

        norm_rows.append({
            'Method':  method,
            'Metric':  metric,
            'W_stat':  round(float(w_val), 4) if not np.isnan(w_val) else np.nan,
            'p_value': round(float(p_norm), 4) if not np.isnan(p_norm) else np.nan,
            'Normal':  normal,
        })
        if metric == 'Acc':
            p_s = f"{p_norm:.4f}" if not np.isnan(p_norm) else '   N/A'
            w_s = f"{w_val:.4f}"  if not np.isnan(w_val)  else '   N/A'
            print(f"{method:<22} {metric:<6} {w_s:>8} {p_s:>8} {normal:>8}")

df_norm = pd.DataFrame(norm_rows)
df_norm.to_csv(input_dir / 'normality_tests.csv', index=False)
print(f"\nNormality tests saved: {input_dir / 'normality_tests.csv'}")

non_normal = df_norm[(df_norm['Normal'] == 'NO') & (df_norm['Metric'] == 'Acc')]
if not non_normal.empty:
    print(f"  {len(non_normal)} method(s) non-normal → Wilcoxon (non-parametric) is justified.")
else:
    print(f"  All normally distributed → Wilcoxon still valid (conservative paired test).")

# ─── Friedman test (R9) ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FRIEDMAN TEST — non-parametric ANOVA  [R9]")
print(f"{'='*60}")
friedman_rows = []

for metric in metrics:
    method_vals = [df[df['Exp'] == m][metric].values for m in methods]
    min_n = min(len(v) for v in method_vals) if method_vals else 0

    if min_n < 3 or len(methods) < 3:
        chi2, p_fr = np.nan, np.nan
        note_fr = 'insufficient data'
    else:
        aligned = [v[:min_n] for v in method_vals]
        try:
            stat_fr = friedmanchisquare(*aligned)
            chi2    = float(stat_fr.statistic)
            p_fr    = float(stat_fr.pvalue)
            note_fr = ''
        except Exception as e:
            chi2, p_fr = np.nan, np.nan
            note_fr = str(e)

    sig_fr = ('***' if not np.isnan(p_fr) and p_fr < 0.001 else
              '**'  if not np.isnan(p_fr) and p_fr < 0.01  else
              '*'   if not np.isnan(p_fr) and p_fr < ALPHA  else
              'ns'  if not np.isnan(p_fr) else 'N/A')

    friedman_rows.append({
        'Metric':        metric,
        'Friedman_chi2': round(chi2, 4) if not np.isnan(chi2) else np.nan,
        'p_value':       round(p_fr,  4) if not np.isnan(p_fr)  else np.nan,
        'Significant':   sig_fr,
        'N_per_method':  min_n,
        'N_methods':     len(methods),
        'Note':          note_fr,
    })
    chi_s = f"{chi2:.4f}" if not np.isnan(chi2) else 'N/A'
    p_s   = f"{p_fr:.4f}" if not np.isnan(p_fr)  else 'N/A'
    print(f"  {metric:<5}: Friedman χ²={chi_s}  p={p_s}  {sig_fr}")
    if not np.isnan(p_fr) and p_fr < ALPHA:
        print(f"           → Significant global difference; post-hoc: pairwise Wilcoxon.")

pd.DataFrame(friedman_rows).to_csv(input_dir / 'friedman_tests.csv', index=False)
print(f"\nFriedman tests saved: {input_dir / 'friedman_tests.csv'}")

# ─── Ranking table ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RANKING TABLE (sorted by F1)")
print(f"{'='*60}")
rank_rows = []
for method in methods:
    m_df = df[df['Exp'] == method]
    row  = {'Method': method, 'N_evals': len(m_df)}
    for metric in metrics:
        vals = m_df[metric].values
        row[f'{metric}_mean'] = round(float(np.mean(vals)), 4)
        row[f'{metric}_std']  = round(float(np.std(vals)),  4)
        row[f'{metric}_str']  = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rank_rows.append(row)

df_rank = pd.DataFrame(rank_rows).sort_values('F1_mean', ascending=False)
df_rank.to_csv(input_dir / 'ranking_table.csv', index=False)

print(f"\n{'Method':<22} {'N':>5} {'Acc':>18} {'F1':>18} {'MCC':>18} {'AUC':>18}")
print('─' * 103)
for _, row in df_rank.iterrows():
    print(f"{row['Method']:<22} {int(row['N_evals']):>5} {row['Acc_str']:>18} "
          f"{row['F1_str']:>18} {row['MCC_str']:>18} {row['AUC_str']:>18}")
print(f"\nRanking table saved: {input_dir / 'ranking_table.csv'}")

# ─── Per-class analysis (R10) ─────────────────────────────────────────────────
pc_file = input_dir / 'per_class_metrics.csv'
df_pc_sum = None
classes_found = []

if pc_file.exists():
    df_pc       = pd.read_csv(pc_file)
    df_pc_valid = df_pc[df_pc['F1'] >= 0].copy()
    classes_found = sorted(df_pc_valid['Class'].unique())

    print(f"\n{'='*60}")
    print("PER-CLASS ANALYSIS  [R10: Early Blight vs Late Blight]")
    print(f"{'='*60}")
    print(f"  Classes: {classes_found}")

    pc_summary = []
    for cls in classes_found:
        for exp in methods:
            sub = df_pc_valid[(df_pc_valid['Class'] == cls) &
                              (df_pc_valid['Exp']   == exp)]
            if sub.empty:
                continue
            pc_summary.append({
                'Class':   cls,
                'Method':  exp,
                'F1_mean': round(sub['F1'].mean(), 4),
                'F1_std':  round(sub['F1'].std(),  4),
                'P_mean':  round(sub['Precision'].mean(), 4),
                'R_mean':  round(sub['Recall'].mean(), 4),
            })

    df_pc_sum = pd.DataFrame(pc_summary)
    df_pc_sum.to_csv(input_dir / 'per_class_analysis.csv', index=False)
    print(f"  Saved: per_class_analysis.csv")

    # Per-class F1 table
    print(f"\n  Per-class F1 (mean ± std):")
    header = f"  {'Class':<40}" + ''.join(f" {m[:12]:>14}" for m in methods)
    print(header)
    print('  ' + '─' * (40 + 15 * len(methods)))
    for cls in classes_found:
        row_str = f"  {cls:<40}"
        for m in methods:
            sub = df_pc_sum[(df_pc_sum['Class'] == cls) & (df_pc_sum['Method'] == m)]
            if not sub.empty:
                row_str += f" {sub['F1_mean'].values[0]:.4f}±{sub['F1_std'].values[0]:.4f}"
            else:
                row_str += f"{'N/A':>14}"
        print(row_str)

    # EB ↔ LB confusion rates
    for tag in ['EB_confused_as_LB', 'LB_confused_as_EB']:
        sub_conf = df_pc[df_pc['Class'] == tag]
        if not sub_conf.empty:
            label = ("Early→Late Blight" if tag == 'EB_confused_as_LB'
                     else "Late→Early Blight")
            print(f"\n  Confusion rate: {label} (target <0.10):")
            print(f"  {'Method':<22} {'Rate mean±std':>16}")
            print('  ' + '─' * 40)
            for m in methods:
                sub_m = sub_conf[sub_conf['Exp'] == m]
                if sub_m.empty:
                    print(f"  {m:<22} {'N/A':>16}")
                else:
                    r_mean = sub_m['Precision'].mean()
                    r_std  = sub_m['Precision'].std()
                    flag   = '  ← HIGH' if r_mean > 0.10 else ''
                    print(f"  {m:<22} {r_mean:.3f}±{r_std:.3f}{flag}")
else:
    print(f"\n[SKIP] Per-class analysis: {pc_file} not found.")
    print(f"       Re-run 03_run_experiments.py (updated) to generate per_class_metrics.csv.")

# ─── Significance bar plot ────────────────────────────────────────────────────
COLOR_MAP = {
    'baseline':        '#2ecc71',
    'tda_x5':          '#3498db',
    'sd_x5':           '#e74c3c',
    'mixup':           '#f39c12',
    'cutmix':          '#9b59b6',
    'randaugment':     '#1abc9c',
    'autoaugment':     '#e91e63',
    'augmix':          '#ff5722',
    'sd_labelonly_x5': '#e67e22',
}
LABEL_MAP = {
    'baseline':        'Baseline',
    'tda_x5':          'TDA×5',
    'sd_x5':           'SD×5 (LLM)',
    'mixup':           'MixUp',
    'cutmix':          'CutMix',
    'randaugment':     'RandAugment',
    'autoaugment':     'AutoAugment',
    'augmix':          'AugMix',
    'sd_labelonly_x5': 'SD×5 (Label)',
}

try:
    plot_metrics = ['Acc', 'F1', 'MCC', 'AUC']
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 6))
    if len(plot_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, plot_metrics):
        means      = df_rank[f'{metric}_mean'].values
        stds       = df_rank[f'{metric}_std'].values
        names      = df_rank['Method'].values
        names_list = list(names)
        x          = np.arange(len(names))

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[COLOR_MAP.get(n, '#95a5a6') for n in names],
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_MAP.get(n, n) for n in names],
                           rotation=30, ha='right', fontsize=8)
        ax.set_title(metric, fontweight='bold')
        ylim = (-1, 1.25) if metric == 'MCC' else (0, 1.25)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.3)

        sig_pairs = df_stats[
            (df_stats['Metric'] == metric) &
            (df_stats['Method_A'] == 'baseline') &
            (df_stats['Significant'].isin(['*', '**', '***']))
        ]
        for _, sp in sig_pairs.iterrows():
            idx = names_list.index(sp['Method_B']) if sp['Method_B'] in names_list else -1
            if idx >= 0:
                ax.text(idx, means[idx] + stds[idx] + 0.04,
                        sp['Significant'], ha='center', fontsize=9,
                        color='#c0392b', fontweight='bold')

    n_dict = df.groupby('Exp')['Trial'].count().to_dict()
    n_note = ', '.join(f"{m}:n={n_dict.get(m,'?')}" for m in list(methods)[:3])
    plt.suptitle(
        f'Method Comparison — EfficientNet-B0, Few-Shot\n'
        f'(* p<0.05, ** p<0.01, *** p<0.001 vs Baseline  |  {n_note})',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(input_dir / 'statistical_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {input_dir / 'statistical_comparison.png'}")
except Exception as e:
    print(f"Warning: could not generate plot: {e}")

# ─── Per-class F1 bar chart (R10) ─────────────────────────────────────────────
if df_pc_sum is not None and not df_pc_sum.empty:
    try:
        classes_plot = [c for c in classes_found
                        if not c.startswith('EB_') and not c.startswith('LB_')]
        n_cls  = len(classes_plot)
        n_cols = min(3, n_cls)
        n_rows = (n_cls + n_cols - 1) // n_cols if n_cls > 0 else 1
        if n_cls > 0:
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 4 * n_rows),
                                     squeeze=False)
            flat_axes = [axes[r][c]
                         for r in range(n_rows) for c in range(n_cols)]
            for i, cls in enumerate(classes_plot):
                ax  = flat_axes[i]
                sub = df_pc_sum[df_pc_sum['Class'] == cls]
                x   = np.arange(len(sub))
                ax.bar(x,
                       sub['F1_mean'].values,
                       yerr=sub['F1_std'].values,
                       capsize=3,
                       color=[COLOR_MAP.get(m, '#95a5a6') for m in sub['Method'].values],
                       edgecolor='black', linewidth=0.8, alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels([LABEL_MAP.get(m, m) for m in sub['Method'].values],
                                   rotation=30, ha='right', fontsize=8)
                cls_short = cls.replace('Tomato___', '').replace('_', ' ')
                ax.set_title(cls_short, fontsize=10, fontweight='bold')
                ax.set_ylim(0, 1.15)
                ax.set_ylabel('F1 (mean±std)', fontsize=9)
                ax.grid(axis='y', alpha=0.3)
            for i in range(n_cls, len(flat_axes)):
                flat_axes[i].axis('off')
            plt.suptitle('Per-Class F1 by Augmentation Method  [R10]',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(input_dir / 'per_class_comparison.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Per-class plot saved: {input_dir / 'per_class_comparison.png'}")
    except Exception as e:
        print(f"Warning: per-class plot failed: {e}")

print(f"\nStatistical analysis complete.")

