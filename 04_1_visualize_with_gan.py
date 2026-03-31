import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
args = parser.parse_args()

input_dir = Path(args.input_dir)
base_dir = Path(__file__).parent.resolve()
results_dir = base_dir / 'Results'

if not input_dir.exists():
    print(f"Error: {input_dir} does not exist")
    sys.exit(1)

main_metrics_file = input_dir / 'metrics_summary.csv'
if not main_metrics_file.exists():
    print(f"Error: {main_metrics_file} does not exist")
    sys.exit(1)

df_main = pd.read_csv(main_metrics_file)

gan_metrics_file = results_dir / 'gan_metrics_summary.csv'
if not gan_metrics_file.exists():
    print(f"Error: {gan_metrics_file} does not exist. Run 03_1_run_gan_experiment.py first.")
    sys.exit(1)

df_gan = pd.read_csv(gan_metrics_file)

df_all = pd.concat([df_main, df_gan], ignore_index=True)

experiments = ['baseline', 'tda_x5', 'sd_x5', 'gan_x5']
exp_labels = {'baseline': 'Baseline', 'tda_x5': 'TDA x5', 'sd_x5': 'SD x5', 'gan_x5': 'GAN x5'}
colors = {'baseline': '#2ecc71', 'tda_x5': '#3498db', 'sd_x5': '#e74c3c', 'gan_x5': '#9b59b6'}

avg_data = df_all[df_all['Trial'] == 'AVG'].copy()
std_data = df_all[df_all['Trial'] == 'STD'].copy()

available_experiments = [exp for exp in experiments if exp in avg_data['Exp'].values]

if len(available_experiments) == 0:
    print("Error: No experiment data found")
    sys.exit(1)

metrics = ['Acc', 'F1', 'MCC', 'AUC']
metric_titles = ['Accuracy', 'F1-Score', 'MCC', 'AUC-ROC']

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[idx]

    x_pos = np.arange(len(available_experiments))
    means = []
    stds = []

    for exp in available_experiments:
        avg_row = avg_data[avg_data['Exp'] == exp]
        std_row = std_data[std_data['Exp'] == exp]

        if len(avg_row) > 0:
            means.append(avg_row[metric].values[0])
            stds.append(std_row[metric].values[0] if len(std_row) > 0 else 0)
        else:
            means.append(0)
            stds.append(0)

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=[colors[exp] for exp in available_experiments],
                  edgecolor='black', linewidth=1.2, width=0.6)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp_labels[exp] for exp in available_experiments], fontsize=10, rotation=15)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    if metric in ['Acc', 'F1', 'AUC']:
        ax.set_ylim(0, 1.15)
    else:
        ax.set_ylim(-1, 1.15)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Baseline vs TDA vs SD vs GAN: Performance Comparison\n(EfficientNet-B0, Few-Shot, 5 Trials)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = results_dir / 'summary_comparison_with_gan.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

print("\nSummary (AVG +/- STD):")
print(f"{'Experiment':<12} {'Accuracy':<18} {'F1':<18} {'MCC':<18} {'AUC':<18}")
print("-" * 78)
for exp in available_experiments:
    avg_row = avg_data[avg_data['Exp'] == exp]
    std_row = std_data[std_data['Exp'] == exp]
    if len(avg_row) > 0:
        a_m = avg_row['Acc'].values[0]
        a_s = std_row['Acc'].values[0] if len(std_row) > 0 else 0
        f_m = avg_row['F1'].values[0]
        f_s = std_row['F1'].values[0] if len(std_row) > 0 else 0
        m_m = avg_row['MCC'].values[0]
        m_s = std_row['MCC'].values[0] if len(std_row) > 0 else 0
        u_m = avg_row['AUC'].values[0]
        u_s = std_row['AUC'].values[0] if len(std_row) > 0 else 0
        print(f"{exp_labels[exp]:<12} {a_m:.4f}+-{a_s:.4f}    {f_m:.4f}+-{f_s:.4f}    {m_m:.4f}+-{m_s:.4f}    {u_m:.4f}+-{u_s:.4f}")

print(f"\nVisualization complete.")

