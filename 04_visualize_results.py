import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
args = parser.parse_args()

input_dir = Path(args.input_dir)

if not input_dir.exists():
    print(f"Error: {input_dir} does not exist")
    sys.exit(1)

metrics_file = input_dir / 'metrics_summary.csv'
curves_file = input_dir / 'training_curves.csv'

if not metrics_file.exists():
    print(f"Error: {metrics_file} does not exist")
    sys.exit(1)

df_metrics = pd.read_csv(metrics_file)
df_curves = pd.read_csv(curves_file) if curves_file.exists() else None

experiments = ['baseline', 'tda_x5', 'sd_x5']
exp_labels = {'baseline': 'Baseline', 'tda_x5': 'TDA x5', 'sd_x5': 'SD x5'}
colors = {'baseline': '#2ecc71', 'tda_x5': '#3498db', 'sd_x5': '#e74c3c'}

avg_data = df_metrics[df_metrics['Trial'] == 'AVG'].copy()
std_data = df_metrics[df_metrics['Trial'] == 'STD'].copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metrics = ['Acc', 'Prec', 'Rec', 'F1', 'MCC', 'AUC']
metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUC-ROC']

for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[idx // 3, idx % 3]

    x_pos = np.arange(len(experiments))
    means = []
    stds = []

    for exp in experiments:
        avg_row = avg_data[avg_data['Exp'] == exp]
        std_row = std_data[std_data['Exp'] == exp]

        if len(avg_row) > 0:
            means.append(avg_row[metric].values[0])
            stds.append(std_row[metric].values[0] if len(std_row) > 0 else 0)
        else:
            means.append(0)
            stds.append(0)

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=[colors[exp] for exp in experiments],
                  edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp_labels[exp] for exp in experiments], fontsize=10)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    if metric in ['Acc', 'Prec', 'Rec', 'F1', 'AUC']:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(-1, 1.1)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('GenAI vs TDA: Performance Metrics Comparison\n(MobileNetV3-Small, Few-Shot, 5 Trials)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(input_dir / 'summary_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: summary_comparison.png")

if df_curves is not None and len(df_curves) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp in experiments:
        exp_data = df_curves[df_curves['Exp'] == exp]
        if len(exp_data) == 0:
            continue

        avg_by_epoch = exp_data.groupby('Epoch').agg({
            'Train_Loss': ['mean', 'std'],
            'Val_Loss': ['mean', 'std'],
            'Train_Acc': ['mean', 'std'],
            'Val_Acc': ['mean', 'std']
        }).reset_index()
        avg_by_epoch.columns = ['Epoch', 'TL_mean', 'TL_std', 'VL_mean', 'VL_std',
                                 'TA_mean', 'TA_std', 'VA_mean', 'VA_std']

        epochs = avg_by_epoch['Epoch']

        axes[0].plot(epochs, avg_by_epoch['TL_mean'],
                     label=f'{exp_labels[exp]} Train', color=colors[exp], linestyle='-', linewidth=2)
        axes[0].fill_between(epochs,
                             avg_by_epoch['TL_mean'] - avg_by_epoch['TL_std'],
                             avg_by_epoch['TL_mean'] + avg_by_epoch['TL_std'],
                             color=colors[exp], alpha=0.2)
        axes[0].plot(epochs, avg_by_epoch['VL_mean'],
                     label=f'{exp_labels[exp]} Val', color=colors[exp], linestyle='--', linewidth=2)

        axes[1].plot(epochs, avg_by_epoch['TA_mean'],
                     label=f'{exp_labels[exp]} Train', color=colors[exp], linestyle='-', linewidth=2)
        axes[1].fill_between(epochs,
                             avg_by_epoch['TA_mean'] - avg_by_epoch['TA_std'],
                             avg_by_epoch['TA_mean'] + avg_by_epoch['TA_std'],
                             color=colors[exp], alpha=0.2)
        axes[1].plot(epochs, avg_by_epoch['VA_mean'],
                     label=f'{exp_labels[exp]} Val', color=colors[exp], linestyle='--', linewidth=2)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle('Learning Curves (Average ± Std over 5 Trials)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(input_dir / 'loss_acc_aggregated.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: loss_acc_aggregated.png")

cm_files = list(input_dir.glob('cm_aggregate_*.png'))
if cm_files:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, exp in enumerate(experiments):
        cm_file = input_dir / f'cm_aggregate_{exp}.png'
        if cm_file.exists():
            img = plt.imread(cm_file)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(exp_labels[exp], fontsize=14, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
            axes[idx].axis('off')

    plt.suptitle('Aggregate Confusion Matrices (5 Trials)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(input_dir / 'confusion_matrix_aggregated.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix_aggregated.png")

print(f"\nVisualization complete. All plots saved to {input_dir}")
