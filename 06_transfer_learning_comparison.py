"""
06_transfer_learning_comparison.py
===================================
Compare THREE training configurations on EfficientNet-B0:

  Config 1 — Transfer Learning + Partial Freezing  (MAIN config used in all paper results)
              pretrained=True, freeze all layers EXCEPT last 3 feature blocks + classifier
  Config 2 — Training from Scratch
              pretrained=False, all layers trainable
  Config 3 — Fine-tuning All Layers
              pretrained=True, all layers trainable (no freezing)

Purpose: justify why Config 1 was chosen as the primary training strategy.
Datasets tested: 'baseline' (always) + 'combined_tda_sd' (if available)

Usage:
  # Standalone — run after 01_data_setup.py (+ optionally 02_1_gen_tda.py + 02_2_gen_sd.py)
  python tomato_vs/06_transfer_learning_comparison.py

  # With custom output
  python tomato_vs/06_transfer_learning_comparison.py --output_dir Results/my_config_run

Output: Results/training_config_comparison/
"""
import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import argparse
import random
import os
import platform
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from tqdm import tqdm

from src.models.efficientnet_b0 import EfficientNetB0Model

# ─────────────────────────── CLI args ────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='06_transfer_learning_comparison: Compare 3 training configurations')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: Results/training_config_comparison/')
args = parser.parse_args()

# ─────────────────────────── Hyperparameters ─────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 50
PATIENCE      = 10
LEARNING_RATE = 1e-4   # consistent with 03_run_experiments.py
WEIGHT_DECAY  = 1e-4
NUM_TRIALS    = 5

# ── GPU / performance constants ───────────────────────────────────────────────
NUM_WORKERS = 0 if platform.system() == 'Windows' else min(4, os.cpu_count() or 4)
USE_AMP     = True   # Mixed-precision (fp16 forward / fp32 update); consistent with 03_run_experiments.py

device = torch.device('cuda')

base_dir     = Path(__file__).parent.resolve()
datasets_dir = base_dir / 'datasets'          # ← fixed from legacy 'Data_ST'
test_dir     = datasets_dir / 'test'

baseline_train_dir  = datasets_dir / 'baseline'        / 'train'
combined_train_dir  = datasets_dir / 'combined_tda_sd' / 'train'   # optional

results_base = base_dir / 'Results'
results_base.mkdir(parents=True, exist_ok=True)

if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = results_base / 'training_config_comparison'
output_dir.mkdir(parents=True, exist_ok=True)

if not test_dir.exists():
    print(f"Error: {test_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)
if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)

# ─────────────────────────── Transforms ──────────────────────────────────────
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────── Training config definitions ─────────────────────
TRAINING_CONFIGS = {
    'Config1_PartialFreezing': {
        'pretrained': True,
        'freeze_mode': 'partial',
        'description': 'Transfer Learning + Partial Freezing  [MAIN config]',
        'note': 'Freeze all except last 3 feature blocks + classifier (~2.2M / 5.3M trainable)',
    },
    'Config2_FromScratch': {
        'pretrained': False,
        'freeze_mode': 'none',
        'description': 'Training from Scratch',
        'note': 'No ImageNet weights; all 5.3M parameters randomly initialised',
    },
    'Config3_FineTuneAll': {
        'pretrained': True,
        'freeze_mode': 'none',
        'description': 'Fine-tuning All Layers',
        'note': 'ImageNet pre-trained; all 5.3M parameters fine-tuned end-to-end',
    },
}

# ─────────────────────────── Single-trial experiment ─────────────────────────
def run_experiment(dataset_dir, exp_name, seed, trial,
                   pretrained_flag, freeze_mode,
                   out_dir, training_curves_data):
    """
    Train + evaluate for one (config, dataset, trial) combination.
    freeze_mode : 'partial' → Config 1 selective freezing
                  'none'    → all layers trainable (Config 2 or 3)
    """
    torch.manual_seed(seed);    np.random.seed(seed)
    random.seed(seed);          torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    full_dataset = datasets.ImageFolder(str(dataset_dir), transform=transform_train)
    test_dataset = datasets.ImageFolder(str(test_dir),    transform=transform_test)
    num_classes  = len(full_dataset.classes)
    class_names  = full_dataset.classes

    val_size   = max(2, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=generator)

    model = EfficientNetB0Model(num_classes=num_classes, pretrained=pretrained_flag)

    if freeze_mode == 'partial':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.classifier.parameters():
            p.requires_grad = True
        for p in model.model.features[-3:].parameters():
            p.requires_grad = True
    # else: all layers trainable (freeze_mode == 'none')

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trial == 1:
        pd.DataFrame([{
            'exp_name':        exp_name,
            'total_params':    total_params,
            'trainable_params': trainable_params,
            'frozen_params':   total_params - trainable_params,
            'pretrained':      pretrained_flag,
            'freeze_mode':     freeze_mode,
        }]).to_csv(out_dir / f'model_params_{exp_name}.csv', index=False)
        print(f"  [{exp_name}] Total: {total_params:,} | "
              f"Trainable: {trainable_params:,} | Frozen: {total_params - trainable_params:,}")

    model = model.to(device)

    tr_loader  = DataLoader(train_ds,    batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                            persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(NUM_WORKERS > 0))
    te_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(NUM_WORKERS > 0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []
    best_vl, pat_ctr, best_state = float('inf'), 0, None

    # AMP GradScaler — consistent with 03_run_experiments.py
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)  # PyTorch < 2.4 fallback

    pbar = tqdm(range(EPOCHS), desc=f"{exp_name} T{trial}", leave=False)
    for epoch in pbar:
        model.train()
        run_loss = correct = total = 0
        for inputs, labels in tr_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out  = model(inputs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        tl = run_loss / len(tr_loader);  ta = correct / total
        tr_losses.append(tl);  tr_accs.append(ta)

        model.eval()
        vl_run = vl_cor = vl_tot = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    out = model(inputs)
                vl_run += criterion(out.float(), labels).item()
                vl_cor += (out.argmax(1) == labels).sum().item()
                vl_tot += labels.size(0)

        vl = vl_run / len(val_loader);  va = vl_cor / vl_tot
        vl_losses.append(vl);  vl_accs.append(va)
        scheduler.step()

        training_curves_data.append({
            'Exp': exp_name, 'Trial': trial, 'Epoch': epoch + 1,
            'Train_Loss': tl, 'Train_Acc': ta, 'Val_Loss': vl, 'Val_Acc': va,
        })
        pbar.set_postfix({'TL': f'{tl:.3f}', 'TA': f'{ta:.2f}',
                          'VL': f'{vl:.3f}', 'VA': f'{va:.2f}'})

        if vl < best_vl:
            best_vl = vl;  pat_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_ctr += 1
            if pat_ctr >= PATIENCE:
                break
    pbar.close()

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tr_losses, label='Train', color='blue')
    axes[0].plot(vl_losses, label='Val',   color='orange')
    axes[0].set_title(f'{exp_name} - Loss (T{trial})')
    axes[0].legend();  axes[0].grid(alpha=0.3)
    axes[1].plot(tr_accs, label='Train', color='blue')
    axes[1].plot(vl_accs, label='Val',   color='orange')
    axes[1].set_title(f'{exp_name} - Acc (T{trial})')
    axes[1].legend();  axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'loss_curve_{exp_name}_t{trial}.png', dpi=150)
    plt.close()

    # Test evaluation
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for inputs, labels in te_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(inputs)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.softmax(out.float(), 1).cpu().numpy())

    probs_all = np.array(probs_all)
    acc  = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, average='weighted', zero_division=0)
    rec  = recall_score(labels_all, preds_all, average='weighted', zero_division=0)
    f1   = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    mcc  = matthews_corrcoef(labels_all, preds_all)
    try:
        auc = (roc_auc_score(labels_all, probs_all[:, 1])
               if num_classes == 2
               else roc_auc_score(labels_all, probs_all,
                                  multi_class='ovr', average='weighted'))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(labels_all, preds_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted');  plt.ylabel('Actual')
    plt.title(f'{exp_name} - Confusion Matrix (T{trial})')
    plt.tight_layout()
    plt.savefig(out_dir / f'cm_{exp_name}_t{trial}.png', dpi=150)
    plt.close()

    del model, optimizer, scheduler, tr_loader, val_loader, te_loader
    torch.cuda.empty_cache();  gc.collect()
    return acc, prec, rec, f1, mcc, auc, cm


# ─────────────────────────── Run one training config ─────────────────────────
def run_config(config_name, config_cfg, dataset_dirs, out_base):
    pretrained  = config_cfg['pretrained']
    freeze_mode = config_cfg['freeze_mode']

    config_out = out_base / config_name
    config_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"  {config_cfg['description']}")
    print(f"  {config_cfg['note']}")
    print(f"{'='*60}")

    results           = []
    cms               = {exp: [] for exp in dataset_dirs}
    history           = {exp: {'acc': [], 'prec': [], 'rec': [],
                                'f1':  [], 'mcc':  [], 'auc': []}
                         for exp in dataset_dirs}
    training_curves   = []

    for trial in range(1, NUM_TRIALS + 1):
        seed = 42 + trial
        print(f"\n  TRIAL {trial}/{NUM_TRIALS}  seed={seed}")

        for exp_name, train_dir in dataset_dirs.items():
            if not train_dir.exists():
                print(f"    Skipping {exp_name} – not found")
                continue
            try:
                acc, prec, rec, f1, mcc, auc, cm = run_experiment(
                    train_dir, f"{config_name}_{exp_name}",
                    seed, trial, pretrained, freeze_mode,
                    config_out, training_curves)
                for k, v in zip(['acc','prec','rec','f1','mcc','auc'],
                                 [acc, prec, rec, f1, mcc, auc]):
                    history[exp_name][k].append(v)
                cms[exp_name].append(cm)
                results.append({'Config': config_name, 'Dataset': exp_name,
                                 'Trial': trial,
                                 'Acc': acc, 'Prec': prec, 'Rec': rec,
                                 'F1': f1, 'MCC': mcc, 'AUC': auc})
                print(f"    {exp_name} T{trial} | Acc={acc:.4f} F1={f1:.4f} "
                      f"MCC={mcc:.4f} AUC={auc:.4f}")
            except Exception as e:
                print(f"    Error {exp_name} T{trial}: {e}")
                import traceback; traceback.print_exc()
                torch.cuda.empty_cache();  gc.collect()

    # Aggregate statistics
    for exp_name in dataset_dirs:
        h = history[exp_name]
        if not h['acc']:
            continue
        means = {k: float(np.mean(h[k])) for k in h}
        stds  = {k: float(np.std( h[k])) for k in h}
        for kind, d in [('AVG', means), ('STD', stds)]:
            results.append({'Config': config_name, 'Dataset': exp_name,
                             'Trial': kind,
                             'Acc': d['acc'], 'Prec': d['prec'], 'Rec': d['rec'],
                             'F1': d['f1'], 'MCC': d['mcc'], 'AUC': d['auc']})
        print(f"\n  {exp_name} AVG | Acc={means['acc']:.4f}±{stds['acc']:.4f}  "
              f"F1={means['f1']:.4f}±{stds['f1']:.4f}  MCC={means['mcc']:.4f}")

    pd.DataFrame(results).to_csv(config_out / 'metrics_summary.csv', index=False)
    pd.DataFrame(training_curves).to_csv(config_out / 'training_curves.csv', index=False)

    for exp_name in dataset_dirs:
        if cms[exp_name]:
            agg = np.sum(cms[exp_name], axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(agg, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{config_name}/{exp_name} - Aggregate CM ({NUM_TRIALS} trials)')
            plt.tight_layout()
            plt.savefig(config_out / f'cm_aggregate_{exp_name}.png', dpi=150)
            plt.close()

    return results


# ─────────────────────────── Main ────────────────────────────────────────────
if __name__ == '__main__':
    # Note: test set is created by 01_data_setup.py — do NOT call create_test_set() here.
    dataset_dirs = {'baseline': baseline_train_dir}
    if combined_train_dir.exists():
        dataset_dirs['combined_tda_sd'] = combined_train_dir
        print("  combined_tda_sd found — will also compare CDA per config.")
    else:
        print("  combined_tda_sd not found — using baseline only.")
        print("  (Run 02_2_gen_sd.py then create combined_tda_sd to add CDA comparison.)")

    all_results = []
    for config_name, config_cfg in TRAINING_CONFIGS.items():
        all_results.extend(run_config(config_name, config_cfg, dataset_dirs, output_dir))

    # ── Combined summary across all configs ───────────────────────────────────
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_dir / 'all_configs_comparison.csv', index=False)

    df_avg = df_all[df_all['Trial'] == 'AVG'].copy()
    print(f"\n{'='*70}")
    print("TRAINING CONFIG COMPARISON — FINAL SUMMARY")
    print(f"{'='*70}")
    print(df_avg[['Config', 'Dataset', 'Acc', 'F1', 'MCC', 'AUC']].to_string(index=False))

    # Summary bar chart
    for dataset_name in dataset_dirs:
        sub = df_avg[df_avg['Dataset'] == dataset_name].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cfg_colors = ['#2ecc71', '#e74c3c', '#3498db']
        for ax, metric in zip(axes, ['Acc', 'F1']):
            bars = ax.bar(range(len(sub)), sub[metric].values,
                          color=cfg_colors[:len(sub)], edgecolor='black', alpha=0.85)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub['Config'].values, rotation=20, ha='right', fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(f'Config Comparison — {metric} ({dataset_name})')
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, sub[metric].values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / f'config_comparison_{dataset_name}.png', dpi=150)
        plt.close()
        print(f"  Saved: config_comparison_{dataset_name}.png")

    print(f"\nAll results → {output_dir}")
    best_row = df_avg.loc[df_avg['Acc'].idxmax()]
    print(f"Best: {best_row['Config']} on {best_row['Dataset']}  "
          f"Acc={best_row['Acc']:.4f}  F1={best_row['F1']:.4f}")
