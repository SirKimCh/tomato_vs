"""
06_transfer_learning_comparison.py
===================================
Compare THREE training configurations on EfficientNet-B0:

  Config 1 – Transfer Learning + Partial Freezing  (MAIN config used in all paper results)
              pretrained=True, freeze all layers EXCEPT last 3 feature blocks + classifier
  Config 2 – Training from Scratch
              pretrained=False, all layers trainable
  Config 3 – Fine-tuning All Layers
              pretrained=True, all layers trainable (no freezing)

Purpose: justify why Config 1 was chosen as the primary training strategy.
Datasets tested: 'baseline' (always) + 'combined_tda_sd' (if available = CDA experiment)

Cross-validation: RepeatedStratifiedKFold (n_splits=5, n_repeats=3) = 15 folds
  - Fold base: 20 baseline originals per class
  - combined_tda_sd: augmented images from held-out originals are EXCLUDED (no leakage)
  - Consistent with 03_run_experiments.py --use_kfold (R3.1 / R3.6 reviewer requirement)

Usage:
  # Standalone – run after 01_data_setup.py (+ optionally 02_1_gen_tda.py + 02_2_gen_sd.py)
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
import re
import os
import platform
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from src.models.efficientnet_b0 import EfficientNetB0Model

# ─── CLI args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='06_transfer_learning_comparison: Compare 3 training configurations')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: Results/training_config_comparison/')
args = parser.parse_args()

# ─── Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 50
PATIENCE      = 10
LEARNING_RATE = 1e-4   # consistent with 03_run_experiments.py
WEIGHT_DECAY  = 1e-4
N_SPLITS      = 5      # RepeatedStratifiedKFold k
N_REPEATS     = 3      # repeats → 15 folds total (consistent with R3.1/R3.6)
N_FOLDS       = N_SPLITS * N_REPEATS   # = 15
AUG_LIMIT     = 4      # max augmented images per original per type (for CDA fold sampling)

# ─── GPU / performance constants ─────────────────────────────────────────────
NUM_WORKERS = 0 if platform.system() == 'Windows' else min(4, os.cpu_count() or 4)
USE_AMP     = True   # Mixed-precision (fp16 forward / fp32 update); consistent with 03_run_experiments.py

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

device = torch.device('cuda')

base_dir     = Path(__file__).parent.resolve()
datasets_dir = base_dir / 'datasets'
test_dir     = datasets_dir / 'test'

baseline_train_dir  = datasets_dir / 'baseline'        / 'train'
combined_train_dir  = datasets_dir / 'combined_tda_sd' / 'train'   # optional (CDA)

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

# ─── Transforms ──────────────────────────────────────────────────────────────
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

# ─── Training config definitions ─────────────────────────────────────────────
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

# ─── FoldDataset ─────────────────────────────────────────────────────────────
class FoldDataset(Dataset):
    """Dataset from explicit (path, label) sample list — used for k-fold."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── K-fold helpers ───────────────────────────────────────────────────────────
def get_baseline_samples():
    """Return (paths, labels, class_names) for all baseline training originals."""
    bl_dir = baseline_train_dir
    class_names = sorted(d.name for d in bl_dir.iterdir() if d.is_dir())
    if not class_names:
        print(f"Error: No class directories found in {bl_dir}.")
        sys.exit(1)
    c2i = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for cls in class_names:
        for f in sorted((bl_dir / cls).iterdir()):
            if f.is_file() and f.suffix.lower() in VALID_EXT:
                paths.append(str(f))
                labels.append(c2i[cls])
    if not paths:
        print(f"Error: No images found in {bl_dir}.")
        sys.exit(1)
    return paths, labels, class_names


def get_fold_aug_samples(aug_dir, train_stems, class_names, aug_limit=4, per_type=False):
    """
    Collect (path, label) from augmented dataset, keeping only images whose
    source original stem is in `train_stems` (prevents data leakage from
    held-out originals).  `aug_limit` caps augmented variants per original.

    per_type=True : count TDA and SD families independently (for combined_tda_sd).
      aug_limit=4 → 4 TDA + 4 SD = 8 aug per original → 9× dataset (cda_x9).
    per_type=False: single shared counter for all aug types (for tda_x5 / sd_x5).
    """
    c2i = {c: i for i, c in enumerate(class_names)}
    SUFFIX_PATTERNS = [
        re.compile(r'_aug\d+$'),
        re.compile(r'_sdlo\d+$'),
        re.compile(r'_sd\d+$'),
        re.compile(r'_ra\d+$'),
    ]
    TDA_PAT_STRS = {'_aug\\d+$', '_ra\\d+$'}
    samples = []

    for cls_dir in sorted(aug_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = c2i.get(cls_dir.name, -1)
        if label < 0:
            continue

        aug_cnt     = {}
        tda_aug_cnt = {}
        sd_aug_cnt  = {}

        for f in sorted(cls_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_EXT:
                continue

            stem        = f.stem
            source_stem = stem
            is_aug      = False
            matched_pat = None

            for pat in SUFFIX_PATTERNS:
                m = pat.search(stem)
                if m:
                    source_stem = stem[:m.start()]
                    is_aug      = True
                    matched_pat = pat
                    break

            if source_stem not in train_stems:
                continue

            if is_aug:
                if per_type and matched_pat is not None:
                    is_tda_type = matched_pat.pattern in TDA_PAT_STRS
                    cnt_dict    = tda_aug_cnt if is_tda_type else sd_aug_cnt
                    cnt = cnt_dict.get(source_stem, 0)
                    if cnt >= aug_limit:
                        continue
                    cnt_dict[source_stem] = cnt + 1
                else:
                    cnt = aug_cnt.get(source_stem, 0)
                    if cnt >= aug_limit:
                        continue
                    aug_cnt[source_stem] = cnt + 1

            samples.append((str(f), label))

    return samples


# ─── Core training + evaluation (shared by all folds) ────────────────────────
def _train_and_eval(tr_ds, vl_ds, class_names, pretrained_flag, freeze_mode,
                    exp_label, fold_idx, out_dir, training_curves_data,
                    log_params=False):
    """
    Train EfficientNet-B0 on tr_ds, validate on vl_ds, evaluate on fixed test set.
    Returns (acc, prec, rec, f1, mcc, auc, cm).
    """
    seed = 42 + fold_idx
    torch.manual_seed(seed);    np.random.seed(seed)
    random.seed(seed);          torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    num_classes  = len(class_names)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=transform_test)

    model = EfficientNetB0Model(num_classes=num_classes, pretrained=pretrained_flag)

    if freeze_mode == 'partial':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.classifier.parameters():
            p.requires_grad = True
        for p in model.model.features[-3:].parameters():
            p.requires_grad = True
    # else freeze_mode == 'none': all layers trainable

    if log_params:
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pd.DataFrame([{
            'exp_label':       exp_label,
            'total_params':    total_params,
            'trainable_params': trainable_params,
            'frozen_params':   total_params - trainable_params,
            'pretrained':      pretrained_flag,
            'freeze_mode':     freeze_mode,
        }]).to_csv(out_dir / f'model_params_{exp_label}.csv', index=False)
        print(f"  [{exp_label}] Total: {total_params:,} | "
              f"Trainable: {trainable_params:,} | "
              f"Frozen: {total_params - trainable_params:,}")

    model = model.to(device)

    tr_loader  = DataLoader(tr_ds,       batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                            persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(vl_ds,       batch_size=BATCH_SIZE, shuffle=False,
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

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    pbar = tqdm(range(EPOCHS), desc=f"{exp_label} F{fold_idx+1}", leave=False)
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
            'Exp': exp_label, 'Fold': fold_idx + 1, 'Epoch': epoch + 1,
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
    axes[0].set_title(f'{exp_label} - Loss (F{fold_idx+1})')
    axes[0].legend();  axes[0].grid(alpha=0.3)
    axes[1].plot(tr_accs, label='Train', color='blue')
    axes[1].plot(vl_accs, label='Val',   color='orange')
    axes[1].set_title(f'{exp_label} - Acc (F{fold_idx+1})')
    axes[1].legend();  axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'loss_curve_{exp_label}_f{fold_idx+1}.png', dpi=150)
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
    plt.title(f'{exp_label} - Confusion Matrix (F{fold_idx+1})')
    plt.tight_layout()
    plt.savefig(out_dir / f'cm_{exp_label}_f{fold_idx+1}.png', dpi=150)
    plt.close()

    del model, optimizer, scheduler, tr_loader, val_loader, te_loader
    torch.cuda.empty_cache();  gc.collect()
    return acc, prec, rec, f1, mcc, auc, cm


# ─── Run one training config across all datasets × 15 folds ──────────────────
def run_config(config_name, config_cfg, dataset_dirs,
               bl_paths, bl_labels, class_names, out_base):
    """
    Run `config_name` on each dataset in `dataset_dirs` using 15-fold CV.
    bl_paths / bl_labels / class_names: output of get_baseline_samples().
    """
    pretrained  = config_cfg['pretrained']
    freeze_mode = config_cfg['freeze_mode']

    config_out = out_base / config_name
    config_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"  {config_cfg['description']}")
    print(f"  {config_cfg['note']}")
    print(f"  Cross-validation: RepeatedStratifiedKFold "
          f"(k={N_SPLITS}, repeats={N_REPEATS}) = {N_FOLDS} folds")
    print(f"{'='*60}")

    results         = []
    cms             = {exp: [] for exp in dataset_dirs}
    history         = {exp: {'acc': [], 'prec': [], 'rec': [],
                             'f1':  [], 'mcc':  [], 'auc': []}
                       for exp in dataset_dirs}
    training_curves = []
    params_logged   = set()   # log model params once per (config, dataset)

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

    for fold_idx, (tr_idx, vl_idx) in enumerate(
        rskf.split(bl_paths, bl_labels)
    ):
        print(f"\n  FOLD {fold_idx + 1}/{N_FOLDS}")

        train_stems = {Path(bl_paths[i]).stem for i in tr_idx}
        val_samples = [(bl_paths[i], bl_labels[i]) for i in vl_idx]
        vl_ds       = FoldDataset(val_samples, transform_test)

        for exp_name, train_dir in dataset_dirs.items():
            if not train_dir.exists():
                print(f"    Skipping {exp_name} – not found")
                continue

            # Build training sample list for this fold
            if exp_name == 'baseline':
                tr_samples = [(bl_paths[i], bl_labels[i]) for i in tr_idx]
            elif exp_name == 'combined_tda_sd':
                # per_type=True: count TDA and SD aug separately → 4 TDA + 4 SD = 9×
                tr_samples = get_fold_aug_samples(
                    train_dir, train_stems, class_names,
                    aug_limit=AUG_LIMIT, per_type=True)
            else:
                # Generic aug dataset (tda_x5, sd_x5, etc. if added later)
                tr_samples = get_fold_aug_samples(
                    train_dir, train_stems, class_names, aug_limit=AUG_LIMIT)

            if not tr_samples:
                print(f"    Skipping {exp_name} – no training samples for this fold")
                continue

            tr_ds     = FoldDataset(tr_samples, transform_train)
            exp_label = f"{config_name}_{exp_name}"
            log_p     = (exp_label not in params_logged)
            if log_p:
                params_logged.add(exp_label)

            try:
                acc, prec, rec, f1, mcc, auc, cm = _train_and_eval(
                    tr_ds, vl_ds, class_names,
                    pretrained, freeze_mode,
                    exp_label, fold_idx,
                    config_out, training_curves,
                    log_params=log_p)
                for k, v in zip(['acc','prec','rec','f1','mcc','auc'],
                                 [acc, prec, rec, f1, mcc, auc]):
                    history[exp_name][k].append(v)
                cms[exp_name].append(cm)
                results.append({
                    'Config': config_name, 'Dataset': exp_name,
                    'Fold': fold_idx + 1,
                    'Acc': acc, 'Prec': prec, 'Rec': rec,
                    'F1': f1, 'MCC': mcc, 'AUC': auc,
                })
                print(f"    {exp_name} F{fold_idx+1} | "
                      f"Acc={acc:.4f} F1={f1:.4f} MCC={mcc:.4f} AUC={auc:.4f}")
            except Exception as e:
                print(f"    Error {exp_name} F{fold_idx+1}: {e}")
                import traceback; traceback.print_exc()
                torch.cuda.empty_cache();  gc.collect()

    # Aggregate statistics across 15 folds
    for exp_name in dataset_dirs:
        h = history[exp_name]
        if not h['acc']:
            continue
        means = {k: float(np.mean(h[k])) for k in h}
        stds  = {k: float(np.std( h[k])) for k in h}
        for kind, d in [('AVG', means), ('STD', stds)]:
            results.append({'Config': config_name, 'Dataset': exp_name,
                             'Fold': kind,
                             'Acc': d['acc'], 'Prec': d['prec'], 'Rec': d['rec'],
                             'F1': d['f1'], 'MCC': d['mcc'], 'AUC': d['auc']})
        print(f"\n  {exp_name} AVG ({N_FOLDS} folds) | "
              f"Acc={means['acc']:.4f}±{stds['acc']:.4f}  "
              f"F1={means['f1']:.4f}±{stds['f1']:.4f}  "
              f"MCC={means['mcc']:.4f}±{stds['mcc']:.4f}")

    pd.DataFrame(results).to_csv(config_out / 'metrics_summary.csv', index=False)
    pd.DataFrame(training_curves).to_csv(config_out / 'training_curves.csv', index=False)

    for exp_name in dataset_dirs:
        if cms[exp_name]:
            agg = np.sum(cms[exp_name], axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(agg, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{config_name}/{exp_name} - Aggregate CM ({N_FOLDS} Folds)')
            plt.tight_layout()
            plt.savefig(config_out / f'cm_aggregate_{exp_name}.png', dpi=150)
            plt.close()

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"06_transfer_learning_comparison.py")
    print(f"  Cross-validation : RepeatedStratifiedKFold "
          f"(k={N_SPLITS} × {N_REPEATS} = {N_FOLDS} folds)")
    print(f"  AUG limit        : {AUG_LIMIT} per type "
          f"(CDA: {AUG_LIMIT} TDA + {AUG_LIMIT} SD = {AUG_LIMIT*2} aug/original → "
          f"{AUG_LIMIT*2 + 1}× dataset)")
    print(f"{'='*60}")

    # Gather baseline originals for fold splitting
    bl_paths, bl_labels, class_names = get_baseline_samples()
    print(f"  Baseline images  : {len(bl_paths)} total "
          f"({len(bl_paths)//len(class_names)} per class, {len(class_names)} classes)")

    dataset_dirs = {'baseline': baseline_train_dir}
    if combined_train_dir.exists():
        dataset_dirs['combined_tda_sd'] = combined_train_dir
        print("  combined_tda_sd found → CDA experiment will run alongside baseline.")
    else:
        print("  combined_tda_sd not found → baseline only.")
        print("  (Run 02_1_gen_tda.py + 02_2_gen_sd.py, then 07_master_run.py to create CDA.)")

    all_results = []
    for config_name, config_cfg in TRAINING_CONFIGS.items():
        all_results.extend(
            run_config(config_name, config_cfg, dataset_dirs,
                       bl_paths, bl_labels, class_names, output_dir))

    # Combined summary across all configs
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_dir / 'all_configs_comparison.csv', index=False)

    df_avg = df_all[df_all['Fold'] == 'AVG'].copy()
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIG COMPARISON — FINAL SUMMARY ({N_FOLDS} folds)")
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
            ax.set_title(f'Config Comparison — {metric} ({dataset_name}, {N_FOLDS} folds)')
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, sub[metric].values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / f'config_comparison_{dataset_name}.png', dpi=150)
        plt.close()
        print(f"  Saved: config_comparison_{dataset_name}.png")

    print(f"\nAll results → {output_dir}")
    if not df_avg.empty:
        best_row = df_avg.loc[df_avg['Acc'].idxmax()]
        print(f"Best: {best_row['Config']} on {best_row['Dataset']}  "
              f"Acc={best_row['Acc']:.4f}  F1={best_row['F1']:.4f}")
