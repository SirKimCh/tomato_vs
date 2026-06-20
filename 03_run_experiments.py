"""
03_run_experiments.py
Train and evaluate EfficientNet-B0 for each augmentation method.

Original experiments (always run):
  baseline  — 20 real images per class, standard training transforms
  tda_x5    — TDA-augmented 5× dataset
  sd_x5     — Stable-Diffusion 5× dataset

Extended experiments (--extra_baselines):
  mixup        — MixUp  applied ONLINE to baseline data during training
  cutmix       — CutMix applied ONLINE to baseline data during training
  randaugment  — RandAugment pre-generated 5× dataset (or online if missing)

Ablation (--ablation_prompt):
  sd_labelonly_x5 — SD with label-only prompts vs Gemini LLM prompts

Cross-validation modes:
  Default     : 5 independent trials with random seeds (matches submitted paper)
  --use_kfold : RepeatedStratifiedKFold (n_splits=5, n_repeats=3 → 15 folds)
                K-fold splits original baseline images; augmented images
                derived from held-out originals are excluded from training,
                preventing data leakage.  PRIMARY mode for revision (R3.1/R3.6).

Quantity ablation (--aug_limit):
  Limit augmented images per original  (1→2×, 2→3×, 3→4×, 4→5×)
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import argparse
import random
import re
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
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from src.models.efficientnet_b0 import EfficientNetB0Model

# ─────────────────────────── CLI args ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',      type=str,  required=True)
parser.add_argument('--train_count',     type=int,  default=20,
                    help='Original training images per class (for metadata only; default 20)')
parser.add_argument('--use_kfold',       action='store_true', default=False,
                    help='Use RepeatedStratifiedKFold instead of fixed trials')
parser.add_argument('--n_splits',        type=int,  default=5,
                    help='K for k-fold  (default 5)')
parser.add_argument('--n_repeats',       type=int,  default=3,
                    help='Repeats for repeated k-fold  (default 3)')
parser.add_argument('--extra_baselines', action='store_true', default=False,
                    help='Also run MixUp, CutMix, RandAugment experiments')
parser.add_argument('--ablation_prompt', action='store_true', default=False,
                    help='Also run sd_labelonly_x5 for prompt-type ablation')
parser.add_argument('--aug_limit',       type=int,  default=4,
                    help='Max augmented images per original (1–4). '
                         'ONLY respected in --use_kfold mode via get_fold_aug_samples(); '
                         'fixed-trial mode (default) loads the full dataset directory '
                         'regardless of aug_limit. Use --use_kfold for sensitivity analysis.')
args = parser.parse_args()

# Runtime guard: aug_limit is silently ignored in fixed-trial mode (by design — fixed-trial
# loads the full dataset directory).  Warn loudly so the user doesn't run sensitivity
# analysis without k-fold and get misleading results.
if args.aug_limit != 4 and not args.use_kfold:
    print("=" * 70)
    print("  WARNING: --aug_limit is IGNORED in fixed-trial mode.")
    print("  aug_limit is only respected by get_fold_aug_samples() in k-fold mode.")
    print("  For sensitivity analysis, always pass --use_kfold as well.")
    print("  Add --use_kfold to your command, or accept that aug_limit has no effect.")
    print("=" * 70)

# ─────────────────────────── Hyper-parameters ────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 50
PATIENCE      = 10
LEARNING_RATE = 1e-3   # original paper setting (AdamW, matches submitted manuscript)
WEIGHT_DECAY  = 1e-4
NUM_TRIALS    = 5      # fixed-trial mode seeds (matches submitted paper; k-fold gives 15 folds for R3.1)

device       = torch.device('cuda')
base_dir     = Path(__file__).parent.resolve()
datasets_dir = base_dir / 'datasets'
test_dir     = datasets_dir / 'test'
output_dir   = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if not test_dir.exists():
    print(f"Error: {test_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)

# ─────────────────────────── Transforms ──────────────────────────────────────
tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tf_randaug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tf_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VALID_EXT = {'.jpg', '.jpeg', '.png'}

# ─────────────────────────── Custom Dataset ───────────────────────────────────
class FoldDataset(Dataset):
    """Dataset from explicit (path, label) sample list – used for k-fold."""
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

# ─────────────────────────── K-fold helpers ───────────────────────────────────
def get_baseline_samples():
    """Return (paths, labels, class_names) for all baseline training images."""
    bl_dir = datasets_dir / 'baseline' / 'train'
    if not bl_dir.exists():
        print(f"Error: {bl_dir} does not exist.  Run 01_data_setup.py first.")
        sys.exit(1)
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


def get_fold_aug_samples(aug_dir, train_stems, class_names, aug_limit=4):
    """
    Collect (path, label) from augmented dataset, keeping only images whose
    source original stem is in `train_stems` (prevents data leakage from
    held-out originals into training).  `aug_limit` caps augmented variants.

    Naming conventions recognised (regex anchored at end-of-stem):
      original:      <stem>.jpg
      tda:           <stem>_augN.jpg
      sd gemini:     <stem>_sdN.jpg
      sd label-only: <stem>_sdloN.jpg
      randaugment:   <stem>_raN.jpg

    Order matters: check _sdlo before _sd to avoid _sd matching _sdlo filenames.
    """
    c2i     = {c: i for i, c in enumerate(class_names)}
    # Ordered from most-specific to least-specific (prevents _sd from eating _sdlo)
    SUFFIX_PATTERNS = [
        re.compile(r'_aug\d+$'),
        re.compile(r'_sdlo\d+$'),
        re.compile(r'_sd\d+$'),
        re.compile(r'_ra\d+$'),
    ]
    samples = []

    for cls_dir in sorted(aug_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = c2i.get(cls_dir.name, -1)
        if label < 0:
            continue

        aug_cnt = {}

        for f in sorted(cls_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_EXT:
                continue

            stem        = f.stem
            source_stem = stem
            is_aug      = False

            for pat in SUFFIX_PATTERNS:
                m = pat.search(stem)
                if m:
                    source_stem = stem[:m.start()]
                    is_aug      = True
                    break

            if source_stem not in train_stems:
                continue

            if is_aug:
                cnt = aug_cnt.get(source_stem, 0)
                if cnt >= aug_limit:
                    continue
                aug_cnt[source_stem] = cnt + 1

            samples.append((str(f), label))

    return samples

# ─────────────────────────── MixUp / CutMix ──────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[idx]
    W, H     = x.size(3), x.size(2)
    cut_w    = int(W * np.sqrt(1 - lam))
    cut_h    = int(H * np.sqrt(1 - lam))
    cx, cy   = np.random.randint(W), np.random.randint(H)
    x1 = max(0, cx - cut_w // 2);  x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2);  y2 = min(H, cy + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y_a, y_b, lam_adj


def mixed_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

# ─────────────────────────── Shared training / eval loop ─────────────────────
training_curves_data = []
per_class_results    = []    # accumulated per-class metrics for R10 analysis


def _train_eval(train_ds, val_ds, test_ds,
                num_classes, class_names,
                exp_name, trial_id, aug_mode):
    """
    Core training + evaluation.
    trial_id : trial number (fixed mode) or fold index (k-fold mode).
    aug_mode : None | 'mixup' | 'cutmix' | 'randaugment'
    Returns  : (acc, prec, rec, f1, mcc, auc, cm)
    """
    seed = 42 + trial_id
    torch.manual_seed(seed);    np.random.seed(seed)
    random.seed(seed);          torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    model = EfficientNetB0Model(num_classes=num_classes, pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.classifier.parameters():
        p.requires_grad = True
    for p in model.model.features[-3:].parameters():
        p.requires_grad = True
    model = model.to(device)

    tr_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    te_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []
    best_vl, patience, best_state = float('inf'), 0, None

    pbar = tqdm(range(EPOCHS), desc=f"{exp_name} T{trial_id}", leave=False)
    for epoch in pbar:
        model.train()
        run_loss = correct = total = 0

        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if aug_mode == 'mixup':
                Xm, ya, yb, lam = mixup_data(X, y)
                out  = model(Xm)
                loss = mixed_loss(criterion, out, ya, yb, lam)
                pred = out.argmax(1)
            elif aug_mode == 'cutmix':
                Xc, ya, yb, lam = cutmix_data(X, y)
                out  = model(Xc)
                loss = mixed_loss(criterion, out, ya, yb, lam)
                pred = out.argmax(1)
            else:
                out  = model(X)
                loss = criterion(out, y)
                pred = out.argmax(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            correct  += (pred == y).sum().item()
            total    += y.size(0)

        tl = run_loss / len(tr_loader)
        ta = correct / total
        tr_losses.append(tl);  tr_accs.append(ta)

        model.eval()
        vl_run = vl_cor = vl_tot = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out   = model(X)
                vl_run += criterion(out, y).item()
                vl_cor += (out.argmax(1) == y).sum().item()
                vl_tot += y.size(0)

        vl = vl_run / len(val_loader)
        va = vl_cor / vl_tot
        vl_losses.append(vl);  vl_accs.append(va)
        scheduler.step()

        training_curves_data.append({
            'Exp': exp_name, 'Trial': trial_id, 'Epoch': epoch + 1,
            'Train_Loss': tl, 'Train_Acc': ta, 'Val_Loss': vl, 'Val_Acc': va,
        })
        pbar.set_postfix({'TL': f'{tl:.3f}', 'TA': f'{ta:.2f}',
                          'VL': f'{vl:.3f}', 'VA': f'{va:.2f}'})

        if vl < best_vl:
            best_vl = vl;  patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    pbar.close()
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Loss / Acc curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tr_losses, label='Train', color='blue')
    axes[0].plot(vl_losses, label='Val',   color='orange')
    axes[0].set_title(f'{exp_name} - Loss (T{trial_id})')
    axes[0].legend();  axes[0].grid(alpha=0.3)
    axes[1].plot(tr_accs, label='Train', color='blue')
    axes[1].plot(vl_accs, label='Val',   color='orange')
    axes[1].set_title(f'{exp_name} - Acc (T{trial_id})')
    axes[1].legend();  axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{exp_name}_t{trial_id}.png', dpi=150)
    plt.close()

    # Test evaluation
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for X, y in te_loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            probs_all.extend(torch.softmax(out, 1).cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

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
    plt.title(f'{exp_name} - Confusion Matrix (T{trial_id})')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{exp_name}_t{trial_id}.png', dpi=150)
    plt.close()

    # ── Per-class metrics (R10: Early Blight vs Late Blight confusion) ────────
    report = classification_report(
        labels_all, preds_all, target_names=class_names,
        output_dict=True, zero_division=0)
    for cls in class_names:
        if cls in report:
            per_class_results.append({
                'Exp': exp_name, 'Trial': trial_id, 'Class': cls,
                'Precision': round(report[cls]['precision'], 4),
                'Recall':    round(report[cls]['recall'],    4),
                'F1':        round(report[cls]['f1-score'],  4),
                'Support':   int(report[cls]['support']),
            })

    # Specific Early Blight ↔ Late Blight confusion rates
    EB = 'Tomato___Early_blight'
    LB = 'Tomato___Late_blight'
    if EB in class_names and LB in class_names:
        ei, li = class_names.index(EB), class_names.index(LB)
        eb_total = cm[ei].sum() + 1e-8
        lb_total = cm[li].sum() + 1e-8
        per_class_results.append({
            'Exp': exp_name, 'Trial': trial_id,
            'Class': 'EB_confused_as_LB',
            'Precision': round(float(cm[ei, li] / eb_total), 4),
            'Recall': -1, 'F1': -1,
            'Support': int(cm[ei, li]),
        })
        per_class_results.append({
            'Exp': exp_name, 'Trial': trial_id,
            'Class': 'LB_confused_as_EB',
            'Precision': round(float(cm[li, ei] / lb_total), 4),
            'Recall': -1, 'F1': -1,
            'Support': int(cm[li, ei]),
        })

    del model, optimizer, scheduler, tr_loader, val_loader, te_loader
    torch.cuda.empty_cache();  gc.collect()
    return acc, prec, rec, f1, mcc, auc, cm


def run_experiment(dataset_dir, exp_name, seed, trial, aug_mode=None):
    """
    Fixed-trial wrapper — backward-compatible with the original script.
    Uses random_split on the full augmented dataset for validation.
    """
    torch.manual_seed(seed);  np.random.seed(seed)
    random.seed(seed);        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    tr    = tf_randaug if aug_mode == 'randaugment' else tf_train
    full  = datasets.ImageFolder(str(dataset_dir), transform=tr)
    test  = datasets.ImageFolder(str(test_dir),    transform=tf_test)

    num_classes = len(full.classes)
    class_names = full.classes
    val_sz      = max(2, int(0.2 * len(full)))
    gen         = torch.Generator().manual_seed(seed)
    tr_ds, vl_ds = random_split(full, [len(full) - val_sz, val_sz], generator=gen)

    return _train_eval(tr_ds, vl_ds, test, num_classes, class_names,
                       exp_name, trial, aug_mode)


def run_fold_experiment(train_samples, val_samples, class_names,
                        exp_name, fold_idx, aug_mode=None):
    """K-fold variant using explicit (path, label) sample lists."""
    tr = tf_randaug if aug_mode == 'randaugment' else tf_train
    tr_ds = FoldDataset(train_samples, tr)
    vl_ds = FoldDataset(val_samples,   tf_test)
    te_ds = datasets.ImageFolder(str(test_dir), transform=tf_test)
    return _train_eval(tr_ds, vl_ds, te_ds, len(class_names), class_names,
                       exp_name, fold_idx, aug_mode)


# ─────────────────────────── Main ────────────────────────────────────────────
if __name__ == '__main__':

    core_exps = ['baseline', 'tda_x5', 'sd_x5']
    ext_exps  = ['mixup', 'cutmix', 'randaugment'] if args.extra_baselines else []
    abl_exps  = []
    if args.ablation_prompt:
        sdlo = datasets_dir / 'sd_labelonly_x5' / 'train'
        if sdlo.exists():
            abl_exps = ['sd_labelonly_x5']
        else:
            print("Warning: sd_labelonly_x5 not found.  Skipping prompt ablation.")

    experiments = core_exps + ext_exps + abl_exps

    AUG_MODE = {'mixup': 'mixup', 'cutmix': 'cutmix', 'randaugment': 'randaugment'}

    RA_DIR = datasets_dir / 'randaugment_x5' / 'train'
    EXP_DIR = {
        'baseline':        datasets_dir / 'baseline'         / 'train',
        'tda_x5':          datasets_dir / 'tda_x5'           / 'train',
        'sd_x5':           datasets_dir / 'sd_x5'            / 'train',
        'sd_labelonly_x5': datasets_dir / 'sd_labelonly_x5'  / 'train',
        'mixup':           datasets_dir / 'baseline'         / 'train',
        'cutmix':          datasets_dir / 'baseline'         / 'train',
        'randaugment':     RA_DIR if RA_DIR.exists()
                           else datasets_dir / 'baseline'    / 'train',
    }

    print(f"\nExperiments   : {experiments}")
    print(f"K-Fold mode   : {args.use_kfold}", end='')
    if args.use_kfold:
        print(f"  ({args.n_splits} × {args.n_repeats} = "
              f"{args.n_splits * args.n_repeats} folds)", end='')
    print(f"\nAug limit     : {args.aug_limit}  ({args.aug_limit + 1}×)")

    results  = []
    cm_store = {exp: [] for exp in experiments}
    history  = {exp: {'acc': [], 'prec': [], 'rec': [],
                      'f1':  [], 'mcc':  [], 'auc': []}
                for exp in experiments}

    # ── MODE A: Fixed trials ─────────────────────────────────────────────────
    if not args.use_kfold:
        for trial in range(1, NUM_TRIALS + 1):
            seed = 42 + trial
            print(f"\n{'='*60}")
            print(f"TRIAL {trial}/{NUM_TRIALS}  (seed={seed})")
            print(f"{'='*60}")

            for exp in experiments:
                d = EXP_DIR.get(exp)
                if d is None or not d.exists():
                    print(f"  Skipping {exp} – directory not found")
                    continue
                try:
                    acc, prec, rec, f1, mcc, auc, cm = run_experiment(
                        d, exp, seed, trial, aug_mode=AUG_MODE.get(exp))
                    for k, v in zip(['acc','prec','rec','f1','mcc','auc'],
                                    [acc, prec, rec, f1, mcc, auc]):
                        history[exp][k].append(v)
                    cm_store[exp].append(cm)
                    results.append({'Exp': exp, 'Trial': trial,
                                    'Acc': acc, 'Prec': prec, 'Rec': rec,
                                    'F1': f1, 'MCC': mcc, 'AUC': auc,
                                    'Train_Count': args.train_count})
                    print(f"  {exp} T{trial} | Acc={acc:.4f} F1={f1:.4f} "
                          f"MCC={mcc:.4f} AUC={auc:.4f}")
                except Exception as e:
                    print(f"  Error {exp} T{trial}: {e}")
                    import traceback; traceback.print_exc()
                    torch.cuda.empty_cache(); gc.collect()

    # ── MODE B: Repeated Stratified K-Fold ───────────────────────────────────
    else:
        bl_paths, bl_labels, class_names = get_baseline_samples()
        rskf    = RepeatedStratifiedKFold(n_splits=args.n_splits,
                                          n_repeats=args.n_repeats,
                                          random_state=42)
        n_folds = args.n_splits * args.n_repeats

        for fold_idx, (tr_idx, vl_idx) in enumerate(
            rskf.split(bl_paths, bl_labels)
        ):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{n_folds}")
            print(f"{'='*60}")

            train_stems = {Path(bl_paths[i]).stem for i in tr_idx}
            val_samples = [(bl_paths[i], bl_labels[i]) for i in vl_idx]

            for exp in experiments:
                d        = EXP_DIR.get(exp)
                aug_mode = AUG_MODE.get(exp)

                if d is None or not d.exists():
                    print(f"  Skipping {exp} – directory not found")
                    continue

                if exp in ('mixup', 'cutmix'):
                    tr_samples = [(bl_paths[i], bl_labels[i]) for i in tr_idx]
                else:
                    tr_samples = get_fold_aug_samples(
                        d, train_stems, class_names, aug_limit=args.aug_limit)

                if not tr_samples:
                    print(f"  Skipping {exp} – no training samples for fold")
                    continue

                try:
                    acc, prec, rec, f1, mcc, auc, cm = run_fold_experiment(
                        tr_samples, val_samples, class_names,
                        exp, fold_idx, aug_mode=aug_mode)
                    for k, v in zip(['acc','prec','rec','f1','mcc','auc'],
                                    [acc, prec, rec, f1, mcc, auc]):
                        history[exp][k].append(v)
                    cm_store[exp].append(cm)
                    results.append({'Exp': exp, 'Trial': fold_idx + 1,
                                    'Acc': acc, 'Prec': prec, 'Rec': rec,
                                    'F1': f1, 'MCC': mcc, 'AUC': auc,
                                    'Train_Count': args.train_count})
                    print(f"  {exp} F{fold_idx+1} | Acc={acc:.4f} F1={f1:.4f} "
                          f"MCC={mcc:.4f} AUC={auc:.4f}")
                except Exception as e:
                    print(f"  Error {exp} F{fold_idx+1}: {e}")
                    import traceback; traceback.print_exc()
                    torch.cuda.empty_cache(); gc.collect()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AVERAGE RESULTS")
    print(f"{'='*60}")

    for exp in experiments:
        h = history[exp]
        if not h['acc']:
            continue
        means = {k: float(np.mean(h[k])) for k in h}
        stds  = {k: float(np.std( h[k])) for k in h}

        for kind, d in [('AVG', means), ('STD', stds)]:
            results.append({'Exp': exp, 'Trial': kind,
                            'Acc': d['acc'],  'Prec': d['prec'],
                            'Rec': d['rec'],  'F1':   d['f1'],
                            'MCC': d['mcc'],  'AUC':  d['auc'],
                            'Train_Count': args.train_count})

        print(f"  {exp:<20} | Acc={means['acc']:.4f}±{stds['acc']:.4f}  "
              f"F1={means['f1']:.4f}±{stds['f1']:.4f}  "
              f"MCC={means['mcc']:.4f}  AUC={means['auc']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(
        output_dir / 'metrics_summary.csv', index=False)
    pd.DataFrame(training_curves_data).to_csv(
        output_dir / 'training_curves.csv', index=False)

    # Per-class metrics CSV (R10)
    if per_class_results:
        df_pc = pd.DataFrame(per_class_results)
        df_pc.to_csv(output_dir / 'per_class_metrics.csv', index=False)
        # Summary: mean ± std per (Exp, Class) across folds/trials
        pc_summary_rows = []
        for (exp, cls), grp in df_pc[df_pc['F1'] >= 0].groupby(['Exp', 'Class']):
            pc_summary_rows.append({
                'Exp': exp, 'Class': cls,
                'Precision_mean': round(grp['Precision'].mean(), 4),
                'Precision_std':  round(grp['Precision'].std(),  4),
                'Recall_mean':    round(grp['Recall'].mean(),    4),
                'Recall_std':     round(grp['Recall'].std(),     4),
                'F1_mean':        round(grp['F1'].mean(),        4),
                'F1_std':         round(grp['F1'].std(),         4),
            })
        pd.DataFrame(pc_summary_rows).to_csv(
            output_dir / 'per_class_summary.csv', index=False)
        # Confusion confusion-rate (EB↔LB)
        for tag in ['EB_confused_as_LB', 'LB_confused_as_EB']:
            sub = df_pc[df_pc['Class'] == tag]
            if not sub.empty:
                for exp_n, grp in sub.groupby('Exp'):
                    rate_mean = grp['Precision'].mean()
                    rate_std  = grp['Precision'].std()
                    print(f"  {exp_n:<22} {tag}: {rate_mean:.3f}±{rate_std:.3f}")

    for exp in experiments:
        if cm_store[exp]:
            agg = np.sum(cm_store[exp], axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(agg, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted');  plt.ylabel('Actual')
            plt.title(f'{exp} - Aggregate Confusion Matrix ({len(cm_store[exp])} evals)')
            plt.tight_layout()
            plt.savefig(output_dir / f'cm_aggregate_{exp}.png', dpi=150)
            plt.close()

    print(f"\nResults saved to: {output_dir}")
    print("  metrics_summary.csv  |  training_curves.csv  |  confusion matrices")
