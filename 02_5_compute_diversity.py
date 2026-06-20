"""
02_5_compute_diversity.py
Compute dataset diversity metrics to quantify how augmented datasets
differ from the original baseline in terms of intra-class variability:

  1. LPIPS intra-class diversity  – average pairwise LPIPS between images
     of the SAME class (higher = more diverse)
  2. Feature dispersion           – mean std of EfficientNet-B0 feature
     embeddings within each class (higher = more spread in feature space)

Compares: baseline / tda_x5 / sd_x5 / randaugment_x5 (if available)

Optional: pip install lpips

Usage:
  python tomato_vs/02_5_compute_diversity.py --run_dir Results/my_run
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import gc
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

HAS_LPIPS = False
try:
    import lpips as lpips_lib
    HAS_LPIPS = True
    print("[OK] lpips available – LPIPS diversity will be computed.")
except ImportError:
    print("[WARN] lpips not installed → LPIPS diversity skipped.")
    print("       Install: pip install lpips")

from src.models.efficientnet_b0 import EfficientNetB0Model

# ---------- CLI args --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True,
                    help='Directory to save diversity_metrics.csv')
parser.add_argument('--n_pairs', type=int, default=100,
                    help='Max random pairs per class for LPIPS intra-class')
args = parser.parse_args()

random.seed(42)
np.random.seed(42)

device       = torch.device('cuda')
base_dir     = Path(__file__).parent.resolve()
datasets_dir = base_dir / 'datasets'
run_dir      = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)

VALID_EXT = {'.jpg', '.jpeg', '.png'}

baseline_train_dir = datasets_dir / 'baseline' / 'train'
if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)

classes     = sorted([d.name for d in baseline_train_dir.iterdir() if d.is_dir()])
num_classes = len(classes)

# ---------- transforms ------------------------------------------------------
tf_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
tf_feat = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- models ----------------------------------------------------------
feat_model = EfficientNetB0Model(num_classes=num_classes, pretrained=True)
feat_model.to(device).eval()

lpips_fn = None
if HAS_LPIPS:
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

# ---------- helpers ---------------------------------------------------------
@torch.no_grad()
def extract_features(paths, batch_size=16):
    all_f = []
    for i in range(0, len(paths), batch_size):
        imgs = []
        for p in paths[i:i+batch_size]:
            try:
                imgs.append(tf_feat(Image.open(p).convert('RGB')))
            except Exception:
                pass
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        x = feat_model.model.features(x)
        x = feat_model.model.avgpool(x).flatten(1)
        all_f.append(x.cpu().numpy())
    return np.vstack(all_f) if all_f else np.zeros((0, 1280), dtype=np.float32)


def lpips_intraclass(paths, n_pairs=100):
    """
    Average pairwise LPIPS within a class (random sampling for speed).

    IMPORTANT for fair comparison across datasets of different sizes:
    call with the same subsample size for all datasets.
    """
    if not HAS_LPIPS or lpips_fn is None or len(paths) < 2:
        return -1.0
    n = len(paths)
    pool = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pool) > n_pairs:
        pool = random.sample(pool, n_pairs)
    scores = []
    for i, j in pool:
        try:
            a = tf_lpips(Image.open(paths[i]).convert('RGB'))
            b = tf_lpips(Image.open(paths[j]).convert('RGB'))
            a = (a * 2 - 1).unsqueeze(0).to(device)
            b = (b * 2 - 1).unsqueeze(0).to(device)
            with torch.no_grad():
                scores.append(float(lpips_fn(a, b).item()))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else -1.0


def feature_dispersion(paths):
    """Mean of per-dimension feature std across images (higher = more spread)."""
    if len(paths) < 2:
        return -1.0
    feats = extract_features(paths)
    if feats.shape[0] < 2:
        return -1.0
    return float(np.mean(np.std(feats, axis=0)))


# ---------- datasets to analyse ---------------------------------------------
DATASETS = ['baseline', 'tda_x5', 'sd_x5', 'randaugment_x5']

print(f"\n{'='*60}")
print("DIVERSITY METRICS")
print(f"{'='*60}")

rows = []

# Determine baseline class sizes for fair equal-sampling comparison
# We subsample all datasets to the same number of images per class as baseline,
# so LPIPS intra-class comparisons are not confounded by dataset size.
SUBSAMPLE_PER_CLASS = {}
baseline_dir_inner = datasets_dir / 'baseline' / 'train'
if baseline_dir_inner.exists():
    for cls in classes:
        cls_dir = baseline_dir_inner / cls
        if cls_dir.exists():
            n = len([f for f in cls_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in VALID_EXT])
            SUBSAMPLE_PER_CLASS[cls] = n

print("\nSubsample sizes for fair LPIPS comparison:")
for cls, n in SUBSAMPLE_PER_CLASS.items():
    print(f"  {cls}: {n} imgs")

for ds in DATASETS:
    ds_dir = datasets_dir / ds / 'train'
    if not ds_dir.exists():
        print(f"\n[{ds}] Not found – skipped.")
        continue

    print(f"\n[{ds}]")
    for cls in classes:
        cls_dir = ds_dir / cls
        if not cls_dir.exists():
            continue
        all_paths = sorted(str(f) for f in cls_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in VALID_EXT)
        if not all_paths:
            continue

        # Subsample to equal size for LPIPS (fair comparison)
        subsample_n = SUBSAMPLE_PER_CLASS.get(cls, len(all_paths))
        paths_for_lpips = (random.sample(all_paths, subsample_n)
                           if len(all_paths) > subsample_n else all_paths)

        lp  = lpips_intraclass(paths_for_lpips, n_pairs=args.n_pairs)
        fd  = feature_dispersion(all_paths)   # dispersion uses all images
        print(f"  {cls:<40}: LPIPS_intra={lp:+.4f}  FeatDisp={fd:.4f}  "
              f"(LPIPS on {len(paths_for_lpips)}, FeatDisp on {len(all_paths)})")
        rows.append({
            'Dataset':            ds,
            'Class':              cls,
            'N_Images_Total':     len(all_paths),
            'N_Images_LPIPS':     len(paths_for_lpips),
            'LPIPS_IntraClass':   lp,
            'Feature_Dispersion': fd,
        })

    torch.cuda.empty_cache()
    gc.collect()

df = pd.DataFrame(rows)
df.to_csv(run_dir / 'diversity_metrics.csv', index=False)

# ---------- summary table ---------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY (mean across classes)")
print(f"{'Dataset':<22} {'LPIPS_intra':>12} {'FeatDisp':>10}")
print('-' * 46)
for ds in DATASETS:
    sub = df[df['Dataset'] == ds]
    if sub.empty:
        continue
    ml = sub['LPIPS_IntraClass'  ][sub['LPIPS_IntraClass']   >= 0].mean()
    md = sub['Feature_Dispersion'][sub['Feature_Dispersion'] >= 0].mean()
    print(f"  {ds:<20} {ml:>12.4f} {md:>10.4f}")
print(f"{'='*60}")

# ---------- bar plot --------------------------------------------------------
try:
    present_ds = [d for d in DATASETS if d in df['Dataset'].values]
    if len(present_ds) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

        for ax_idx, (metric, title) in enumerate([
            ('LPIPS_IntraClass',   'Intra-class LPIPS Diversity'),
            ('Feature_Dispersion', 'Feature Dispersion (std)'),
        ]):
            means, stds, labels = [], [], []
            for ds in present_ds:
                sub = df[(df['Dataset'] == ds) & (df[metric] >= 0)]
                means.append(sub[metric].mean())
                stds.append(sub[metric].std())
                labels.append(ds)

            x = range(len(labels))
            axes[ax_idx].bar(x, means, yerr=stds, capsize=4,
                             color=colors[:len(labels)], edgecolor='black')
            axes[ax_idx].set_xticks(list(x))
            axes[ax_idx].set_xticklabels(labels, rotation=20, ha='right')
            axes[ax_idx].set_title(title, fontweight='bold')
            axes[ax_idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Dataset Diversity Comparison', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(run_dir / 'diversity_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {run_dir / 'diversity_comparison.png'}")
except Exception as e:
    print(f"Plot skipped: {e}")

print(f"Saved: {run_dir / 'diversity_metrics.csv'}")
print("Diversity computation complete.")

