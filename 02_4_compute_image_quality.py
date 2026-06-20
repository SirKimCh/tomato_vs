"""
02_4_compute_image_quality.py
Compute image quality metrics for SD-generated images vs baseline originals:
  - FID  (Fréchet Inception Distance)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - Label Noise Proxy  (feature-space nearest-neighbour classification)

These metrics are computed for EACH Strength / Guidance combination and saved
to the run directory for inclusion in the paper.

Optional packages:
  pip install torchmetrics[image] lpips

Usage:
  python tomato_vs/02_4_compute_image_quality.py \\
      --strength 0.35 --guidance 7.5 --run_dir Results/my_run
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# ---------- optional imports ------------------------------------------------
HAS_FID, HAS_LPIPS = False, False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
    print("[OK] torchmetrics available – FID will be computed.")
except ImportError:
    print("[WARN] torchmetrics not installed → FID skipped.")
    print("       Install: pip install torchmetrics[image]")

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
    print("[OK] lpips available – LPIPS will be computed.")
except ImportError:
    print("[WARN] lpips not installed → LPIPS skipped.")
    print("       Install: pip install lpips")

from src.models.efficientnet_b0 import EfficientNetB0Model

# ---------- CLI args --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--strength', type=float, default=0.35,
                    help='SD img2img strength used for this run')
parser.add_argument('--guidance', type=float, default=7.5,
                    help='SD guidance scale used for this run')
parser.add_argument('--run_dir', type=str, required=True,
                    help='Output directory for metric CSVs')
args = parser.parse_args()

# ---------- paths -----------------------------------------------------------
device        = torch.device('cuda')
base_dir      = Path(__file__).parent.resolve()
datasets_dir  = base_dir / 'datasets'
baseline_dir  = datasets_dir / 'baseline' / 'train'
sd_dir        = datasets_dir / 'sd_x5'    / 'train'
run_dir       = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)

for d, label in [(baseline_dir, 'baseline/train'), (sd_dir, 'sd_x5/train')]:
    if not d.exists():
        print(f"Error: {d} does not exist.  Run prior steps first.")
        sys.exit(1)

VALID_EXT = {'.jpg', '.jpeg', '.png'}
classes      = sorted([d.name for d in baseline_dir.iterdir() if d.is_dir()])
num_classes  = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

print(f"\n{'='*60}")
print(f"IMAGE QUALITY METRICS  (Strength={args.strength}, Guidance={args.guidance})")
print(f"Classes : {classes}")
print(f"{'='*60}")

# ---------- transforms ------------------------------------------------------
tf_fid   = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()           # [0,1] – FID with normalize=True
])
tf_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()           # converted to [-1,1] inside loop
])
tf_feat  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_batch(paths, tf, max_n=None):
    """Load images → stacked tensor.  Returns None if nothing loaded."""
    if max_n:
        paths = paths[:max_n]
    imgs = []
    for p in paths:
        try:
            imgs.append(tf(Image.open(p).convert('RGB')))
        except Exception:
            pass
    return torch.stack(imgs) if imgs else None

# ---------- feature extractor (EfficientNet-B0 avgpool output) --------------
feat_model = EfficientNetB0Model(num_classes=num_classes, pretrained=True)
feat_model.to(device).eval()

@torch.no_grad()
def extract_features(paths, batch_size=16):
    all_f = []
    for i in range(0, len(paths), batch_size):
        b = load_batch(paths[i:i+batch_size], tf_feat)
        if b is None:
            continue
        b = b.to(device)
        x = feat_model.model.features(b)
        x = feat_model.model.avgpool(x).flatten(1)
        all_f.append(x.cpu().numpy())
    return np.vstack(all_f) if all_f else np.zeros((0, 1280), dtype=np.float32)

# ---------- collect file paths ----------------------------------------------
real_by_cls  = {}
gen_by_cls   = {}
pairs_by_cls = {}          # list of (gen_path, source_original_path)

for cls in classes:
    r = sorted(str(f) for f in (baseline_dir / cls).iterdir()
               if f.is_file() and f.suffix.lower() in VALID_EXT)
    g = sorted(str(f) for f in (sd_dir / cls).iterdir()
               if f.is_file() and '_sd' in f.stem
               and f.suffix.lower() in VALID_EXT)
    real_by_cls[cls] = r
    gen_by_cls[cls]  = g

    stem_map = {Path(p).stem: p for p in r}
    pairs_by_cls[cls] = [
        (gf, stem_map[Path(gf).stem.split('_sd')[0]])
        for gf in g
        if Path(gf).stem.split('_sd')[0] in stem_map
    ]

# ---------- FID (global, all classes) ----------------------------------------
fid_value = -1.0
if HAS_FID:
    print("\n[FID] Computing …")
    try:
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        all_real = [f for cls in classes for f in real_by_cls[cls]]
        all_gen  = [f for cls in classes for f in gen_by_cls[cls]]

        BS = 8
        for i in range(0, len(all_real), BS):
            b = load_batch(all_real[i:i+BS], tf_fid)
            if b is not None:
                fid_metric.update(b.to(device), real=True)
        for i in range(0, len(all_gen), BS):
            b = load_batch(all_gen[i:i+BS], tf_fid)
            if b is not None:
                fid_metric.update(b.to(device), real=False)

        fid_value = float(fid_metric.compute().item())
        del fid_metric
        torch.cuda.empty_cache(); gc.collect()
        print(f"  FID = {fid_value:.4f}")
    except Exception as e:
        print(f"  FID failed: {e}")
        fid_value = -1.0
else:
    print("\n[FID] Skipped.")

# ---------- LPIPS initialisation --------------------------------------------
lpips_fn = None
if HAS_LPIPS:
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

# ---------- per-class LPIPS + label-noise proxy ----------------------------
class_rows = []

# Pre-extract real features for label-noise NN classifier
all_real_paths  = [f for cls in classes for f in real_by_cls[cls]]
all_real_labels = [class_to_idx[cls] for cls in classes for _ in real_by_cls[cls]]
print("\n[Features] Extracting reference features …")
real_feats_all = extract_features(all_real_paths)
real_labels_arr = np.array(all_real_labels[:real_feats_all.shape[0]])

# Normalise once for cosine similarity
real_norm = real_feats_all / (np.linalg.norm(real_feats_all, axis=1, keepdims=True) + 1e-8)

print("\n[Per-class LPIPS + Label Noise]")
for cls in classes:
    pairs = pairs_by_cls[cls]

    # LPIPS: generated vs its source original (capped at 40 pairs for speed)
    lpips_scores = []
    if HAS_LPIPS and lpips_fn is not None:
        for gf, sf in pairs[:40]:
            try:
                g_t = load_batch([gf], tf_lpips)
                s_t = load_batch([sf], tf_lpips)
                if g_t is not None and s_t is not None:
                    g_t = (g_t * 2 - 1).to(device)
                    s_t = (s_t * 2 - 1).to(device)
                    with torch.no_grad():
                        d = lpips_fn(g_t, s_t)
                    lpips_scores.append(float(d.item()))
            except Exception:
                pass
    mean_lpips = float(np.mean(lpips_scores)) if lpips_scores else -1.0

    # Label-noise proxy: NN in feature space (all real images as reference)
    noise_rate = -1.0
    gen_paths_sample = [gf for gf, _ in pairs[:50]]
    if gen_paths_sample and real_feats_all.shape[0] > 0:
        gen_feats = extract_features(gen_paths_sample)
        if gen_feats.shape[0] > 0:
            gen_norm = gen_feats / (np.linalg.norm(gen_feats, axis=1, keepdims=True) + 1e-8)
            sims     = cosine_similarity(gen_norm, real_norm)    # (n_gen, n_real)
            nn_lbl   = real_labels_arr[np.argmax(sims, axis=1)]
            target   = class_to_idx[cls]
            noise_rate = float(np.sum(nn_lbl != target) / len(gen_feats))

    print(f"  {cls:<40}: LPIPS={mean_lpips:+.4f}  LabelNoise={noise_rate:.3f}")
    class_rows.append({
        'Strength':          args.strength,
        'Guidance':          args.guidance,
        'Class':             cls,
        'Mean_LPIPS':        mean_lpips,
        'Label_Noise_Rate':  noise_rate,
        'N_Generated':       len(gen_by_cls[cls]),
    })

# ---------- summary ---------------------------------------------------------
valid_lpips = [r['Mean_LPIPS']       for r in class_rows if r['Mean_LPIPS'] >= 0]
valid_noise = [r['Label_Noise_Rate'] for r in class_rows if r['Label_Noise_Rate'] >= 0]

summary = {
    'Strength':                  args.strength,
    'Guidance':                  args.guidance,
    'FID':                       fid_value,
    'Mean_LPIPS':                float(np.mean(valid_lpips))  if valid_lpips else -1.0,
    'Std_LPIPS':                 float(np.std(valid_lpips))   if valid_lpips else -1.0,
    'Mean_Label_Noise_Rate':     float(np.mean(valid_noise))  if valid_noise else -1.0,
}

tag = f"s{args.strength}_g{args.guidance}"
pd.DataFrame(class_rows).to_csv(run_dir / f'image_quality_{tag}.csv', index=False)
pd.DataFrame([summary]).to_csv(run_dir / f'image_quality_summary_{tag}.csv', index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"  FID                  : {fid_value:.4f}")
print(f"  Mean LPIPS (gen→src) : {summary['Mean_LPIPS']:.4f} ± {summary['Std_LPIPS']:.4f}")
print(f"  Label Noise Rate     : {summary['Mean_Label_Noise_Rate']:.3f}")
print(f"{'='*60}")
print(f"Saved: {run_dir}/image_quality_{tag}.csv")
print(f"Saved: {run_dir}/image_quality_summary_{tag}.csv")
print("Image quality computation complete.")

