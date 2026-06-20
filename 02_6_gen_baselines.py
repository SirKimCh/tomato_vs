"""
02_6_gen_baselines.py
Pre-generate additional baseline augmentation datasets for fair comparison:

  randaugment_x5  — RandAugment applied 4× to each original = 5× dataset
                    (same scale as tda_x5 and sd_x5)

These datasets allow apples-to-apples comparison in Table 3 of the paper.
MixUp / CutMix are applied online during training (see 03_run_experiments.py)
and therefore do not need pre-generated datasets.

Usage:
  python tomato_vs/02_6_gen_baselines.py
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import shutil
from pathlib import Path
from PIL import Image
from torchvision import transforms

torch.manual_seed(42)

base_dir            = Path(__file__).parent.resolve()
baseline_train_dir  = base_dir / 'datasets' / 'baseline' / 'train'
datasets_dir        = base_dir / 'datasets'

if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)

classes = [d.name for d in baseline_train_dir.iterdir() if d.is_dir()]
if not classes:
    print(f"Error: No class directories found in {baseline_train_dir}")
    sys.exit(1)

MULTIPLIER = 4    # 4 augmented + 1 original = 5×

# RandAugment transform  (magnitude=9 is a standard mid-range value)
tf_rand = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def denorm_save(tensor_img, path):
    """Denormalise ImageNet-normalised tensor and save as JPEG."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = torch.clamp(tensor_img * std + mean, 0.0, 1.0)
    transforms.ToPILImage()(img).save(str(path))


# ─────────────────────────── randaugment_x5 ──────────────────────────────────
print(f"Generating randaugment_x5  (5× = original + {MULTIPLIER}× RandAugment)...")

out_dir = datasets_dir / 'randaugment_x5' / 'train'
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for cls in classes:
    cls_out = out_dir / cls
    cls_out.mkdir(parents=True, exist_ok=True)
    cls_in  = baseline_train_dir / cls

    imgs = sorted(f for f in os.listdir(cls_in)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    # Copy originals unchanged
    for fn in imgs:
        shutil.copy(str(cls_in / fn), str(cls_out / fn))

    # Generate RandAugment variants
    aug_count = 0
    for fn in imgs:
        stem, ext = os.path.splitext(fn)
        for k in range(MULTIPLIER):
            pil  = Image.open(cls_in / fn).convert('RGB')
            aug  = tf_rand(pil)
            denorm_save(aug, cls_out / f"{stem}_ra{k}{ext}")
            aug_count += 1

    total = len(imgs) + aug_count
    print(f"  {cls}: {len(imgs)} original + {aug_count} RandAugment = {total} total")

print("\nrandaugment_x5 generation complete.")

