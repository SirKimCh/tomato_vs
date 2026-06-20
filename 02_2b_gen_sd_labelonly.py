"""
02_2b_gen_sd_labelonly.py
Generate SD img2img images using SIMPLE LABEL-ONLY PROMPTS (no Gemini LLM).
Used for ABLATION STUDY comparing:

  sd_x5          → Gemini-LLM-generated expert prompts  (02_2_gen_sd.py)
  sd_labelonly_x5 → minimal template prompts  (this script)

Prompt template:  "tomato leaf {class_name_cleaned}, disease symptoms, macro photography"
No external API needed.

Usage:
  python tomato_vs/02_2b_gen_sd_labelonly.py --strength 0.35 --guidance 7.5
"""

import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import gc
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--strength', type=float, default=0.35)
parser.add_argument('--guidance', type=float, default=7.5)
args = parser.parse_args()

base_dir           = Path(__file__).parent.resolve()
baseline_train_dir = base_dir / 'datasets' / 'baseline' / 'train'
datasets_dir       = base_dir / 'datasets'

if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist.  Run 01_data_setup.py first.")
    sys.exit(1)

classes = [d.name for d in baseline_train_dir.iterdir() if d.is_dir()]
if not classes:
    print(f"Error: No class directories found in {baseline_train_dir}")
    sys.exit(1)

# ── Label-only prompt (NO Gemini, just class name cleaned up) ──────────────
def label_prompt(class_name: str) -> str:
    clean = class_name.replace('Tomato___', '').replace('_', ' ').strip().lower()
    return f"tomato leaf {clean}, disease symptoms, macro photography"

label_prompts = {cls: label_prompt(cls) for cls in classes}
print("Label-only prompts (no LLM):")
for cls, p in label_prompts.items():
    print(f"  {cls}: {p}")

NEGATIVE = (
    "cartoon, drawing, painting, illustration, blur, low quality, distorted, "
    "watermark, text, artificial, tomato fruit, red fruit, stem, stalk, branch"
)

MULTIPLIER   = 4
DATASET_NAME = 'sd_labelonly_x5'

print(f"\nLoading Stable Diffusion model …")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
print("Model loaded.")

out_dir = datasets_dir / DATASET_NAME / 'train'
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating {DATASET_NAME}  (strength={args.strength}, guidance={args.guidance}) …")

for cls in classes:
    cls_out = out_dir / cls
    cls_out.mkdir(parents=True, exist_ok=True)
    cls_in  = baseline_train_dir / cls

    imgs = sorted(f for f in os.listdir(cls_in)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    # copy originals
    for fn in imgs:
        shutil.copy(str(cls_in / fn), str(cls_out / fn))

    prompt = label_prompts[cls]
    print(f"\n  {cls}")
    print(f"    Prompt: {prompt}")

    gen_count = 0
    for fn in tqdm(imgs, desc="  Generating", leave=False):
        init = Image.open(cls_in / fn).convert('RGB').resize((512, 512))
        stem, ext = os.path.splitext(fn)
        for k in range(MULTIPLIER):
            with torch.no_grad():
                out_img = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    image=init,
                    strength=args.strength,
                    guidance_scale=args.guidance,
                    num_inference_steps=50,
                ).images[0]
            out_img.save(cls_out / f"{stem}_sdlo{k}{ext}")
            gen_count += 1
            if gen_count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print(f"    Result: {len(imgs)} original + {gen_count} generated")
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{DATASET_NAME} generation complete.")

