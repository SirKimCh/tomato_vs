"""
00_check_requirements.py
========================
Environment verification & auto-installer for the Tomato Leaf Disease pipeline.

Designed for:
  • NVIDIA GeForce RTX/GTX 3050  6 GB VRAM
  • CUDA 12.x  (tested with 12.1 / 12.2)
  • Python 3.8 – 3.11

Run THIS SCRIPT FIRST, before any experiment:
  python tomato_vs/00_check_requirements.py
  python tomato_vs/00_check_requirements.py --install   # auto-install missing pkgs

What it checks:
  ✔  Python version (≥3.8)
  ✔  NVIDIA GPU & CUDA availability
  ✔  GPU VRAM (warn if <6 GB)
  ✔  PyTorch version & CUDA backend
  ✔  All required Python packages
  ✔  .env file (Gemini API key)
  ✔  Data_OG directory (PlantVillage classes)
  ✔  Quick EfficientNet-B0 forward-pass smoke test
"""

import sys
import subprocess
import importlib
import argparse
from pathlib import Path

# ─── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Check and optionally install all pipeline dependencies')
parser.add_argument('--install', action='store_true',
                    help='Auto-install / upgrade missing packages')
parser.add_argument('--skip_smoke', action='store_true',
                    help='Skip the EfficientNet smoke test (faster)')
args = parser.parse_args()

BASE_DIR = Path(__file__).parent.resolve()
PASS = "[OK]"
FAIL = "[!!]"
WARN = "[--]"

issues = []   # will hold strings for summary

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Python version
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. Python version")
print("="*60)
v = sys.version_info
ver_str = f"{v.major}.{v.minor}.{v.micro}"
if v.major < 3 or (v.major == 3 and v.minor < 8):
    print(f"{FAIL} Python {ver_str}  —  need ≥ 3.8")
    issues.append("Python < 3.8")
else:
    print(f"{PASS} Python {ver_str}")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  NVIDIA GPU / CUDA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. GPU & CUDA")
print("="*60)

def nvidia_smi():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None

smi = nvidia_smi()
if smi:
    print(f"{PASS} nvidia-smi: {smi}")
else:
    print(f"{FAIL} nvidia-smi not found or no NVIDIA GPU detected")
    issues.append("No NVIDIA GPU / nvidia-smi missing")

# Check torch CUDA support
try:
    import torch
    if torch.cuda.is_available():
        dev       = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver  = torch.version.cuda
        torch_ver = torch.__version__
        print(f"{PASS} PyTorch {torch_ver}  —  CUDA {cuda_ver}")
        print(f"{PASS} GPU  : {dev}")
        print(f"{PASS} VRAM : {vram_gb:.1f} GB")
        if vram_gb < 5.5:
            print(f"{WARN} VRAM < 6 GB — Stable Diffusion may OOM during generation")
            issues.append(f"Low VRAM ({vram_gb:.1f} GB) — SD may be slow/fail")
        # Check CUDA version matches user's hardware (GTX 3050 → CUDA 12.x)
        if cuda_ver and not cuda_ver.startswith('12'):
            print(f"{WARN} CUDA {cuda_ver} detected. GTX 3050 works best with CUDA 12.x")
            print(f"      Recommended: pip install torch torchvision "
                  f"--index-url https://download.pytorch.org/whl/cu121")
            issues.append(f"PyTorch compiled with CUDA {cuda_ver} (recommend cu121)")
    else:
        print(f"{FAIL} PyTorch installed but CUDA not available — check drivers")
        issues.append("CUDA not available in PyTorch")
except ImportError:
    print(f"{FAIL} PyTorch not installed")
    issues.append("PyTorch missing")
    if args.install:
        print("  → Installing PyTorch with CUDA 12.1 (for GTX 3050) …")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ])

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Required Python packages
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. Python packages")
print("="*60)

# (name to import, pip install name, minimum version string or None)
REQUIRED = [
    ('torch',           'torch',                    '2.0'),
    ('torchvision',     'torchvision',              '0.15'),
    ('diffusers',       'diffusers',                '0.20'),
    ('transformers',    'transformers',             '4.30'),
    ('accelerate',      'accelerate',               '0.20'),
    ('PIL',             'pillow',                   '9.0'),
    ('numpy',           'numpy',                    '1.21'),
    ('pandas',          'pandas',                   '1.5'),
    ('matplotlib',      'matplotlib',               '3.6'),
    ('seaborn',         'seaborn',                  '0.12'),
    ('sklearn',         'scikit-learn',             '1.2'),
    ('scipy',           'scipy',                    '1.9'),
    ('tqdm',            'tqdm',                     '4.60'),
    ('dotenv',          'python-dotenv',            None),
    ('torchmetrics',    'torchmetrics[image]',      '1.0'),
    ('lpips',           'lpips',                    '0.1'),
    ('google.genai',    'google-genai',             None),
    ('safetensors',     'safetensors',              None),
]

missing = []
for import_name, pip_name, min_ver in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, '__version__', '?')
        # Minimal version check
        if min_ver and ver != '?':
            try:
                parts_got  = [int(x) for x in ver.split('.')[:2]]
                parts_need = [int(x) for x in min_ver.split('.')[:2]]
                if parts_got < parts_need:
                    print(f"{WARN} {import_name:<20} {ver}  (need ≥ {min_ver})")
                    missing.append((import_name, pip_name, 'upgrade'))
                    continue
            except Exception:
                pass
        print(f"{PASS} {import_name:<20} {ver}")
    except ImportError:
        print(f"{FAIL} {import_name:<20} NOT installed  (pip: {pip_name})")
        missing.append((import_name, pip_name, 'install'))

if missing and args.install:
    print(f"\nInstalling {len(missing)} missing/outdated package(s) …")
    for import_name, pip_name, action in missing:
        print(f"  {action}: {pip_name}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', pip_name])
elif missing:
    print(f"\n{WARN} {len(missing)} package(s) missing/outdated.")
    print("  Run with --install to auto-install, or:")
    for _, pip_name, _ in missing:
        print(f"    pip install {pip_name}")
    # Check if critical packages are missing
    critical = {'torch', 'torchvision', 'PIL', 'numpy', 'sklearn', 'scipy'}
    if any(imp in critical for imp, _, _ in missing):
        issues.append(f"Missing critical packages: {[p for i,p,_ in missing if i in critical]}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  .env file (Gemini API key)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. .env / Gemini API key")
print("="*60)
env_path = BASE_DIR / '.env'
if env_path.exists():
    content = env_path.read_text()
    if 'GEMINI_API_KEY' in content and 'your_key' not in content:
        print(f"{PASS} .env found with GEMINI_API_KEY")
    else:
        print(f"{WARN} .env found but GEMINI_API_KEY may not be set correctly")
        print(f"      Edit {env_path} and set GEMINI_API_KEY=<your_key>")
        issues.append(".env GEMINI_API_KEY not configured (required for SD generation)")
else:
    print(f"{FAIL} .env not found at {env_path}")
    print(f"      Create {env_path} with:  GEMINI_API_KEY=<your_key>")
    print(f"      Get key at: https://aistudio.google.com/apikey")
    issues.append(".env missing — SD generation (02_2_gen_sd.py) will fail")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Data_OG directory
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. Data_OG (PlantVillage source data)")
print("="*60)
data_og = BASE_DIR / 'Data_OG'
EXPECTED_CLASSES = [
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

if data_og.exists():
    found_classes = [d.name for d in data_og.iterdir() if d.is_dir()]
    for cls in EXPECTED_CLASSES:
        cls_dir = data_og / cls
        if cls_dir.exists():
            n = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.JPG'))
                    + list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpeg')))
            if n < 120:   # need at least 20 train + 100 test
                print(f"{FAIL} {cls}: {n} images  (need ≥ 120)")
                issues.append(f"Too few images in {cls} ({n})")
            else:
                print(f"{PASS} {cls}: {n} images")
        else:
            print(f"{FAIL} {cls}: directory not found")
            issues.append(f"Missing class: {cls}")
else:
    print(f"{FAIL} Data_OG/ not found at {data_og}")
    print(f"      Download PlantVillage from Kaggle and place 5 tomato classes in Data_OG/")
    issues.append("Data_OG/ missing — cannot run any experiments")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  EfficientNet-B0 smoke test
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. EfficientNet-B0 smoke test")
print("="*60)
if args.skip_smoke:
    print(f"{WARN} Skipped (--skip_smoke)")
else:
    try:
        project_root = BASE_DIR.parent
        sys.path.insert(0, str(project_root))
        from src.models.efficientnet_b0 import EfficientNetB0Model
        import torch as _torch

        device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        model  = EfficientNetB0Model(num_classes=5, pretrained=True).to(device)
        dummy  = _torch.randn(2, 3, 224, 224, device=device)
        with _torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 5), f"unexpected output shape {out.shape}"
        print(f"{PASS} EfficientNet-B0 forward pass OK  (device={device})")
        del model, dummy, out
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception as e:
        print(f"{FAIL} EfficientNet-B0 smoke test failed: {e}")
        issues.append(f"EfficientNet smoke test failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if not issues:
    print(f"{PASS} All checks passed!  Environment is ready.")
    print("\nNext step:")
    print("  python tomato_vs/07_master_run.py")
else:
    print(f"{WARN} {len(issues)} issue(s) found:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nFix the issues above, then run again.")
    if not args.install and any('package' in i.lower() or 'missing' in i.lower()
                                 for i in issues if 'Data_OG' not in i):
        print("\nTip: run with --install to auto-install missing packages:")
        print("  python tomato_vs/00_check_requirements.py --install")

print("\nNote on CUDA / PyTorch for GTX 3050 (6 GB, CUDA 12.x):")
print("  pip install torch torchvision torchaudio "
      "--index-url https://download.pytorch.org/whl/cu121")
print()

