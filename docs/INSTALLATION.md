# Detailed Installation Guide (Revised for CUDA 12.x / RTX 3050 Ti / RTX 5060 Ti)

## 1. Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA CUDA | **RTX 3050 Ti 4 GB** / **RTX 5060 Ti 16 GB** |
| CUDA | 12.1+ (Ampere) / 12.8+ (Blackwell) | **12.1 / 12.8** |
| RAM | 8 GB | 16 GB |
| Disk | 15 GB | 30 GB |
| OS | Windows 10/11 or Ubuntu 20.04+ | Windows 11 |
| Python | 3.8+ | **3.13** |

> All scripts check `torch.cuda.is_available()` at startup and exit with `"No GPU Found"` if no CUDA device is detected.

---

## 2. Automated Check (recommended)

```bash
cd leaf-disease-ai
python tomato_vs/00_check_requirements.py --install
```

This script verifies all dependencies and auto-installs missing packages.

---

## 3. Manual Installation

### 3.1. Clone the parent project

```bash
git clone https://github.com/junayed-hasan/leaf-disease-ai.git
cd leaf-disease-ai
git clone https://github.com/SirKimCh/tomato_vs.git tomato_vs
```

Directory structure:
```
leaf-disease-ai/
├── src/
│   ├── models/efficientnet_b0.py
│   └── configurations/augmentation_config.py
└── tomato_vs/
    ├── 00_check_requirements.py
    └── ...
```

### 3.2. Create virtual environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3.3. Install PyTorch (choose per GPU)

```bash
# RTX 3050 Ti (Ampere, CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RTX 5060 Ti (Blackwell, CUDA 12.8 — requires PyTorch >= 2.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Run `nvidia-smi` to check your driver version.  
> Blackwell GPUs (RTX 50xx) **require** CUDA 12.8 and PyTorch ≥ 2.6.

### 3.4. Install all other dependencies

```bash
pip install -r requirements.txt
pip install google-genai accelerate transformers safetensors
```

This installs:

| Package | Purpose |
|---------|---------|
| `diffusers` | Stable Diffusion pipeline |
| `torchmetrics[image]` | FID computation |
| `lpips` | LPIPS perceptual similarity |
| `scikit-learn` | Metrics + RepeatedStratifiedKFold + classification_report |
| `scipy` | Wilcoxon, Shapiro-Wilk, Friedman tests [R9] |
| `pandas`, `numpy` | Data handling |
| `matplotlib`, `seaborn` | Visualization |
| `google-genai` | Gemini API for prompt generation |
| `python-dotenv` | Read `.env` API key file |

---

## 4. Gemini API Key

```bash
# Create tomato_vs/.env
echo "GEMINI_API_KEY=your_key_here" > tomato_vs/.env
```

Get a free key at: https://aistudio.google.com/apikey

> Without a Gemini key, `02_2_gen_sd.py` will fail.  
> `02_2b_gen_sd_labelonly.py` (ablation) does NOT require a Gemini key.

---

## 5. Prepare PlantVillage Data

Download from: https://www.kaggle.com/datasets/emmarex/plantdisease

Copy these 5 directories to `tomato_vs/Data_OG/`:
```
Tomato___Early_blight/
Tomato___healthy/
Tomato___Late_blight/
Tomato___Leaf_Mold/
Tomato___Tomato_Yellow_Leaf_Curl_Virus/
```

Verify (minimum 120 images per class for train=20 + test=100):
```bash
python -c "
from pathlib import Path
d = Path('tomato_vs/Data_OG')
for c in sorted(d.iterdir()):
    if c.is_dir():
        n = len(list(c.glob('*.jpg')) + list(c.glob('*.JPG')) + list(c.glob('*.png')))
        status = 'OK' if n >= 120 else 'TOO FEW'
        print(f'{status}: {c.name}: {n} images')
"
```

---

## 6. Verify Installation

```bash
python tomato_vs/00_check_requirements.py
```

Expected output (no `[!!]` errors):
```
[OK] Python 3.13.x
[OK] nvidia-smi: NVIDIA GeForce RTX 3050 Ti, 4096 MiB, ...   (or RTX 5060 Ti, 16376 MiB)
[OK] PyTorch 2.x.x+cu121  —  CUDA 12.1
[OK] GPU  : NVIDIA GeForce RTX 3050 Ti
[OK] VRAM : 4.0 GB
[OK] torch                 2.x.x
[OK] torchvision           0.x.x
...
[OK] All checks passed!  Environment is ready.
```

---

## 7. Common Issues

See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
