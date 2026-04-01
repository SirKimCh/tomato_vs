# Detailed Installation Guide

## 1. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with CUDA | NVIDIA RTX 3060 or higher |
| VRAM | 4 GB | 8 GB+ (for Stable Diffusion) |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB free | 20 GB+ |
| OS | Windows 10/11 or Ubuntu 20.04+ | — |

> **Note:** All scripts check for GPU at the beginning of the file. If no CUDA GPU is found, the script will exit immediately with the message `"No GPU Found"`.

---

## 2. Python & Dependencies Installation

### 2.1. Clone the parent project

The `tomato_vs/` directory depends on source code from the `leaf-disease-ai` project. Specifically, the following files are imported:

- `src/models/efficientnet_b0.py` → Class `EfficientNetB0Model`
- `src/configurations/augmentation_config.py` → Class `AugmentationConfig`

```bash
# Clone the parent project
git clone https://github.com/junayed-hasan/leaf-disease-ai.git
cd leaf-disease-ai

# Clone tomato_vs inside it
git clone https://github.com/SirKimCh/tomato_vs.git tomato_vs
```

Final directory structure:

```
leaf-disease-ai/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── efficientnet_b0.py
│   └── configurations/
│       └── augmentation_config.py
├── requirements.txt
└── tomato_vs/
    ├── 01_data_setup.py
    └── ...
```

### 2.2. Create a Virtual Environment

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### 2.3. Install PyTorch (matching your CUDA version)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the appropriate install command. For example:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.4. Install other dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `pillow` | Image processing |
| `numpy` | Numerical computing |
| `pandas` | Tabular data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Metrics (accuracy, F1, confusion matrix, etc.) |
| `tqdm` | Progress bars |
| `diffusers` | Stable Diffusion pipeline |
| `python-dotenv` | Read `.env` files |

### 2.5. Install additional packages for Stable Diffusion & Gemini

```bash
pip install google-genai
pip install accelerate transformers safetensors
```

---

## 3. Gemini API Key Configuration

Script `02_2_gen_sd.py` uses the **Gemini API** to automatically generate disease description prompts for Stable Diffusion.

### Step 1: Get an API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with a Google account
3. Create a new API key

### Step 2: Create the `.env` file

Create the file `tomato_vs/.env`:

```
GEMINI_API_KEY=AIzaSy...your_key_here
```

> **Security:** Do not commit the `.env` file to GitHub. Add `.env` to `.gitignore`.

> **If you do not have a Gemini API key:** The script has fallback prompts hardcoded in the source code. However, the current version requires a valid key. You can modify the code to bypass this requirement.

---

## 4. Preparing the PlantVillage Dataset

### Step 1: Download the dataset

Download from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Step 2: Extract the 5 tomato classes

From the downloaded dataset, copy the following **5 directories** into `tomato_vs/Data_OG/`:

```
tomato_vs/Data_OG/
├── Tomato___Early_blight/
├── Tomato___healthy/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
└── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
```

### Step 3: Verify

```bash
# Check the number of images in each class
python -c "
import os
data_dir = 'tomato_vs/Data_OG'
for d in sorted(os.listdir(data_dir)):
    full = os.path.join(data_dir, d)
    if os.path.isdir(full):
        count = len([f for f in os.listdir(full) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f'{d}: {count} images')
"
```

Expected output:

```
Tomato___Early_blight: 1000 images
Tomato___healthy: 1591 images
Tomato___Late_blight: 1909 images
Tomato___Leaf_Mold: 952 images
Tomato___Tomato_Yellow_Leaf_Curl_Virus: 5357 images
```

---

## 5. Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import torchvision, numpy, pandas, matplotlib, seaborn, sklearn, tqdm, PIL
print('All packages imported successfully!')
"
```

---

## 6. Common Installation Issues

See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
