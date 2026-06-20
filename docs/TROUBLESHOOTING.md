# Troubleshooting — Common Errors & Solutions

## 1. "No GPU Found" — All scripts exit immediately

**Cause:** No NVIDIA GPU / PyTorch without CUDA support.

```bash
nvidia-smi                                    # check GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Reinstall PyTorch for GTX 3050 / CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. CUDA Out of Memory

**Cause:** GTX 3050 6 GB is tight for Stable Diffusion.

**Solutions:**
- `02_2_gen_sd.py` uses `enable_model_cpu_offload()` automatically — this offloads weights to CPU when not in use. Ensure no other VRAM-heavy apps are running.
- Training scripts: reduce `BATCH_SIZE` in the file from 8 to 4.

---

## 3. ModuleNotFoundError: No module named 'src'

**Cause:** Running script from wrong directory.

```bash
# WRONG:
cd tomato_vs && python 03_run_experiments.py

# CORRECT:
cd leaf-disease-ai
python tomato_vs/07_master_run.py
```

---

## 4. Gemini API Error (02_2_gen_sd.py)

```
Error: GEMINI_API_KEY not found in .env file
```

**Solution:** Create `tomato_vs/.env`:
```
GEMINI_API_KEY=AIzaSy...your_key
```

> `02_2b_gen_sd_labelonly.py` (label-only ablation) does NOT require a Gemini key.

---

## 5. "Data_OG does not exist"

See [INSTALLATION.md](INSTALLATION.md#5-prepare-plantvillage-data).

---

## 6. "datasets/baseline/train does not exist"

```bash
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
```

> **Warning:** This script deletes and recreates `datasets/` entirely.

---

## 7. torchmetrics / lpips not found (FID/LPIPS steps)

```bash
pip install torchmetrics[image] lpips
```

These are optional for the FID/LPIPS analysis steps. If not installed, those steps are skipped with a warning.

---

## 8. scipy not found (statistical analysis step)

```bash
pip install scipy
```

---

## 9. Master run interrupted mid-way

Results from completed combinations are saved in `Results/{timestamp}_s{s}_g{g}/`. To resume from a specific point:

```bash
# Skip Phase 0 if data + TDA + RandAugment already generated
python tomato_vs/07_master_run.py --skip_data_setup --skip_tda --skip_randaug

# For a single combination manually:
python tomato_vs/02_2_gen_sd.py --strength 0.35 --guidance 7.5 --output_log_dir tomato_vs/Results/run1
python tomato_vs/03_run_experiments.py --output_dir tomato_vs/Results/run1 --train_count 20 --use_kfold --extra_baselines
python tomato_vs/03_3_analyze_results.py --input_dir tomato_vs/Results/run1
```

---

## 10. Stable Diffusion model download

First run downloads `runwayml/stable-diffusion-v1-5` (~4 GB) from HuggingFace. Cached after first download.

```bash
# Manual download (optional):
pip install huggingface-hub
huggingface-cli download runwayml/stable-diffusion-v1-5
```

---

## 11. Results vary between runs

Despite fixed seeds, minor variations occur due to GPU floating-point precision. The code sets:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Accept < 0.5% variation as normal.

---

## 12. Run 00_check_requirements.py first

If any package is missing, the check script can auto-install:
```bash
python tomato_vs/00_check_requirements.py --install
```
