# Troubleshooting — Common Errors & Solutions

## 1. "No GPU Found" — All scripts exit immediately

### Symptom
```
No GPU Found
```

### Cause
- No NVIDIA GPU present
- CUDA driver not installed
- PyTorch installed without CUDA support

### Solution

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. CUDA Out of Memory

### Symptom
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

### Solution
- **SD script (`02_2_gen_sd.py`):** Requires ≥6 GB VRAM. The script already uses `enable_model_cpu_offload()`.
- **Training scripts:** Reduce `BATCH_SIZE` in the code (default = 8, try reducing to 4 or 2).
- Close other applications using the GPU.

---

## 3. ModuleNotFoundError: No module named 'src'

### Symptom
```
ModuleNotFoundError: No module named 'src.models.efficientnet_b0'
```

### Cause
The script cannot find the `src/` package from `leaf-disease-ai`.

### Solution
- **Run from the `leaf-disease-ai/` directory** (parent project):
  ```bash
  cd leaf-disease-ai
  python tomato_vs/03_run_experiments.py --output_dir tomato_vs/Results/test
  ```
- **Do NOT run from `tomato_vs/`:**
  ```bash
  # WRONG:
  cd tomato_vs
  python 03_run_experiments.py

  # CORRECT:
  cd leaf-disease-ai
  python tomato_vs/03_run_experiments.py
  ```

> Note: The scripts already include `sys.path.insert(0, str(project_root))`, so if the directory structure is correct, this error should not occur.

---

## 4. Gemini API Error (02_2_gen_sd.py)

### Symptom
```
Error: GEMINI_API_KEY not found in .env file
```
or
```
Error: google-genai package not installed
```

### Solution

1. Create the file `tomato_vs/.env`:
   ```
   GEMINI_API_KEY=AIzaSy...your_key
   ```

2. Install the package:
   ```bash
   pip install google-genai python-dotenv
   ```

3. Verify that the API key is valid at [Google AI Studio](https://aistudio.google.com/apikey)

---

## 5. "Error: Data_OG does not exist"

### Cause
The PlantVillage dataset has not been downloaded and placed in the correct location.

### Solution
See the instructions at [INSTALLATION.md](INSTALLATION.md#4-preparing-the-plantvillage-dataset).

---

## 6. "Error: datasets/baseline/train does not exist"

### Cause
Step 1 (`01_data_setup.py`) has not been run yet.

### Solution
```bash
python tomato_vs/01_data_setup.py --train_count 20 --test_count 100
```

---

## 7. "datasets/gan_x5/train does not exist"

### Cause
The GAN generation step (`02_3_gen_gan.py`) has not been run yet.

### Solution
```bash
python tomato_vs/02_3_gen_gan.py
```

---

## 8. Results differ between runs

### Cause
Despite setting seeds, results may vary slightly due to:
- Different GPUs (floating-point precision)
- Different cuDNN versions
- Different PyTorch versions

### Solution
- Ensure `torch.backends.cudnn.deterministic = True` (already included in the code)
- Use the same PyTorch and CUDA versions
- Accept minor differences (typically < 1–2%)

---

## 9. Stable Diffusion model download is too slow

### Cause
On the first run, the model `runwayml/stable-diffusion-v1-5` (~4 GB) is downloaded from HuggingFace.

### Solution
- Wait for the download to complete (only happens once; it is cached afterwards)
- If network is slow, download the model manually:
  ```bash
  pip install huggingface-hub
  huggingface-cli download runwayml/stable-diffusion-v1-5
  ```

---

## 10. Script 05_master_run.py crashes mid-run

### Solution
- Results from completed runs are still saved in `Results/`
- Re-running the script will create a new folder (does not overwrite)
- Alternatively, run each step manually (see [Pipeline A](PIPELINE_A.md))
