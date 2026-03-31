import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product

print("="*50)
print("TOMATO VS - MASTER RUN")
print("="*50)

print("\nSelect mode:")
print("  1. Random (Run 1 random combination)")
print("  2. Full (Run all 9 combinations)")
print()

while True:
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == '1':
        mode = 'random'
        break
    elif choice == '2':
        mode = 'full'
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")

while True:
    try:
        train_count = int(input("Enter train count per class (default=10): ").strip() or "10")
        if train_count > 0:
            break
        print("Train count must be positive.")
    except ValueError:
        print("Invalid number. Please enter a valid integer.")

while True:
    try:
        test_count = int(input("Enter test count per class (default=50): ").strip() or "50")
        if test_count > 0:
            break
        print("Test count must be positive.")
    except ValueError:
        print("Invalid number. Please enter a valid integer.")

print(f"\nConfiguration:")
print(f"  Mode: {mode}")
print(f"  Train Count: {train_count}")
print(f"  Test Count: {test_count}")

STRENGTHS = [0.35, 0.5, 0.65]
GUIDANCES = [6.0, 7.5, 9.0]

combinations = list(product(STRENGTHS, GUIDANCES))

base_dir = Path(__file__).parent.resolve()
results_dir = base_dir / 'Results'
datasets_dir = base_dir / 'datasets'
results_dir.mkdir(parents=True, exist_ok=True)

python_exe = sys.executable

def cleanup_datasets():
    if datasets_dir.exists():
        shutil.rmtree(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

def run_setup(strength, guidance, train_count, test_count):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = results_dir / f'{timestamp}_train{train_count}_s{strength}_g{guidance}'
    run_dir.mkdir(parents=True, exist_ok=True)

    config_file = run_dir / 'experiment_config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*40}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Train Count: {train_count}\n")
        f.write(f"Test Count: {test_count}\n")
        f.write(f"SD Strength: {strength}\n")
        f.write(f"SD Guidance: {guidance}\n")

    print(f"\n{'='*70}")
    print(f"SETUP: Train={train_count}, Strength={strength}, Guidance={guidance}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}")

    cleanup_datasets()

    print("\n[Step 1/7] Running 01_data_setup.py...")
    result = subprocess.run(
        [python_exe, str(base_dir / '01_data_setup.py'),
         '--train_count', str(train_count),
         '--test_count', str(test_count)],
        cwd=str(base_dir.parent),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in 01_data_setup.py: {result.stderr}")
        return False
    print(result.stdout)

    print("\n[Step 2/7] Running 02_1_gen_tda.py...")
    result = subprocess.run(
        [python_exe, str(base_dir / '02_1_gen_tda.py')],
        cwd=str(base_dir.parent),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in 02_1_gen_tda.py: {result.stderr}")
        return False
    print(result.stdout)

    print("\n[Step 3/7] Running 02_2_gen_sd.py...")
    result = subprocess.run(
        [python_exe, str(base_dir / '02_2_gen_sd.py'),
         '--strength', str(strength),
         '--guidance', str(guidance),
         '--output_log_dir', str(run_dir)],
        cwd=str(base_dir.parent)
    )
    if result.returncode != 0:
        print(f"Error in 02_2_gen_sd.py")
        return False

    print("\n[Step 4/7] Running 03_run_experiments.py...")
    result = subprocess.run(
        [python_exe, str(base_dir / '03_run_experiments.py'),
         '--output_dir', str(run_dir),
         '--train_count', str(train_count)],
        cwd=str(base_dir.parent)
    )
    if result.returncode != 0:
        print(f"Error in 03_run_experiments.py")
        return False

    print("\n[Step 5/7] Backing up generated images...")
    backup_dir = run_dir / 'generated_images_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    sd_source = datasets_dir / 'sd_x5'
    tda_source = datasets_dir / 'tda_x5'
    baseline_source = datasets_dir / 'baseline'

    if sd_source.exists():
        shutil.copytree(str(sd_source), str(backup_dir / 'sd_x5'))
        print(f"  SD images backed up")

    if tda_source.exists():
        shutil.copytree(str(tda_source), str(backup_dir / 'tda_x5'))
        print(f"  TDA images backed up")

    if baseline_source.exists():
        shutil.copytree(str(baseline_source), str(backup_dir / 'baseline'))
        print(f"  Baseline images backed up")

    print("\n[Step 6/7] Running 04_visualize_results.py...")
    result = subprocess.run(
        [python_exe, str(base_dir / '04_visualize_results.py'),
         '--input_dir', str(run_dir)],
        cwd=str(base_dir.parent),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Warning in 04_visualize_results.py: {result.stderr}")
    else:
        print(result.stdout)

    print("\n[Step 7/7] Cleanup datasets...")
    cleanup_datasets()

    print(f"\nSetup complete: {run_dir}")
    return True

if __name__ == '__main__':
    if mode == 'random':
        s, g = random.choice(combinations)
        print(f"\nRandom mode selected: Strength={s}, Guidance={g}")
        try:
            run_setup(s, g, train_count, test_count)
        except Exception as e:
            print(f"Error: {e}")
            cleanup_datasets()

    elif mode == 'full':
        total_setups = len(combinations)
        print(f"\nFull mode: Running {total_setups} combinations SEQUENTIALLY")
        print("="*70)

        completed = 0
        failed = 0

        for idx, (s, g) in enumerate(combinations, 1):
            print(f"\n{'#'*70}")
            print(f"# SETUP {idx}/{total_setups}")
            print(f"# Strength: {s}, Guidance: {g}")
            print(f"# Completed: {completed}, Failed: {failed}, Remaining: {total_setups - idx + 1}")
            print(f"{'#'*70}")

            try:
                success = run_setup(s, g, train_count, test_count)
                if success:
                    completed += 1
                    print(f"\n[SUCCESS] Setup {idx}/{total_setups} completed")
                else:
                    failed += 1
                    print(f"\n[FAILED] Setup {idx}/{total_setups} failed")
            except Exception as e:
                failed += 1
                print(f"\n[ERROR] Setup {idx}/{total_setups}: {e}")
                cleanup_datasets()

            print(f"\nProgress: {idx}/{total_setups} | Completed: {completed} | Failed: {failed}")
            print("="*70)

        print("\n" + "="*70)
        print("MASTER RUN COMPLETED")
        print(f"Total Setups: {total_setups}")
        print(f"Successful: {completed}")
        print(f"Failed: {failed}")
        print("="*70)

    else:
        print("\n" + "="*70)
        print("MASTER RUN COMPLETED")
        print("="*70)
