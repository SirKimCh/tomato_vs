import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import argparse
import random
import shutil
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from tqdm import tqdm

from src.models.efficientnet_b0 import EfficientNetB0Model

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()

BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_TRIALS = 5
TEST_COUNT_PER_CLASS = 100

device = torch.device('cuda')

base_dir = Path(__file__).parent.resolve()
data_og_dir = base_dir / 'Data_OG'
data_st_dir = base_dir / 'Data_ST'
baseline_train_dir = data_st_dir / 'baseline' / 'train'
sd_x5_train_dir = data_st_dir / 'sd_x5' / 'train'
tda_x5_train_dir = data_st_dir / 'tda_x5' / 'train'
test_dir = data_st_dir / 'test'
combined_train_dir = data_st_dir / 'combined_tda_sd' / 'train'
results_dir = base_dir / 'Results'
results_dir.mkdir(parents=True, exist_ok=True)

if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = results_dir / f'{timestamp}_final_comparison'
output_dir.mkdir(parents=True, exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_curves_data = []

def get_baseline_filenames():
    baseline_files = {}
    if baseline_train_dir.exists():
        for class_folder in baseline_train_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                baseline_files[class_name] = set()
                for img_file in class_folder.iterdir():
                    if img_file.is_file():
                        baseline_files[class_name].add(img_file.name)
    return baseline_files

def create_test_set():
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    baseline_files = get_baseline_filenames()

    for class_folder in data_og_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            test_class_dir = test_dir / class_name
            test_class_dir.mkdir(parents=True, exist_ok=True)

            excluded_names = baseline_files.get(class_name, set())

            available_images = []
            for img_file in class_folder.iterdir():
                if img_file.is_file() and img_file.name not in excluded_names:
                    available_images.append(img_file)

            random.shuffle(available_images)
            selected_images = available_images[:TEST_COUNT_PER_CLASS]

            for img_file in selected_images:
                shutil.copy2(str(img_file), str(test_class_dir / img_file.name))

            print(f"Test set - {class_name}: {len(selected_images)} images")

def create_combined_dataset():
    if combined_train_dir.exists():
        shutil.rmtree(combined_train_dir)
    combined_train_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in baseline_train_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            combined_class_dir = combined_train_dir / class_name
            combined_class_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_folder.iterdir():
                if img_file.is_file():
                    shutil.copy2(str(img_file), str(combined_class_dir / img_file.name))

    for class_folder in sd_x5_train_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            combined_class_dir = combined_train_dir / class_name
            combined_class_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_folder.iterdir():
                if img_file.is_file() and '_sd' in img_file.stem:
                    shutil.copy2(str(img_file), str(combined_class_dir / img_file.name))

    for class_folder in tda_x5_train_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            combined_class_dir = combined_train_dir / class_name
            combined_class_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_folder.iterdir():
                if img_file.is_file() and '_aug' in img_file.stem:
                    shutil.copy2(str(img_file), str(combined_class_dir / img_file.name))

    for class_folder in combined_train_dir.iterdir():
        if class_folder.is_dir():
            count = len(list(class_folder.iterdir()))
            print(f"Combined dataset - {class_folder.name}: {count} images")

def run_experiment(dataset_dir, exp_name, seed, trial):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    full_dataset = datasets.ImageFolder(str(dataset_dir), transform=transform_train)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=transform_test)

    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    val_size = max(2, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    model = EfficientNetB0Model(num_classes=num_classes, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.model.classifier.parameters():
        param.requires_grad = True

    for param in model.model.features[-3:].parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    if trial == 1:
        param_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'pretrained': True,
            'freezing': 'partial'
        }
        param_df = pd.DataFrame([param_info])
        param_df.to_csv(output_dir / 'model_params.csv', index=False)
        print(f"Model Parameters - Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    pbar = tqdm(range(EPOCHS), desc=f"{exp_name} T{trial}", leave=False)

    for epoch in pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        training_curves_data.append({
            'Exp': exp_name,
            'Trial': trial,
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Train_Acc': train_acc,
            'Val_Loss': val_loss,
            'Val_Acc': val_acc
        })

        pbar.set_postfix({'TL': f'{train_loss:.3f}', 'TA': f'{train_acc:.2f}', 'VL': f'{val_loss:.3f}', 'VA': f'{val_acc:.2f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    pbar.close()

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{exp_name} - Loss (Trial {trial})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(train_accs, label='Train Acc', color='blue')
    axes[1].plot(val_accs, label='Val Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{exp_name} - Accuracy (Trial {trial})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{exp_name}_trial{trial}.png', dpi=150)
    plt.close()

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{exp_name} - Confusion Matrix (Trial {trial})')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{exp_name}_trial{trial}.png', dpi=150)
    plt.close()

    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader
    del test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return acc, prec, rec, f1, mcc, auc, cm

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 1: Creating Test Set")
    print("=" * 60)
    create_test_set()

    print("\n" + "=" * 60)
    print("STEP 2: Creating Combined Dataset")
    print("=" * 60)
    create_combined_dataset()

    print("\n" + "=" * 60)
    print("STEP 3: Running Experiments")
    print("=" * 60)

    experiments = ['baseline', 'combined_tda_sd']
    experiment_dirs = {
        'baseline': baseline_train_dir,
        'combined_tda_sd': combined_train_dir
    }

    results = []
    confusion_matrices = {exp: [] for exp in experiments}
    history = {exp: {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'mcc': [], 'auc': []} for exp in experiments}

    for trial in range(1, NUM_TRIALS + 1):
        seed = 42 + trial
        print(f"\n{'='*60}")
        print(f"TRIAL {trial}/{NUM_TRIALS} (Seed: {seed})")
        print(f"{'='*60}")

        for exp_name in experiments:
            train_dir = experiment_dirs[exp_name]

            if not train_dir.exists():
                print(f"Skipping {exp_name} - not found")
                continue

            try:
                acc, prec, rec, f1, mcc, auc, cm = run_experiment(train_dir, exp_name, seed, trial)

                history[exp_name]['acc'].append(acc)
                history[exp_name]['prec'].append(prec)
                history[exp_name]['rec'].append(rec)
                history[exp_name]['f1'].append(f1)
                history[exp_name]['mcc'].append(mcc)
                history[exp_name]['auc'].append(auc)
                confusion_matrices[exp_name].append(cm)

                results.append({
                    'Exp': exp_name,
                    'Trial': trial,
                    'Acc': acc,
                    'Prec': prec,
                    'Rec': rec,
                    'F1': f1,
                    'MCC': mcc,
                    'AUC': auc
                })

                print(f"{exp_name} | Trial {trial} | Acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {auc:.4f}")

            except Exception as e:
                print(f"Error: {exp_name} Trial {trial}: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()

    print(f"\n{'='*60}")
    print("AVERAGE RESULTS")
    print(f"{'='*60}")

    for exp_name in experiments:
        if history[exp_name]['acc']:
            avg_acc = np.mean(history[exp_name]['acc'])
            std_acc = np.std(history[exp_name]['acc'])
            avg_prec = np.mean(history[exp_name]['prec'])
            std_prec = np.std(history[exp_name]['prec'])
            avg_rec = np.mean(history[exp_name]['rec'])
            std_rec = np.std(history[exp_name]['rec'])
            avg_f1 = np.mean(history[exp_name]['f1'])
            std_f1 = np.std(history[exp_name]['f1'])
            avg_mcc = np.mean(history[exp_name]['mcc'])
            std_mcc = np.std(history[exp_name]['mcc'])
            avg_auc = np.mean(history[exp_name]['auc'])
            std_auc = np.std(history[exp_name]['auc'])

            results.append({
                'Exp': exp_name,
                'Trial': 'AVG',
                'Acc': avg_acc,
                'Prec': avg_prec,
                'Rec': avg_rec,
                'F1': avg_f1,
                'MCC': avg_mcc,
                'AUC': avg_auc
            })

            results.append({
                'Exp': exp_name,
                'Trial': 'STD',
                'Acc': std_acc,
                'Prec': std_prec,
                'Rec': std_rec,
                'F1': std_f1,
                'MCC': std_mcc,
                'AUC': std_auc
            })

            print(f"{exp_name} | AVG | Acc: {avg_acc:.4f}±{std_acc:.4f} | F1: {avg_f1:.4f}±{std_f1:.4f} | MCC: {avg_mcc:.4f} | AUC: {avg_auc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'metrics_summary.csv', index=False)

    df_curves = pd.DataFrame(training_curves_data)
    df_curves.to_csv(output_dir / 'training_curves.csv', index=False)

    for exp_name in experiments:
        if confusion_matrices[exp_name]:
            agg_cm = np.sum(confusion_matrices[exp_name], axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'{exp_name} - Aggregate Confusion Matrix (5 Trials)')
            plt.tight_layout()
            plt.savefig(output_dir / f'cm_aggregate_{exp_name}.png', dpi=150)
            plt.close()

    print(f"\nResults saved to {output_dir}")
    print(f"  - metrics_summary.csv")
    print(f"  - training_curves.csv")
    print(f"  - confusion matrices & loss curves")

