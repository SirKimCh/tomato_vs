# Model Architecture & Training Configuration

## 1. Model: EfficientNet-B0

### Overview

EfficientNet-B0 is the baseline model of the EfficientNet family, designed with compound scaling (depth + width + resolution). This study uses the `torchvision.models` implementation with ImageNet pre-trained weights.

### Source Code

```python
# src/models/efficientnet_b0.py
class EfficientNetB0Model(BaseModel):
    def __init__(self, num_classes, pretrained=True):
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, num_classes
        )
```

### Model Specifications

| Specification | Value |
|---------------|-------|
| Total Parameters | ~5.3 M |
| Trainable (partial fine-tuning) | ~2.2 M |
| Frozen (partial fine-tuning) | ~3.1 M |
| Input Size | 224 × 224 × 3 |
| Output | 5 classes |
| Pre-trained | ImageNet (EfficientNet_B0_Weights.DEFAULT) |

---

## 2. Training Configuration

### Hyperparameters (all scripts: 03, 03_1, 05, 06)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | lr=1e-4, weight_decay=1e-4 |
| **LR Scheduler** | CosineAnnealingWarmRestarts | T_0=10, T_mult=2, η_min=1e-7 |
| **Loss** | CrossEntropyLoss | MixUp/CutMix use mixed-label loss |
| **Epochs** | 50 | Maximum |
| **Early Stopping** | Patience=10 | Based on validation loss |
| **Batch Size** | 8 | — |
| **Gradient Clipping** | max_norm=1.0 | — |

### Cross-Validation (default rigorous mode)

| Parameter | Value |
|-----------|-------|
| Method | RepeatedStratifiedKFold |
| n_splits | 5 |
| n_repeats | 3 |
| Total evaluations | 15 per method |
| Seeds | fold_idx: seed = 42 + fold_idx |
| Report | mean ± std across 15 folds |

### Partial Freezing Strategy (03_run_experiments, 03_3, 05_final, 06_transfer)

```python
# Freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier (Linear head)
for param in model.model.classifier.parameters():
    param.requires_grad = True

# Unfreeze last 3 feature blocks
for param in model.model.features[-3:].parameters():
    param.requires_grad = True
# Result: ~2.2M trainable / ~3.1M frozen
```

### No-Freeze (06_transfer_learning_comparison only — for ablation)

```python
model = EfficientNetB0Model(num_classes=5, pretrained=pretrained_flag)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

### Training-time Data Augmentation (applied to all datasets except test)

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

> **Note:** For MixUp/CutMix experiments, no additional training-time transform is added (online mixing already provides augmentation). For RandAugment experiments, `transforms.RandAugment(num_ops=2, magnitude=9)` replaces the standard ColorJitter.

---

## 3. Evaluation Metrics

| Metric | Library | Notes |
|--------|---------|-------|
| Accuracy | `sklearn.metrics.accuracy_score` | Overall correctness |
| Precision | `precision_score(weighted)` | Class-imbalance aware |
| Recall | `recall_score(weighted)` | Class-imbalance aware |
| F1-Score | `f1_score(weighted)` | **Primary metric** |
| MCC | `matthews_corrcoef` | Best single metric for imbalanced classes |
| AUC-ROC | `roc_auc_score(OvR, weighted)` | Discrimination ability |
| FID | `torchmetrics.image.FrechetInceptionDistance` | Image quality (lower = better) |
| LPIPS | `lpips.LPIPS(net='alex')` | Perceptual similarity |
| Wilcoxon p | `scipy.stats.wilcoxon` | Statistical significance |
| Cohen's d | Pooled std formula | Effect size |

### Aggregate Confusion Matrix

- Sum (not average) of per-fold confusion matrices
- Number of evaluations in title: e.g., "(15 evals)" for k-fold

---

## 4. Reproducibility

### Random Seeds

```python
# K-fold mode: seed = 42 + fold_idx  (fold_idx ∈ 0..14)
# Fixed-trial mode: seed = 42 + trial  (trial ∈ 1..5)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Data Split Reproducibility

```python
# For fixed-trial mode (20% val split from augmented dataset)
generator = torch.Generator().manual_seed(seed)
train_ds, val_ds = random_split(full_dataset, [train_sz, val_sz], generator=generator)

# For k-fold mode (split on ORIGINAL 20 baseline images per class)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

> **Note:** Despite fixing seeds, slight numerical differences may occur across GPUs due to floating-point precision.
