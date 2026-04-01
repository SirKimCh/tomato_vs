# Model Architecture & Training Configuration

## 1. Model: EfficientNet-B0

### Overview

EfficientNet-B0 is the baseline model of the EfficientNet family, designed with compound scaling (balancing depth, width, and resolution). This study uses the implementation from `torchvision.models`.

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
| Total Parameters | ~5.3M |
| Input Size | 224 × 224 × 3 |
| Output | 5 classes |
| Pretrained | ImageNet (EfficientNet_B0_Weights.DEFAULT) |

---

## 2. Training Configurations

### Main Config (03_run_experiments, 03_1_run_gan_experiment, 06_final_comparison)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | lr=1e-4, weight_decay=1e-4 |
| **Scheduler** | CosineAnnealingWarmRestarts | T_0=10, T_mult=2, eta_min=1e-7 |
| **Loss** | CrossEntropyLoss | — |
| **Epochs** | 50 | Maximum |
| **Early Stopping** | Patience=10 | Based on val_loss |
| **Batch Size** | 8 | — |
| **Gradient Clipping** | max_norm=1.0 | — |
| **Validation Split** | 20% | Minimum 2 samples |
| **Trials** | 5 | Seeds: 43, 44, 45, 46, 47 |

### Freezing Strategy

**Scripts 03, 03_1, 06 (Partial Freeze):**
```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier layer
for param in model.model.classifier.parameters():
    param.requires_grad = True

# Unfreeze last 3 feature blocks
for param in model.model.features[-3:].parameters():
    param.requires_grad = True
```

**Script 07 (No Freeze):**
```python
# All parameters are trainable
model = EfficientNetB0Model(num_classes=5, pretrained=pretrained_flag)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

### Data Augmentation During Training

Regardless of the data generation method used, the training pipeline always applies light augmentation:

```python
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
```

---

## 3. Evaluation Metrics

| Metric | Function | Averaging | Description |
|--------|----------|-----------|-------------|
| Accuracy | `accuracy_score` | — | Ratio of correct predictions |
| Precision | `precision_score` | weighted | Ratio of true positives among predicted positives |
| Recall | `recall_score` | weighted | Ratio of true positives among actual positives |
| F1-Score | `f1_score` | weighted | Harmonic mean of Precision and Recall |
| MCC | `matthews_corrcoef` | — | Balanced metric, range [-1, 1] |
| AUC-ROC | `roc_auc_score` | weighted, ovr | Area under the ROC curve |

### Confusion Matrix

- Computed using `sklearn.metrics.confusion_matrix`
- Plotted using `seaborn.heatmap` (cmap='Blues', fmt='d')
- **Per trial:** Separate matrix for each trial
- **Aggregate:** Sum of matrices across 5 trials (sum, not average)

---

## 4. Reproducibility

### Random Seeds

```python
# Each trial uses seed = 42 + trial_number
# Trial 1: seed=43, Trial 2: seed=44, ..., Trial 5: seed=47

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Data Split

```python
# Validation split uses a PyTorch Generator with the seed
generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
```

> **Note:** Despite setting seeds, results may vary slightly between different GPUs due to floating-point precision differences.
