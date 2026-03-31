import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from PIL import Image
from torchvision import transforms
from src.configurations.augmentation_config import AugmentationConfig

torch.manual_seed(42)

base_dir = Path(__file__).parent.resolve()
baseline_train_dir = base_dir / 'datasets' / 'baseline' / 'train'
datasets_dir = base_dir / 'datasets'

if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist")
    sys.exit(1)

classes = [d.name for d in baseline_train_dir.iterdir() if d.is_dir()]

if len(classes) == 0:
    print(f"Error: No class directories found in {baseline_train_dir}")
    sys.exit(1)

aug_config = {
    "horizontal_flip": {"p": AugmentationConfig.GEOMETRIC_TRANSFORMS["horizontal_flip"]["ranges"]["p"][0]},
    "rotation": {"degrees": AugmentationConfig.GEOMETRIC_TRANSFORMS["rotation"]["ranges"]["degrees"][1]},
    "color_combined": {
        "brightness": AugmentationConfig.PHOTOMETRIC_TRANSFORMS["color_combined"]["ranges"]["brightness"][0],
        "contrast": AugmentationConfig.PHOTOMETRIC_TRANSFORMS["color_combined"]["ranges"]["contrast"][0],
        "saturation": AugmentationConfig.PHOTOMETRIC_TRANSFORMS["color_combined"]["ranges"]["saturation"][0],
        "hue": AugmentationConfig.PHOTOMETRIC_TRANSFORMS["color_combined"]["ranges"]["hue"][0]
    }
}

transform_aug = AugmentationConfig.get_augmented_transforms(
    image_size=224,
    augmentation_config=aug_config,
    train=True
)

def denormalize_and_save(tensor_img, save_path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor_img * std + mean
    img = torch.clamp(img, 0, 1)
    img = transforms.ToPILImage()(img)
    img.save(save_path)

tda_name = 'tda_x5'
multiplier = 4

print(f"Generating {tda_name} (5x = original + {multiplier}x augmented)...")

output_dir = datasets_dir / tda_name / 'train'
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

for class_name in classes:
    class_output_dir = output_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    class_input_dir = baseline_train_dir / class_name
    image_files = sorted([f for f in os.listdir(class_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    num_original = len(image_files)
    num_aug_total = num_original * multiplier

    for img_file in image_files:
        img_path = class_input_dir / img_file
        shutil.copy(str(img_path), str(class_output_dir / img_file))

    aug_count = 0
    aug_per_image = multiplier

    for img_file in image_files:
        img_path = class_input_dir / img_file
        img_basename = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]

        for aug_idx in range(aug_per_image):
            img = Image.open(img_path).convert('RGB')
            img_aug_tensor = transform_aug(img)
            output_path_aug = class_output_dir / f"{img_basename}_aug{aug_idx}{img_ext}"
            denormalize_and_save(img_aug_tensor, output_path_aug)
            aug_count += 1

    print(f"  {class_name}: {num_original} original + {aug_count} augmented = {num_original + aug_count} total")

print("TDA generation complete.")
