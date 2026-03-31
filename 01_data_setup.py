import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import random
import shutil
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--train_count', type=int, default=20)
parser.add_argument('--test_count', type=int, default=100)
args = parser.parse_args()

random.seed(42)

base_dir = Path(__file__).parent.resolve()
data_og_dir = base_dir / 'Data_OG'
datasets_dir = base_dir / 'datasets'
baseline_train_dir = datasets_dir / 'baseline' / 'train'
test_dir = datasets_dir / 'test'

if not data_og_dir.exists():
    print(f"Error: {data_og_dir} does not exist")
    sys.exit(1)

if datasets_dir.exists():
    shutil.rmtree(datasets_dir)

baseline_train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

classes = [d.name for d in data_og_dir.iterdir() if d.is_dir()]

if len(classes) == 0:
    print(f"Error: No class directories found in {data_og_dir}")
    sys.exit(1)

required_count = args.train_count + args.test_count
dataset_info = {
    'train_count': args.train_count,
    'test_count': args.test_count,
    'classes': {}
}

for class_name in classes:
    class_path = data_og_dir / class_name

    all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(all_images) < required_count:
        print(f"Error: {class_name} has only {len(all_images)} images, need at least {required_count}")
        sys.exit(1)

    random.shuffle(all_images)

    selected_images = all_images[:required_count]

    train_images = selected_images[:args.train_count]
    test_images = selected_images[args.train_count:args.train_count + args.test_count]

    train_set = set(train_images)
    test_set = set(test_images)
    assert len(train_set.intersection(test_set)) == 0, f"Overlap detected in {class_name}"

    train_class_dir = baseline_train_dir / class_name
    test_class_dir = test_dir / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    for img in train_images:
        src = class_path / img
        dst = train_class_dir / img
        shutil.copy(str(src), str(dst))

    for img in test_images:
        src = class_path / img
        dst = test_class_dir / img
        shutil.copy(str(src), str(dst))

    dataset_info['classes'][class_name] = {
        'train': len(train_images),
        'test': len(test_images)
    }

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

with open(datasets_dir / 'dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)

print("Data setup complete.")
