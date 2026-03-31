import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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

IMG_SIZE = 64
LATENT_DIM = 100
NUM_EPOCHS = 500
BATCH_SIZE = 16
LR = 0.0002
BETA1 = 0.5
GEN_PER_CLASS = 80
OUTPUT_SIZE = 224

device = torch.device('cuda')


class SingleClassDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = sorted([
            image_dir / f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

to_pil = transforms.Compose([
    transforms.Normalize([-1, -1, -1], [2, 2, 2]),
    transforms.ToPILImage()
])

gan_name = 'gan_x5'
output_dir = datasets_dir / gan_name / 'train'
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating {gan_name} (5x = 20 original + 80 GAN per class)...")
print(f"Classes: {classes}")
print("=" * 60)

for class_name in classes:
    print(f"\nTraining DCGAN for: {class_name}")

    class_input_dir = baseline_train_dir / class_name
    class_output_dir = output_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(class_input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for img_file in image_files:
        shutil.copy(str(class_input_dir / img_file), str(class_output_dir / img_file))

    dataset = SingleClassDataset(class_input_dir, transform_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    netG = Generator(LATENT_DIM).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    real_label_val = 0.9
    fake_label_val = 0.0

    last_d_loss = 0.0
    last_g_loss = 0.0

    pbar = tqdm(range(NUM_EPOCHS), desc=f"  {class_name}", leave=False)
    for epoch in pbar:
        for real_batch in dataloader:
            real_batch = real_batch.to(device)
            b_size = real_batch.size(0)

            netD.zero_grad()
            label_real = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
            output_real = netD(real_batch)
            lossD_real = criterion(output_real, label_real)
            lossD_real.backward()

            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake_batch = netG(noise)
            label_fake = torch.full((b_size,), fake_label_val, dtype=torch.float, device=device)
            output_fake = netD(fake_batch.detach())
            lossD_fake = criterion(output_fake, label_fake)
            lossD_fake.backward()
            optimizerD.step()

            netG.zero_grad()
            label_gen = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            output_gen = netD(fake_batch)
            lossG = criterion(output_gen, label_gen)
            lossG.backward()
            optimizerG.step()

            last_d_loss = (lossD_real + lossD_fake).item()
            last_g_loss = lossG.item()

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'D_loss': f'{last_d_loss:.3f}', 'G_loss': f'{last_g_loss:.3f}'})

    pbar.close()

    netG.eval()
    gen_count = 0
    with torch.no_grad():
        while gen_count < GEN_PER_CLASS:
            remaining = GEN_PER_CLASS - gen_count
            batch_gen = min(BATCH_SIZE, remaining)
            noise = torch.randn(batch_gen, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise).cpu()

            for i in range(batch_gen):
                img_tensor = fake_images[i]
                img_pil = to_pil(img_tensor)
                img_pil = img_pil.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BICUBIC)
                save_path = class_output_dir / f"gan_generated_{gen_count:04d}.jpg"
                img_pil.save(save_path, quality=95)
                gen_count += 1

    del netG, netD, optimizerD, optimizerG
    torch.cuda.empty_cache()

    total = len(image_files) + GEN_PER_CLASS
    print(f"  {class_name}: {len(image_files)} original + {GEN_PER_CLASS} GAN = {total} total")

print("\nGAN generation complete.")

