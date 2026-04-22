import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch import Unet

import glob
import os
from tqdm import tqdm

from src.augment import augmentation, val_augmentation
# from src.model import UNet
from src.dataset import TreeDataset
from src.utils import loss_fn

EPOCH = 100

image_paths = sorted(glob.glob("dataset/images/*"))
mask_paths = sorted(glob.glob("dataset/masks/*"))

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = augmentation()
val_transform = val_augmentation()

model = Unet(
    encoder_name="resnet18",       
    encoder_weights="imagenet",     
    in_channels=3,
    classes=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths,
    mask_paths,
    test_size=0.2,
    random_state=42
)

train_dataset = TreeDataset(train_imgs, train_masks, transform=train_transform)
test_dataset = TreeDataset(val_imgs, val_masks, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

best_loss = float("inf")
num_params = sum(p.numel() for p in model.parameters())

print("="*40)
print("Training")
print("="*40)
print(f"Device: {device}")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {train_loader.batch_size}")
print(f"Epoch: {EPOCH}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Total parameters: {num_params:,}")
print("="*40)


for epoch in range(EPOCH):
    model.train()
    train_loss = 0

    train_prog = tqdm(train_loader, desc="Training", leave=False) 

    for img, mask in train_prog:
        img = img.to(device).float()
        mask = mask.to(device).float()

        pred = model(img)
        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_prog.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    val_prog = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for img, mask in val_prog:
            img = img.to(device).float()
            mask = mask.to(device).float()

            pred = model(img)
            loss = loss_fn(pred, mask)
            
            val_loss += loss.item()

            val_prog.set_postfix(loss=loss.item())

    val_loss /= len(val_loader)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/model.pth")