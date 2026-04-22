import torch
from torch.utils.data import DataLoader
from src.model import UNet
from src.dataset import TreeDataset
from src.utils import loss_fn
import glob
import os
from src.augment import augmentation

image_paths = sorted(glob.glob("data/images/*"))
mask_paths = sorted(glob.glob("data/masks/*"))

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = augmentation()

model = UNet().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = TreeDataset(image_paths, mask_paths, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

best_loss = float("inf")

for epoch in range(10):
    epoch_loss = 0

    for img, mask in loader:
        img = img.to(device).float()
        mask = mask.to(device).float()

        pred = model(img)
        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "checkpoints/model.pth")