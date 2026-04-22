import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch

# from src.model import UNet
from segmentation_models_pytorch import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("checkpoints/model.pth", map_location=device, weights_only=True))
model.eval()

os.makedirs("outputs", exist_ok=True)

image_paths = sorted(glob.glob("data/images/*"))
mask_paths = sorted(glob.glob("data/masks/*"))

for i in range(5):
    img = cv2.imread(image_paths[i])
    img_resized = cv2.resize(img, (256, 256))

    img_norm = img_resized / 255.0
    img_tensor = np.transpose(img_norm, (2,0,1))

    img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8)

    gt = cv2.imread(mask_paths[i], 0)
    gt = cv2.resize(gt, (256, 256))
    gt = (gt > 127).astype(np.uint8)

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

    plt.subplot(1,3,2)
    plt.title("GT")
    plt.imshow(gt, cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Pred")
    plt.imshow(pred_mask, cmap="gray")

    plt.savefig(f"outputs/sample_{i}.png")
    plt.close()