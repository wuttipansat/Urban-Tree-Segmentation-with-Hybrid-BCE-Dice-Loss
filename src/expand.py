import cv2
import os
import glob
import albumentations as A


"""This is offline augmentation for increasing dataset. Be aware to use it carefully preventing overfitting"""

image_paths = sorted(glob.glob("data/images/*"))
mask_paths = sorted(glob.glob("data/masks/*"))

out_img_dir = "dataset/images"
out_mask_dir = "dataset/masks"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

transforms = [
    A.HorizontalFlip(p=1.0),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0
    ),

    A.RandomRotate90(p=1.0),
]

for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    base_name = f"{i:04d}"

    cv2.imwrite(f"{out_img_dir}/{base_name}_orig.png", image)
    cv2.imwrite(f"{out_mask_dir}/{base_name}_orig.png", mask)

    for j, transform in enumerate(transforms):
        augmented = transform(image=image, mask=mask)

        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        cv2.imwrite(f"{out_img_dir}/{base_name}_aug{j}.png", aug_img)
        cv2.imwrite(f"{out_mask_dir}/{base_name}_aug{j}.png", aug_mask)

print("Offline augmentation finished.")