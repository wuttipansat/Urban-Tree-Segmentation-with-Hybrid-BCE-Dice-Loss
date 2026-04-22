import albumentations as A
from albumentations.pytorch import ToTensorV2

def augmentation():
    return A.compose([
        A.Resize(256, 256),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),

        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.5),

        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),

        A.Normalize(),
        ToTensorV2()
    ])