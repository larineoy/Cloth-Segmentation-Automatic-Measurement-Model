"""
Albumentations-based augmentation pipelines.
Returns tensors compatible with PyTorch (C, H, W) float32.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512, mask_void_label: int = 255):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            mask_fill_value=mask_void_label,
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms(img_size: int = 512):
    """No mask — for single-image prediction."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
