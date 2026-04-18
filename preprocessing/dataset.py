from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ClothSegDataset(Dataset):
    def __init__(self, data_root: str, split: str = "Train", transform=None):
        """
        Args:
            data_root : path to the top-level 'data/' folder
            split     : "Train" or "Test"  (case-sensitive, matches folder name)
            transform : albumentations pipeline from preprocessing.transforms
        """
        self.img_dir  = Path(data_root) / split / "Image"
        self.mask_dir = Path(data_root) / split / "Segmented"
        self.transform = transform

        # Collect all jpg stems that have a matching segmented file
        self.stems = sorted([
            p.stem                                  # e.g. "189"  (from "189.jpg")
            for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (self.mask_dir / f"{p.name}.png").exists()
            #                     ^^^^^^^^^^^^^^^^^^^
            #   mask filename = image filename + ".png"
            #   e.g. "189.jpg" → "189.jpg.png"
        ])

        if not self.stems:
            raise FileNotFoundError(
                f"No matched image/mask pairs found.\n"
                f"  Image dir : {self.img_dir}\n"
                f"  Mask dir  : {self.mask_dir}\n"
                f"  Expected mask format: <image_filename>.png  e.g. 189.jpg.png"
            )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]                          # e.g. "189"

        img_path  = self.img_dir  / f"{stem}.jpg"
        mask_path = self.mask_dir / f"{stem}.jpg.png"  # e.g. 189.jpg.png

        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # greyscale → class index

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]   # tensor (3, H, W) float32 normalised
            mask = out["mask"]    # tensor (H, W)    int64

        return img, mask


def build_dataloaders(cfg: dict):
    """
    Builds Train and Test DataLoaders from config dict.

    Expected cfg keys:
        root        : path to data/ folder
        img_size    : int
        batch_size  : int
        num_workers : int
    """
    from preprocessing.transforms import get_train_transforms, get_val_transforms

    train_ds = ClothSegDataset(
        data_root=cfg["root"],
        split="Train",
        transform=get_train_transforms(cfg["img_size"]),
    )
    test_ds = ClothSegDataset(
        data_root=cfg["root"],
        split="Test",
        transform=get_val_transforms(cfg["img_size"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}  |  Test samples: {len(test_ds)}")
    return train_loader, test_loader
