"""
Dump tensors exactly as the training dataloader produces them (after transforms).

  python scripts/inspect_preprocessing.py
  python scripts/inspect_preprocessing.py --config configs/default.yaml --split val --num-samples 8
  python scripts/inspect_preprocessing.py --split train --seed 42

Writes PNGs (ImageNet-denormalized RGB, colored mask, overlay) under --out-dir and prints
min/max/mean and unique mask values so you can verify label mapping (no invented pixels).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

from inference.colorize import colorize_mask, overlay_mask
from preprocessing.dataset import build_dataloaders

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _tensor_to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    """Invert Normalize + ToTensorV2 (ImageNet)."""
    x = img_chw.detach().cpu().float()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = (x.clamp(0, 1) * 255).byte().numpy()
    return np.transpose(x, (1, 2, 0))


def _palette_from_cfg(data_cfg: dict) -> dict:
    pal = data_cfg.get("palette") or {}
    return {int(k): tuple(int(c) for c in v) for k, v in pal.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize preprocessed train/val tensors.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="val = same as Test split, deterministic order. train = augmentations (stochastic unless --seed).",
    )
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument(
        "--out-dir",
        default="debug_preprocessing",
        help="Folder under project root for PNG exports",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, fixes Python/NumPy/Torch RNG before sampling train augmentations",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        s = args.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    cfg_path = _ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    _void = cfg["training"].get("ignore_index", 255)
    if _void is not None:
        _void = int(_void)

    train_loader, test_loader = build_dataloaders(cfg["data"], void_label=_void)
    ds = train_loader.dataset if args.split == "train" else test_loader.dataset

    out_root = _ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    palette = _palette_from_cfg(cfg["data"])
    # show ignore pixels in visualization
    palette_vis = {**palette, 255: (48, 48, 48)}

    n = min(args.num_samples, len(ds))
    print(f"Saving {n} sample(s) from {args.split} split -> {out_root}\n")

    for i in range(n):
        img_t, mask_t = ds[i]
        # img_t: (3,H,W) float32 normalized; mask_t: (H,W) long
        rgb = _tensor_to_uint8_rgb(img_t)
        mask_np = mask_t.cpu().numpy().astype(np.int64)

        u, cnt = np.unique(mask_np, return_counts=True)
        frac = cnt.astype(np.float64) / mask_np.size
        u_str = ", ".join(f"{v}({p*100:.1f}%)" for v, p in zip(u, frac))

        print(
            f"[{i}] image  shape={tuple(img_t.shape)} dtype={img_t.dtype} "
            f"min={img_t.min().item():.4f} max={img_t.max().item():.4f} "
            f"mean={img_t.mean().item():.4f}"
        )
        print(f"     mask   shape={tuple(mask_t.shape)} dtype={mask_t.dtype} unique: {u_str}")
        print()

        stem = f"{args.split}_{i:03d}"
        Image.fromarray(rgb).save(out_root / f"{stem}_image.png")
        cmask = colorize_mask(mask_np.astype(np.uint8), palette=palette_vis)
        Image.fromarray(cmask).save(out_root / f"{stem}_mask_color.png")
        ov = overlay_mask(rgb, mask_np.astype(np.uint8), palette=palette_vis, alpha=0.45)
        Image.fromarray(ov).save(out_root / f"{stem}_overlay.png")

    print("Done. Open the PNGs: *_image is denormalized RGB; *_mask_color is class ids via palette; *_overlay blends them.")


if __name__ == "__main__":
    main()
