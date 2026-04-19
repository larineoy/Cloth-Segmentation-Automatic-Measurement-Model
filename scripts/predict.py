"""
Usage:
    python scripts/predict.py --image path/to/photo.jpg
    python scripts/predict.py --image path/to/photo.jpg --overlay
    python scripts/predict.py --folder path/to/images/  --overlay
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml
from PIL import Image

from inference.predictor import Predictor
from inference.colorize  import save_result, overlay_mask


def _palette_from_cfg(data_cfg: dict) -> dict:
    pal = data_cfg.get("palette") or {}
    return {int(k): tuple(int(c) for c in v) for k, v in pal.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--image",      default=None,  help="Single image path")
    p.add_argument("--folder",     default=None,  help="Folder of images")
    p.add_argument("--overlay",    action="store_true",
                   help="Blend mask over original image instead of flat color")
    return p.parse_args()


def run_one(predictor, image_path, out_dir, overlay=False, palette=None):
    mask = predictor.predict(image_path)
    stem = Path(image_path).stem
    out_path = os.path.join(out_dir, f"{stem}_mask.png")
    ckw = {"palette": palette} if palette else {}

    if overlay:
        orig = np.array(Image.open(image_path).convert("RGB"))
        # Resize original to match mask if needed
        if orig.shape[:2] != mask.shape:
            orig = np.array(
                Image.fromarray(orig).resize(
                    (mask.shape[1], mask.shape[0]), Image.BILINEAR
                )
            )
        blended = overlay_mask(orig, mask, **ckw)
        Image.fromarray(blended).save(out_path)
        print(f"Saved overlay → {out_path}")
    else:
        save_result(mask, out_path, **ckw)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    predictor = Predictor(
        checkpoint  = cfg["inference"]["checkpoint"],
        num_classes = cfg["data"]["num_classes"],
        img_size    = cfg["data"]["img_size"],
    )
    pal = _palette_from_cfg(cfg["data"])

    out_dir = cfg["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.image:
        run_one(predictor, args.image, out_dir, overlay=args.overlay, palette=pal)

    elif args.folder:
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts]
        print(f"Running inference on {len(paths)} images...")
        for p in paths:
            run_one(predictor, str(p), out_dir, overlay=args.overlay, palette=pal)
    else:
        print("Provide --image or --folder")


if __name__ == "__main__":
    main()
