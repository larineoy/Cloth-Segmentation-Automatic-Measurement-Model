"""
Usage:
    cd cloth_seg/
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
import torch

from preprocessing.dataset import (
    build_dataloaders,
    class_weights_inverse_freq,
    count_train_class_pixels,
)
from model.segmodel        import ClothSegModel
from training.trainer      import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    _void = cfg["training"].get("ignore_index", 255)
    if _void is not None:
        _void = int(_void)
    train_loader, test_loader = build_dataloaders(cfg["data"], void_label=_void)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ClothSegModel(
        num_classes=cfg["data"]["num_classes"],
        backbone   =cfg["model"]["backbone"],
        pretrained =cfg["model"]["pretrained"],
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {total:.1f}M parameters")

    # ── Train ─────────────────────────────────────────────────────────────────
    train_cfg = {
        **cfg["training"],
        "num_classes": cfg["data"]["num_classes"],
    }
    _cw_cfg = train_cfg.get("class_weights")
    if _cw_cfg == "auto":
        print("Computing inverse-frequency class weights from Train masks (no aug)...")
        counts = count_train_class_pixels(cfg["data"], _void)
        w = class_weights_inverse_freq(counts)
        print(f"  pixel counts: {counts.tolist()}")
        print(f"  CE weights (mean≈1): {w.tolist()}")
        train_cfg = {**train_cfg, "ce_class_weights": torch.from_numpy(w)}
    elif isinstance(_cw_cfg, list):
        if len(_cw_cfg) != cfg["data"]["num_classes"]:
            raise ValueError(
                f"training.class_weights list length must match data.num_classes "
                f"({cfg['data']['num_classes']}), got {len(_cw_cfg)}"
            )
        train_cfg = {**train_cfg, "ce_class_weights": torch.tensor(_cw_cfg, dtype=torch.float32)}
    elif _cw_cfg is not None and _cw_cfg not in (False, "none", "off"):
        raise ValueError(
            "training.class_weights must be 'auto', a list of length num_classes, or null"
        )

    trainer = Trainer(model, train_loader, test_loader, train_cfg, device)
    trainer.run()


if __name__ == "__main__":
    main()
