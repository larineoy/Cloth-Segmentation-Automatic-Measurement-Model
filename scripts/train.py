"""
Usage:
    cd cloth_seg/
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import yaml
import torch

from preprocessing.dataset import build_dataloaders
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
    train_loader, test_loader = build_dataloaders(cfg["data"])

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
    trainer = Trainer(model, train_loader, test_loader, train_cfg, device)
    trainer.run()


if __name__ == "__main__":
    main()
