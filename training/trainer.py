"""
Train / validation loop with:
  - mixed precision (AMP)
  - encoder freeze/unfreeze schedule
  - best-checkpoint saving
  - tqdm progress bars
"""

import os
import torch
from tqdm import tqdm

from training.loss    import SegLoss
from training.metrics import compute_miou, compute_pixel_acc


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict, device: str):
        """
        Args:
            model        : ClothSegModel
            train_loader : DataLoader
            val_loader   : DataLoader
            cfg          : training section of default.yaml (as dict)
            device       : "cuda" or "cpu"
        """
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        self.criterion = SegLoss(num_classes=cfg["num_classes"])
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["epochs"]
        )
        self.scaler    = torch.cuda.amp.GradScaler(enabled=(cfg["amp"] and device == "cuda"))
        self.best_miou = 0.0
        os.makedirs("checkpoints", exist_ok=True)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(self.train_loader, desc="  train", leave=False):
            imgs  = imgs.to(self.device)
            masks = masks.long().to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(self.device, enabled=self.cfg["amp"]):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        total_loss, total_miou, total_acc = 0.0, 0.0, 0.0
        for imgs, masks in tqdm(self.val_loader, desc="  val  ", leave=False):
            imgs  = imgs.to(self.device)
            masks = masks.long().to(self.device)
            with torch.autocast(self.device, enabled=self.cfg["amp"]):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)
            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            total_miou += compute_miou(preds, masks, self.cfg["num_classes"])
            total_acc  += compute_pixel_acc(preds, masks)
        n = len(self.val_loader)
        return total_loss / n, total_miou / n, total_acc / n

    def _save(self, epoch: int, miou: float, tag: str = "best"):
        path = f"checkpoints/{tag}.pth"
        torch.save({
            "epoch"     : epoch,
            "model"     : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "best_miou" : miou,
        }, path)
        return path

    def run(self):
        epochs              = self.cfg["epochs"]
        freeze_until        = self.cfg.get("freeze_encoder_epochs", 5)

        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Encoder frozen for first {freeze_until} epochs")
        self.model.freeze_encoder()

        for epoch in range(1, epochs + 1):
            # Phase switch: unfreeze encoder after freeze_until epochs
            if epoch == freeze_until + 1:
                self.model.unfreeze_encoder()
                print(f"\n[Epoch {epoch}] Encoder unfrozen — fine-tuning end-to-end")

            train_loss = self._train_epoch()
            val_loss, val_miou, val_acc = self._val_epoch()
            self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mIoU={val_miou:.4f} | "
                f"acc={val_acc:.4f}"
            )

            # Save best checkpoint
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                path = self._save(epoch, val_miou, tag="best")
                print(f"  ↑ New best mIoU={val_miou:.4f}  saved → {path}")

            # Always save latest
            self._save(epoch, val_miou, tag="latest")

        print(f"\nTraining complete. Best mIoU: {self.best_miou:.4f}")
