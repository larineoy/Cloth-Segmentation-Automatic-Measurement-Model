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
        _ig = cfg.get("ignore_index", 255)
        _esp = cfg.get("early_stopping_patience")
        if _esp is not None:
            _esp = int(_esp)
        self.cfg          = {
            **cfg,
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "epochs": int(cfg["epochs"]),
            "freeze_encoder_epochs": int(cfg.get("freeze_encoder_epochs", 5)),
            "num_classes": int(cfg["num_classes"]),
            "amp": bool(cfg["amp"]),
            "ignore_index": None if _ig is None else int(_ig),
            "early_stopping_patience": _esp,
            "early_stopping_min_delta": float(cfg.get("early_stopping_min_delta", 0.0)),
        }
        self.device       = device

        c = self.cfg
        _cw = c.get("ce_class_weights")
        _cw_t = None
        if _cw is not None:
            _cw_t = _cw if isinstance(_cw, torch.Tensor) else torch.tensor(_cw, dtype=torch.float32)
        self.criterion = SegLoss(
            num_classes=c["num_classes"],
            ignore_index=c["ignore_index"],
            class_weights=_cw_t,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=c["epochs"]
        )
        self.scaler    = torch.amp.GradScaler(
            "cuda", enabled=(c["amp"] and device == "cuda")
        )
        self.best_miou = None  # type: float | None
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
            total_miou += compute_miou(
                preds, masks, self.cfg["num_classes"], self.cfg["ignore_index"]
            )
            total_acc += compute_pixel_acc(preds, masks, self.cfg["ignore_index"])
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
        es_patience         = self.cfg.get("early_stopping_patience")
        es_delta            = self.cfg["early_stopping_min_delta"]
        es_enabled          = es_patience is not None and es_patience > 0
        stall               = 0

        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Encoder frozen for first {freeze_until} epochs")
        if es_enabled:
            print(
                f"Early stopping: patience={es_patience} epochs "
                f"(val mIoU, min_delta={es_delta})"
            )
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

            best = self.best_miou
            improved = best is None or val_miou > best + es_delta

            # Save best checkpoint
            if improved:
                self.best_miou = val_miou
                path = self._save(epoch, val_miou, tag="best")
                print(f"  ↑ New best mIoU={val_miou:.4f}  saved → {path}")
                stall = 0
            elif es_enabled:
                stall += 1

            # Always save latest
            self._save(epoch, val_miou, tag="latest")

            if es_enabled and stall >= es_patience:
                print(
                    f"\nEarly stopping at epoch {epoch}: "
                    f"val mIoU did not improve by more than {es_delta} for {es_patience} epochs."
                )
                break

        best_miou = self.best_miou if self.best_miou is not None else float("nan")
        print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
