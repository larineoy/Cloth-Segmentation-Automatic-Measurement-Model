"""
Combined Cross-Entropy + Dice loss for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, smooth: float = 1e-6,
                 ignore_index: int | None = 255,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.num_classes   = num_classes
        self.ce_weight     = ce_weight
        self.dice_weight   = dice_weight
        self.smooth        = smooth
        self.ignore_index  = ignore_index
        cw = class_weights.float() if class_weights is not None else None
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss(weight=cw)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=cw)

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs  : (B, C, H, W) softmax probabilities
            targets: (B, H, W)    integer class labels
        """
        if self.ignore_index is None:
            one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            inter = (probs * one_hot).sum(dims)
            union = probs.sum(dims) + one_hot.sum(dims)
            dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()

        valid = targets != self.ignore_index
        if not valid.any():
            return (probs * 0).sum()

        valid_f = valid.unsqueeze(1).float()
        probs_m = probs * valid_f
        tc = targets.clone()
        tc[~valid] = 0
        one_hot = F.one_hot(tc, self.num_classes).permute(0, 3, 1, 2).float() * valid_f
        dims    = (0, 2, 3)
        inter   = (probs_m * one_hot).sum(dims)
        union   = probs_m.sum(dims) + one_hot.sum(dims)
        dice    = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, C, H, W) raw model output
            targets: (B, H, W)    integer class labels
        Returns:
            scalar loss
        """
        if self.ignore_index is None:
            loss_ce = self.ce(logits, targets)
        else:
            valid = targets != self.ignore_index
            if not valid.any():
                # CrossEntropyLoss(all ignore) returns nan — avoid poisoning the run
                loss_ce = (logits * 0).sum()
            else:
                loss_ce = self.ce(logits, targets)
        loss_ce *= self.ce_weight
        probs     = logits.softmax(dim=1)
        loss_dice = self.dice_loss(probs, targets) * self.dice_weight
        return loss_ce + loss_dice
