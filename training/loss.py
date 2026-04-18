"""
Combined Cross-Entropy + Dice loss for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.num_classes  = num_classes
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.smooth       = smooth
        self.ce           = nn.CrossEntropyLoss()

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs  : (B, C, H, W) softmax probabilities
            targets: (B, H, W)    integer class labels
        """
        one_hot = F.one_hot(targets, self.num_classes)      # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()       # (B, C, H, W)
        dims    = (0, 2, 3)
        inter   = (probs * one_hot).sum(dims)
        union   = probs.sum(dims) + one_hot.sum(dims)
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
        loss_ce   = self.ce(logits, targets) * self.ce_weight
        probs     = logits.softmax(dim=1)
        loss_dice = self.dice_loss(probs, targets) * self.dice_weight
        return loss_ce + loss_dice
