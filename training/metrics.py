"""
mIoU and pixel accuracy computed from prediction tensors.
"""

import torch
import numpy as np

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, ignore_index: int | None = None) -> float:
    """
    Args:
        preds  : (B, H, W) integer predictions  (argmax of logits)
        targets: (B, H, W) integer ground-truth labels
    Returns:
        mean IoU over all classes (ignores classes with no GT pixels)
    """
    if ignore_index is not None:
        valid = targets != ignore_index
        preds = preds[valid].cpu().numpy().flatten()
        targets = targets[valid].cpu().numpy().flatten()
    else:
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

    ious = []
    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        inter    = (pred_c & target_c).sum()
        union    = (pred_c | target_c).sum()
        if union == 0:
            continue  # class not present — skip
        ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_acc(preds: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int | None = None) -> float:
    """Overall pixel accuracy (optionally excluding ignore-labelled pixels)."""
    if ignore_index is not None:
        valid = targets != ignore_index
        if not valid.any():
            return 0.0
        correct = ((preds == targets) & valid).sum().item()
        total   = valid.sum().item()
    else:
        correct = (preds == targets).sum().item()
        total   = targets.numel()
    return correct / total
