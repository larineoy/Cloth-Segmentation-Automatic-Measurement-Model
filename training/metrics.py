"""
mIoU and pixel accuracy computed from prediction tensors.
"""

import torch
import numpy as np

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int) -> float:
    """
    Args:
        preds  : (B, H, W) integer predictions  (argmax of logits)
        targets: (B, H, W) integer ground-truth labels
    Returns:
        mean IoU over all classes (ignores classes with no GT pixels)
    """
    ious = []
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        inter    = (pred_c & target_c).sum()
        union    = (pred_c | target_c).sum()
        if union == 0:
            continue  # class not present — skip
        ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_acc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Overall pixel accuracy."""
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total
