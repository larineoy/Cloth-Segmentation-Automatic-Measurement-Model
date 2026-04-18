"""
Assembles encoder + decoder + segmentation head into one nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import EfficientNetEncoder
from model.decoder import UNetDecoder


class ClothSegModel(nn.Module):
    def __init__(self, num_classes: int = 7, backbone: str = "efficientnet_b4",
                 pretrained: bool = True):
        super().__init__()
        self.encoder = EfficientNetEncoder(backbone=backbone, pretrained=pretrained)
        self.decoder = UNetDecoder(encoder_channels=self.encoder.out_channels)
        self.head    = nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) normalised image tensor
        Returns:
            logits: (B, num_classes, H, W)  — raw, un-softmaxed
        """
        features = self.encoder(x)
        decoded  = self.decoder(features)
        return self.head(decoded)

    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_encoder(self):
        self.encoder.unfreeze()

    def load_checkpoint(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state["model"])
        print(f"Loaded checkpoint from {path}  (epoch {state.get('epoch', '?')})")
        return state.get("best_miou", 0.0)
