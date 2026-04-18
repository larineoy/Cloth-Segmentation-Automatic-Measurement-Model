"""
EfficientNet-B4 feature extractor via timm.
Returns 4 multi-scale feature maps at strides 4, 8, 16, 32.
"""

import timm
import torch.nn as nn


class EfficientNetEncoder(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        # B4 channel sizes: [32, 56, 160, 448]
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        """Returns list [s1, s2, s3, s4] of feature maps, coarsest last."""
        return self.backbone(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
