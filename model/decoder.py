"""
UNet-style decoder with skip connections.
Accepts the 4-scale feature list from EfficientNetEncoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNetBlock(nn.Module):
    """Upsample × 2  →  concat skip  →  2 × ConvBnRelu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims mismatch (odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels: list):
        """
        Args:
            encoder_channels: list of 4 ints from EfficientNetEncoder.out_channels
                               e.g. [32, 56, 160, 448]
        """
        super().__init__()
        ch = encoder_channels
        self.dec4 = UNetBlock(ch[3], ch[2], 256)  # 448  → up + skip 160
        self.dec3 = UNetBlock(256,   ch[1], 128)  # 256  → up + skip 56
        self.dec2 = UNetBlock(128,   ch[0],  64)  # 128  → up + skip 32
        self.dec1 = nn.Sequential(
            ConvBnRelu(64, 32),
            ConvBnRelu(32, 32),
        )
        self.out_channels = 32

    def forward(self, features: list):
        """
        Args:
            features: [s1, s2, s3, s4] from encoder  (finest → coarsest)
        Returns:
            Tensor (B, 32, H/4, W/4) — will be 4× upsampled in segmodel
        """
        s1, s2, s3, s4 = features
        x = self.dec4(s4, s3)
        x = self.dec3(x,  s2)
        x = self.dec2(x,  s1)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.dec1(x)
