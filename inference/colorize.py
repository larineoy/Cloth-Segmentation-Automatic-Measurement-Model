"""
Converts a class-index mask (H, W) → an RGB image (H, W, 3).
Palette matches configs/default.yaml.
"""

import numpy as np
from PIL import Image

# Default palette — override by passing your own dict
DEFAULT_PALETTE = {
    0: (75,  0,   130),   # background
    1: (70,  130, 180),   # top
    2: (64,  224, 208),   # bottom
    3: (255, 105, 180),   # dress
    4: (255, 215,   0),   # skin
    5: (34,  139,  34),   # hair
    6: (173, 255,  47),   # shoes
}


def colorize_mask(mask: np.ndarray,
                  palette: dict = DEFAULT_PALETTE) -> np.ndarray:
    """
    Args:
        mask   : (H, W) uint8 array of class indices
        palette: dict mapping class_index → (R, G, B)
    Returns:
        rgb: (H, W, 3) uint8 numpy array
    """
    h, w  = mask.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 palette: dict = DEFAULT_PALETTE, alpha: float = 0.5) -> np.ndarray:
    """
    Blends the colorized mask over the original image.
    Args:
        image: (H, W, 3) uint8 original RGB image
        mask : (H, W)    uint8 class-index mask
        alpha: mask opacity (0 = invisible, 1 = opaque)
    Returns:
        blended: (H, W, 3) uint8
    """
    colored = colorize_mask(mask, palette).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colored
    return blended.clip(0, 255).astype(np.uint8)


def save_result(mask: np.ndarray, out_path: str,
                palette: dict = DEFAULT_PALETTE):
    """Convenience: colorize and save to disk."""
    rgb = colorize_mask(mask, palette)
    Image.fromarray(rgb).save(out_path)
    print(f"Saved → {out_path}")
