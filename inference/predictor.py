"""
Single-image inference wrapper.
"""

import numpy as np
import torch
from PIL import Image

from data.transforms  import get_inference_transforms
from model.segmodel   import ClothSegModel


class Predictor:
    def __init__(self, checkpoint: str, num_classes: int = 7,
                 img_size: int = 512, device: str = "cpu"):
        self.device   = device
        self.img_size = img_size
        self.transform = get_inference_transforms(img_size)

        self.model = ClothSegModel(num_classes=num_classes, pretrained=False)
        self.model.load_checkpoint(checkpoint, device=device)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> np.ndarray:
        """
        Args:
            image_path: path to a .jpg / .png fashion image
        Returns:
            mask: (H, W) uint8 numpy array of class indices
                  at the ORIGINAL image resolution
        """
        img = np.array(Image.open(image_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        tensor = self.transform(image=img)["image"]          # (3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)         # (1, 3, H, W)

        logits = self.model(tensor)                          # (1, C, H, W)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize back to original resolution (nearest = no class bleeding)
        pred = np.array(
            Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST)
        )
        return pred
