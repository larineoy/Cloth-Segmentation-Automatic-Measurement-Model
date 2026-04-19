"""
sRGB (0–255) -> CIE Lab, vectorized. Used for perceptually better nearest-color label mapping.
"""

import numpy as np


def _srgb_channel_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Args:
        rgb: (..., 3) uint8 or float in 0–255
    Returns:
        Lab with same leading dimensions, approximate D65
    """
    x = np.asarray(rgb, dtype=np.float64)
    if x.max() > 1.5:
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
    else:
        r, g, b = x[..., 0] * 255.0, x[..., 1] * 255.0, x[..., 2] * 255.0

    R = _srgb_channel_to_linear(r)
    G = _srgb_channel_to_linear(g)
    B = _srgb_channel_to_linear(b)

    # sRGB -> XYZ (D65)
    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041

    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x_r = X / xn
    y_r = Y / yn
    z_r = Z / zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x_r), f(y_r), f(z_r)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b2 = 200.0 * (fy - fz)
    return np.stack([L, a, b2], axis=-1)
