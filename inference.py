"""
Loads best.pth, runs segmentation on a camera-captured image, saves the
coloured segmentation overlay, then calculates cloth lengths from the
user's height input.

Usage (standalone):
    python inference.py --image path/to/photo.jpg --checkpoint checkpoints/best.pth

Or import and call run_inference() from your camera pipeline:
    from inference import run_inference
    results = run_inference(image=np_bgr_array, checkpoint="checkpoints/best.pth")
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# 1.  Class / colour definitions  — edit to match your training config
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "background",   # 0
    "top",          # 1
    "bottom",       # 2
    "hair",         # 3
    "skin",         # 4
    "arms",         # 5
    "shoes",        # 6
]

# Which classes are actual garments we want to measure
GARMENT_CLASSES = {1: "top", 2: "bottom"}

# Visualisation colours (BGR for OpenCV) — one per class
CLASS_COLORS_BGR = [
    (89,   5,  69),   # 0 background  — dark purple
    (74, 212, 134),   # 1 top         — yellow-green
    (139, 152,  30),  # 2 bottom      — teal
    (141, 104,  48),  # 3 hair        — steel blue
    (102, 197,  83),  # 4 skin        — green
    (36,  230, 253),  # 5 arms        — yellow
    (113, 190,  68),  # 6 shoes       — green
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 512   # must match training img_size


# ---------------------------------------------------------------------------
# 2.  Model loading  — swap the body of _build_model() for your architecture
# ---------------------------------------------------------------------------

def _build_model(num_classes: int, checkpoint_path: str, device: torch.device):
    """
    Load model architecture and restore weights from checkpoint.

    Edit this function to match whatever architecture you trained.
    The checkpoint is expected to contain either:
      - a raw state_dict, or
      - a dict with a "model_state_dict" / "state_dict" key.
    """
    # ---- Replace this block with your actual model ----
    # Example: torchvision DeepLabV3+ with ResNet-50 backbone
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(num_classes=num_classes, weights=None)
    except Exception as e:
        raise ImportError(
            "Could not build the default model (DeepLabV3-ResNet50). "
            "Edit _build_model() in inference.py to match your architecture."
        ) from e
    # ---------------------------------------------------

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle various checkpoint formats
    if isinstance(ckpt, dict):
        sd = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt          # assume the whole dict is the state_dict
        )
    else:
        sd = ckpt

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"[inference] Loaded checkpoint: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# 3.  Preprocessing (mirrors get_inference_transforms but without albumentations
#     dependency at runtime — change if you prefer to keep albumentations)
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(img_bgr: np.ndarray, img_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    BGR uint8 → normalised (1, 3, H, W) float32 tensor.
    Returns tensor and original (H, W) for rescaling the mask back.
    """
    orig_hw = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return x, orig_hw


# ---------------------------------------------------------------------------
# 4.  Segmentation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def segment(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Run model on one BGR image.
    Returns an int64 mask of shape (orig_H, orig_W) with class ids.
    """
    x, orig_hw = _preprocess(img_bgr, img_size)
    x = x.to(device)
    out = model(x)

    # Support both plain tensor output and torchvision dict output
    if isinstance(out, dict):
        logits = out["out"]          # torchvision segmentation models
    else:
        logits = out                 # plain (B, C, H, W)

    # Upsample back to original resolution
    logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return mask


# ---------------------------------------------------------------------------
# 5.  Visualisation
# ---------------------------------------------------------------------------

def colorise_mask(mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Convert integer class mask → solid-colour BGR image (uint8).
    """
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS_BGR):
        canvas[mask == cls_id] = color
    return canvas


def overlay_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Blend original image with colour mask (alpha = mask opacity).
    """
    colour = colorise_mask(mask)
    colour_resized = cv2.resize(colour, (img_bgr.shape[1], img_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, colour_resized, alpha, 0)
    return blended


# ---------------------------------------------------------------------------
# 6.  Measurement
# ---------------------------------------------------------------------------

def _person_pixel_height(mask: np.ndarray) -> tuple[int, int, int]:
    """
    Find the vertical pixel span of the whole person (all non-background pixels).
    Returns (top_row, bottom_row, span_px).  span_px = 0 if no person found.
    """
    person_pixels = mask > 0
    rows = np.where(person_pixels.any(axis=1))[0]
    if len(rows) == 0:
        return 0, 0, 0
    return int(rows[0]), int(rows[-1]), int(rows[-1] - rows[0] + 1)


def _largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Given a boolean 2-D mask, return a boolean mask containing only the
    largest connected component (8-connectivity).
    This filters out small disconnected blobs (e.g. a bag labelled as pants).
    Falls back to the original mask if cv2 is unavailable.
    """
    bw = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        return binary_mask  # nothing or only background
    # stats[0] is background — skip it; find largest foreground component
    fg_sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(fg_sizes)) + 1   # +1 because we skipped bg
    return labels == largest_label


def measure_garments(
    mask: np.ndarray,
    person_height_cm: float,
) -> dict[str, dict]:
    """
    For each garment class:
      1. Isolate the LARGEST connected blob of that class (drops stray pixels
         like a bag that shares the same colour as pants).
      2. Measure its vertical span (length) and horizontal span (width) in px.
      3. Scale to cm using:
             cm_per_px = person_height_cm / person_pixel_height
         where person_pixel_height = top of highest non-bg pixel
                                     → bottom of lowest non-bg pixel.

    Returns:
        {
          class_name: {
            "length_cm":    float,   # top-to-bottom of garment blob
            "width_cm":     float,   # left-to-right of garment blob
            "pixel_height": int,
            "pixel_width":  int,
            "top_row":      int,     # absolute row in original mask
            "bottom_row":   int,
            "left_col":     int,
            "right_col":    int,
          }
        }
    """
    top_row, bottom_row, person_px_h = _person_pixel_height(mask)
    if person_px_h == 0:
        print("[measure] Warning: no person detected in mask; cannot scale.")
        return {}

    cm_per_px = person_height_cm / person_px_h
    print(f"[measure] Person pixel height: {person_px_h}px  →  "
          f"scale: {cm_per_px:.4f} cm/px  (rows {top_row}–{bottom_row})")

    results = {}

    for cls_id, cls_name in GARMENT_CLASSES.items():
        raw_region = mask == cls_id
        if not raw_region.any():
            continue

        # Keep only the main garment blob — drops bags, accessories, etc.
        region = _largest_connected_component(raw_region)
        if not region.any():
            region = raw_region   # fallback: use everything

        rows = np.where(region.any(axis=1))[0]
        cols = np.where(region.any(axis=0))[0]

        r_top,  r_bot = int(rows[0]),  int(rows[-1])
        c_left, c_right = int(cols[0]), int(cols[-1])
        px_h = r_bot - r_top + 1
        px_w = c_right - c_left + 1

        results[cls_name] = {
            "length_cm":    round(px_h * cm_per_px, 1),
            "width_cm":     round(px_w * cm_per_px, 1),
            "pixel_height": px_h,
            "pixel_width":  px_w,
            "top_row":      r_top,
            "bottom_row":   r_bot,
            "left_col":     c_left,
            "right_col":    c_right,
        }

    return results


def annotate_measurements(
    vis: np.ndarray,
    mask: np.ndarray,
    measurements: dict[str, dict],
) -> np.ndarray:
    """
    Draw bounding boxes and measurement labels on the visualisation image.
    Uses the pre-computed bounding box coords stored in measurements dict.
    """
    out = vis.copy()
    cls_id_lookup = {v: k for k, v in GARMENT_CLASSES.items()}

    for cls_name, info in measurements.items():
        cls_id = cls_id_lookup[cls_name]
        color = CLASS_COLORS_BGR[cls_id]

        y1 = info["top_row"]
        y2 = info["bottom_row"]
        x1 = info["left_col"]
        x2 = info["right_col"]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Arrow showing the length dimension on the left edge
        mid_x = x1 - 12
        cv2.arrowedLine(out, (mid_x, y1), (mid_x, y2), color, 2, tipLength=0.05)

        label = (
            f"{cls_name}: "
            f"len={info['length_cm']}cm  "
            f"w={info['width_cm']}cm"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# 7.  Top-level entry point (called by camera pipeline or CLI)
# ---------------------------------------------------------------------------

def run_inference(
    image: np.ndarray,
    checkpoint: str = "checkpoints/best.pth",
    output_path: str | None = None,
    person_height_cm: float | None = None,
    img_size: int = IMG_SIZE,
    device_str: str = "auto",
) -> dict:
    """
    Full pipeline: segment → visualise → measure.

    Args:
        image            : BGR uint8 numpy array from camera / cv2.imread
        checkpoint       : path to best.pth
        output_path      : where to save the annotated image (None = don't save)
        person_height_cm : if provided, skips the interactive height prompt
        img_size         : must match training resolution
        device_str       : "auto" | "cpu" | "cuda" | "mps"

    Returns:
        {
          "mask":         np.ndarray (H, W) int64,
          "vis":          np.ndarray (H, W, 3) BGR uint8  — overlay,
          "annotated":    np.ndarray (H, W, 3) BGR uint8  — overlay + boxes,
          "measurements": dict  (empty if height not provided),
          "output_path":  str | None,
        }
    """
    # --- device ---
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"[inference] Using device: {device}")

    # --- model ---
    model = _build_model(NUM_CLASSES, checkpoint, device)

    # --- segment ---
    mask = segment(model, image, device, img_size)

    # --- visualise ---
    vis = overlay_mask(image, mask, alpha=0.55)

    # --- height prompt ---
    if person_height_cm is None:
        person_height_cm = _ask_height()

    # --- measure ---
    measurements: dict = {}
    if person_height_cm is not None and person_height_cm > 0:
        measurements = measure_garments(mask, person_height_cm)
        _print_measurements(measurements)

    # --- annotate ---
    annotated = annotate_measurements(vis, mask, measurements) if measurements else vis

    # --- save ---
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"[inference] Saved annotated image → {output_path}")

    return {
        "mask":         mask,
        "vis":          vis,
        "annotated":    annotated,
        "measurements": measurements,
        "output_path":  output_path,
    }


# ---------------------------------------------------------------------------
# 8.  Helper I/O
# ---------------------------------------------------------------------------

def _ask_height() -> float | None:
    """Prompt user for height in cm (or feet+inches). Returns cm float or None."""
    print("\n─── Height Input ───────────────────────────────────────")
    print("Enter your height so we can calculate garment lengths.")
    print("Examples:  '175'  or  '175cm'  or  5'9\"  or  5ft9in")
    raw = input("Your height: ").strip()
    if not raw:
        print("[measure] No height entered — skipping measurements.")
        return None
    return _parse_height(raw)


def _parse_height(raw: str) -> float | None:
    """
    Parse height string to centimetres.
    Accepts:  175  |  175cm  |  5'9  |  5'9"  |  5ft9in  |  1.75m
    """
    import re
    raw = raw.lower().replace('"', '').replace('\u2019', "'")

    # metres: 1.75m
    m = re.fullmatch(r"(\d+\.?\d*)\s*m", raw)
    if m:
        return float(m.group(1)) * 100

    # centimetres: 175 or 175cm
    m = re.fullmatch(r"(\d+\.?\d*)\s*(?:cm)?", raw)
    if m:
        val = float(m.group(1))
        if val > 100:          # already cm
            return val
        # ambiguous small number — ask again
        print(f"[parse] {val} looks too small for cm — did you mean {val*100:.0f} cm?")
        confirm = input("Enter height again in cm: ").strip()
        return _parse_height(confirm)

    # feet & inches: 5'9  or  5ft9  or  5ft9in
    m = re.fullmatch(r"(\d+)\s*(?:ft|')?\s*(\d+)\s*(?:in|')?", raw)
    if m:
        ft, inch = int(m.group(1)), int(m.group(2))
        return round((ft * 12 + inch) * 2.54, 1)

    # feet only: 6ft or 6'
    m = re.fullmatch(r"(\d+)\s*(?:ft|')", raw)
    if m:
        return round(int(m.group(1)) * 30.48, 1)

    print(f"[parse] Could not understand '{raw}'. Skipping measurements.")
    return None


def _print_measurements(measurements: dict) -> None:
    print("\n─── Garment Measurements ───────────────────────────────")
    for name, info in measurements.items():
        print(f"  {name.capitalize():<10}  "
              f"length = {info['length_cm']:>6.1f} cm   "
              f"width  = {info['width_cm']:>6.1f} cm")
    print("────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Cloth segmentation inference + measurement")
    p.add_argument("--image",      required=True,               help="Path to input image")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--output",     default="output/result.jpg", help="Where to save annotated image")
    p.add_argument("--height",     type=float, default=None,    help="Person height in cm (skips prompt)")
    p.add_argument("--img-size",   type=int,   default=IMG_SIZE)
    p.add_argument("--device",     default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"[error] Could not read image: {args.image}")

    run_inference(
        image=img,
        checkpoint=args.checkpoint,
        output_path=args.output,
        person_height_cm=args.height,
        img_size=args.img_size,
        device_str=args.device,
    )
