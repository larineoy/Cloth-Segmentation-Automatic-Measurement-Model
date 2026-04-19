from pathlib import Path
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def _load_mask_pixels(mask_pil: Image.Image) -> np.ndarray:
    """
    Load label image without destroying palette indices.
    ``convert('L')`` on paletted (P) PNGs uses luminance, not class / palette index,
    which produces arbitrary values (e.g. 30) that are not semantic labels.
    """
    if mask_pil.mode == "P":
        return np.asarray(mask_pil, dtype=np.int64)
    if mask_pil.mode in ("L", "1"):
        return np.asarray(mask_pil, dtype=np.int64)
    if mask_pil.mode == "RGBA":
        return np.asarray(mask_pil)[..., 0].astype(np.int64)
    return np.asarray(mask_pil.convert("L"), dtype=np.int64)


def _remap_mask_to_classes(
    mask: np.ndarray,
    raw_class_values: list[int],
    void_label: int,
) -> np.ndarray:
    """
    Map dataset-specific raw pixel values to training ids 0..num_classes-1.
    Any other value becomes void_label (typically ignored in the loss).
    """
    out = np.full(mask.shape, void_label, dtype=np.int64)
    for cls_id, raw in enumerate(raw_class_values):
        out[mask == raw] = cls_id
    return out


def _remap_mask_palette_cycle(
    mask: np.ndarray,
    offset: int,
    span: int,
    num_classes: int,
    void_label: int,
) -> np.ndarray:
    """
    Map each raw value in ``[offset, offset + span)`` to class ``(raw - offset) % num_classes``.
    Use when a tool emits many palette indices (e.g. 30–69) that cycle through the K classes.
    All other raw values stay ``void_label``.
    """
    out = np.full(mask.shape, void_label, dtype=np.int64)
    for raw_val in range(offset, offset + span):
        cls = (raw_val - offset) % num_classes
        out[mask == raw_val] = cls
    return out


def _apply_extra_raw_map(
    remapped: np.ndarray,
    raw: np.ndarray,
    extra: dict[int, int],
) -> np.ndarray:
    """After base remapping, assign extra raw palette values to class ids (overrides void)."""
    out = remapped.copy()
    for raw_val, cls_id in extra.items():
        out[raw == int(raw_val)] = int(cls_id)
    return out


def _remap_rgb_mask_nearest(
    rgb: np.ndarray,
    palette: np.ndarray,
    void_rgb: np.ndarray | None,
    void_label: int,
    *,
    distance: str = "rgb",
    void_if_min_sqdist_gt: float | None = None,
) -> np.ndarray:
    """
    Map each pixel's RGB to the nearest class color in ``palette`` (rows = class ids).
    Pixels exactly equal to ``void_rgb`` (e.g. white background) become ``void_label``.

    ``distance``: ``rgb`` (L2 in RGB 0–255) or ``lab`` (L2 in CIE Lab — better for similar hues).
    ``void_if_min_sqdist_gt``: if set, pixels whose **squared** distance to the nearest
    anchor exceeds this become ``void_label`` (drops ambiguous / anti-aliased pixels).
    """
    from preprocessing.colors import rgb_uint8_to_lab

    h, w, _ = rgb.shape
    flat_u8 = rgb.reshape(-1, 3).astype(np.uint8)
    pal = np.asarray(palette, dtype=np.float32)
    if void_rgb is not None:
        v = void_rgb.astype(np.float32).reshape(1, 3)
        flat_f = flat_u8.astype(np.float32)
        is_void = np.all(flat_f == v, axis=1)
    else:
        is_void = np.zeros(flat_u8.shape[0], dtype=bool)

    if distance == "lab":
        flat_x = rgb_uint8_to_lab(flat_u8)
        pal_x = rgb_uint8_to_lab(pal.reshape(-1, 3).astype(np.uint8))
        diff = flat_x[:, None, :] - pal_x[None, :, :]
    elif distance == "rgb":
        diff = flat_u8.astype(np.float32)[:, None, :] - pal[None, :, :]
    else:
        raise ValueError(f"distance must be 'rgb' or 'lab', got {distance!r}")

    d2 = (diff * diff).sum(axis=2)
    md = d2.min(axis=1)
    cls = np.argmin(d2, axis=1).astype(np.int64)
    cls[is_void] = int(void_label)
    if void_if_min_sqdist_gt is not None:
        thr = float(void_if_min_sqdist_gt)
        far = (md > thr) & (~is_void)
        cls[far] = int(void_label)
    return cls.reshape(h, w)


def _load_mask_rgb_hwc(mask_pil: Image.Image) -> np.ndarray:
    """RGB or RGBA label image: (H, W, 3) uint8."""
    arr = np.asarray(mask_pil)
    if arr.ndim == 2:
        raise ValueError(f"Expected RGB mask, got single channel (mode={mask_pil.mode})")
    return arr[..., :3].copy()


def probe_mask_encoding(data_root: str, train_split: str = "Train", seg_folder: str = "Segmented") -> str:
    """Return ``rgb_nearest`` if the first PNG in ``seg_folder`` is RGB/RGBA, else ``indexed``."""
    seg = Path(data_root) / train_split / seg_folder
    if not seg.is_dir():
        return "indexed"
    for p in sorted(seg.glob("*.png")):
        with Image.open(p) as im:
            if im.mode in ("RGB", "RGBA"):
                return "rgb_nearest"
            return "indexed"
    return "indexed"


class ClothSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "Train",
        transform=None,
        *,
        raw_class_values: list[int] | None = None,
        num_classes: int = 7,
        void_label: int = 255,
        extra_raw_class_map: dict[int, int] | None = None,
        cycle_offset: int | None = None,
        cycle_span: int | None = None,
        mask_encoding: str = "indexed",
        palette_rgb: np.ndarray | None = None,
        void_mask_rgb: list[int] | np.ndarray | None = None,
        rgb_distance: str = "lab",
        rgb_void_far_sqdist: float | None = None,
        img_folder: str = "Image",
        mask_folder: str = "Segmented",
        stem_suffix: str = ".jpg",       # extension appended when building image path from stem
        mask_name_style: str = "double", # "double" → stem.jpg.png | "single" → stem.png
    ):
        """
        Args:
            data_root : path to the top-level 'data/' folder
            split     : "Train" or "Test"  (case-sensitive, matches folder name)
            transform : albumentations pipeline from preprocessing.transforms
            raw_class_values : pixel value for each training class in order (0..K-1).
                Defaults to [0,1,...,num_classes-1]. Ignored when ``cycle_span`` is set.
            void_label : raw pixels not listed here are replaced with this (match training ignore_index).
            extra_raw_class_map : optional raw palette value -> class id (for IDs outside the main 7).
            cycle_offset, cycle_span : if span is set, use palette cycle remap on ``[offset, offset+span)``.
            mask_encoding : ``indexed`` (P/L grayscale) or ``rgb_nearest`` (RGB/RGB masks vs ``palette_rgb``).
            palette_rgb : (num_classes, 3) reference RGB for each class (from config ``data.palette``).
            void_mask_rgb : RGB treated as unlabeled (e.g. white ``[255,255,255]``) -> void_label.
            rgb_distance : ``rgb`` or ``lab`` (perceptual) for nearest-color mapping.
            rgb_void_far_sqdist : if set, pixels too far from every anchor become void (reduces AA noise).
            img_folder  : subfolder name for images inside split dir (default "Image", alt "Image2").
            mask_folder : subfolder name for masks inside split dir (default "Segmented", alt "Segmented2").
            stem_suffix : image file extension used when reconstructing path from stem (default ".jpg").
            mask_name_style : "double" means mask is <stem><stem_suffix>.png (e.g. 189.jpg.png);
                              "single" means mask is <stem>.png (e.g. 189.png). Use "single" for Image2/Segmented2.
        """
        self.img_dir  = Path(data_root) / split / img_folder
        self.mask_dir = Path(data_root) / split / mask_folder
        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"Image folder does not exist: {self.img_dir}\n"
                f"  (split={split!r}, img_folder={img_folder!r}). "
                "For Image2/Segmented2, create e.g. data/Test/Image2 with paired masks in Segmented2."
            )
        self.stem_suffix = stem_suffix
        self.mask_name_style = mask_name_style
        self.transform = transform
        self.num_classes = int(num_classes)
        self.mask_encoding = mask_encoding
        if mask_encoding == "rgb_nearest":
            if palette_rgb is None:
                raise ValueError("palette_rgb is required when mask_encoding='rgb_nearest'")
            self.palette_rgb = np.asarray(palette_rgb, dtype=np.float32)
            if self.palette_rgb.shape != (self.num_classes, 3):
                raise ValueError(
                    f"palette_rgb must be shape ({self.num_classes}, 3), got {self.palette_rgb.shape}"
                )
            self.void_mask_rgb = (
                np.asarray(void_mask_rgb, dtype=np.float32).reshape(3)
                if void_mask_rgb is not None
                else None
            )
            self.cycle_span = None
            self.cycle_offset = None
            self.raw_class_values = list(range(num_classes))
            self.rgb_distance = str(rgb_distance)
            self.rgb_void_far_sqdist = (
                float(rgb_void_far_sqdist)
                if rgb_void_far_sqdist is not None
                else None
            )
        else:
            self.palette_rgb = None
            self.void_mask_rgb = None
            self.cycle_span = int(cycle_span) if cycle_span is not None else None
            self.cycle_offset = int(cycle_offset) if cycle_offset is not None else None
            if self.cycle_span is not None:
                if self.cycle_offset is None:
                    raise ValueError("cycle_offset is required when cycle_span is set")
                self.raw_class_values: list[int] = []
            else:
                if raw_class_values is None:
                    raw_class_values = list(range(num_classes))
                self.raw_class_values = [int(x) for x in raw_class_values]
            self.rgb_distance = "rgb"
            self.rgb_void_far_sqdist = None
        self.void_label = int(void_label)
        self.extra_raw_class_map = dict(extra_raw_class_map or {})
        for _rv, cid in self.extra_raw_class_map.items():
            if not (0 <= int(cid) < self.num_classes):
                raise ValueError(
                    f"extra_raw_class_map class id {cid} must be in [0, {self.num_classes - 1}]"
                )

        # Collect all image stems that have a matching segmented file
        def _mask_path(img_path: Path) -> Path:
            if self.mask_name_style == "single":
                return self.mask_dir / f"{img_path.stem}.png"
            else:  # "double": e.g. 189.jpg -> 189.jpg.png
                return self.mask_dir / f"{img_path.name}.png"

        self.stems = sorted([
            p.stem
            for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and _mask_path(p).exists()
        ])

        if not self.stems:
            raise FileNotFoundError(
                f"No matched image/mask pairs found.\n"
                f"  Image dir : {self.img_dir}\n"
                f"  Mask dir  : {self.mask_dir}\n"
                f"  mask_name_style={self.mask_name_style!r}\n"
                + (
                    "  Expected mask format: <stem>.png  e.g. 189.png"
                    if self.mask_name_style == "single"
                    else "  Expected mask format: <image_filename>.png  e.g. 189.jpg.png"
                )
            )

    def _resolve_paths(self, stem: str) -> tuple[Path, Path]:
        img_path = self.img_dir / f"{stem}{self.stem_suffix}"
        if self.mask_name_style == "single":
            mask_path = self.mask_dir / f"{stem}.png"
        else:
            mask_path = self.mask_dir / f"{stem}{self.stem_suffix}.png"
        return img_path, mask_path

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path, mask_path = self._resolve_paths(stem)

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)
        if mask_pil.size != img_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)

        img = np.array(img_pil)
        if self.mask_encoding == "rgb_nearest":
            rgb = _load_mask_rgb_hwc(mask_pil)
            mask = _remap_rgb_mask_nearest(
                rgb,
                self.palette_rgb,
                self.void_mask_rgb,
                self.void_label,
                distance=self.rgb_distance,
                void_if_min_sqdist_gt=self.rgb_void_far_sqdist,
            )
        else:
            raw = _load_mask_pixels(mask_pil)
            if self.cycle_span is not None:
                mask = _remap_mask_palette_cycle(
                    raw,
                    self.cycle_offset,
                    self.cycle_span,
                    self.num_classes,
                    self.void_label,
                )
            else:
                mask = _remap_mask_to_classes(raw, self.raw_class_values, self.void_label)
            if self.extra_raw_class_map:
                mask = _apply_extra_raw_map(mask, raw, self.extra_raw_class_map)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]   # tensor (3, H, W) float32 normalised
            mask = out["mask"]    # tensor (H, W)    int64

        return img, mask

    def raw_and_remapped_mask(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """For diagnostics: raw pixel values vs class indices after remapping."""
        stem = self.stems[idx]
        img_path, mask_path = self._resolve_paths(stem)
        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)
        if mask_pil.size != img_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
        if self.mask_encoding == "rgb_nearest":
            rgb = _load_mask_rgb_hwc(mask_pil)
            remapped = _remap_rgb_mask_nearest(
                rgb,
                self.palette_rgb,
                self.void_mask_rgb,
                self.void_label,
                distance=self.rgb_distance,
                void_if_min_sqdist_gt=self.rgb_void_far_sqdist,
            )
            return None, remapped
        raw = _load_mask_pixels(mask_pil)
        if self.cycle_span is not None:
            remapped = _remap_mask_palette_cycle(
                raw,
                self.cycle_offset,
                self.cycle_span,
                self.num_classes,
                self.void_label,
            )
        else:
            remapped = _remap_mask_to_classes(raw, self.raw_class_values, self.void_label)
        if self.extra_raw_class_map:
            remapped = _apply_extra_raw_map(remapped, raw, self.extra_raw_class_map)
        return raw, remapped


def resolve_dataset_kwargs(cfg: dict, void_label: int | None, seg_folder: str = "Segmented") -> tuple[dict, str]:
    """
    Shared config for ``ClothSegDataset`` (train/val/counts). Returns ``(ds_kw, mask_encoding)``.
    """
    nc = int(cfg["num_classes"])
    enc_cfg = cfg.get("mask_encoding", "auto")
    if enc_cfg == "auto":
        enc = probe_mask_encoding(cfg["root"], cfg.get("train_split", "Train"), seg_folder)
    elif enc_cfg in ("rgb_nearest", "indexed"):
        enc = enc_cfg
    else:
        raise ValueError(
            f"data.mask_encoding must be auto, rgb_nearest, or indexed, got {enc_cfg!r}"
        )

    raw = cfg.get("raw_class_values")
    off = cfg.get("label_id_offset")
    span_cfg = cfg.get("label_palette_span")

    cycle_offset: int | None = None
    cycle_span: int | None = None
    palette_arr: np.ndarray | None = None
    void_mask_rgb_cfg = cfg.get("void_mask_rgb")
    if void_mask_rgb_cfg is not None:
        void_mask_rgb_cfg = [int(x) for x in void_mask_rgb_cfg]

    if enc == "rgb_nearest":
        pal = cfg.get("palette")
        if not pal or len(pal) < nc:
            raise ValueError(
                "RGB label masks require `data.palette` with one [R,G,B] per class index 0..num_classes-1."
            )
        palette_arr = np.array([list(pal[i]) for i in range(nc)], dtype=np.float32)
        raw = None
        if off is not None or span_cfg is not None:
            print(
                "Note: ignoring `label_id_offset` / `label_palette_span` "
                "(using RGB nearest-color mapping vs `data.palette`)."
            )
    else:
        if off is not None and raw is not None:
            raise ValueError(
                "Set only one of `data.label_id_offset` or `data.raw_class_values`, not both."
            )
        if span_cfg is not None and raw is not None:
            raise ValueError(
                "Set only one of `data.label_palette_span` or `data.raw_class_values`, not both."
            )

        if span_cfg is not None:
            if off is None:
                raise ValueError(
                    "`data.label_palette_span` requires `data.label_id_offset` (start of palette range)."
                )
            cycle_span = int(span_cfg)
            cycle_offset = int(off)
            raw = None
        elif off is not None:
            off = int(off)
            raw = [off + k for k in range(nc)]
        elif raw is not None:
            raw = [int(x) for x in raw]
            if len(raw) != nc:
                raise ValueError(
                    f"data.raw_class_values length ({len(raw)}) must match "
                    f"data.num_classes ({nc})"
                )

    vl = 255 if void_label is None else int(void_label)

    extras = None
    _ex = cfg.get("extra_raw_class_map")
    if _ex and enc != "rgb_nearest":
        extras = {int(k): int(v) for k, v in _ex.items()}
        for cid in extras.values():
            if not (0 <= int(cid) < nc):
                raise ValueError(
                    f"extra_raw_class_map targets class {cid}; must be in [0, {nc - 1}]"
                )
    elif _ex and enc == "rgb_nearest":
        print("Note: ignoring `extra_raw_class_map` for RGB label masks.")

    _vf = cfg.get("rgb_void_far_sqdist")
    ds_kw = dict(
        raw_class_values=raw,
        num_classes=nc,
        void_label=vl,
        extra_raw_class_map=extras,
        cycle_offset=cycle_offset,
        cycle_span=cycle_span,
        mask_encoding=enc,
        palette_rgb=palette_arr,
        void_mask_rgb=void_mask_rgb_cfg,
        rgb_distance=str(cfg.get("rgb_distance", "lab")),
        rgb_void_far_sqdist=float(_vf) if _vf is not None else None,
    )
    return ds_kw, enc


def count_train_class_pixels(cfg: dict, void_label: int | None) -> np.ndarray:
    """Pixel counts per class (excluding void) over the Train split, no augmentation.
    Mirrors ``build_dataloaders`` / ``cfg.sources`` (primary, secondary, or both).
    """
    def _count_ds(ds: ClothSegDataset) -> np.ndarray:
        counts = np.zeros(ds.num_classes, dtype=np.int64)
        vlab = int(ds.void_label)
        for i in range(len(ds)):
            _img, m = ds[i]
            if hasattr(m, "numpy"):
                m = m.numpy()
            m = np.asarray(m)
            valid = m != vlab
            if valid.any():
                bc = np.bincount(m[valid].ravel().astype(np.int64), minlength=ds.num_classes)
                counts += bc[: ds.num_classes]
        return counts

    cfg2 = cfg.get("data2")
    default_sources = "both" if cfg2 else "primary"
    sources = cfg.get("sources", default_sources)
    if sources in ("secondary", "both") and cfg2 is None:
        raise ValueError("count_train_class_pixels: sources includes secondary but data2 is missing.")

    split = cfg.get("train_split", "Train")
    counts = np.zeros(int(cfg["num_classes"]), dtype=np.int64)

    if sources in ("primary", "both"):
        ds_kw, _ = resolve_dataset_kwargs(cfg, void_label, seg_folder="Segmented")
        ds1 = ClothSegDataset(
            data_root=cfg["root"], split=split, transform=None,
            img_folder="Image", mask_folder="Segmented",
            mask_name_style="double", **ds_kw,
        )
        counts += _count_ds(ds1)

    if sources in ("secondary", "both"):
        merged = {**cfg, **cfg2}
        ds_kw2, _ = resolve_dataset_kwargs(merged, void_label, seg_folder="Segmented2")
        ds2 = ClothSegDataset(
            data_root=merged["root"], split=split, transform=None,
            img_folder="Image2", mask_folder="Segmented2",
            mask_name_style="single", **ds_kw2,
        )
        counts += _count_ds(ds2)

    return counts


def class_weights_inverse_freq(counts: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Normalize so mean weight is 1.0 (helps CE with imbalanced regions)."""
    c = np.maximum(counts.astype(np.float64), eps)
    w = 1.0 / c
    w *= len(w) / np.sum(w)
    return w.astype(np.float32)


def _audit_train_masks(train_ds: ClothSegDataset) -> None:
    """Fail fast with a clear hint if remapping turns every pixel into void."""
    if len(train_ds) == 0:
        return
    n_check = min(16, len(train_ds))
    any_labeled = False
    first_raw: np.ndarray | None = None
    for i in range(n_check):
        raw, remapped = train_ds.raw_and_remapped_mask(i)
        if first_raw is None and raw is not None:
            first_raw = raw
        if (remapped != train_ds.void_label).any():
            any_labeled = True
            break
    if any_labeled:
        return
    u = sorted(np.unique(first_raw).tolist()) if first_raw is not None else []
    hint_rgb = (
        "\nRGB masks: check `data.palette` matches your label colors and `void_mask_rgb` "
        "(e.g. [255,255,255] for white background)."
    )
    raise ValueError(
        "After remapping, the first checked masks contain no training labels — only "
        f"void ({train_ds.void_label}). Cross-entropy would be NaN.\n"
        + (
            f"Indexed mask sample values: {u[:40]}"
            + (" ..." if len(u) > 40 else "")
            + "\nFix: set `data.label_id_offset` / `label_palette_span` or `raw_class_values`."
            if u
            else hint_rgb
        )
    )


def build_dataloaders(cfg: dict, void_label: int | None = None):
    """
    Builds Train and Test DataLoaders from config dict.

    Expected cfg keys:
        root        : path to data/ folder
        img_size    : int
        batch_size  : int
        num_workers : int
        num_classes : int
        sources     : ``primary`` | ``secondary`` | ``both``. Default: ``both`` if ``data2`` is set, else ``primary``.
        raw_class_values  : optional list of length num_classes (mask pixel → class id)
        label_id_offset   : optional int; if set, raw value (offset + k) → class k
        data2             : sub-config for Image2/Segmented2 (required for ``secondary`` or ``both``)

    data2 sub-config keys (merged with top-level ``data``; same root, optional palette/void overrides):
        palette           : list of [R,G,B] per class for the second source's masks
        mask_encoding     : usually "rgb_nearest"
        void_mask_rgb     : e.g. [0,0,0] for black border in second source
        (other keys inherited from top-level cfg if not overridden)
    """
    from preprocessing.transforms import get_train_transforms, get_val_transforms

    cfg2 = cfg.get("data2")
    default_sources = "both" if cfg2 else "primary"
    sources = cfg.get("sources", default_sources)
    if sources not in ("both", "primary", "secondary"):
        raise ValueError(f"cfg.sources must be 'both', 'primary', or 'secondary', got {sources!r}")
    if sources in ("secondary", "both") and cfg2 is None:
        raise ValueError(f"cfg.sources={sources!r} requires a 'data2' sub-config in cfg.")

    merged_cfg2 = {**cfg, **cfg2} if cfg2 else None
    ds_kw1 = enc1 = None
    ds_kw2 = enc2 = None
    if sources in ("primary", "both"):
        ds_kw1, enc1 = resolve_dataset_kwargs(cfg, void_label, seg_folder="Segmented")
    if sources in ("secondary", "both"):
        ds_kw2, enc2 = resolve_dataset_kwargs(merged_cfg2, void_label, seg_folder="Segmented2")

    vl = 255 if void_label is None else int(void_label)
    train_tfm = get_train_transforms(cfg["img_size"], mask_void_label=vl)
    val_tfm   = get_val_transforms(cfg["img_size"])
    split_tr  = cfg.get("train_split", "Train")
    split_te  = cfg.get("test_split",  "Test")
    if sources == "secondary":
        test_img2 = Path(cfg["root"]) / split_te / "Image2"
        if not test_img2.is_dir():
            held = split_te
            warnings.warn(
                f"Missing folder {test_img2} — using train_split={split_tr!r} for the val/test "
                f"loader (metrics are not a true hold-out). Add data/{held}/Image2 and "
                f"data/{held}/Segmented2 when you have a held-out set.",
                UserWarning,
                stacklevel=2,
            )
            split_te = split_tr

    train_parts = []

    # --- Primary: Image / Segmented (mask naming: 189.jpg → 189.jpg.png) ---
    if sources in ("primary", "both"):
        train_ds1 = ClothSegDataset(
            data_root=cfg["root"], split=split_tr, transform=train_tfm,
            img_folder="Image", mask_folder="Segmented",
            mask_name_style="double", **ds_kw1,
        )
        _audit_train_masks(train_ds1)
        train_parts.append(train_ds1)
        print(f"Primary   (Image/Segmented):   {len(train_ds1)} samples  |  encoding: {enc1}")

    # --- Secondary: Image2 / Segmented2 (mask naming: stem.jpg → stem.png) ---
    if sources in ("secondary", "both"):
        train_ds2 = ClothSegDataset(
            data_root=merged_cfg2["root"], split=split_tr, transform=train_tfm,
            img_folder="Image2", mask_folder="Segmented2",
            mask_name_style="single", **ds_kw2,
        )
        _audit_train_masks(train_ds2)
        train_parts.append(train_ds2)
        print(f"Secondary (Image2/Segmented2): {len(train_ds2)} samples  |  encoding: {enc2}")

    # Val/test: same layout as training when using one source only
    if sources == "secondary":
        test_ds = ClothSegDataset(
            data_root=merged_cfg2["root"], split=split_te, transform=val_tfm,
            img_folder="Image2", mask_folder="Segmented2",
            mask_name_style="single", **ds_kw2,
        )
    else:
        test_ds = ClothSegDataset(
            data_root=cfg["root"], split=split_te, transform=val_tfm,
            img_folder="Image", mask_folder="Segmented",
            mask_name_style="double", **ds_kw1,
        )

    train_combined = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_combined,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
    )

    print(f"Total train: {len(train_combined)}  |  Test: {len(test_ds)}")
    return train_loader, test_loader

# from pathlib import Path

# import numpy as np
