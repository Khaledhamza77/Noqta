from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ClustererConfig:
    # --- Rendering / Binarization ---
    dpi: int = 200                      # modest DPI as requested
    invert: bool = False                # set True if content is white-on-black
    use_otsu: bool = True
    fixed_threshold: Optional[int] = None  # 0..255; overrides Otsu if set

    # --- Dilation (smudge) ---
    use_dilation: bool = True
    dilate_radius_px: int = 2           # kernel radius in pixels (size = 2r+1)
    dilation_iterations: int = 1

    # --- Hierarchical clustering ---
    hier_distance_threshold: float = 6.0  # pixels; akin to “merge radius”
    hier_linkage: str = "single"          # 'single'|'complete'|'average'|'ward'

    # --- Plotting ---
    plot_point_size: float = 0.15         # smaller for dense pages
    random_state: int = 42

    # --- Safety / Performance ---
    max_points_warn: int = 3_000_000      # warn if too many black pixels

@dataclass
class ChunkerConfig:
    high_dpi: int = 600                 # DPI to render the high-res page
    low_dpi: Optional[int] = None       # DPI used for low-res clusters (optional)
    padding_low_px: int = 2             # extra padding around each cluster (in LOW-DPI pixels)
    save_format: str = "PNG"            # PNG or TIFF, etc.
    save_mode_1bit: bool = False        # if True, convert crops to 1-bit (useful for OCR)
    ensure_min_size_px: int = 1         # ensure width/height >= this many pixels after scaling


@dataclass
class SuppressorConfig:
    overlap_threshold: float = 0.60  # fraction of the smaller box area required to merge
    clamp: Optional[Tuple[int, int]] = None  # (W, H) to clamp boxes within image bounds
    pad_union_px: int = 0  # extra pixels to pad the union when enlarging bigger box
    remove_duplicates: bool = True  # drop exact-duplicate boxes after processing