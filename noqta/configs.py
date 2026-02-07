import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ClustererConfig:
    # --- Rendering / Binarization ---
    max_px: int = 600                   # Longest side of page in pixels for speed
    invert: bool = False                # set True if content is white-on-black
    use_otsu: bool = True
    fixed_threshold: Optional[int] = None  # 0..255; overrides Otsu if set

    # --- frame removal ---
    sobel_kernel_size: int = 3
    rho: float = 1.0
    theta: float = np.pi / 180
    threshold: int = 100
    angle_tolerance_degree: float = 1.0
    max_line_gap: int = 5
    min_ratio_to_side: float = 0.9
    edge_thickness: int = 2

    # --- Smudge ---
    use_smudging: bool = True
    type_smudging: str = "edt"          # 'edt' or 'dilation'
    threshold_edt: int = 1              # pixels; max distance to black for EDT smudging
    dilate_radius_px: int = 2           # kernel radius in pixels (size = 2r+1)
    dilation_iterations: int = 1

    # --- Hierarchical clustering ---
    hier_distance_threshold: float = 6.0  # pixels; akin to “merge radius”
    hier_linkage: str = "single"          # 'single'|'complete'|'average'|'ward'

@dataclass
class ChunkerConfig:
    zoom_rate: float = 5.0              # how much larger the page size (to crop from) compared to page size of clusterer
    padding_low_px: int = 2             # extra padding around each cluster (in LOW-DPI pixels)
    save_format: str = "PNG"            # PNG or TIFF, etc.
    save_mode_1bit: bool = False        # if True, convert crops to 1-bit (useful for OCR)
    ensure_min_size_px: int = 1         # ensure width/height >= this many pixels after scaling

@dataclass
class SuppressorConfig:
    ratio_to_image_threshold: float = 0.6  # max fraction of page acceptable for height or width of a box
    overlap_threshold: float = 0.60  # fraction of the smaller box area required to merge
    clamp: Optional[Tuple[int, int]] = None  # (W, H) to clamp boxes within image bounds
    pad_union_px: int = 0  # extra pixels to pad the union when enlarging bigger box
    remove_duplicates: bool = True  # drop exact-duplicate boxes after processing
    min_area_ratio: float = 0.05 # percentage of og image area as minimum area of a box to not be considered large
    dist_ratio: float = 0.05 # percentage of maximum (width, height) of og image o be the minimum distance for two centers of bboxes to be considered close.