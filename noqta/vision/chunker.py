from typing import Any, Dict, List, Optional, Tuple
from ..configs import ChunkerConfig
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os

class Chunker:
    """
    Given cluster points/labels derived from a LOW-DPI image of a PDF page,
    re-render the SAME page at HIGH-DPI and crop per-cluster regions by scaling
    low-DPI boxes up to high-DPI coordinates.

    Usage flow:
      - You already ran Clusterer at low DPI and obtained:
          * points_xy : np.ndarray of shape (N,2) in low-DPI pixel coordinates
          * labels    : np.ndarray of shape (N,), >=0 are clusters; -1 ignored
          * low_size  : (w_low, h_low) of the low-DPI image used for points
      - Chunker renders the page at high DPI and crops mapped boxes.
    """

    def __init__(self, cfg: ChunkerConfig):
        self.cfg = cfg

    # --------------------- Render high-DPI page ---------------------

    def _render_page_high(self, page: fitz.Page) -> Image.Image:
        scale = self.cfg.high_dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        # Render in GRAY to reduce memory; keep alpha=False for deterministic size
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        return img

    # --------------------- Compute boxes from clusters ---------------------

    @staticmethod
    def _boxes_from_labels(
        points_xy_low: np.ndarray,
        labels: np.ndarray,
        padding_low_px: int,
        low_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of low-DPI boxes (x0,y0,x1,y1), padded and clamped to low image bounds.
        """
        w_low, h_low = low_size; img_area = w_low * h_low
        boxes: List[Tuple[int, int, int, int]] = []
        if points_xy_low.size == 0 or labels.size == 0:
            return boxes

        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return boxes

        for lb in np.unique(labels[valid_mask]):
            pts = points_xy_low[labels == lb]
            if pts.size == 0:
                continue
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
            # pad in LOW-DPI pixels
            x0 -= padding_low_px
            y0 -= padding_low_px
            x1 += padding_low_px + 1
            y1 += padding_low_px + 1
            # clamp to low image
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(w_low, x1); y1 = min(h_low, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            area = abs(x0 - x1) * abs(y0 - y1)
            if area < 0.6 * img_area:
                boxes.append((x0, y0, x1, y1))
        return boxes

    # --------------------- Scale low→high DPI boxes ---------------------

    @staticmethod
    def _scale_boxes_low_to_high(
        boxes_low: List[Tuple[int, int, int, int]],
        low_size: Tuple[int, int],
        high_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Scale list of boxes from low-DPI pixel coordinates to high-DPI pixel coordinates
        using independent x/y scaling to avoid rounding drift.
        """
        w_low, h_low = low_size
        w_high, h_high = high_size
        sx = w_high / max(1, w_low)
        sy = h_high / max(1, h_low)

        boxes_high: List[Tuple[int, int, int, int]] = []
        for (x0, y0, x1, y1) in boxes_low:
            X0 = int(round(x0 * sx))
            Y0 = int(round(y0 * sy))
            X1 = int(round(x1 * sx))
            Y1 = int(round(y1 * sy))
            # clamp to high image bounds
            X0 = max(0, min(X0, w_high))
            Y0 = max(0, min(Y0, h_high))
            X1 = max(0, min(X1, w_high))
            Y1 = max(0, min(Y1, h_high))
            if X1 <= X0 or Y1 <= Y0:
                continue
            boxes_high.append((X0, Y0, X1, Y1))
        return boxes_high