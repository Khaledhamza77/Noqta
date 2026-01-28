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

    def __init__(self, pdf_path: str, out_dir: str, cfg: Optional[ChunkerConfig] = None):
        self.pdf_path = pdf_path
        self.out_dir = out_dir
        self.cfg = cfg or ChunkerConfig()
        os.makedirs(self.out_dir, exist_ok=True)

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

    # --------------------- Crop & save ---------------------

    def _save_crop(self, high_img: Image.Image, box_high: Tuple[int, int, int, int], page_idx: int, cluster_idx: int) -> str:
        crop = high_img.crop(box_high)
        if self.cfg.save_mode_1bit and crop.mode != "1":
            crop = crop.convert("1")
        filename = f"page_{page_idx:04d}_chunk_{cluster_idx:03d}.{self.cfg.save_format.lower()}"
        path = os.path.join(self.out_dir, filename)
        crop.save(path, format=self.cfg.save_format.upper())
        return path

    # --------------------- Main API ---------------------

    def chunk_page(
        self,
        page_idx: int,
        points_xy_low: np.ndarray,
        labels: np.ndarray,
        low_size: Tuple[int, int],
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        For a single page:
          1) Re-render high-DPI page image
          2) Compute minimal low-DPI boxes per cluster (labels >= 0), with padding
          3) Scale boxes to high-DPI coordinates and crop from high image
          4) Save crops and (optionally) return metadata

        Args:
          page_idx: index of the page in the PDF
          points_xy_low: (N,2) black-pixel coordinates in low-DPI pixel units
          labels: (N,) cluster labels (>=0 = valid clusters; -1 ignored)
          low_size: (w_low, h_low) of the image used to produce points_xy_low
          return_metadata: if True, returns mapping with low/high boxes and file paths

        Returns:
          A list of dicts (one per crop) with file path and box mappings.
        """
        if points_xy_low.ndim != 2 or points_xy_low.shape[1] != 2:
            raise ValueError("points_xy_low must be (N,2) array of pixel coordinates.")
        if labels.ndim != 1 or labels.shape[0] != points_xy_low.shape[0]:
            raise ValueError("labels must be (N,) and aligned with points_xy_low.")
        if len(low_size) != 2:
            raise ValueError("low_size must be (w_low, h_low).")

        with fitz.open(self.pdf_path) as doc:
            page = doc.load_page(page_idx)
            high_img = self._render_page_high(page)
            w_high, h_high = high_img.size
            w_low, h_low = low_size

            # Build low-DPI boxes and scale them to high-DPI
            boxes_low = self._boxes_from_labels(points_xy_low, labels, self.cfg.padding_low_px, (w_low, h_low))
            boxes_high = self._scale_boxes_low_to_high(boxes_low, (w_low, h_low), (w_high, h_high))

            # Optionally enforce minimum size to avoid empty/degenerate crops
            sanitized_high = []
            for (X0, Y0, X1, Y1) in boxes_high:
                if (X1 - X0) < self.cfg.ensure_min_size_px:
                    X1 = min(w_high, X0 + self.cfg.ensure_min_size_px)
                if (Y1 - Y0) < self.cfg.ensure_min_size_px:
                    Y1 = min(h_high, Y0 + self.cfg.ensure_min_size_px)
                if X1 > X0 and Y1 > Y0:
                    sanitized_high.append((X0, Y0, X1, Y1))

            # Save crops
            results: List[Dict[str, Any]] = []
            for i, box_high in enumerate(sanitized_high):
                path = self._save_crop(high_img, box_high, page_idx, i)
                if return_metadata:
                    results.append({
                        "page_index": page_idx,
                        "cluster_index": i,
                        "low_box": tuple(map(int, boxes_low[i])),
                        "high_box": tuple(map(int, box_high)),
                        "high_image_size": (w_high, h_high),
                        "path": path
                    })

        return results