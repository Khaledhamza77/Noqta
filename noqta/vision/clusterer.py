from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import distance_transform_edt
from ..configs import ClustererConfig
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import Tuple
import fitz  # PyMuPDF
import numpy as np
import os


class Clusterer:
    """
    Clusterer pipeline:
      PDF page -> grayscale -> binarize -> (optional) smudge -> (x,y) black pixels ->
      Agglomerative (hierarchical) clustering -> save plots (points and clusters).
    """

    def __init__(self, cfg: ClustererConfig):
        self.cfg = cfg

    # --------------- Rendering ---------------

    @staticmethod
    def _calculate_dpi(page: fitz.Page, max_px: int) -> int:
        w_pt = page.rect.width
        h_pt = page.rect.height
        return (max_px * 72.0) / max(w_pt, h_pt)


    def _render_page_gray(self, doc: fitz.Document, page_index: int) -> Image.Image:
        page = doc.load_page(page_index)
        scale = self._calculate_dpi(page, self.cfg.max_px) / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        if self.cfg.invert:
            img = Image.eval(img, lambda p: 255 - p)
        return img

    # --------------- Binarization ---------------

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        total = gray.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0
        w_b = 0.0
        var_max = 0.0
        threshold = 0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = t
        return threshold

    def _to_binary_L(self, img_gray: Image.Image) -> Image.Image:
        """
        Return 'L' mode image with values in {0,255}, where black=0, white=255.
        This polarity is convenient for dilation using MinFilter (expands dark regions).
        """
        arr = np.asarray(img_gray, dtype=np.uint8)
        thr = self.cfg.fixed_threshold if self.cfg.fixed_threshold is not None \
              else (self._otsu_threshold(arr) if self.cfg.use_otsu else 128)
        bin_arr = np.where(arr <= thr, 0, 255).astype(np.uint8)
        return Image.fromarray(bin_arr, mode="L")

    # --------------- EDT / Dilation (smudge) ---------------

    def _edt(self, bin_L: Image.Image) -> Image.Image:
        """
        Euclidean Distance Transform for black foreground.
        """
        dist = distance_transform_edt(~np.array(bin_L))   # distance from black
        return Image.fromarray(dist < self.cfg.threshold_edt)

    def _dilate(self, bin_L: Image.Image) -> Image.Image:
        """
        Morphological dilation for black foreground using MinFilter on {black=0, white=255}.
        Kernel size = 2*radius + 1; repeat for iterations.
        """
        if not self.cfg.use_dilation or self.cfg.dilate_radius_px <= 0:
            return bin_L
        size = 2 * int(self.cfg.dilate_radius_px) + 1
        out = bin_L
        for _ in range(max(1, self.cfg.dilation_iterations)):
            out = out.filter(ImageFilter.MinFilter(size=size))
        return out

    # --------------- Extract black pixel coordinates ---------------

    @staticmethod
    def _extract_black_xy_from_L(pil_binary_L: Image.Image) -> np.ndarray:
        # black pixels are 0 in 'L' image
        arr = np.asarray(pil_binary_L, dtype=np.uint8)
        ys, xs = np.where(arr == 0)
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.column_stack((xs.astype(np.int32), ys.astype(np.int32)))

    # --------------- Clustering ---------------

    def _hierarchical_cluster(self, pts: np.ndarray) -> np.ndarray:
        """
        Agglomerative clustering with a distance threshold (in pixels).
        Returns labels for each point (0..K-1), or empty if no points.
        """
        if pts.shape[0] == 0:
            return np.empty((0,), dtype=int)

        ac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(self.cfg.hier_distance_threshold),
            linkage=self.cfg.hier_linkage
        )
        labels = ac.fit_predict(pts)
        return labels