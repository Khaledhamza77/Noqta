from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, List, Tuple
from ..configs import ClustererConfig
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import Tuple
import fitz  # PyMuPDF
import numpy as np
import cv2
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
        return img, scale

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
    
    # --------------- Removing Frames -----------------------

    def _remove_frames(self, img: Image):
        W, H = img.size
        img = np.array(img)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=self.cfg.sobel_kernel_size)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=self.cfg.sobel_kernel_size)
        
        # Gradient magnitude
        mag = cv2.magnitude(gx, gy)  # float32
        # Normalize to 8-bit for thresholding / visualization
        mag_u8 = cv2.convertScaleAbs(mag)
        _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x = Image.fromarray(edges)
        min_line_length_h = int(W * self.cfg.min_ratio_to_side)
        min_line_length_v = int(H * self.cfg.min_ratio_to_side)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.cfg.rho,
            theta=self.cfg.theta,
            threshold=self.cfg.threshold,
            maxLineGap=self.cfg.max_line_gap
        )
        horizontals: List[Tuple[int, int, int, int]] = []
        verticals: List[Tuple[int, int, int, int]] = []
        if lines is not None:
            angle_tol = np.deg2rad(self.cfg.angle_tolerance_degree)
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length < 2:  # ignore tiny segments
                    continue
                angle = np.arctan2(dy, dx)  # radians
            
                # Normalize angle to [-pi/2, pi/2] for easier checks
                a = ((angle + np.pi/2) % np.pi) - np.pi/2
            
                # Horizontal if near 0°, Vertical if near ±90°
                if abs(a) <= angle_tol:
                    # Horizontal: check projected span in X
                    span = abs(x2 - x1)
                    if span >= min_line_length_h:
                        # Normalize y to average to reduce small skew artifacts
                        y = int(round((y1 + y2) / 2))
                        horizontals.append((min(x1, x2), y, max(x1, x2), y))
                elif abs(abs(a) - np.pi/2) <= angle_tol:
                    # Vertical: check projected span in Y
                    span = abs(y2 - y1)
                    if span >= min_line_length_v:
                        x = int(round((x1 + x2) / 2))
                        verticals.append((x, min(y1, y2), x, max(y1, y2)))
        
        #out1 og image with detected edges
        out1 = np.array(img.copy())
        out1 = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
        for x1,y,x2,_ in horizontals:
            cv2.line(out1, (x1,y), (x2,y), (255,0,0), self.cfg.edge_thickness)
        for x, y1, _, y2 in verticals:
            cv2.line(out1, (x,y1), (x,y2), (255,0,0), self.cfg.edge_thickness)

        #out2 og with edges removed (whitened)
        out2 = np.array(img.copy())
        for x1,y,x2,_ in horizontals:
            cv2.line(out2, (x1,y), (x2,y), 255, self.cfg.edge_thickness)
        for x, y1, _, y2 in verticals:
            cv2.line(out2, (x,y1), (x,y2), 255, self.cfg.edge_thickness)
        
        return Image.fromarray(out1), Image.fromarray(out2, mode="L")

    # --------------- EDT / Dilation (smudge) ---------------

    def _edt(self, bin_L: Image.Image) -> Image.Image:
        """
        Euclidean Distance Transform for black foreground.
        """
        dist = distance_transform_edt(~np.array(bin_L))   # distance from black
        img = (dist < self.cfg.threshold_edt).astype(np.uint8) * 255
        return Image.fromarray(img, mode="L")

    def _dilate(self, bin_L: Image.Image) -> Image.Image:
        """
        Morphological dilation for black foreground using MinFilter on {black=0, white=255}.
        Kernel size = 2*radius + 1; repeat for iterations.
        """
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