from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Optional
from ..configs import ScissorsConfig
from PIL import Image
import numpy as np
import cv2
import os


class Scissors:
    def __init__(self, cfg: Optional[ScissorsConfig] = None):
        self.cfg = cfg

    @staticmethod #UNDER REVIEW NOT USED OR TESTED
    def order_boxes(
        boxes: List[Tuple[float, float, float, float]],
        *,
        row_tolerance: float = 0.5,
        row_tolerance_is_fraction_of_height: bool = True,
        reverse_x: bool = False,   # set True for right→left within row
        reverse_y: bool = False    # set True for bottom→top rows
    ) -> List[int]:
        """
        Return the indices of `boxes` ordered row-wise.
        
        Ordering logic (default):
        • Rows are formed from TOP to BOTTOM using centers' y with a tolerance.
        • Within each row, boxes are ordered LEFT to RIGHT.
        • Result: left-most upper-most is first; right-most lower-most is last.

        Args:
            boxes: List of boxes in either xyxy or xywh format.
            fmt: 'xyxy' or 'xywh'.
            row_tolerance:
                If row_tolerance_is_fraction_of_height=True (default), this is a fraction
                of the median box height (e.g., 0.5 = half the median height).
                If False, it is in absolute pixels.
            row_tolerance_is_fraction_of_height: See above.
            reverse_x: If True, order within rows right→left.
            reverse_y: If True, order rows bottom→top.

        Returns:
            A list of indices (into the original `boxes` list) in the desired order.
        """
        if not boxes:
            return []

        # compute center and size
        centers = []
        heights = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            h = max(1e-6, (y2 - y1))
            centers.append((i, cx, cy))
            heights.append(h)

        # Determine row tolerance in pixels
        if row_tolerance_is_fraction_of_height:
            # Use median height for stability
            sorted_h = sorted(heights)
            mid = len(sorted_h) // 2
            median_h = (sorted_h[mid] if len(sorted_h) % 2 == 1
                        else 0.5 * (sorted_h[mid - 1] + sorted_h[mid]))
            y_tol = median_h * row_tolerance
        else:
            y_tol = float(row_tolerance)

        # Sort by center y first (top→bottom), then by center x (left→right)
        centers.sort(key=lambda t: (t[2], t[1]))

        # Group into rows using y tolerance
        rows = []  # list of lists of tuples (i, cx, cy)
        for item in centers:
            i, cx, cy = item
            if not rows:
                rows.append([item])
                continue
            # Compare to the running mean cy of the last row
            last_row = rows[-1]
            mean_cy = sum(p[2] for p in last_row) / len(last_row)
            if abs(cy - mean_cy) <= y_tol:
                last_row.append(item)
            else:
                rows.append([item])

        # Order rows (top→bottom by default)
        if reverse_y:
            rows = rows[::-1]

        # Sort within each row by x (left→right by default)
        ordered_indices: List[int] = []
        for row in rows:
            row.sort(key=lambda t: t[1], reverse=reverse_x)  # x sort
            ordered_indices.extend([t[0] for t in row])

        return ordered_indices
    
    def _prepare_image(self, img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
        """
        Crop to box and resize to new_w while maintaining aspect ratio.
        """
        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))
        w, h = crop.size
        new_w = self.cfg.new_w
        new_h = int(round((h / w) * new_w))
        return crop, self._edt(crop.resize((new_w, new_h))) if self.cfg.use_edt else crop.resize((new_w, new_h))
    
    def _find_horizontal_edges(
        self,
        img: Image.Image
    ):
        W, H = img.size
        img = np.array(img)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=self.cfg.sobel_kernel_size)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=self.cfg.sobel_kernel_size)

        # Gradient magnitude
        mag = cv2.magnitude(gx, gy)  # float32
        # Normalize to 8-bit for thresholding / visualization
        mag_u8 = cv2.convertScaleAbs(mag)
        _, edges = cv2.threshold(mag_u8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                )
        min_line_length_h = int(W * self.cfg.min_ratio_to_side)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.cfg.rho,
            theta=self.cfg.theta,
            threshold=self.cfg.threshold,
            maxLineGap=self.cfg.max_line_gap
        )
        horizontals: List[Tuple[int, int, int, int]] = []
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
                    # Normalize y to average to reduce small skew artifacts
                    if span >= min_line_length_h:
                        y = int(round((y1 + y2) / 2))
                        horizontals.append((min(x1, x2), y, max(x1, x2), y))

        # out1 og image with detected edges
        out1 = np.array(img.copy())
        out1 = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
        for x1, y, x2, _ in horizontals:
            cv2.line(out1, (x1, y), (x2, y), (255, 0, 0), 4)

        return Image.fromarray(out1), horizontals

    @staticmethod
    def _clip_box_to_image(
        xyxy: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Clamp box to image bounds and convert to integer pixel coordinates.
        Ensures x1<x2 and y1<y2 by max/min.
        """
        W, H = image_size
        x1, y1, x2, y2 = xyxy

        x1 = max(0, min(W, int(round(x1))))
        y1 = max(0, min(H, int(round(y1))))
        x2 = max(0, min(W, int(round(x2))))
        y2 = max(0, min(H, int(round(y2))))

        # Ensure non-empty box
        if x2 <= x1: x2 = min(W, x1 + 1)
        if y2 <= y1: y2 = min(H, y1 + 1)

        return (x1, y1, x2, y2)
    
    def _edt(self, bin_L: Image.Image) -> Image.Image:
        """
        Euclidean Distance Transform for black foreground.
        """
        dist = distance_transform_edt(~np.array(bin_L))   # distance from black
        img = (dist < self.cfg.edt_threshold).astype(np.uint8) * 255
        return Image.fromarray(img, mode="L")
    
    def _find_splitting_points(
        self,
        img_size: Tuple[int, int] = None,
        edges: List[Tuple[float, float, float, float]] = None
    ) -> List[Tuple[float, float, float, float]]:
        w, h = img_size

        n = int(round(max(2, ((9 * h) / (5 * w))), 0))
        new_h = int(round(h / n, 0))
        potential_points = list(range(0, h, new_h))[1:]

        Hs_y = [edge[1] for edge in edges]
        splitting_points = []
        for point in potential_points:
            splitting_points.append(edges[np.argmin([abs(point - y) for y in Hs_y])])
        
        return splitting_points
    
    @staticmethod
    def _crop_at_high(
        box_index: int,
        high_lines: List[Tuple[float, float, float, float]],
        high_img: Image.Image,
        path: str
    ) -> None:
        
        y_cuts: List[float] = [t[1] for t in high_lines]

        if y_cuts[len(y_cuts) - 1] < high_img.size[1] - 30:
            y_cuts.append(high_img.size[1])
        else:
            y_cuts[len(y_cuts) - 1] = high_img.size[1]

        y_start = 0
        w, _ = high_img.size
        for i, cut in enumerate(y_cuts):
            if cut - y_start > 30:
                crop = high_img.crop((0, y_start, w, cut))
                print(f"Crop {i}: y_start={y_start}, cut={cut}")
                y_start = cut
                crop.save(f'{path}/box_{box_index}_{i}.jpg')
    
    @staticmethod
    def _scale_lines(
        low_size: Tuple[int, int],
        high_size: Tuple[int, int],
        lines: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        low_w, low_h = low_size
        high_w, high_h = high_size
        
        sx = high_w / low_w
        sy = high_h / low_h

        high_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            high_lines.append(
                (
                    x1 * sx,
                    y1 * sy,
                    x2 * sx,
                    y2 * sy
                )
            )

        return high_lines

    @staticmethod
    def crop_image_by_boxes(
        i: int,
        image: Image.Image,
        box: Tuple[float, float, float, float],
        path_to_save: str,
    ) -> List[Image.Image]:
        """
        Crop an image into multiple patches given bounding boxes.

        Args:
            image: PIL Image.
            boxes: List of boxes. Depending on 'fmt':
                - 'xyxy': (x1, y1, x2, y2)
                - 'xywh': (x, y, w, h)
                If relative=True, values are in [0,1] normalized by image size.
            fmt: Box format, 'xyxy' or 'xywh'.
            relative: Whether boxes are normalized to [0,1].
            pad: Padding amount around each box. Pixels if pad_is_fraction=False.
            pad_is_fraction: If True, 'pad' is fraction of box width/height.
            output_size: If provided, each crop is resized to (width, height).
            resample: Resampling filter for resizing (e.g., Image.BILINEAR).

        Returns:
            List of PIL Image crops in the same order as input boxes.
        """
        W, H = image.size
        x1, y1, x2, y2 = Scissors._clip_box_to_image(box, (W, H))
        # Crop and save
        crop = image.crop((x1, y1, x2, y2))
        crop.save(os.path.join(path_to_save, f"box_{i}.jpeg"))