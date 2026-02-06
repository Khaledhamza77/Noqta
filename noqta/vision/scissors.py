from typing import List, Tuple
from PIL import Image
import os


class Scissors:
    def __init__(self):
        pass

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

    @staticmethod
    def crop_image_by_boxes(
        image: Image.Image,
        boxes: List[Tuple[float, float, float, float]],
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
        for i, box in enumerate(boxes):
            # Clip to image bounds
            x1, y1, x2, y2 = Scissors._clip_box_to_image(box, (W, H))
            # Crop and save
            crop = image.crop((x1, y1, x2, y2))
            crop.save(os.path.join(path_to_save, f"box_{i}.jpeg"))
