from typing import List, Tuple, Optional
from ..configs import SuppressorConfig


Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive/exclusive OK if consistent

class Suppressor:
    """
    Post-process a set of boxes with:
      1) Non-maximum suppression: remove boxes completely encompassed by others.
      2) High-overlap merging: if intersection / area_of_smaller >= overlap_threshold,
         enlarge the bigger box to the union (optionally padded), then remove the smaller.
    """

    def __init__(self, cfg: Optional[SuppressorConfig] = None):
        self.cfg = cfg or SuppressorConfig()

    # ------------------------------ Geometry helpers ------------------------------

    @staticmethod
    def _area(b: Box) -> int:
        x0, y0, x1, y1 = b
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        return w * h

    @staticmethod
    def _intersect(a: Box, b: Box) -> Box:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0)  # no overlap
        return (x0, y0, x1, y1)

    @staticmethod
    def _union(a: Box, b: Box) -> Box:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        x0 = min(ax0, bx0)
        y0 = min(ay0, by0)
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        return (x0, y0, x1, y1)

    @staticmethod
    def _is_enclosed(inner: Box, outer: Box) -> bool:
        ix0, iy0, ix1, iy1 = inner
        ox0, oy0, ox1, oy1 = outer
        return (ix0 >= ox0) and (iy0 >= oy0) and (ix1 <= ox1) and (iy1 <= oy1)

    def _pad_and_clamp(self, b: Box) -> Box:
        x0, y0, x1, y1 = b
        p = int(self.cfg.pad_union_px)
        x0 -= p; y0 -= p; x1 += p; y1 += p

        if self.cfg.clamp is not None:
            W, H = self.cfg.clamp
            x0 = max(0, min(x0, W))
            y0 = max(0, min(y0, H))
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))

        # ensure non-degenerate
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        return (x0, y0, x1, y1)
    # ------------------------------ Step 1: Remove oversized boxes ------------------------------

    def remove_oversized(self, boxes: List[Box], image_size: Tuple[int, int]) -> List[Box]:
        """
        Remove boxes that exceed a certain fraction of the image size in width or height.
        """
        w_img, h_img = image_size
        max_area = self.cfg.ratio_to_image_threshold * (w_img * h_img)

        filtered_boxes: List[Box] = []
        for b in boxes:
            x0, y0, x1, y1 = b
            box_w = x1 - x0
            box_h = y1 - y0
            area = box_w * box_h
            if area <= max_area:
                filtered_boxes.append(b)

        return filtered_boxes

    # ------------------------------ Step 2: Enclosure NMS ------------------------------

    def remove_fully_enclosed(self, boxes: List[Box]) -> List[Box]:
        """
        Remove any box that is completely encompassed by another box.
        """
        n = len(boxes)
        if n <= 1:
            return boxes[:]

        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                # If box i is enclosed by box j, drop i
                if self._is_enclosed(boxes[i], boxes[j]):
                    keep[i] = False
                    break

        result = [boxes[i] for i in range(n) if keep[i]]
        if self.cfg.remove_duplicates:
            result = self._deduplicate(result)
        return result

    # ------------------------------ Step 3: Merge high-overlap pairs ------------------------------

    def merge_high_overlap(self, boxes: List[Box]) -> List[Box]:
        """
        For any pair (A, B) with intersection / area(smaller) >= overlap_threshold,
        enlarge the bigger box to the union(A, B), then remove the smaller box.
        Repeat until no more merges occur.
        """
        if len(boxes) <= 1:
            return boxes[:]

        threshold = float(self.cfg.overlap_threshold)
        bxs = boxes[:]
        changed = True

        while changed:
            changed = False
            n = len(bxs)
            if n <= 1:
                break

            to_remove = set()
            # Pairwise check
            for i in range(n):
                if i in to_remove: 
                    continue
                for j in range(i + 1, n):
                    if j in to_remove:
                        continue

                    A = bxs[i]
                    B = bxs[j]
                    inter = self._intersect(A, B)
                    inter_area = self._area(inter)
                    if inter_area == 0:
                        continue

                    area_A = self._area(A)
                    area_B = self._area(B)
                    smaller_area = min(area_A, area_B)

                    overlap_frac = inter_area / max(1, smaller_area)
                    if overlap_frac >= threshold:
                        # Determine bigger box and enlarge it to union
                        if area_A >= area_B:
                            # A is bigger: enlarge A to union(A,B), remove B
                            union_box = self._pad_and_clamp(self._union(A, B))
                            bxs[i] = union_box
                            to_remove.add(j)
                        else:
                            # B is bigger: enlarge B to union(A,B), remove A
                            union_box = self._pad_and_clamp(self._union(A, B))
                            bxs[j] = union_box
                            to_remove.add(i)
                            break  # i removed; break inner loop

                        changed = True

            if to_remove:
                bxs = [b for k, b in enumerate(bxs) if k not in to_remove]
                # After changes, also remove any now-fully-enclosed boxes
                bxs = self.remove_fully_enclosed(bxs)

        if self.cfg.remove_duplicates:
            bxs = self._deduplicate(bxs)
        return bxs

    # ------------------------------ Utilities ------------------------------

    @staticmethod
    def _deduplicate(boxes: List[Box]) -> List[Box]:
        seen = set()
        unique = []
        for b in boxes:
            if b not in seen:
                seen.add(b)
                unique.append(b)
        return unique

    def process(self, boxes: List[Box], image_size: Tuple[int, int]) -> List[Box]:
        """
        Full pipeline:
          1) Remove oversized boxes.
          2) Remove fully encompassed boxes.
          3) Merge high-overlap pairs (≥ threshold), enlarging bigger box, then remove enclosed.
        """
        b = self.remove_oversized(boxes, image_size)
        b = self.remove_fully_enclosed(b)
        b = self.merge_high_overlap(b)
        return b