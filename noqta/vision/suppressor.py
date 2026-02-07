from typing import List, Tuple, Optional
from ..configs import SuppressorConfig
import networkx as nx
import numpy as np


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

    # ------------------------ Combining Small boxes ------------------------
    
    @staticmethod
    def find_small_boxes(boxes, min_area):
        boxes = np.asarray(boxes)

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = w * h

        small_mask = area < min_area

        return boxes[small_mask], boxes[~small_mask]
    
    @staticmethod
    def box_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @staticmethod
    def boxes_intersect(b1, b2):
        return not (
            b1[2] < b2[0] or b1[0] > b2[2] or
            b1[3] < b2[1] or b1[1] > b2[3]
        )

    @staticmethod
    def build_graph(boxes, dist_thresh):
        G = nx.Graph()

        centers = np.array([Suppressor.box_center(b) for b in boxes])

        for i in range(len(boxes)):
            G.add_node(i)

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                dist = np.linalg.norm(centers[i] - centers[j])
                intersect = Suppressor.boxes_intersect(boxes[i], boxes[j])

                if dist < dist_thresh or intersect:
                    G.add_edge(i, j)

        return G
    
    @staticmethod
    def merge_components(boxes, graph, min_area):
        merged_boxes = []

        for comp in nx.connected_components(graph):
            if len(comp) == 1:
                comp_box = boxes[list(comp)[0]]
                x1 = comp_box[0]; x2 = comp_box[2]
                y1 = comp_box[1]; y2 = comp_box[3]
                w = x2 - x1
                h = y2 - y1
                area = w * h
                if area <= min_area:
                    continue
                else:
                    merged_boxes.append([x1, y1, x2, y2])

            comp_boxes = boxes[list(comp)]
            x1 = comp_boxes[:, 0].min()
            y1 = comp_boxes[:, 1].min()
            x2 = comp_boxes[:, 2].max()
            y2 = comp_boxes[:, 3].max()

            merged_boxes.append([x1, y1, x2, y2])

        return np.array(merged_boxes)
    
    def merge_and_remove_small(self, boxes, img_size):
        w, h = img_size
        min_area = self.cfg.min_area_ratio * (w * h)
        dist_thresh = self.cfg.dist_ratio * max(w,  h)
        min_min_area = self.cfg.min_area_ratio_to_remove * (w * h)

        # Step 1: find small boxes
        small_boxes, large_boxes = Suppressor.find_small_boxes(boxes, min_area=min_area)

        # Step 3: build graph
        G = Suppressor.build_graph(small_boxes, dist_thresh=dist_thresh)

        # Step 4: Merge small and remove singletons
        merged_small_boxes = Suppressor.merge_components(small_boxes, G, min_min_area)

        if len(merged_small_boxes) > 0:
            final_boxes = np.vstack([large_boxes, merged_small_boxes])
        else:
            final_boxes = large_boxes.copy()

        return small_boxes, merged_small_boxes, final_boxes

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