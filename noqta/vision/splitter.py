import os
import numpy as np
from typing import Literal
from PIL import Image, ImageDraw
from ..configs import SplitterConfig
from scipy.ndimage import distance_transform_edt


class Splitter:
    def __init__(self, cfg: SplitterConfig):
        self.cfg = cfg
    
    @staticmethod
    def _edt(bin_L: Image.Image, edt_threshold: int) -> Image.Image:
        """
        Euclidean Distance Transform for black foreground.
        """
        dist = distance_transform_edt(~np.array(bin_L))   # distance from black
        img = (dist < edt_threshold).astype(np.uint8) * 255
        return Image.fromarray(img, mode="L")
    
    def is_high(self, image_height: float, box: tuple[float, float, float, float]) -> bool:
        """Heuristic to determine if a box is "high" (tall) vs "wide" (short)."""
        _, y0, _, y1 = box
        height = y1 - y0
        return height >= self.cfg.high_box_threshold * image_height
    
    def prep_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        new_h = int(round((h / w) * self.cfg.new_w))
        return self._edt(image.resize((self.cfg.new_w, new_h)), self.cfg.edt_threshold)
    
    @staticmethod
    def _draw_lines(image: Image.Image, lines: list[tuple[float, float, float, float]], col: Literal["red", "blue"]) -> Image.Image:
        img = image.copy()
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for line in lines:
            draw.line(line, fill=col, width=2)
        return img
    
    @staticmethod
    def _upscale_box(
        og_size: tuple[int, int],
        new_size: tuple[int, int],
        box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """
        Upscale horizontal box from low-res image back to original image coordinates.
         - og_size: (width, height) of original image
         - new_size: (width, height) of low-res image where box was detected
         - box: (x0, y0, x1, y1) coordinates of box in low-res image
        Returns: (x0, y0, x1, y1) coordinates of box in original image
        Note: x0 and x1 are not scaled since they span the full width, but y0 and y1 are scaled by the height ratio.
        """
        h_ratio = og_size[1] / new_size[1]
        _, y0, _, y1 = box
        return (0, y0 * h_ratio, og_size[0], y1 * h_ratio)
    
    def run(
        self,
        local_dir: str,
        image: Image.Image,
        boxes: list[tuple[float, float, float, float]]
    ):
        """
        Main function to run the splitter class.
        It takes as input a list of boxes
        finds tall boxes, splits them and saves crops of the split boxes (and non-split boxes) in the output directory.
        It also saves intermediate logs of detected lines and splitting points in subdirectories for each tall box for debugging purposes.
        """
        w, h = image.size
        for i, box in enumerate(boxes):
            if self.is_high(h, box):
                os.makedirs(f"{local_dir}/long_box_{i}", exist_ok=True)  # create subdir for this box's logs
                # Split tall box into multiple smaller boxes
                high_res_box = image.crop(box)
                croppedBox = self.prep_image(high_res_box)
                lines = []
                
                horizontal_white = self.find_horizontal_lines(croppedBox, line_color="white")
                log_img1 = self._draw_lines(croppedBox.copy(), horizontal_white, col="blue")
                log_img1.save(f"{local_dir}/long_box_{i}/1_horizontal_white.png")

                horizontal_black = self.find_horizontal_lines(croppedBox, line_color="black")
                log_img2 = self._draw_lines(croppedBox.copy(), horizontal_black, col="blue")
                log_img2.save(f"{local_dir}/long_box_{i}/2_horizontal_black.png")

                lines.extend(horizontal_white)
                lines.extend(horizontal_black)
                splitting_points = self.find_splitting_points(croppedBox.size, lines)
                
                # sort splitting points by y-coordinate and ensure the last splitting point is not too close to the bottom edge
                splitting_points.sort(key=lambda line: line[1])
                if splitting_points[len(splitting_points) - 1][1] < croppedBox.size[1] - self.cfg.min_length_to_split:
                    splitting_points.append((0, croppedBox.size[1], croppedBox.size[0], croppedBox.size[1]))
                else:
                    splitting_points[len(splitting_points) - 1] = (0, croppedBox.size[1], croppedBox.size[0], croppedBox.size[1])
                
                log_img3 = self._draw_lines(croppedBox.copy(), splitting_points, col="red")
                log_img3.save(f"{local_dir}/long_box_{i}/3_detected_splitting_points.png")

                og_size = high_res_box.size
                new_size = croppedBox.size
                y_start = 0
                for j, spoint in enumerate(splitting_points):
                    _, cut, _, _ = spoint
                    if cut - y_start > self.cfg.min_length_to_split:
                        high_res_box.crop(
                            self._upscale_box(
                                og_size,
                                new_size,
                                (
                                    0, 
                                    y_start,
                                    w,
                                    cut
                                )
                            )
                        ).save(f"{local_dir}/box_{i}_{j}.png")
                        y_start = cut
            else:
                image.crop(box).save(f"{local_dir}/box_{i}.png")
    
    def find_splitting_points(
        self,
        img_size: tuple[int, int],
        edges: list[tuple[float, float, float, float]]
    ) -> list[tuple[float, float, float, float]]:
        w_aspect, h_aspect = self.cfg.max_ratio
        w, h = img_size

        n = int(round(max(2, ((h_aspect * h) / (w_aspect * w))), 0))
        new_h = int(round(h / n, 0))
        potential_points = list(range(0, h, new_h))[1:]

        Hs_y = [edge[1] for edge in edges]
        splitting_points = []
        for point in potential_points:
            splitting_points.append(edges[np.argmin([abs(point - y) for y in Hs_y])])
        
        return splitting_points

    def find_horizontal_lines(
        self,
        gray: Image.Image,
        line_color: Literal["white", "black"] = "white",
    ) -> list[tuple[float, float, float, float]]:
        """
        Find horizontal lines in a binary image that represent empty/full rows of pixels.

        A "line" is a horizontal span of consecutive pixels of the same color (white or black)
        that runs for at least `min_width` of the image width. Isolated pixels of the
        opposite color within that span are dismissed as noise if their run length is
        <= `tolerance`.

        Args:
            gray:          Binary PIL image (will be converted to grayscale + thresholded).
            line_color:    Whether to look for white lines ("white") or black lines ("black").

        Returns:
            List of (x0, y0, x1, y1) tuples (floats) for each qualifying line,
            where x0/x1 are the start/end x-coordinates of the run and y0 == y1
            (same row).
        """
        # --- Normalise to a 1-bit representation -----------------------------------
        width, _ = gray.size
        target_value = 255 if line_color == "white" else 0   # pixel value we look for

        min_run_px = width * self.cfg.min_width
        # Create a binary bool mask where True = target color (white or black)
        arr = np.array(gray)
        mask = (arr == target_value)

        sums = mask.sum(axis=1)  # sum of target-color pixels in each row
        return [(0, y, width, y) for y, count in enumerate(sums) if count >= min_run_px]