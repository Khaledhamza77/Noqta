from .configs import ClustererConfig, ChunkerConfig, SuppressorConfig
from scipy.ndimage import distance_transform_edt
from .cv.suppressor import Suppressor
from matplotlib import pyplot as plt
from .cv.clusterer import Clusterer
from PIL import ImageDraw, Image
from .cv.chunker import Chunker
import numpy as np
import fitz  # PyMuPDF
import os

class NOQTA:
    def __init__(self, path: str, config: dict):
        self.config = config
        if os.path.isfile(path):
            self.paths = [path]
        elif os.path.isdir(path):
            self.paths = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]
        self.local_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.local_dir, 'noqta_crops')
    
    def configure(self):
        clfg = self.config.get("clusterer", {})
        cl_cfg = ClustererConfig(
            max_px=clfg.get("max_px", 10),                 # modest DPI as requested
            invert=clfg.get("invert", False),            # set True if your pages are white-on-black
            # Smudging to connect fragments (tune these)
            use_smudging=clfg.get("use_dilation", True),
            type_smudging=clfg.get("type_smudging", "edt"),  # 'edt' or 'dilation'
            threshold_edt=clfg.get("threshold_edt", 1),
            dilate_radius_px=clfg.get("dilate_radius_px", 1),      # try 2–3 for tables/diagrams
            dilation_iterations=clfg.get("dilation_iterations", 1),

            # Hierarchical clustering parameters (pixel units)
            hier_distance_threshold=clfg.get("hier_distance_threshold", 3.0),
            hier_linkage=clfg.get("hier_linkage", "single"),   # 'single' tends to merge nearby components

            plot_point_size=1,
        )
        cluster = Clusterer(self.output_dir, cl_cfg)

        chcfg = self.config.get("chunker", {})
        ch_cfg = ChunkerConfig(
            high_dpi=chcfg.get("high_dpi", 100),                  # render high-resolution crops
            padding_low_px=chcfg.get("padding_low_px", 0),        # extra margin around clusters in LOW-DPI units
            save_format=chcfg.get("save_format", "PNG"),          # PNG or TIFF, etc.
            save_mode_1bit=chcfg.get("save_mode_1bit", False)     # set True if you want 1-bit crops
        )
        chunker = Chunker(self.output_dir, ch_cfg)

        supcfg = self.config.get("suppressor", {})
        sup_cfg = SuppressorConfig(
            overlap_threshold=supcfg.get("overlap_threshold", 0.60),
            pad_union_px=supcfg.get("pad_union_px", 0),
            remove_duplicates=supcfg.get("remove_duplicates", True)
        )
        suppressor = Suppressor(sup_cfg)
        return cluster, chunker, suppressor

    def run(self):
        c, chunker, suppressor = self.configure(pdf_path)
        for pdf_path in self.paths:
            with fitz.open(pdf_path) as doc:
                page_indices = list(range(doc.page_count))
                for pidx in page_indices:
                    gray = c._render_page_gray(doc, pidx); w, h = gray.size
                    remove_right = (1/20) * w; remove_left = (1/10) * w
                    gray = gray.crop((remove_right, 0, w-remove_left, h))
                    w, h = gray.size

                    gray.show() 
                    bin_l = c._to_binary_L(gray)
                    bin_l.show()
                    if c.cfg.use_smudging:
                        if c.cfg.type_smudging == 'dilation':
                            bin_l_smudged = c._dilate(bin_l)
                        elif c.cfg.type_smudging == 'edt':
                            bin_l_smudged = c._edt(bin_l)
                        else:
                            raise ValueError(f"Unknown smudging type: {c.cfg.type_smudging}")
                    else:
                        bin_l_smudged = bin_l

                    bin_l_smudged.show()

                    points_xy = c._extract_black_xy_from_L(bin_l_smudged)
                    plt.scatter(points_xy[:, 0], points_xy[:, 1],
                                            s=1, c="black", marker="s")
                    plt.gca().invert_yaxis()
                    plt.show()
                    plt.close()

                    labels = c._hierarchical_cluster(points_xy)

                    if points_xy.size > 0 and labels.size == points_xy.shape[0]:
                        rng = np.random.default_rng(42)
                        uniq = np.unique(labels)
                        color_map = {lb: (rng.random(), rng.random(), rng.random()) for lb in uniq}
                        for lb in uniq:
                            mask = labels == lb
                            plt.scatter(points_xy[mask, 0], points_xy[mask, 1],
                                        s=1, c=[color_map[lb]], marker="s", label=f"cluster {lb}")
                        if len(uniq) <= 20:
                            plt.legend(loc="upper right", fontsize="small", markerscale=10)
                    plt.gca().invert_yaxis()
                    plt.show()

                    page = doc.load_page(pidx)
                    high_img = chunker._render_page_high(page)
                    w_high, h_high = high_img.size
                    remove_right = (1/20) * w_high; remove_left = (1/10) * w_high
                    high_img = high_img.crop((remove_right, 0, w_high-remove_left, h_high))
                    w_high, h_high = high_img.size
                    w_low, h_low = w, h

                    boxes_low = chunker._boxes_from_labels(points_xy, labels,
                                                        chunker.cfg.padding_low_px,
                                                        (w_low, h_low))
                    draw = ImageDraw.Draw(gray)
                    for box in boxes_low:
                        draw.rectangle(box, outline='red', width=3)
                    gray.show()
                    boxes_high = chunker._scale_boxes_low_to_high(boxes_low,
                                                                (w_low, h_low),
                                                                (w_high, h_high))
                    draw2 = ImageDraw.Draw(high_img)
                    for box in boxes_high:
                        draw2.rectangle(box, outline='red', width=3)
                    high_img.save('output.png')