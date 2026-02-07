from .configs import ClustererConfig, ChunkerConfig, SuppressorConfig
from .vision.suppressor import Suppressor
from .vision.clusterer import Clusterer
from .vision.scissors import Scissors
from matplotlib import pyplot as plt
from .vision.chunker import Chunker
from PIL import ImageDraw
import fitz  # PyMuPDF
import numpy as np
import logging
import os

class NOQTA:
    def __init__(self, path: str, config: dict):
        logging.basicConfig(level=logging.INFO)
        self.config = config
        if os.path.isfile(path):
            self.paths = [path]
            logging.info(f"Single PDF file provided: {path}")
        elif os.path.isdir(path):
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.pdf')]
            logging.info(f"Directory provided. Found {len(self.paths)} PDF files.")
        self.local_dir = os.getcwd().replace("\\", "/")
        self.output_dir = os.path.join(self.local_dir, 'noqta_crops')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def configure(self):
        clfg = self.config.get("clusterer", {})
        cl_cfg = ClustererConfig(
            max_px=clfg.get("max_px", 600), # Longest side of page in pixels for speed
            invert=clfg.get("invert", False), # set True if your pages are white-on-black
            # Smudging to connect fragments (tune these)
            use_smudging=clfg.get("use_dilation", True),
            type_smudging=clfg.get("type_smudging", "edt"), # 'edt' or 'dilation'
            threshold_edt=clfg.get("threshold_edt", 1),
            dilate_radius_px=clfg.get("dilate_radius_px", 1),
            dilation_iterations=clfg.get("dilation_iterations", 1),

            # Hierarchical clustering parameters (pixel units)
            hier_distance_threshold=clfg.get("hier_distance_threshold", 3.0),
            hier_linkage=clfg.get("hier_linkage", "single") # 'single' tends to merge nearby components
        )
        cluster = Clusterer(cl_cfg)

        chcfg = self.config.get("chunker", {})
        ch_cfg = ChunkerConfig(
            zoom_rate=chcfg.get("zoom_rate", 3.0),                # how much larger the page size (to crop from) compared to page size of clusterer
            padding_low_px=chcfg.get("padding_low_px", 0),        # extra margin around clusters in LOW-DPI units
            save_format=chcfg.get("save_format", "PNG"),          # PNG or TIFF, etc.
            save_mode_1bit=chcfg.get("save_mode_1bit", False)     # set True if you want 1-bit crops
        )
        chunker = Chunker(ch_cfg)

        supcfg = self.config.get("suppressor", {})
        sup_cfg = SuppressorConfig(
            ratio_to_image_threshold=supcfg.get("ratio_to_image_threshold", 0.8),  # max fraction of page acceptable for height or width of a box
            overlap_threshold=supcfg.get("overlap_threshold", 0.60),
            pad_union_px=supcfg.get("pad_union_px", 0),
            remove_duplicates=supcfg.get("remove_duplicates", True)
        )
        suppressor = Suppressor(sup_cfg)
        logging.info("NOQTA configured successfully.")
        return cluster, chunker, suppressor

    @staticmethod
    def _get_doc_name(pdf_path: str) -> str:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_name = doc_name.replace(' ', '_')
        doc_name = doc_name.replace('.pdf', '')
        return doc_name

    def run(self, show_imgs: bool = False):
        clusterer, chunker, suppressor = self.configure()
        for pdf_path in self.paths:
            doc_name = self._get_doc_name(pdf_path)
            logging.info(f"Processing document: {doc_name}")
            os.makedirs(os.path.join(self.output_dir, doc_name), exist_ok=True)
            with fitz.open(pdf_path) as doc:
                page_indices = list(range(doc.page_count))
                for pidx in page_indices:
                    os.makedirs(os.path.join(self.output_dir, doc_name, f"page_{pidx}"), exist_ok=True)

                    gray, scale = clusterer._render_page_gray(doc, pidx)
                    w, h = gray.size
                    logging.info(f"Doc: {doc_name} -> Page {pidx}: rendered (crop and binarization) grayscale image of size {w}x{h}")

                    if show_imgs: gray.show() 
                    gray.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"1_page_{pidx}_gray.png"))

                    if clusterer.cfg.use_smudging:
                        if clusterer.cfg.type_smudging == 'dilation':
                            gray_smudged = clusterer._dilate(gray)
                        elif clusterer.cfg.type_smudging == 'edt':
                            gray_smudged = clusterer._edt(gray)
                        else:
                            raise ValueError(f"Unknown smudging type: {clusterer.cfg.type_smudging}")
                    else:
                        gray_smudged = gray
                    if show_imgs: gray_smudged.show()
                    gray_smudged.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"2_page_{pidx}_smudged.png"))

                    edges_detected, edges_removed = clusterer._remove_frames(img=gray_smudged)
                    edges_detected.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"3_page_{pidx}_detected_frames.png"))
                    edges_removed.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"4_page_{pidx}_frames_removed.png"))
                    if show_imgs: edges_detected.show()
                    del edges_detected

                    if clusterer.cfg.type_smudging != 'edt':
                        bin_l = clusterer._to_binary_L(edges_removed)
                        fp = f"5_page_{pidx}_binarized.png"
                        if show_imgs: bin_l.show()
                    else:
                        bin_l = edges_removed
                        fp = f"5_page_{pidx}_binarized_(eq_to_edt).png"
                        logging.info('Skipped Binarization since EDT is used for smudging!')
                    bin_l.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", fp))

                    points_xy = clusterer._extract_black_xy_from_L(bin_l)
                    logging.info(f"Doc: {doc_name} -> Page {pidx}: extracted {points_xy.shape[0]} black pixel points")
                    plt.scatter(points_xy[:, 0], points_xy[:, 1],
                                            s=1, c="black", marker="s")
                    plt.gca().invert_yaxis()
                    plt.xlim(0, w)
                    plt.ylim(h, 0)
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.title(f"Page {pidx} — Black Pixel Map")
                    plt.xlabel("x (px)")
                    plt.ylabel("y (px)")
                    plt.tight_layout()
                    if show_imgs:
                        plt.show()
                    plt.savefig(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"6_page_{pidx}_points.png"), dpi=150)
                    plt.close()

                    labels = clusterer._hierarchical_cluster(points_xy)
                    logging.info(f"Doc: {doc_name} -> Page {pidx}: formed {len(np.unique(labels))} clusters")

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
                    plt.xlim(0, w)
                    plt.ylim(h, 0)
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.title(f"Page {pidx} — Hierarchical Clusters")
                    plt.xlabel("x (px)")
                    plt.ylabel("y (px)")
                    plt.tight_layout()
                    if show_imgs: plt.show()
                    plt.savefig(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"7_page_{pidx}_clusters.png"), dpi=150)
                    plt.close()

                    page = doc.load_page(pidx)
                    high_img = chunker._render_page_high(page, scale)
                    w_high, h_high = high_img.size
                    w_low, h_low = w, h

                    boxes_low = chunker._boxes_from_labels(points_xy, labels,
                                                        chunker.cfg.padding_low_px,
                                                        (w_low, h_low))
                    
                    cleaned_boxes = suppressor.process(boxes_low, (w_low, h_low))
                    logging.info(f"Doc: {doc_name} -> Page {pidx}: reduced {len(boxes_low)} boxes to {len(cleaned_boxes)} after suppression")
                    os.makedirs(os.path.join(self.output_dir, doc_name, f"page_{pidx}", "boxes"), exist_ok=True)
                    
                    draw = ImageDraw.Draw(gray)
                    for box in cleaned_boxes:
                        draw.rectangle(box, outline='red', width=3)
                    if show_imgs: gray.show()
                    gray.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"8_page_{pidx}_boxes_processed1.png"))

                    small_boxes, merged_small_boxes, final_boxes = suppressor.merge_and_remove_small(self, cleaned_boxes, (w_low, h_low))

                    sb1 = gray.copy(); draw1 = ImageDraw.Draw(sb1)
                    for box in small_boxes:
                        draw1.rectangle(box, outline='red', width=3)
                    if show_imgs: sb1.show()
                    sb1.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"9_page_{pidx}_small_boxes.png"))

                    sb2 = gray.copy(); draw2 = ImageDraw.Draw(sb2)
                    for box in merged_small_boxes:
                        draw2.rectangle(box, outline='red', width=3)
                    if show_imgs: sb2.show()
                    sb2.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"10_page_{pidx}_small_boxes_merged.png"))

                    sb3 = gray.copy(); draw3 = ImageDraw.Draw(sb3)
                    for box in final_boxes:
                        draw3.rectangle(box, outline='red', width=3)
                    if show_imgs: sb3.show()
                    sb3.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"11_page_{pidx}_final_boxes_low.png"))


                    
                    boxes_high = chunker._scale_boxes_low_to_high(final_boxes,
                                                                (w_low, h_low),
                                                                (w_high, h_high))
                    himgBoxes = high_img.copy()
                    draw4 = ImageDraw.Draw(himgBoxes)
                    for box in boxes_high:
                        draw4.rectangle(box, outline='red', width=3)
                    
                    if show_imgs: himgBoxes.show()
                    himgBoxes.save(os.path.join(self.output_dir, doc_name, f"page_{pidx}", f"12_page_{pidx}_final_boxes_high.png"))
                    
                    Scissors.crop_image_by_boxes(
                        high_img,
                        boxes_high,
                        os.path.join(self.output_dir, doc_name, f"page_{pidx}", "boxes")
                    )