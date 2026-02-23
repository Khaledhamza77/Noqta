# Noqta

Noqta is a lightweight Python package that takes PDF files, detects regions of interest (text blocks, figures, tables, stamps, etc.), and produces high-quality crops ready for OCR. It uses a combination of classical computer-vision preprocessing and unsupervised clustering to find robust crop boxes without machine learning models or training data.

---

**Key capabilities**

- **Fast preprocessing**: dynamic DPI calculation, binarization, optional smudging (distance transforms or EDT), and page cropping.
- **Unsupervised clustering**: hierarchical clustering on binarized / smudged pixel coordinates to identify coherent regions.
- **High-resolution chunking**: transform clusters into boxes and extract crops from higher-DPI renderings for better OCR results.
- **Non-maximum suppression & post-processing**: merge and filter overlapping boxes to reduce duplicates and false positives. Also, it splits tall boxes based on horizontal whitespace / lines to improve OCR.

---

**Requirements**

- Python 3.8 or newer
- PyMuPDF (fitz)
- Pillow
- NumPy
- scikit-learn
- matplotlib (optional, used for visualization)
- OpenCV (cv2)

You can install the minimal runtime dependencies with pip (recommended in a virtualenv):

```bash
pip install -r requirements.txt
```

Or install directly from the repository (editable install for development):

```bash
# from repository root
pip install -e .
```

---

**Quick Start (Python API)**

1. Prepare a configuration dictionary (you can save it as JSON and load it):

```json
{
  "clusterer": {
    "max_px": 700,
    "invert": false,
    "use_otsu": true,
    "use_smudging": true,
    "type_smudging": "edt",
    "threshold_edt": 1,
    "hier_distance_threshold": 3.0,
    "hier_linkage": "single"
  },
  "chunker": {
    "high_dpi": 300,
    "padding_low_px": 5,
    "save_format": "PNG"
  },
  "suppressor": {
    "overlap_threshold": 0.60,
    "pad_union_px": 8,
    "remove_duplicates": true
  },
  "splitter": {
    "new_w": 400,
    "edt_threshold": 1,
    "high_box_threshold": 0.5,
    "max_ratio": [5, 9],
    "min_width": 0.95,
    "min_length_to_split": 30
  }
}
```

2. Use the `NOQTA` entrypoint from `noqta.app`:

```python
import json
from pathlib import Path
from noqta.app import NOQTA

config = json.loads(Path('config_noqta.json').read_text())

# input_path can be a folder of PDFs or a single PDF file
NOQTA('./input', config).run()
```

The package is designed so that the `NOQTA` class does the high-level orchestration: it renders pages, runs the clusterer, converts clusters to boxes (chunker), and applies suppression and post-processing (suppressor).

---

**Configuration options (overview)**

- `clusterer`:
  - `max_px` (int): max side length used for downsampling when detecting clusters.
  - `invert` (bool): invert image colors before binarization.
  - `use_otsu` (bool): compute threshold via Otsu when true.
  - `use_smudging` (bool): apply smudging (distance transform or EDT) to connect components.
  - `type_smudging` ("edt"|"blur"): smudging algorithm.
  - `threshold_edt` (float): threshold for EDT-based smudging.
  - `hier_distance_threshold` (float): linkage distance for hierarchical clustering.
  - `hier_linkage` (str): linkage method for hierarchical clustering (e.g., `single`, `complete`).

- `chunker`:
  - `high_dpi` (int): DPI used when extracting final crops (higher yields better OCR results).
  - `padding_low_px` (int): padding to add to low-res boxes before upscaling.
  - `save_format` (str): `PNG` or `JPEG` for saved crops.

- `suppressor`:
  - `overlap_threshold` (float): IoU threshold for non-max suppression.
  - `pad_union_px` (int): pixels to pad bounding boxes when computing unions.
  - `remove_duplicates` (bool): drop duplicate/similar boxes.

- `splitter`:
  - `new_w` (int): width (px) to resize tall/high-resolution boxes before looking for horizontal split lines.
  - `edt_threshold` (int): distance threshold for Euclidean Distance Transform used when preparing boxes for line detection.
  - `high_box_threshold` (float): fraction of page height above which a box is considered "high" (tall) and may be split.
  - `max_ratio` (tuple[int,int]): aspect ratio control used to determine candidate split densities.
  - `min_width` (float): minimum fraction of image width that a horizontal run must cover to be considered a splitting line.
  - `min_length_to_split` (int): minimum vertical length in pixels for a split to be considered valid.

Refer to the inline docstrings in the `noqta/vision` modules for fine-grained parameter behavior.

---

**Output layout**

By default the library writes crops and visualizations under `noqta_crops/` with the structure below:

```
noqta_crops/
├── document_1/
│   ├── page_0/
│   │   ├── 1_page_0_gray.png                   # preprocessed grayscale page used for detection
│   │   ├── 2_page_0_smudged.png                # smudged / connected components image
│   │   ├── 3_page_0_detected_frames.png        # detected page frames/edges visualization
│   │   ├── 4_page_0_frames_removed.png         # image after frame removal
│   │   ├── 5_page_0_binarized.png              # binarized image used to extract black pixels
│   │   ├── 6_page_0_points.png                 # black-pixel point map used for clustering
│   │   ├── 7_page_0_clusters.png               # clusters plotted on point map
│   │   ├── 8_page_0_boxes_processed1.png       # low-res boxes drawn on low-res image
│   │   ├── 9_page_0_small_boxes.png            # small boxes visualization
│   │   ├── 10_page_0_small_boxes_merged.png    # merged small boxes visualization
│   │   ├── 11_page_0_final_boxes_low.png       # final low-res boxes visualization
│   │   ├── 12_page_0_merged_final_boxes_low.png # merged final low-res boxes visualization
│   │   ├── 13_page_0_boxes_high.png            # high-res boxes visualization
│   │   ├── boxes/                              # final crops and split-box logs
│   │   │   ├── box_0.png                       # cropped high-res box (non-split)
│   │   │   ├── box_1.png
│   │   │   ├── long_box_2/                         # tall-box that was split
│   │   │   │   ├── 1_horizontal_white.png          # horizontal white runs detected in resized box
│   │   │   │   ├── 2_horizontal_black.png          # horizontal black runs detected in resized box
│   │   │   │   └── 3_detected_splitting_points.png # detected splitting points overlay
│   │   │   ├── box_2_0.png                         # cropped high-res split part
│   │   │   └── box_2_1.png
│   └── page_1/ ...
└── document_m/
```

Each crop image is saved using the format specified in `chunker.save_format` and file names include the document and page indexes for traceability.

Additionally, `NOQTA.run()` saves a timing CSV for performance inspection at:

```
noqta_crops/timing_records.csv
```

---

**NOQTA.run pipeline (what happens per page)**

The `NOQTA.run()` method orchestrates the whole pipeline and saves intermediate visualizations for debugging and reproducibility. Stages (and the file names the pipeline writes) are:

- `render_and_preprocess`: render page at a low/resized DPI and save the grayscale image (`1_page_{pidx}_gray.png`).
- `smudging_edt_or_dilation`: optionally apply EDT or dilation to connect fragments and save (`2_page_{pidx}_smudged.png`).
- `frame_removal_and_binarization`: detect and remove page frames/edges, then binarize the page; saves frame detections (`3_page_{pidx}_detected_frames.png`), frame-removed image (`4_page_{pidx}_frames_removed.png`) and the binarized image (`5_page_{pidx}_binarized.png`).
- `extract_points_and_cluster`: extract black-pixel coordinates, save the point map (`6_page_{pidx}_points.png`), perform hierarchical clustering and save clusters (`7_page_{pidx}_clusters.png`).
- `box_extraction_and_suppression`: transform clusters to low-res boxes, apply suppressor and save processed-box visualizations (`8_page_{pidx}_boxes_processed1.png`).
- `suppressor_merge_and_remove_small`: merge/clean small boxes and save visualizations (`9_page_{pidx}_small_boxes.png`, `10_page_{pidx}_small_boxes_merged.png`).
- `box_scaling_and_splitting`: render the page at high DPI and extract high-resolution crops for each final box. Crops are written under the page `boxes/` directory as `box_{i}.png` or, for tall boxes, a `long_box_{i}/` subfolder with debugging images and split parts (e.g., `1_horizontal_white.png`, `2_horizontal_black.png`, `3_detected_splitting_points.png`, `box_{i}_{j}.png`).

Notes:

- The `show_imgs` argument to `NOQTA.run()` can be used during development to open images as they are generated.
- The numbering in file names (1..10) corresponds to the major pipeline steps and makes it easy to inspect intermediate outputs when tuning parameters.

---

**Development & Contributing**

- Run tests (if present) with `pytest`.
- Use an editable install for development: `pip install -e .` and update `requirements.txt` when adding packages.
- Contributions: fork, create a feature branch, add tests, then open a pull request. Keep changes focused and include a short example demonstrating behavior.

---

**Troubleshooting**

- If crops contain excessive whitespace, increase `clusterer.max_px` or lower `hier_distance_threshold`.
- If boxes are fragmented, enable `clusterer.use_smudging` and try `type_smudging: "edt"`.
- For fuzzy OCR results, increase `chunker.high_dpi` (e.g., 300) when extracting final crops.