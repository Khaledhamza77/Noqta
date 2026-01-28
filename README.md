# Noqta

This is a python package which chunks and splits any .pdf document into areas of interest where OCR can be performed reliably. This is done through traditional computer vision techniques and unsupervised learning.

---

## Features

- Clusterer: Recieves a pdf page and processes it as a resized image using dynamic DPI calculation. Binarization, Smudging, and cropping are done followed by hierarchical clustering.
- Chunker: Clusters are transformed into boxes and these boxes are upscaled to match a higher resolution image of the page with no smudges or downsizing.
- Suppressor: Boxes are non-maximum suppressed and post-processed to yield best results.

---

## Requirements

- Python >= 3.8
- PyMuPDF
- Pillow
- NumPy
- Scikit-Learn
- MatPlotLib
- OpenCV

---

## Installation

Clone the repository and install dependencies:

```bash
pip install noqta@git+https://github.com/Khaledhamza77/Noqta
```

---

## Sample Usage

This package is highly configurable. It expects a dictionary to configure the application layer NOQTA. Here are the suggested configs based on some sample cases:

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
    "high_dpi": 100,
    "padding_low_px": 0,
    "save_format": "PNG"
  },
  "suppressor": {
    "overlap_threshold": 0.60,
    "pad_union_px": 0,
    "remove_duplicates": true
  }
}
```

After this config is saved and read it should be sent as a parameter to NOQTA, and the following should run the basic use-case where *input* directory has a number of PDF files:

```python
import json
with open("./config_noqta.json", "r") as f:
    config = json.load(f)

from noqta.app import NOQTA
NOQTA('./input', config).run()
```

Afterwards, the output should follow the following format:

```bash
noqta_crops/
├── document_1
│   ├── page_0
│   │   ├── page_0_cluster.png #black pixels plotted and color coded according to cluster lables after hierch. clust.
│   │   ├── page_0_gray.png #grayscale image after dynamic dpi calc., cropping, and converting to grayscale.
│   │   ├── page_0_high_boxes.png #bounding boxes drawn on the higher definition version of the pdf page
│   │   └── page_0_points.png #a plot of black pixels as points in cartesian coordinates after binarization and smudging (this is the input to clustering)
│   ├── page_1
│   │   ├── page_1_cluster.png
│   │   ├── page_1_gray.png
│   │   ├── page_1_high_boxes.png
│   │   └── page_1_points.png
│   │   ...
│   └── page_n
│       ├── page_n_cluster.png
│       ├── page_n_gray.png
│       ├── page_n_high_boxes.png
│       └── page_n_points.png
│   ...
└── document_m
    ├── page_0
    │   ├── page_0_cluster.png
    │   ├── page_0_gray.png
    │   ├── page_0_high_boxes.png
    │   └── page_0_points.png
    ├── page_1
    │   ├── page_1_cluster.png
    │   ├── page_1_gray.png
    │   ├── page_1_high_boxes.png
    │   └── page_1_points.png
    │   ...
    └── page_k
        ├── page_k_cluster.png
        ├── page_k_gray.png
        ├── page_k_high_boxes.png
        └── page_k_points.png
```