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