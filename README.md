# Matcha Receipt OCR

End-to-end scripts to OCR receipt images using classic Tesseract-based OCR and a deep-learning pipeline with `keras-ocr`, plus a simple image pre-processing utility.

---

## Prerequisites

- Python 3.9–3.11 recommended
- System packages:
  - Tesseract OCR engine
  - OpenCV runtime dependency for headless environments

On Debian/Ubuntu:
```bash
sudo apt-get update \
  && sudo apt-get install -y tesseract-ocr libgl1
```

---

## Quickstart

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Scripts

### 1) Image Preprocessing (binarization)

Use thresholding to generate cleaner, high-contrast images that can improve OCR quality.

```bash
python preprocess_images.py <input_dir> <output_dir> [threshold]
```

Notes:
- `threshold` defaults to 128 (0–255). Adjust as needed for your images.
- Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`.

### 2) Phase 1: Classic OCR Extraction (Tesseract)

Extract text from receipt images using Tesseract via `pytesseract`.

```bash
python extract_text.py <input_images_dir> <output_text_dir>
```

Output:
- A `.txt` file per image in `<output_text_dir>` containing the recognized text.

### 3) Phase 2: Text & Layout Extraction (keras-ocr)

Use a deep learning OCR model to extract words and their bounding boxes, then reconstruct reading order for receipt-like layouts.

```bash
python extract_text_with_layout.py <input_images_dir> <output_text_dir>
```

Details:
- Internally initializes a `keras_ocr.pipeline.Pipeline()` and recognizes words with quadrilateral boxes.
- Words are sorted top-to-bottom, then left-to-right per row; output is written as `.txt` per image.

---

## Troubleshooting

- ModuleNotFoundError: Ensure your virtualenv is active and `pip install -r requirements.txt` completed successfully.
- Tesseract not found: Install `tesseract-ocr` (see prerequisites) and make sure the `tesseract` binary is on your `PATH`.
- TensorFlow install issues on Linux: If you have a GPU or specialized environment, you may need a different TensorFlow build. The default requirement targets CPU.
- OpenCV errors about GUI backends: Installing `libgl1` usually resolves headless import errors for `opencv-python`.

---

## Files

- `preprocess_images.py`: Binarize images with a configurable threshold.
- `extract_text.py`: Classic OCR using Tesseract via `pytesseract`.
- `extract_text_with_layout.py`: Deep learning-based OCR with `keras-ocr`, reconstructing reading order.

---

## Contributing

Pull requests are welcome. Please run formatters/linters and include a brief description of changes and testing steps.