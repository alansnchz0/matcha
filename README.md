# Matcha Receipt OCR

## Image Preprocessing for Faded Receipts (Recommended)

Enhance faint thermal prints and uneven backgrounds before OCR.

### Why

- Uses illumination correction, CLAHE (local contrast), unsharp masking, denoising, adaptive/Otsu binarization, and morphology to improve OCR on low-ink receipts.

### Install

```
pip install numpy opencv-python-headless
```

### Usage

```
python preprocess_images.py <input_images_dir> <output_images_dir> [threshold] [--options...]
```

Common examples:

- Faded receipts (auto):
  ```
  python preprocess_images.py ./receipts_raw ./receipts_proc --method auto --scale 1.6 --clahe-clip 3.5 --debug
  ```
- Backward-compatible global thresholding:
  ```
  python preprocess_images.py ./receipts_raw ./receipts_proc 140 --method global
  ```

Key options:

- `--method`: `auto` | `adaptive` | `otsu` | `global`
- `--block-size`, `--C`: adaptive threshold settings (auto block-size if omitted)
- `--scale`: upscale factor to help small, faint text (default 1.5)
- `--clahe-clip`, `--clahe-grid`: local contrast params
- `--unsharp-sigma`, `--unsharp-amount`: sharpening
- `--denoise-h`: strength (0 disables)
- `--illumination-sigma`: background normalization
- `--morph-close`, `--morph-open`: morphology refinement
- `--invert`: invert output
- `--debug`: saves stages to `output_images_dir/_debug/<image>/`

### Recommended workflow

1. Preprocess images for quality normalization.
2. Run OCR on the processed images directory.

Example:

```
python preprocess_images.py ./receipts_raw ./receipts_proc --method auto --scale 1.6
python extract_text.py ./receipts_proc ./text_out
```

## Phase 1: Classic OCR Extraction

Extract text from receipt images using Tesseract OCR.

### Usage

1. **Install dependencies**
    ```
    pip install pytesseract opencv-python
    ```
2. **Run extraction**
    ```
    python extract_text.py <input_images_dir> <output_text_dir>
    ```
    - Each image will produce a `.txt` file with the extracted text.

---

## Phase 2: Text & Layout Extraction with Keras-OCR

**Objective:** Use a deep learning OCR model to extract text and its coordinates.  
This phase remains unchanged.

### Action

1. Use the `keras-ocr` pipeline to get a list of `(word, bounding_box)` tuples for each receipt.
2. Write a helper function to sort these tuples by their coordinates, reconstructing a logically ordered text string for each receipt. Save these as `.txt` files.

### Usage

1. **Install dependencies**
    ```
    pip install tensorflow keras-ocr
    ```
2. **Run extraction**
    ```
    python extract_text_with_layout.py <input_images_dir> <output_text_dir>
    ```
    - Each image will yield a `.txt` file. The text is reconstructed in logical reading order based on bounding box positions.

---

## Files

- `extract_text.py`: Classic OCR extraction with Tesseract.
- `extract_text_with_layout.py`: Deep learning-based extraction with Keras-OCR, including reading order reconstruction.

---

## Example Output

For each receipt image, a `.txt` file is created in the output directory containing the recognized text in logical order.