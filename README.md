# Image Preprocessing & OCR Extraction

This repository provides scripts for:
1. **Preprocessing images:** Convert images to grayscale and apply binarization (thresholding).
2. **Extracting text:** Use Tesseract OCR to convert processed images to raw, machine-readable text.

---

## Requirements

- Python 3.7+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [pytesseract](https://pypi.org/project/pytesseract/)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) (must be installed on your system)

## Installation

1. **Install Python dependencies:**
   ```
   pip install opencv-python pytesseract
   ```
2. **Install Tesseract-OCR:**
   - On Ubuntu:  
     `sudo apt-get install tesseract-ocr`
   - On Mac (Homebrew):  
     `brew install tesseract`
   - On Windows:  
     Download the [installer here](https://github.com/tesseract-ocr/tesseract/wiki).

---

## Phase 1: Image Preprocessing

Convert all images in a directory to grayscale and apply binarization.

**Usage:**
```sh
python preprocess_images.py <input_dir> <output_dir> [threshold]
```
- `<input_dir>`: Folder with original images.
- `<output_dir>`: Folder to save processed images.
- `[threshold]`: (Optional) Binarization threshold (default: 128).

**Example:**
```sh
python preprocess_images.py ./images ./images_bw 150
```

---

## Phase 2: Text Extraction with OCR

Extract raw text from the processed images using Tesseract OCR.

**Usage:**
```sh
python extract_text.py <input_images_dir> <output_text_dir>
```
- `<input_images_dir>`: Folder with binarized images (from Phase 1).
- `<output_text_dir>`: Folder to save extracted `.txt` files.

**Example:**
```sh
python extract_text.py ./images_bw ./ocr_output
```

---

## Notes

- The extracted text may contain errors; that's expected and useful for training AI models to improve OCR accuracy.
- For large batches, ensure enough disk space for output folders.
---
