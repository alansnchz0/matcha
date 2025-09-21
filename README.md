# Matcha Receipt OCR

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

## Phase 2 (REVISED): Text & Layout Extraction with Keras-OCR

Use deep learning OCR to extract both text and its location from receipts.

### Key Concept

Keras-OCR performs two steps:
- **Detection:** Finds where the text is.
- **Recognition:** Reads the text in those locations.

The output is a list of `(word, bounding_box_coordinates)` tuples—this location data is invaluable.

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