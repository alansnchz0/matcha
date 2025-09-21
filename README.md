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