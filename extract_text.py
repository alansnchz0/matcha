import pytesseract
import cv2
import os
import sys

def extract_text_from_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return ""
    # OCR with pytesseract
    text = pytesseract.image_to_string(img)
    return text

def batch_extract_text(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            image_path = os.path.join(input_dir, filename)
            text = extract_text_from_image(image_path)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text saved to: {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_text.py <input_images_dir> <output_text_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch_extract_text(input_dir, output_dir)