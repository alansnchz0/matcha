import cv2
import os
import sys

def preprocess_image(input_path, output_path, threshold=128):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read image: {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binarization (thresholding)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Save the processed image
    cv2.imwrite(output_path, binary)
    print(f"Processed and saved: {output_path}")

def batch_preprocess_images(input_dir, output_dir, threshold=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path, threshold)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_images.py <input_dir> <output_dir> [threshold]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    batch_preprocess_images(input_dir, output_dir, threshold)