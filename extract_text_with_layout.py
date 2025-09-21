import os
import sys
import numpy as np
import keras_ocr

def sort_words_by_layout(words_boxes):
    """
    Sort words top-to-bottom, then left-to-right (like reading English text).
    Each entry is (word, box), where box is a 4x2 numpy array of (x, y) points.
    """
    # Calculate mean y (vertical) of each box for sorting
    lines = []
    for word, box in words_boxes:
        y_mean = np.mean(box[:, 1])
        x_min = np.min(box[:, 0])
        lines.append((y_mean, x_min, word, box))
    # Sort: first by y_mean (row), then by x_min (column)
    lines.sort()
    # Simple line grouping: group words within the same line (tolerance)
    grouped_lines = []
    current_line = []
    last_y = None
    tolerance = 15  # pixels; adjust as needed for your receipts
    for y, x, word, box in lines:
        if last_y is None or abs(y - last_y) < tolerance:
            current_line.append((x, word))
        else:
            grouped_lines.append(current_line)
            current_line = [(x, word)]
        last_y = y
    if current_line:
        grouped_lines.append(current_line)
    # Sort words in each line left-to-right, then join
    text_lines = []
    for line in grouped_lines:
        line.sort()
        text_lines.append(" ".join(word for x, word in line))
    return "\n".join(text_lines)

def extract_with_keras_ocr(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Set up keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        # Read image
        image = keras_ocr.tools.read(image_path)
        # Run pipeline (returns [ [(word, box), ...] ])
        prediction_groups = pipeline.recognize([image])
        words_boxes = prediction_groups[0]
        # Save layout-ordered text
        text = sort_words_by_layout(words_boxes)
        txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Processed {filename}: {len(words_boxes)} words found. Text saved to {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_text_with_layout.py <input_images_dir> <output_text_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    extract_with_keras_ocr(input_dir, output_dir)