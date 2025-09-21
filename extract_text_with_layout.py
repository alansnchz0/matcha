import os
import sys
import keras_ocr
import numpy as np

def sort_boxes(boxes):
    """
    Sorts detected boxes into logical reading order: top-to-bottom, left-to-right (for receipts).
    Each box is a tuple: (word, box), where box is a (4, 2) ndarray of coordinates.
    """
    # Compute the y-center of each box for row assignment
    box_centers = [(i, np.mean(box[:, 1])) for i, (word, box) in enumerate(boxes)]
    # Sort by vertical position (top to bottom)
    box_centers.sort(key=lambda x: x[1])
    sorted_indices = [i for i, _ in box_centers]

    # Assign rows: group boxes that are at similar vertical positions
    rows = []
    current_row = []
    current_y = None
    row_thresh = 20  # pixels, adjust for your receipts

    for idx in sorted_indices:
        _, box = boxes[idx]
        box_y = np.mean(box[:, 1])
        if current_y is None or abs(box_y - current_y) < row_thresh:
            current_row.append(idx)
            current_y = box_y if current_y is None else (current_y + box_y) / 2
        else:
            # Sort current row left-to-right
            current_row.sort(key=lambda i: np.mean(boxes[i][1][:, 0]))
            rows.append(current_row)
            current_row = [idx]
            current_y = box_y
    if current_row:
        current_row.sort(key=lambda i: np.mean(boxes[i][1][:, 0]))
        rows.append(current_row)

    ordered_words = []
    for row in rows:
        ordered_words.extend([boxes[i][0] for i in row])
    return ' '.join(ordered_words)

def extract_text_with_layout(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        images = [keras_ocr.tools.read(image_path)]
        prediction_groups = pipeline.recognize(images)
        boxes = prediction_groups[0]  # list of (word, box)
        ordered_text = sort_boxes(boxes)
        output_file = os.path.splitext(image_file)[0] + '.txt'
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ordered_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_text_with_layout.py <input_images_dir> <output_text_dir>")
        sys.exit(1)
    extract_text_with_layout(sys.argv[1], sys.argv[2])