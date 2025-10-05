import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np

def compute_auto_block_size(
    image_shape: Tuple[int, int],
    *,
    target_fraction: float = 0.02,
    min_size: int = 15,
    max_size: int = 55,
) -> int:
    """Compute an odd block size based on image size, clamped to sensible bounds."""
    height, width = image_shape[:2]
    size = int(round(min(height, width) * target_fraction))
    if size % 2 == 0:
        size += 1
    size = max(min_size, min(size, max_size))
    return size


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    new_size = (
        int(round(image.shape[1] * scale)),
        int(round(image.shape[0] * scale)),
    )
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, new_size, interpolation=interpolation)


def correct_illumination(gray: np.ndarray, *, sigma: float = 15.0) -> np.ndarray:
    """Normalize background by dividing by a heavily blurred version."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gray_f = gray.astype(np.float32)
    background_f = background.astype(np.float32) + 1.0
    corrected = cv2.divide(gray_f, background_f, scale=255.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_clahe(
    gray: np.ndarray, *, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def unsharp_mask(
    image: np.ndarray, *, sigma: float = 1.0, amount: float = 1.5
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray, *, h: int = 10) -> np.ndarray:
    if h <= 0:
        return gray
    return cv2.fastNlMeansDenoising(
        gray, None, h=h, templateWindowSize=7, searchWindowSize=21
    )


def adaptive_binarize(gray: np.ndarray, *, block_size: int, C: int) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def choose_best_binary(adaptive_img: np.ndarray, otsu_img: np.ndarray) -> np.ndarray:
    """Pick the binarization whose black-pixel ratio is within a good range."""

    def black_ratio(img: np.ndarray) -> float:
        return float(np.count_nonzero(img == 0)) / img.size

    r_adapt = black_ratio(adaptive_img)
    r_otsu = black_ratio(otsu_img)
    target_min, target_max = 0.01, 0.35

    def score(r: float) -> float:
        if r < target_min:
            return - (target_min - r) * 10
        if r > target_max:
            return - (r - target_max) * 10
        mid = (target_min + target_max) / 2.0
        return 1.0 - abs(r - mid)

    return adaptive_img if score(r_adapt) >= score(r_otsu) else otsu_img


def morphology_refine(
    binary: np.ndarray, *, close_size: int = 1, open_size: int = 1
) -> np.ndarray:
    result = binary
    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k, iterations=1)
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k, iterations=1)
    return result


class DebugSaver:
    def __init__(self, enabled: bool, base_dir: str, base_name: str):
        self.enabled = enabled
        if enabled:
            self.dir = os.path.join(base_dir, "_debug", base_name)
            os.makedirs(self.dir, exist_ok=True)

    def save(self, image: np.ndarray, stage: str) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.dir, f"{stage}.png")
        cv2.imwrite(path, image)


def preprocess_image(
    input_path,
    output_path,
    threshold=128,
    *,
    method: str = "auto",
    block_size: int | None = None,
    C: int = 10,
    scale: float = 1.5,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,
    denoise_h: int = 10,
    illumination_sigma: float = 15.0,
    morph_close: int = 2,
    morph_open: int = 1,
    invert: bool = False,
    debug_saver: DebugSaver | None = None,
):
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"Could not read image: {input_path}")
        return

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if scale and abs(scale - 1.0) > 1e-3:
        gray = resize_image(gray, scale)
    if debug_saver:
        debug_saver.save(gray, "01_gray")

    corrected = correct_illumination(gray, sigma=illumination_sigma)
    if debug_saver:
        debug_saver.save(corrected, "02_illumination")

    contrasted = apply_clahe(
        corrected, clip_limit=clahe_clip, tile_grid_size=(clahe_grid, clahe_grid)
    )
    if debug_saver:
        debug_saver.save(contrasted, "03_clahe")

    sharpened = unsharp_mask(
        contrasted, sigma=unsharp_sigma, amount=unsharp_amount
    )
    if debug_saver:
        debug_saver.save(sharpened, "04_unsharp")

    denoised = denoise(sharpened, h=denoise_h)
    if debug_saver:
        debug_saver.save(denoised, "05_denoise")

    if method == "global":
        _, binary = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY)
    else:
        if block_size is None:
            block_size = compute_auto_block_size(
                denoised.shape, target_fraction=0.02, min_size=15, max_size=55
            )
        else:
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
        adaptive = adaptive_binarize(denoised, block_size=block_size, C=C)
        otsu = otsu_binarize(denoised)
        if method == "adaptive":
            binary = adaptive
        elif method == "otsu":
            binary = otsu
        else:  # auto
            binary = choose_best_binary(adaptive, otsu)
    if debug_saver:
        debug_saver.save(binary, "06_threshold")

    refined = morphology_refine(
        binary, close_size=morph_close, open_size=morph_open
    )
    if debug_saver:
        debug_saver.save(refined, "07_morph")

    if invert:
        refined = cv2.bitwise_not(refined)
        if debug_saver:
            debug_saver.save(refined, "08_invert")

    cv2.imwrite(output_path, refined)
    print(f"Processed and saved: {output_path}")

def batch_preprocess_images(input_dir, output_dir, threshold=128, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        ):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            base_name = os.path.splitext(filename)[0]
            debug_saver = DebugSaver(kwargs.get("debug", False), output_dir, base_name)
            preprocess_image(
                input_path,
                output_path,
                threshold,
                method=kwargs.get("method", "auto"),
                block_size=kwargs.get("block_size", None),
                C=kwargs.get("C", 10),
                scale=kwargs.get("scale", 1.5),
                clahe_clip=kwargs.get("clahe_clip", 3.0),
                clahe_grid=kwargs.get("clahe_grid", 8),
                unsharp_sigma=kwargs.get("unsharp_sigma", 1.0),
                unsharp_amount=kwargs.get("unsharp_amount", 1.5),
                denoise_h=kwargs.get("denoise_h", 10),
                illumination_sigma=kwargs.get("illumination_sigma", 15.0),
                morph_close=kwargs.get("morph_close", 2),
                morph_open=kwargs.get("morph_open", 1),
                invert=kwargs.get("invert", False),
                debug_saver=debug_saver,
            )

def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Preprocess receipt images for better OCR; robust to faded prints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to write processed images")
    parser.add_argument(
        "threshold",
        nargs="?",
        type=int,
        default=128,
        help="Global threshold (kept for backward compatibility)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "adaptive", "otsu", "global"],
        default="auto",
        help="Binarization method",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Adaptive threshold block size (odd). Auto if omitted.",
    )
    parser.add_argument("--C", type=int, default=10, help="Constant for adaptive threshold")
    parser.add_argument(
        "--scale", type=float, default=1.5, help="Upscale factor to enhance small text"
    )
    parser.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE clip limit")
    parser.add_argument(
        "--clahe-grid", type=int, default=8, help="CLAHE tile grid size (N => NxN)"
    )
    parser.add_argument(
        "--unsharp-sigma", type=float, default=1.0, help="Unsharp mask Gaussian sigma"
    )
    parser.add_argument(
        "--unsharp-amount", type=float, default=1.5, help="Unsharp mask amount"
    )
    parser.add_argument(
        "--denoise-h", type=int, default=10, help="Denoising strength (0 disables)"
    )
    parser.add_argument(
        "--illumination-sigma",
        type=float,
        default=15.0,
        help="Sigma for illumination correction blur",
    )
    parser.add_argument(
        "--morph-close",
        type=int,
        default=2,
        help="Morphological close kernel size (0 to skip)",
    )
    parser.add_argument(
        "--morph-open",
        type=int,
        default=1,
        help="Morphological open kernel size (0 to skip)",
    )
    parser.add_argument("--invert", action="store_true", help="Invert output binary")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate stages to output_dir/_debug/<image>/",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    batch_preprocess_images(
        args.input_dir,
        args.output_dir,
        args.threshold,
        method=args.method,
        block_size=args.block_size,
        C=args.C,
        scale=args.scale,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
        denoise_h=args.denoise_h,
        illumination_sigma=args.illumination_sigma,
        morph_close=args.morph_close,
        morph_open=args.morph_open,
        invert=args.invert,
        debug=args.debug,
    )