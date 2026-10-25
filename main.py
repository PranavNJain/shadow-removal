"""
Smart Shadow Detection and Removal in Outdoor Images
Using Color Space Analysis and Image Inpainting
------------------------------------------------------
CLI entry point.

Usage:
    python main.py --input samples/test1.jpg --outdir outputs

Produces, in outdir/<image_name>/:
    01_original.png
    02_shadow_mask.png
    03_mask_overlay.png
    04_result.png
"""

import argparse
import os
import cv2
import numpy as np

from src.detect_shadow import detect_shadow_mask, refine_mask_ycbcr
from src.remove_shadow import remove_shadow


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255  # BGR -> red
    alpha = 0.4
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, red, alpha, 0)[mask_bool]
    return overlay


def process_image(path: str, outdir: str, use_ycbcr_refine: bool = True):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    name = os.path.splitext(os.path.basename(path))[0]
    save_dir = os.path.join(outdir, name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Detect
    mask = detect_shadow_mask(image_bgr)
    if use_ycbcr_refine:
        mask = refine_mask_ycbcr(image_bgr, mask)

    # 2. Remove
    result = remove_shadow(image_bgr, mask)

    # 3. Save all stages
    cv2.imwrite(os.path.join(save_dir, "01_original.png"), image_bgr)
    cv2.imwrite(os.path.join(save_dir, "02_shadow_mask.png"), mask)
    cv2.imwrite(os.path.join(save_dir, "03_mask_overlay.png"), make_overlay(image_bgr, mask))
    cv2.imwrite(os.path.join(save_dir, "04_result.png"), result)

    print(f"[OK] {path} -> {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Shadow detection & removal pipeline")
    parser.add_argument("--input", required=True, help="Path to an image, or a folder of images")
    parser.add_argument("--outdir", default="outputs", help="Directory to save results")
    parser.add_argument("--no-refine", action="store_true", help="Disable YCbCr refinement step")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.isdir(args.input):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                  if f.lower().endswith(exts)]
        for f in files:
            process_image(f, args.outdir, use_ycbcr_refine=not args.no_refine)
    else:
        process_image(args.input, args.outdir, use_ycbcr_refine=not args.no_refine)


if __name__ == "__main__":
    main()
