"""
Evaluation script for the ISTD test set.

Compares:
- Detection: your predicted mask vs ISTD's ground-truth mask (test_B) -> IoU, F1
- Removal:   your result vs ISTD's ground-truth shadow-free image (test_C) -> RMSE, SSIM
             (computed in LAB space over the shadow region, non-shadow region, and whole image,
              matching how papers report it)

Usage:
    python evaluate.py --istd_root /path/to/ISTD_Dataset --split test --limit 50

    --limit caps how many images to evaluate (omit to run all 540 test images;
    that's slow-ish on CPU, so start with --limit 20-50 to sanity check first)

Expects the ISTD folder layout:
    ISTD_Dataset/
        test/
            test_A/   shadow images
            test_B/   ground-truth shadow masks
            test_C/   ground-truth shadow-free images
"""

import argparse
import os
import csv
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from src.detect_shadow import detect_shadow_mask, refine_mask_ycbcr
from src.remove_shadow import remove_shadow


def iou_f1(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0

    tp = intersection
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return iou, f1


def rmse_lab(result_bgr, gt_bgr, region_mask=None):
    lab_r = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_g = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = (lab_r - lab_g) ** 2
    if region_mask is not None:
        m = region_mask > 0
        if m.sum() == 0:
            return None
        diff = diff[m]
    return float(np.sqrt(diff.mean()))


def evaluate(istd_root, split="test", limit=None):
    a_dir = os.path.join(istd_root, split, f"{split}_A")
    b_dir = os.path.join(istd_root, split, f"{split}_B")
    c_dir = os.path.join(istd_root, split, f"{split}_C")

    for d in (a_dir, b_dir, c_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected folder not found: {d}")

    files = sorted(os.listdir(a_dir))
    if limit:
        files = files[:limit]

    rows = []
    for fname in files:
        shadow_path = os.path.join(a_dir, fname)
        mask_gt_path = os.path.join(b_dir, fname)
        free_gt_path = os.path.join(c_dir, fname)

        if not (os.path.exists(mask_gt_path) and os.path.exists(free_gt_path)):
            print(f"[skip] missing pair for {fname}")
            continue

        img = cv2.imread(shadow_path)
        gt_mask = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
        gt_free = cv2.imread(free_gt_path)

        pred_mask = detect_shadow_mask(img)
        pred_mask = refine_mask_ycbcr(img, pred_mask)
        result = remove_shadow(img, pred_mask)

        # resize gt to match if needed
        if gt_mask.shape[:2] != img.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]))
        if gt_free.shape[:2] != img.shape[:2]:
            gt_free = cv2.resize(gt_free, (img.shape[1], img.shape[0]))

        iou, f1 = iou_f1(pred_mask, gt_mask)

        rmse_whole = rmse_lab(result, gt_free)
        rmse_shadow = rmse_lab(result, gt_free, region_mask=gt_mask)
        rmse_nonshadow = rmse_lab(result, gt_free, region_mask=255 - gt_mask)

        gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_g = cv2.cvtColor(gt_free, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_r, gray_g)

        rows.append({
            "file": fname,
            "iou": iou, "f1": f1,
            "rmse_whole": rmse_whole,
            "rmse_shadow": rmse_shadow,
            "rmse_nonshadow": rmse_nonshadow,
            "ssim": ssim_val,
        })
        print(f"{fname}: IoU={iou:.3f} F1={f1:.3f} RMSE(whole)={rmse_whole:.2f} "
              f"RMSE(shadow)={rmse_shadow:.2f} SSIM={ssim_val:.3f}")

    if not rows:
        print("No images evaluated.")
        return

    # Averages
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== AVERAGES over", len(rows), "images ===")
    for key in ["iou", "f1", "rmse_whole", "rmse_shadow", "rmse_nonshadow", "ssim"]:
        print(f"{key}: {avg(key):.4f}")

    out_csv = "evaluation_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--istd_root", required=True, help="Path to ISTD_Dataset folder")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to evaluate")
    args = parser.parse_args()
    evaluate(args.istd_root, args.split, args.limit)
