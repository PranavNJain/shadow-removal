"""
Shadow Detection using LAB Color Space Analysis
-------------------------------------------------
Based on the classic Murali & Govindan (2013) approach:
"Shadow Detection and Removal from a Single Image Using LAB Color Space"

Key idea:
- Shadows lower Lightness (L) but keep chromaticity (a, b) roughly stable,
  unlike dark objects which affect all channels differently.
- We adaptively pick whether to threshold on L alone, or on L+a+b combined,
  depending on how "colorful" the image is on average.
"""

import cv2
import numpy as np


def detect_shadow_mask(image_bgr: np.ndarray, min_area: int = 150) -> np.ndarray:
    """
    Detect shadow regions in an outdoor image using LAB color space analysis.

    Args:
        image_bgr: Input image in BGR (as read by cv2.imread).
        min_area: Minimum connected-component area (in pixels) to keep as a
                   valid shadow region. Removes small noisy blobs.

    Returns:
        mask: uint8 binary mask (0 = non-shadow, 255 = shadow), same H x W as input.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    mean_L, mean_A, mean_B = L.mean(), A.mean(), B.mean()
    std_L = L.std()

    # LAB in OpenCV: A and B are stored 0-255 with 128 as the neutral point.
    # The original paper works in the -128..127 range, so we recenter for the
    # "how colorful is this image" check.
    colorfulness = (mean_A - 128) + (mean_B - 128)

    if colorfulness <= 15:
        # Low color cast overall -> rely on Lightness only
        threshold = mean_L - (std_L / 3.0)
        shadow_bool = L < threshold
    else:
        # Combine L, A, B (a common extension when the scene has strong color)
        combined = L + A + B
        mean_c, std_c = combined.mean(), combined.std()
        threshold = mean_c - (std_c / 3.0)
        shadow_bool = combined < threshold

    mask = (shadow_bool.astype(np.uint8)) * 255

    # --- Morphological cleanup ---
    # Larger kernel than before: real photos (grass, concrete grain, etc.) have
    # a lot of fine dark texture that a small kernel doesn't remove.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Remove small / spurious connected components ---
    # Textured outdoor surfaces (grass, gravel, rough concrete) commonly throw
    # off dozens of small dark blobs that pass a naive area check individually.
    # Real shadows are usually one dominant contiguous region, so in addition
    # to an absolute area floor (scaled to image size), we drop any component
    # that's much smaller than the largest detected blob.
    h, w = mask.shape[:2]
    abs_min_area = max(min_area, int(0.0015 * h * w))  # ~0.15% of image area
    relative_min_frac = 0.15  # component must be >=15% the size of the biggest one

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    rel_min_area = max_area * relative_min_frac

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= abs_min_area and area >= rel_min_area:
            cleaned[labels == i] = 255

    return cleaned


def refine_mask_ycbcr(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Optional secondary check using YCbCr: shadows typically show Cb (blue-diff)
    higher and Cr (red-diff) lower than the surrounding lit area. We use this
    to drop candidate shadow pixels that don't agree, reducing false positives
    from dark (but non-shadow) objects like tree trunks or dark pavement.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    shadow_pixels = mask > 0
    if shadow_pixels.sum() == 0 or (~shadow_pixels).sum() == 0:
        return mask

    lit_Cb_mean = Cb[~shadow_pixels].mean()
    lit_Cr_mean = Cr[~shadow_pixels].mean()

    agree = (Cb.astype(np.int16) >= int(lit_Cb_mean) - 5) & \
            (Cr.astype(np.int16) <= int(lit_Cr_mean) + 5)

    refined = np.where(shadow_pixels & agree, 255, 0).astype(np.uint8)

    # keep at least the original mask if refinement wipes everything out
    if refined.sum() == 0:
        return mask
    return refined
