"""
Shadow Removal via LAB Relighting + Boundary Inpainting
---------------------------------------------------------
Two-stage approach:

1. Relighting (illumination correction):
   For each LAB channel, compute the mean/std of pixel values in the
   shadow region and in a nearby lit "reference" region (a ring just
   outside the shadow boundary). Linearly rescale shadow pixels so their
   statistics match the lit region. This is the classic approach used in
   "Shadow Removal via Shadow Image Decomposition"-style relighting, and
   works far better than blind inpainting because it preserves texture.

2. Boundary inpainting:
   The relit region and shadow-free area will still show a seam. We
   inpaint a thin band around the shadow boundary using cv2.inpaint
   (Telea's algorithm) to blend the transition smoothly.
"""

import cv2
import numpy as np


def _get_reference_ring(mask: np.ndarray, ring_width: int = 15) -> np.ndarray:
    """Return a boolean mask for a ring of lit pixels just outside the shadow."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = (dilated > 0) & (mask == 0)
    return ring


def relight_shadow(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Channel-wise linear relighting of the shadow region in LAB space.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    shadow_bool = mask > 0
    ring_bool = _get_reference_ring(mask)

    if shadow_bool.sum() == 0:
        return image_bgr.copy()

    # Fallback to the whole non-shadow area if the ring is too small
    if ring_bool.sum() < 50:
        ring_bool = ~shadow_bool

    out_lab = lab.copy()
    for c in range(3):
        channel = lab[:, :, c]
        shadow_vals = channel[shadow_bool]
        lit_vals = channel[ring_bool]

        mean_s, std_s = shadow_vals.mean(), shadow_vals.std() + 1e-6
        mean_l, std_l = lit_vals.mean(), lit_vals.std() + 1e-6

        scale = std_l / std_s
        # Dampen extreme scale factors to avoid blowing out contrast
        scale = np.clip(scale, 0.5, 3.0)

        new_vals = (shadow_vals - mean_s) * scale + mean_l
        out_channel = channel.copy()
        out_channel[shadow_bool] = new_vals
        out_lab[:, :, c] = out_channel

    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    return result


def inpaint_boundary(image_bgr: np.ndarray, mask: np.ndarray, band_width: int = 7) -> np.ndarray:
    """
    Smooth the seam left by relighting using inpainting on a thin band
    around the shadow boundary (not the whole shadow region, so we don't
    lose the texture that relighting already recovered).
    """
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, edge_kernel)
    result = cv2.inpaint(image_bgr, boundary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result


def remove_shadow(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Full pipeline: relight the shadow region, then inpaint the boundary seam.
    """
    relit = relight_shadow(image_bgr, mask)
    final = inpaint_boundary(relit, mask)
    return final
