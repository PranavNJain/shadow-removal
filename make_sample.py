"""
Generates a synthetic test image (ground plane + object casting a soft
shadow) so the pipeline can be validated before you plug in the real
ISTD/SRD dataset. Not a substitute for real data in your final report.
"""

import numpy as np
import cv2

def make_synthetic_shadow_image(path="samples/synthetic1.jpg", size=500):
    h = w = size
    # Base "ground" texture: sunlit pavement/grass tone with slight noise
    base_color = np.array([90, 160, 200], dtype=np.float32)  # BGR, warm lit ground
    img = np.tile(base_color, (h, w, 1))
    noise = np.random.normal(0, 6, (h, w, 3))
    img += noise

    # Add a subtle texture pattern (tile lines) so inpainting has something to preserve
    for i in range(0, h, 25):
        img[i:i+1, :, :] -= 10
    for j in range(0, w, 25):
        img[:, j:j+1, :] -= 10

    # Create an elliptical soft shadow (simulating a tree/building shadow)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (int(w*0.55), int(h*0.55)), (140, 90), 30, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)  # soft penumbra edge

    # Shadow effect: darken + slight blue shift (like open-sky illumination)
    shadow_strength = 0.45
    img[:, :, 0] += mask * 15          # B up slightly (bluer shadow)
    img[:, :, 1] -= mask * base_color[1] * shadow_strength
    img[:, :, 2] -= mask * base_color[2] * shadow_strength

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    print(f"Saved synthetic test image to {path}")

if __name__ == "__main__":
    import os
    os.makedirs("samples", exist_ok=True)
    make_synthetic_shadow_image()
