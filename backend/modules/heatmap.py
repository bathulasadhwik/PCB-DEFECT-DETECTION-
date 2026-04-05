import os
from typing import Dict, List

import cv2
import numpy as np


def generate_heatmap_overlay(image: np.ndarray, defects: List[Dict], alpha: float = 0.45) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Invalid image for heatmap generation")

    h, w = image.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for defect in defects:
        x1, y1, x2, y2 = defect.get("bbox", [0, 0, 0, 0])
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(defect.get("confidence", 0.5))
        density[y1:y2, x1:x2] += confidence

    if np.max(density) > 0:
        density = density / np.max(density)

    heat_uint8 = np.uint8(density * 255)
    colored_map = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    if not defects:
        # Keep subtle overlay when there are no defects.
        colored_map = cv2.applyColorMap(np.zeros_like(heat_uint8), cv2.COLORMAP_BONE)
        alpha = 0.15

    overlay = cv2.addWeighted(image, 1 - alpha, colored_map, alpha, 0)
    return overlay


def save_heatmap(path: str, heatmap_image: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, heatmap_image)
    if not ok or not os.path.exists(path):
        raise IOError(f"Failed to write heatmap image: {path}")
    return path
