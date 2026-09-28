"""CBEM ground-truth edges. Caller: detect. Owns hand contour plus auto-Canny. Must not open files."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CbemParams:
    sigma: float = 0.33
    blur_ksize: int = 5


def _odd(ksize: int) -> int:
    ksize = max(1, int(ksize))
    return ksize if ksize % 2 else ksize + 1


def run_cbem(image: np.ndarray, polygon: list[tuple[int, int]], params: CbemParams):
    """Caller: propagate. Owns one CBEM frame. Must not open files."""
    height, width = image.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [np.array(polygon, dtype=np.int32)], 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (_odd(params.blur_ksize), _odd(params.blur_ksize)), 0)
    median = float(np.median(blur))
    lower = int(max(0, (1.0 - params.sigma) * median))
    upper = int(min(255, (1.0 + params.sigma) * median))
    if upper <= lower:
        upper = min(255, lower + 1)
    edges = cv2.bitwise_and(cv2.Canny(blur, lower, upper), roi)
    ys, xs = np.where(edges > 0)
    coords = np.column_stack((xs, ys)).astype(np.int32) if xs.size else np.zeros((0, 2), dtype=np.int32)
    return edges, coords, {"roi": roi, "blur": blur, "canny": edges}, f"sigma {params.sigma:.2f}"
