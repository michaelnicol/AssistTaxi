"""CDLEM one-frame detector. Caller: detect. Owns contour, Canny, Hough, RDP. Must not open files."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CdlemParams:
    sigma: float = 0.33
    blur_ksize: int = 5
    hough_threshold: int = 80
    min_line_length: int = 30
    max_line_gap: int = 10
    epsilon: float = 5.0


def _auto_canny(gray: np.ndarray, sigma: float) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        upper = min(255, lower + 1)
    return cv2.Canny(gray, lower, upper)


def _odd(ksize: int) -> int:
    ksize = max(1, int(ksize))
    return ksize if ksize % 2 else ksize + 1


def run_cdlem(image: np.ndarray, polygon: list[tuple[int, int]], params: CdlemParams):
    """Caller: propagate. Owns one CDLEM frame. Must not open files."""
    height, width = image.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [np.array(polygon, dtype=np.int32)], 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=roi)
    blur = cv2.GaussianBlur(masked, (_odd(params.blur_ksize), _odd(params.blur_ksize)), 0)
    edges = cv2.bitwise_and(_auto_canny(blur, params.sigma), roi)
    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    hough_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        params.hough_threshold,
        minLineLength=params.min_line_length,
        maxLineGap=params.max_line_gap,
    )
    if lines is not None:
        for x1, y1, x2, y2 in np.reshape(lines, (-1, 4)):
            cv2.line(pixel_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
            cv2.line(hough_view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rdp_view = hough_view.copy()
    for contour in contours:
        if len(contour) < 2:
            continue
        approx = cv2.approxPolyDP(contour, params.epsilon, False)
        cv2.polylines(pixel_mask, [approx], False, 255, 1)
        cv2.polylines(rdp_view, [approx], False, (255, 0, 0), 1)
    pixel_mask = cv2.bitwise_and(pixel_mask, roi)
    ys, xs = np.where(pixel_mask > 0)
    coords = np.column_stack((xs, ys)).astype(np.int32) if xs.size else np.zeros((0, 2), dtype=np.int32)
    stages = {"roi": roi, "blur": blur, "canny": edges, "hough": hough_view, "rdp": rdp_view, "mask": pixel_mask}
    note = f"{0 if lines is None else len(lines)} hough lines"
    return pixel_mask, coords, stages, note
