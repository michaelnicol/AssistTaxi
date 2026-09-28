"""ALINA one-frame detector. Caller: detect. Owns warp, HSV, CIRCLEDAT. Must not open files."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlinaParams:
    peak_pixel_threshold: int = 50
    min_white_pixels: int = 200
    circular_threshold: int = 15
    yellow_lower: tuple[int, int, int] = (0, 70, 170)
    yellow_upper: tuple[int, int, int] = (255, 255, 255)
    mask_ignore_left_columns: int = 300
    dst_bottom_left: tuple[int, int] = (50, 800)
    dst_top_left: tuple[int, int] = (50, 100)
    dst_top_right: tuple[int, int] = (1200, 100)
    dst_bottom_right: tuple[int, int] = (1200, 800)

    def destination(self) -> np.ndarray:
        return np.array(
            [
                self.dst_bottom_left,
                self.dst_top_left,
                self.dst_top_right,
                self.dst_bottom_right,
            ],
            dtype=np.float32,
        )


def _normalize_hsv(image: np.ndarray) -> np.ndarray:
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    hue = cv2.normalize(hue, None, 0, 255, cv2.NORM_MINMAX)
    sat = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX)
    val = cv2.normalize(val, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack((hue, sat, val), axis=-1)


def _histogram(mask: np.ndarray, min_white_pixels: int) -> tuple[list[int], int, np.ndarray]:
    binary = np.where(mask == 255, 1, 0)
    projection = np.sum(binary, axis=0)
    peak_value = int(np.max(projection)) if projection.size else 0
    peak_index = int(np.argmax(projection)) if projection.size else 0
    column = binary[:, peak_index] if projection.size else np.zeros(0, dtype=np.int32)
    white_rows = np.flatnonzero(column == 1)
    if white_rows.size > min_white_pixels:
        return [peak_index, int(white_rows.mean())], peak_value, projection
    return [], peak_value, projection


def _circledat(mask: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    current = np.zeros_like(mask)
    current[y, x] = 255
    while True:
        grown = cv2.bitwise_and(cv2.dilate(current, kernel), mask)
        if np.array_equal(grown, current):
            return current
        current = grown


def _projection_image(projection: np.ndarray, peak_index: int, height: int = 80) -> np.ndarray:
    width = int(projection.size)
    image = np.zeros((height, max(width, 1)), dtype=np.uint8)
    if width == 0:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    peak = float(projection.max()) or 1.0
    ys = (height - 1 - (projection.astype(np.float32) / peak) * (height - 1)).astype(np.int32)
    image[ys, np.arange(width)] = 255
    if 0 <= peak_index < width:
        image[:, peak_index] = 128
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack((xs, ys)).astype(np.int32)


def run_alina(image: np.ndarray, quad: list[tuple[int, int]], params: AlinaParams):
    """Caller: propagate. Owns one ALINA frame. Must not open files."""
    height, width = image.shape[:2]
    source = np.array(quad, dtype=np.float32).reshape(4, 2)
    destination = params.destination()
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (width, height))
    normalized = _normalize_hsv(warped)
    mask = cv2.inRange(normalized, np.array(params.yellow_lower), np.array(params.yellow_upper))
    if params.mask_ignore_left_columns > 0:
        mask[:, : params.mask_ignore_left_columns] = 0
    seed, peak_value, projection = _histogram(mask, params.min_white_pixels)
    stages = {
        "warp": warped,
        "hsv": normalized,
        "threshold": mask,
        "histogram": _projection_image(projection, seed[0] if seed else int(np.argmax(projection) if projection.size else 0)),
    }
    empty = np.zeros((height, width), dtype=np.uint8)
    if not seed or peak_value <= params.peak_pixel_threshold:
        why = "column too thin" if not seed else "below peak gate"
        stages["circledat"] = empty
        stages["unwarped"] = empty
        return empty, np.zeros((0, 2), dtype=np.int32), stages, f"no peak ({why}, max {peak_value})"

    binary = mask.copy()
    binary[seed[1], seed[0]] = 255
    found = _circledat(binary, seed[0], seed[1], max(1, params.circular_threshold))
    blob = cv2.dilate(found, np.ones((3, 3), dtype=np.uint8))
    inverse = cv2.getPerspectiveTransform(destination, source)
    unwarped = cv2.warpPerspective(blob, inverse, (width, height))
    out = np.where(unwarped > 0, 255, 0).astype(np.uint8)
    stages["circledat"] = blob
    stages["unwarped"] = out
    return out, _coords_from_mask(out), stages, f"peak {peak_value}"
