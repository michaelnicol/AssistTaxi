"""Lane text. One line is the cubic through four points on the visible part of a lane."""

import numpy as np

WIDTH = 1276
HEIGHT = 717


def lane_line(markers: list) -> np.ndarray | None:
    buckets: dict[float, list[float]] = {}
    for marker in markers:
        for end in ("pixel_start", "pixel_end"):
            x = float(marker[end]["x"]) / WIDTH
            y = float(marker[end]["y"]) / HEIGHT
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                buckets.setdefault(round(y, 4), []).append(x)
    ys = sorted(buckets)
    if len(ys) < 2 or ys[-1] - ys[0] < 0.002:
        return None
    xs = [sum(buckets[y]) / len(buckets[y]) for y in ys]
    dense = np.linspace(ys[0], ys[-1], 32)
    # Messy labels make a cubic swing past the points, so drop a degree until it stays near them.
    for degree in range(min(3, len(ys) - 1), 0, -1):
        coef = np.polyfit(ys, xs, degree)
        curve = np.polyval(coef, dense)
        if curve.min() >= min(xs) - 0.05 and curve.max() <= max(xs) + 0.05:
            break
    heights = np.linspace(ys[0], ys[-1], 4)
    return np.concatenate(([ys[0], ys[-1]], np.polyval(coef, heights)))


def spline_answer(label: dict) -> str | None:
    rows = [row for lane in label.get("lanes") or [] if (row := lane_line(lane.get("markers") or [])) is not None]
    if not rows:
        return None
    rows.sort(key=lambda row: float(row[-1]))
    return "\n".join(" ".join(f"{value:.3f}" for value in row) for row in rows)


def read_lanes(text: str) -> list[np.ndarray] | None:
    lanes = []
    for line in text.strip().splitlines():
        try:
            row = np.array([float(part) for part in line.split()])
        except ValueError:
            return None
        if len(row) != 6 or not np.isfinite(row).all() or not 0.0 <= row[0] < row[1] <= 1.0:
            return None
        lanes.append(row)
    return lanes or None


def lane_x(row: np.ndarray, heights: np.ndarray) -> np.ndarray:
    return np.polyval(np.polyfit(np.linspace(row[0], row[1], 4), row[2:], 3), heights)


def lane_errors(gold: list[np.ndarray], predicted: list[np.ndarray]) -> list[float]:
    """Pixel error of each gold lane against the closest predicted lane, capped at one image width."""
    errors = []
    for lane in gold:
        heights = np.linspace(lane[0], lane[1], 8)
        want = lane_x(lane, heights)
        best = min([float(np.abs(lane_x(guess, heights) - want).mean()) for guess in predicted] + [1.0])
        errors.append(best * WIDTH)
    return errors
