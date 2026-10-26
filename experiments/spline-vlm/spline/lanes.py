"""Lane text and geometry, in whole pixels of each photo.

A lane is a polyline written as one line of text, ``x1 y1 x2 y2 ... xk yk``, points
from top to bottom. An answer is any number of lanes, one per line. To judge an
answer, every predicted lane is pooled into one point set and every gold lane into
another, so lanes are never matched one to one (see :func:`overlap`).
"""

import itertools
from typing import NamedTuple

import numpy as np

from spline.config import JudgeSettings

DEFAULT_JUDGE = JudgeSettings(
    rdp_px=2.0,
    step_px=2.0,
    radius_px=15.0,
    tau_px=50.0,
    distance="euclidean px from each sampled point to the nearest sampled point of the other pooled set",
    cost=(
        "truncated symmetric chamfer: the mean of both directions, each distance capped at tau_px; "
        "an unreadable or empty answer costs tau_px"
    ),
    why={
        "rdp_px": "gold lanes keep every bend larger than this, and straight runs shrink to their two ends",
        "step_px": "measuring to the nearest sample instead of the true line adds at most half this",
        "radius_px": "half the 30 px lane width of the official CULane and LLAMAS metric",
        "tau_px": "caps any one point, so one wild line cannot dominate a photo's cost",
    },
)
# Distance-matrix cells computed at once (about 32 MB), so a very long answer cannot exhaust memory.
_CELLS = 2_000_000


class Overlap(NamedTuple):
    cost_px: float
    prec: float
    rec: float


def _rdp(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: the fewest points that stay within ``tolerance`` px of the rest."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        chord = points[last] - points[first]
        rest = points[first + 1 : last] - points[first]
        length = float(np.hypot(*chord))
        if length:
            gaps = np.abs(chord[0] * rest[:, 1] - chord[1] * rest[:, 0]) / length
        else:
            gaps = np.hypot(rest[:, 0], rest[:, 1])
        worst = int(np.argmax(gaps))
        if gaps[worst] > tolerance:
            middle = first + 1 + worst
            keep[middle] = True
            stack += [(first, middle), (middle, last)]
    return points[keep]


def lane_points(markers: list, width: int, height: int, rdp_px: float) -> np.ndarray | None:
    """One gold lane as an integer (k, 2) array of x y pixels, top to bottom, or None if too short.

    Marker endpoints outside the photo are dropped. Endpoints on the same pixel row
    become one point at their mean x; rounding that mean moves a point by at most 0.5 px.
    """
    rows: dict[int, list[float]] = {}
    for marker in markers:
        for end in ("pixel_start", "pixel_end"):
            x, y = float(marker[end]["x"]), float(marker[end]["y"])
            if 0 <= x < width and 0 <= y < height:
                rows.setdefault(round(y), []).append(x)
    ys = sorted(rows)
    if len(ys) < 2 or ys[-1] - ys[0] < 2:
        return None
    points = np.array([(sum(rows[y]) / len(rows[y]), y) for y in ys])
    kept = np.rint(_rdp(points, rdp_px)).astype(int)
    kept[:, 0] = np.clip(kept[:, 0], 0, width - 1)
    return kept


def spline_answer(label: dict, width: int, height: int, rdp_px: float) -> str | None:
    """The gold answer text for one LLAMAS label, or None if it has no usable lane."""
    lanes = [
        points
        for lane in label.get("lanes") or []
        if (points := lane_points(lane.get("markers") or [], width, height, rdp_px)) is not None
    ]
    if not lanes:
        return None
    lanes.sort(key=lambda points: int(points[-1, 0]))
    return "\n".join(" ".join(f"{x} {y}" for x, y in points) for points in lanes)


def read_lanes(text: str, width: int, height: int) -> list[np.ndarray] | None:
    """Lanes from answer text, or None if any line is not an even count of at least 4 whole pixels inside the photo.

    Blank lines are skipped.
    """
    lanes = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 4 or len(parts) % 2 or not all(part.isascii() and part.isdigit() for part in parts):
            return None
        points = np.array([int(part) for part in parts]).reshape(-1, 2)
        if (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
            return None
        lanes.append(points)
    return lanes or None


def readable_prefix(text: str, width: int, height: int) -> bool:
    """False once no continuation of ``text`` can pass read_lanes: a bad character, a number past the photo,
    or a finished line with an odd count or fewer than 4 numbers. The last number may still grow."""
    lines = text.lstrip().split("\n")
    for index, line in enumerate(lines):
        parts = line.split()
        if not all(part.isascii() and part.isdigit() for part in parts):
            return False
        if any(int(part) >= (height if place % 2 else width) for place, part in enumerate(parts)):
            return False
        if index < len(lines) - 1 and parts and (len(parts) < 4 or len(parts) % 2):
            return False
    return True


def densify(lane: np.ndarray, step_px: float) -> np.ndarray:
    """Float points along the lane's segments, at most ``step_px`` apart, ends included."""
    points = lane.astype(float)
    parts = [points[:1]]
    for start, stop in itertools.pairwise(points):
        steps = max(1, int(np.ceil(np.hypot(*(stop - start)) / step_px)))
        parts.append(start + (stop - start) * (np.arange(1, steps + 1) / steps)[:, None])
    return np.concatenate(parts)


def nearest(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest of ``others``."""
    rows = max(1, _CELLS // len(others))
    return np.concatenate(
        [
            np.sqrt(((points[i : i + rows, None, :] - others[None, :, :]) ** 2).sum(-1).min(1))
            for i in range(0, len(points), rows)
        ]
    )


def overlap(pred: list[np.ndarray], gold: list[np.ndarray], judge: JudgeSettings) -> Overlap:
    """Pool all predicted lanes and all gold lanes, and compare the two point sets.

    The cost is the mean of both nearest-point directions, each distance capped at
    ``tau_px``. Precision and recall are the shares of points within ``radius_px``.
    """
    if not pred or not gold:
        return Overlap(judge.tau_px, 0.0, 0.0)
    p = np.concatenate([densify(lane, judge.step_px) for lane in pred])
    g = np.concatenate([densify(lane, judge.step_px) for lane in gold])
    to_gold, to_pred = nearest(p, g), nearest(g, p)
    cost = 0.5 * (np.minimum(to_gold, judge.tau_px).mean() + np.minimum(to_pred, judge.tau_px).mean())
    return Overlap(float(cost), float((to_gold <= judge.radius_px).mean()), float((to_pred <= judge.radius_px).mean()))
