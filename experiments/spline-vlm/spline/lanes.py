"""Lane text and geometry, in whole pixels of each photo.

A lane is a polyline written as one line of text, ``x1 y1 x2 y2 ... xk yk``, points
from top to bottom. An answer is any number of lanes, one per line. To judge an
answer, every predicted lane is pooled with the others and every gold lane with
the others, so lanes are never matched one to one (see :func:`overlap`).

A judge with ``stroke_px`` set paints both pools that many pixels thick and
takes the mean squared distance from each painted model pixel to the nearest
painted label pixel. A judge with ``stroke_px`` null keeps the older capped
two-way distance between centerline samples.
"""

import itertools
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from spline.config import JudgeSettings

DEFAULT_JUDGE = JudgeSettings(
    rdp_px=2.0,
    step_px=2.0,
    radius_px=15.0,
    tau_px=50.0,
    stroke_px=3.0,
    distance="euclidean px from each painted model pixel to the nearest painted gold pixel",
    cost=(
        "mean squared distance from each painted model pixel to the nearest painted gold pixel; "
        "both strokes are stroke_px wide; an unreadable or empty answer costs tau_px squared"
    ),
    why={
        "rdp_px": "gold lanes keep every bend larger than this, and straight runs shrink to their two ends",
        "step_px": (
            "painted pixels are kept where both coordinates fall on this grid, "
            "so the mean square uses fewer pixels"
        ),
        "radius_px": (
            "a painted model pixel is a hit at most this far from a painted label pixel, "
            "half the 30 px CULane and LLAMAS lane width"
        ),
        "tau_px": "an empty or unreadable answer costs the square of this",
        "stroke_px": "both strokes are this thick, a small odd width that sits on the center of the lane",
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


def paint_pixels(lanes: list[np.ndarray], width: int, height: int, stroke_px: float) -> np.ndarray:
    """Pixel centers of the lanes drawn ``stroke_px`` thick, as an (n, 2) array of x y.

    Pixels outside the photo are dropped. An empty list of lanes gives an empty array.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    thick = max(1, int(round(stroke_px)))
    for lane in lanes:
        if len(lane) < 2:
            continue
        draw.line([tuple(int(value) for value in point) for point in lane], fill=1, width=thick)
    ys, xs = np.nonzero(np.asarray(mask))
    if len(xs) == 0:
        return np.zeros((0, 2))
    return np.stack((xs, ys), axis=1).astype(float)


def _on_grid(points: np.ndarray, step_px: float) -> np.ndarray:
    """Pixels whose x and y both fall on the ``step_px`` grid. The full set is kept if the grid hits none."""
    step = max(1, int(round(step_px)))
    if step <= 1 or len(points) == 0:
        return points
    coords = np.rint(points).astype(int)
    chosen = points[(coords[:, 0] % step == 0) & (coords[:, 1] % step == 0)]
    return chosen if len(chosen) else points


def stroke_points(lanes: list[np.ndarray], width: int, height: int, judge: JudgeSettings) -> np.ndarray:
    """The painted pixels the stroke cost averages, on the judge's sample grid."""
    return _on_grid(paint_pixels(lanes, width, height, judge.stroke_px), judge.step_px)


def _stroke_overlap(
    pred: list[np.ndarray], gold: list[np.ndarray], judge: JudgeSettings, width: int, height: int
) -> Overlap:
    """Mean squared distance from each sampled model pixel to the nearest sampled label pixel."""
    penalty = judge.tau_px**2
    if not pred or not gold:
        return Overlap(penalty, 0.0, 0.0)
    painted_pred = stroke_points(pred, width, height, judge)
    painted_gold = stroke_points(gold, width, height, judge)
    if len(painted_pred) == 0 or len(painted_gold) == 0:
        return Overlap(penalty, 0.0, 0.0)
    to_gold = nearest(painted_pred, painted_gold)
    to_pred = nearest(painted_gold, painted_pred)
    return Overlap(
        float(np.square(to_gold).mean()),
        float((to_gold <= judge.radius_px).mean()),
        float((to_pred <= judge.radius_px).mean()),
    )


def overlap(pred: list[np.ndarray], gold: list[np.ndarray], judge: JudgeSettings, width: int, height: int) -> Overlap:
    """Pool all predicted lanes and all gold lanes, and compare the two sets.

    With ``stroke_px`` set, both pools are painted that thick and the cost is the
    mean squared distance from each model pixel to the nearest label pixel. Without
    it, the cost is the mean of both nearest-point directions along the centerlines,
    each distance capped at ``tau_px``. Precision and recall are the shares of
    points within ``radius_px``.
    """
    if judge.stroke_px:
        return _stroke_overlap(pred, gold, judge, width, height)
    if not pred or not gold:
        return Overlap(judge.tau_px, 0.0, 0.0)
    p = np.concatenate([densify(lane, judge.step_px) for lane in pred])
    g = np.concatenate([densify(lane, judge.step_px) for lane in gold])
    to_gold, to_pred = nearest(p, g), nearest(g, p)
    cost = 0.5 * (np.minimum(to_gold, judge.tau_px).mean() + np.minimum(to_pred, judge.tau_px).mean())
    return Overlap(float(cost), float((to_gold <= judge.radius_px).mean()), float((to_pred <= judge.radius_px).mean()))
