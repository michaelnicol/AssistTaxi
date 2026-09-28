"""Read and write x y label files. Caller: Session. Owns the text format. Must not choose a detector or open the log."""

from pathlib import Path

import numpy as np

from labeler.io.guard import assert_output_allowed


def parse_xy(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    if not rows:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def format_xy(coords: np.ndarray) -> str:
    if len(coords) == 0:
        return ""
    return "".join(f"{int(x):6d} {int(y):6d}\n" for x, y in coords)


def read_xy(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 2), dtype=np.int32)
    return parse_xy(path.read_text())


def raster(coords: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if len(coords) == 0:
        return mask
    height, width = shape_hw
    xs = coords[:, 0].astype(np.int32)
    ys = coords[:, 1].astype(np.int32)
    keep = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    mask[ys[keep], xs[keep]] = 255
    return mask


def write_xy(path: Path, coords: np.ndarray) -> None:
    path = Path(path)
    assert_output_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_xy(coords))
