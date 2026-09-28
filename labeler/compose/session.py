"""One labeling session. Caller: GUI. Owns frame index work and save path. Must not draw widgets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from labeler.compose.propagate import detect, pixel_recall
from labeler.io.frames import list_frames, read_bgr
from labeler.io.labels import raster, read_xy, write_xy
from labeler.methods.alina import AlinaParams
from labeler.methods.cbem import CbemParams
from labeler.methods.cdlem import CdlemParams


@dataclass
class ShowResult:
    image: np.ndarray
    mask: np.ndarray
    coords: np.ndarray
    stages: dict
    note: str
    recall: float | None
    path: Path


class Session:
    def __init__(self, frames_dir: Path, labels_dir: Path, out_dir: Path):
        self.frames_dir = Path(frames_dir)
        self.labels_dir = Path(labels_dir)
        self.out_dir = Path(out_dir)

    def list_frames(self) -> list[Path]:
        return list_frames(self.frames_dir)

    def show(
        self,
        path: Path,
        method: str,
        roi: list[tuple[int, int]],
        alina: AlinaParams,
        cdlem: CdlemParams,
        cbem: CbemParams,
    ) -> ShowResult:
        image = read_bgr(path)
        if method == "existing":
            coords = read_xy(self.labels_dir / f"{path.stem}.txt")
            mask = raster(coords, image.shape[:2])
            stages: dict = {}
            note = "repo label" if len(coords) else "empty label file"
        else:
            mask, coords, stages, note = detect(image, method, roi, alina, cdlem, cbem)
        gt = raster(read_xy(self.labels_dir / f"{path.stem}.txt"), image.shape[:2])
        return ShowResult(image, mask, coords, stages, note, pixel_recall(mask, gt), path)

    def save(self, stem: str, coords: np.ndarray, method: str) -> Path:
        out = self.out_dir / "cbem" if method == "cbem" else self.out_dir
        path = out / f"{stem}.txt"
        write_xy(path, coords)
        return path
