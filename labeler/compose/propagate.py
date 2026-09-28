"""Pick a detector for one frame. Caller: Session. Owns method dispatch. Must not open files or draw."""

import numpy as np

from labeler.methods.alina import AlinaParams, run_alina
from labeler.methods.cbem import CbemParams, run_cbem
from labeler.methods.cdlem import CdlemParams, run_cdlem


def detect(image, method: str, roi: list[tuple[int, int]], alina: AlinaParams, cdlem: CdlemParams, cbem: CbemParams):
    if method == "alina":
        if len(roi) < 4:
            return _empty(image, "click bottom-left, top-left, top-right, bottom-right")
        return run_alina(image, roi[:4], alina)
    if method == "cdlem":
        if len(roi) < 3:
            return _empty(image, "click a contour on the first frame")
        return run_cdlem(image, roi, cdlem)
    if method == "cbem":
        if len(roi) < 3:
            return _empty(image, "trace a ground-truth contour")
        return run_cbem(image, roi, cbem)
    raise ValueError(method)


def pixel_recall(pred: np.ndarray, gt: np.ndarray) -> float | None:
    truth = int((gt > 0).sum())
    if truth == 0:
        return None
    hit = int(np.logical_and(pred > 0, gt > 0).sum())
    return hit / truth


def _empty(image, note: str):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    coords = np.zeros((0, 2), dtype=np.int32)
    return mask, coords, {}, note
