"""The LLAMAS photo lists, and lookups from any photo back to its raw label.

``split.json`` is made once from ``labels.zip`` and never rewritten. Every run
copies it into its own folder, and its sha256 is part of the run config.

A run trains on the first ``train_percent`` of the ``train`` list. The same
percent is the same photos. A smaller percent is a prefix of a larger one, so
10% is the start of 25%. Tune and test photos are always the first N of their
lists, whatever the train percent, so every run is measured on the same photos.
A second epoch shuffles that same train list again. It does not draw new photos.

Each row is ``[photo path, gold answer text, width, height]``. The size is read
from the photo file, not assumed, and gold lanes are clipped to it. ``split.json``
also counts every size seen (``sizes``) and lists each photo whose size is not the
most common one (``odd_size``), so a photo that arrived at the wrong size is visible.

The audit page in the GUI can open any photo. :func:`label_for` finds the raw
LLAMAS label behind a photo path, and :func:`marker_segments` turns it into the
pixel segments the annotators drew.
"""

import json
import logging
import random
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from PIL import Image

from spline.lanes import DEFAULT_JUDGE, spline_answer
from spline.runs import file_sha256

log = logging.getLogger("spline.data")

LABELS = Path("/home/michaelnicol/github/LLAMAS/labels.zip")
IMAGES = Path("/home/michaelnicol/github/LLAMAS/color_images")
SPLIT_NAMES = ("train", "tune", "test")
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
# split.json has its own version, separate from the run schema: the photo lists
# must stay the same across schema bumps so old and new runs share one split.
SPLIT_VERSION = 3
# Photo headers read at once while making the split. The read is disk bound.
SIZE_READERS = 16

# One photo in a split: [absolute photo path, gold answer text, width, height].
Row = list


def image_path(label_name: str) -> Path:
    """Where the color photo for a label lives.

    ``labels/train/<drive>/<stamp>.json`` maps to
    ``color_images/train/<drive>/<stamp>_color_rect.png``.
    """
    rel = Path(*Path(label_name).parts[1:])
    return IMAGES / rel.parent / f"{rel.stem}_color_rect.png"


def split_path(output: Path) -> Path:
    return output / "split.json"


def photo_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the photo's header, or None if there is no such file."""
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.size


def make_split(output: Path) -> Path:
    """Write split.json if it is missing, and return its path.

    Labels whose photo is not on disk, or with no usable lane inside that photo, are
    left out and counted. The rest are shuffled with SPLIT_SEED and cut by SPLIT_FRACTIONS.
    """
    dest = split_path(output)
    if dest.exists():
        log.debug("using existing %s", dest)
        return dest
    rdp_px = DEFAULT_JUDGE.rdp_px
    log.info("making %s from %s; this reads every label and photo header once", dest, LABELS)
    with zipfile.ZipFile(LABELS) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        with ThreadPoolExecutor(SIZE_READERS) as pool:
            sizes = list(pool.map(photo_size, (image_path(name) for name in names)))
        log.info("read %d photo headers", len(names))
        rows, no_lanes, no_photo = [], 0, 0
        for count, (name, size) in enumerate(zip(names, sizes, strict=True), 1):
            if size is None:
                no_photo += 1
            elif answer := spline_answer(json.loads(archive.read(name)), *size, rdp_px):
                rows.append([str(image_path(name)), answer, *size])
            else:
                no_lanes += 1
            if count % 10000 == 0:
                log.info("read %d of %d labels", count, len(names))
    counts = Counter(f"{width}x{height}" for _path, _gold, width, height in rows)
    common = counts.most_common(1)[0][0]
    odd = [row[0] for row in rows if f"{row[2]}x{row[3]}" != common]
    if odd:
        log.warning("%d photos are not %s; they are kept and listed in odd_size", len(odd), common)
    random.Random(SPLIT_SEED).shuffle(rows)
    cut1 = round(len(rows) * SPLIT_FRACTIONS[0])
    cut2 = round(len(rows) * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    split = {
        "schema_version": SPLIT_VERSION,
        "seed": SPLIT_SEED,
        "fractions": list(SPLIT_FRACTIONS),
        "labels_zip": str(LABELS),
        "labels_sha256": file_sha256(LABELS),
        "images_dir": str(IMAGES),
        "rdp_px": rdp_px,
        "label_files": len(names),
        "skipped_no_lanes": no_lanes,
        "skipped_no_photo": no_photo,
        "sizes": dict(counts),
        "odd_size": odd,
        "train": rows[:cut1],
        "tune": rows[cut1:cut2],
        "test": rows[cut2:],
    }
    output.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(split))
    log.info(
        "wrote %s: train %d, tune %d, test %d; skipped %d with no lanes, %d with no photo; sizes %s",
        dest,
        len(split["train"]),
        len(split["tune"]),
        len(split["test"]),
        no_lanes,
        no_photo,
        dict(counts),
    )
    return dest


def reference_size(split: dict) -> tuple[int, int]:
    """The most common (width, height) in a split."""
    width, height = max(split["sizes"].items(), key=lambda item: item[1])[0].split("x")
    return int(width), int(height)


def read_split(path: Path) -> dict:
    """A split.json, refused if it was made by another split version."""
    split = json.loads(path.read_text())
    if split.get("schema_version") != SPLIT_VERSION:
        raise ValueError(f"{path} is schema {split.get('schema_version')}; move it away to make a new split")
    return split


def first(rows: list[Row], count: int, what: str) -> list[Row]:
    """The first ``count`` rows, or an error naming ``what`` if there are fewer."""
    if count < 1 or count > len(rows):
        raise ValueError(f"{what}: asked for {count} photos, the split has {len(rows)}")
    return rows[:count]


def train_count(split: dict, percent: int) -> int:
    """Photos in the first ``percent`` of the train list, at least one."""
    if percent < 1 or percent > 100:
        raise ValueError("data percent must be from 1 to 100")
    return max(1, len(split["train"]) * percent // 100)


@cache
def _label_names() -> dict[str, str]:
    """Photo path -> label name inside labels.zip, for every label. Read once per process."""
    with zipfile.ZipFile(LABELS) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
    log.debug("indexed %d label names", len(names))
    return {str(image_path(name)): name for name in names}


def label_for(photo: str) -> dict | None:
    """The raw LLAMAS label behind a photo, or None for a photo with no label (test, or not LLAMAS)."""
    name = _label_names().get(str(Path(photo)))
    if name is None:
        return None
    with zipfile.ZipFile(LABELS) as archive:
        label = json.loads(archive.read(name))
    label["_label_file"] = name
    return label


def marker_segments(label: dict) -> list[list[tuple[float, float, float, float]]]:
    """Per lane, the annotated marker segments as (x1, y1, x2, y2) in photo pixels."""
    lanes = []
    for lane in label.get("lanes") or []:
        segments = []
        for marker in lane.get("markers") or []:
            start, end = marker["pixel_start"], marker["pixel_end"]
            segments.append((float(start["x"]), float(start["y"]), float(end["x"]), float(end["y"])))
        lanes.append(segments)
    return lanes
