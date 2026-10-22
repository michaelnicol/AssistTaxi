"""The photo lists. The full split is saved once. A percent run saves its own list and then reuses it."""

import json
import random
import zipfile
from pathlib import Path

from tqdm import tqdm

from spline.lanes import spline_answer

LABELS = Path("/home/michaelnicol/github/LLAMAS/labels.zip")
IMAGES = Path("/home/michaelnicol/github/LLAMAS/color_images")
SPLIT_NAMES = ("train", "tune", "test")
SEED = 0


def image_path(label_name: str) -> Path:
    rel = Path(*Path(label_name).parts[1:])
    return IMAGES / rel.parent / f"{rel.stem}_color_rect.png"


def load_split(output: Path) -> dict[str, list[list[str]]]:
    dest = output / "split.json"
    if dest.exists():
        return json.loads(dest.read_text())
    pairs = []
    with zipfile.ZipFile(LABELS) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for name in tqdm(names, desc="Reading labels"):
            answer = spline_answer(json.loads(archive.read(name)))
            photo = image_path(name)
            if answer and photo.is_file():
                pairs.append([str(photo), answer])
    random.Random(SEED).shuffle(pairs)
    cut1, cut2 = len(pairs) * 8 // 10, len(pairs) * 9 // 10
    split = {"train": pairs[:cut1], "tune": pairs[cut1:cut2], "test": pairs[cut2:]}
    output.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"seed": SEED, **split}))
    print({name: len(split[name]) for name in SPLIT_NAMES}, "photos written to", dest)
    return split


def dataset_path(output: Path, percent: int) -> Path:
    return output / f"dataset-{percent}.json"


def load_dataset(output: Path, percent: int) -> dict[str, list[list[str]]]:
    """The same percent always returns the same photos, for every model."""
    if percent < 1 or percent > 100:
        raise ValueError("data percent must be from 1 to 100")
    full = load_split(output)
    if percent == 100:
        return {name: full[name] for name in SPLIT_NAMES}
    dest = dataset_path(output, percent)
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved.get("percent") == percent and saved.get("seed") == SEED:
            return {name: saved[name] for name in SPLIT_NAMES}
    saved = {"percent": percent, "seed": SEED}
    for name in SPLIT_NAMES:
        rows = full[name]
        saved[name] = rows[: max(1, len(rows) * percent // 100)]
    output.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(saved))
    print({name: len(saved[name]) for name in SPLIT_NAMES}, f"photos at {percent}% written to", dest)
    return {name: saved[name] for name in SPLIT_NAMES}
