"""Where each run's files go. Datasets are shared in output/; each run gets output/runs/<name>/."""

import time
from pathlib import Path


def run_name(model: str, lora: bool, data_percent: int, tiles: int | None) -> str:
    short = model.rstrip("/").split("/")[-1].lower()
    tile_part = f"-t{tiles}" if tiles else ""
    return f"{short}-{'lora' if lora else 'full'}-d{data_percent}{tile_part}"


def run_dir(output: Path, name: str) -> Path:
    return output / "runs" / name


def append_error(output: Path, run: str, text: str) -> None:
    path = output / "errors.md"
    output.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if fresh:
            handle.write("# Training errors\n")
        handle.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} — {run}\n\n```\n{text.rstrip()}\n```\n")
