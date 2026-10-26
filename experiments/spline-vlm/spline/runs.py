"""Run folders: hashes, write-once files, the environment record, and run summaries.

Each run lives in ``output/runs/<name>/``::

    spec.json            the spec it was started from, normalized
    config.json          RunConfig, everything that shapes the weights
    split.json           byte copy of output/split.json
    data.json            the exact train and tune-loss rows
    checkpoints/         while training only
    model/               the saved adapter or full weights
    train.json           TrainResult
    score-<split>.jsonl  header line, then one answer per photo
    score-<split>.json   ScoreResult
    live/<stage>.json    live status for the GUI (see spline/logs.py)
    live/train.jsonl     one line per training log step, for live plots
    logs/                rotating debug and error logs per step

Files that describe a finished step are written once. Writing one again with
different content is an error, so a run folder can never mix two runs.
"""

import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import torch

log = logging.getLogger("spline.runs")

HERE = Path(__file__).resolve().parent.parent
OUTPUT = HERE / "output"
PACKAGES = ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "pillow", "numpy", "safetensors")


def run_dir(output: Path, name: str) -> Path:
    return output / "runs" / name


def now() -> str:
    """Local time with its UTC offset, for every timestamp the project writes."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def weights_digest(saved: Path) -> tuple[str, str]:
    """(file or folder inside the run, sha256) of the saved weights.

    An adapter is one file. A full model may be split over several safetensors
    files, so its hash covers the sorted list of "name hash" lines.
    """
    adapter = saved / "adapter_model.safetensors"
    if adapter.is_file():
        return f"{saved.name}/{adapter.name}", file_sha256(adapter)
    parts = sorted(saved.glob("*.safetensors"))
    if not parts:
        raise RuntimeError(f"no weights in {saved}")
    listing = "".join(f"{part.name} {file_sha256(part)}\n" for part in parts)
    return saved.name, text_sha256(listing)


def weights_identity(config, folder: Path) -> tuple[str, str]:
    """(weights_file, weights_sha256) of the weights a run answers with.

    A base run answers with the hub weights, named by their commit. A trained run's
    saved weights must hash to what train.json recorded.
    """
    if config.model.method == "base":
        return f"hub:{config.model.id}", config.model.revision
    trained = _read(folder / "train.json")
    if trained is None:
        raise RuntimeError(f"{folder} has no train.json; train the run before scoring it")
    recorded = (trained["weights_file"], trained["weights_sha256"])
    if weights_digest(folder / "model") != recorded:
        raise RuntimeError(f"the weights in {folder / 'model'} do not match the hash in train.json")
    return recorded


def write_once(path: Path, text: str) -> None:
    """Write a file, or check that the existing one already says exactly this."""
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"{path} already exists with other contents; this run is not the one that made it")
        log.debug("%s already written with the same contents", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    log.debug("wrote %s (%d bytes)", path, len(text))


def dump(data) -> str:
    return json.dumps(data, indent=2) + "\n"


def _gpu() -> str | None:
    """Name, driver, and memory of each visible GPU, as nvidia-smi reports them."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def environment() -> dict:
    """What ran the step: versions, hardware, command line, and a hash of every code file.

    Comparing ``code_sha256`` between two runs shows whether the code changed between them.
    """
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    code = sorted([*HERE.glob("spline_*.py"), *(HERE / "spline").rglob("*.py")])
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": _gpu(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "code_sha256": {str(path.relative_to(HERE)): file_sha256(path) for path in code if path.is_file()},
    }


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def summary(folder: Path) -> dict:
    """What the GUI and the report show for one run folder, read from its JSON files.

    Keys: name, folder, config, spec, train, scores {tune, test}, live {stage: status}.
    Missing files give None, so a run that is still training still has a summary.
    """
    live = {}
    for path in sorted((folder / "live").glob("*.json")):
        status = _read(path)
        if status:
            live[path.stem] = status
    return {
        "name": folder.name,
        "folder": folder,
        "config": _read(folder / "config.json"),
        "spec": _read(folder / "spec.json"),
        "train": _read(folder / "train.json"),
        "scores": {split: _read(folder / f"score-{split}.json") for split in ("tune", "test")},
        "live": live,
    }


def all_runs(output: Path) -> list[dict]:
    """A summary of every run folder that has a config.json, newest first."""
    folders = [path for path in (output / "runs").glob("*/") if (path / "config.json").exists()]
    folders.sort(key=lambda path: (path / "config.json").stat().st_mtime, reverse=True)
    return [summary(folder) for folder in folders]
