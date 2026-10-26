"""What the GUI pages share: run lists, which split a photo is in, stored answers, and a worker thread.

Everything here reads files written by training and scoring. The heavy objects
(split.json, each run's data.json and score jsonl) are read once and cached,
keyed by path and modification time so a file that changes is read again.
"""

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from spline.data import split_path
from spline.runs import OUTPUT, all_runs

log = logging.getLogger("spline.gui")

SPLITS = ("train", "tune", "test")


def _cached(path: Path, cache: dict, read: Callable[[Path], object]):
    """``read(path)``, reused until the file's modification time changes."""
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    hit = cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    started = time.monotonic()
    value = read(path)
    log.debug("read %s in %.2f s", path, time.monotonic() - started)
    cache[path] = (stamp, value)
    return value


_split_cache: dict = {}
_index_cache: dict = {}
_data_cache: dict = {}
_answer_cache: dict = {}


def split() -> dict | None:
    """output/split.json, or None before the first run has made it."""
    return _cached(split_path(OUTPUT), _split_cache, lambda path: json.loads(path.read_text()))


def split_place(photo: str) -> tuple[str, int, str] | None:
    """(split name, index in that split, gold answer) for a photo in split.json."""
    data = split()
    if not data:
        return None
    index = _cached(split_path(OUTPUT), _index_cache, _index_split)
    return index.get(photo)


def _index_split(path: Path) -> dict:
    data = split()
    return {row[0]: (name, at, row[1]) for name in SPLITS for at, row in enumerate(data[name])}


RUN_PHOTO_KEYS = ("train", "tune_loss", "tune_cost")


def run_photos(folder: Path) -> dict[str, set[str]]:
    """{"train": trained on, "tune_loss": loss logged at each check, "tune_cost": answered to pick the checkpoint}."""

    def read(path: Path) -> dict[str, set[str]]:
        data = json.loads(path.read_text())
        return {key: {row[0] for row in data[key]} for key in RUN_PHOTO_KEYS}

    return _cached(folder / "data.json", _data_cache, read) or {key: set() for key in RUN_PHOTO_KEYS}


def stored_answers(folder: Path) -> dict[str, dict]:
    """Photo path -> its line from the run's score-tune.jsonl and score-test.jsonl, plus the decode header."""
    found = {}
    for split_name in ("tune", "test"):
        path = folder / f"score-{split_name}.jsonl"

        def read(path: Path) -> dict:
            lines = path.read_text().splitlines()
            header = json.loads(lines[0]) if lines else {}
            rows = {}
            for line in lines[1:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                rows[row["image"]] = {**row, "file": path.name, "header": header}
            return rows

        found.update(_cached(path, _answer_cache, read) or {})
    return found


def answering_runs() -> list[dict]:
    """Run summaries that can answer a photo, newest first: trained runs with saved weights, and base runs."""
    return [
        run
        for run in all_runs(OUTPUT)
        if run["config"]["model"]["method"] == "base" or (run["train"] and (run["folder"] / "model").is_dir())
    ]


@dataclass
class Job:
    """One piece of slow work. ``done`` runs on the Tk thread with the result, ``failed`` with the error."""

    name: str
    work: Callable[[], object]
    done: Callable[[object], None]
    failed: Callable[[BaseException], None]


class Worker:
    """One background thread that runs jobs in order, so the GPU sees one request at a time.

    The Tk thread calls :meth:`poll` every few hundred milliseconds to deliver
    finished results; Tk widgets are never touched from the worker thread.
    """

    def __init__(self) -> None:
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.results: queue.Queue[tuple[Job, object, BaseException | None]] = queue.Queue()
        self.busy = threading.Event()
        threading.Thread(target=self._loop, name="gui-worker", daemon=True).start()

    def submit(self, job: Job) -> None:
        log.debug("queued job %s", job.name)
        self.jobs.put(job)

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            self.busy.set()
            started = time.monotonic()
            try:
                result, error = job.work(), None
            except BaseException as caught:
                log.exception("job %s failed", job.name)
                result, error = None, caught
            log.debug("job %s took %.2f s", job.name, time.monotonic() - started)
            self.busy.clear()
            self.results.put((job, result, error))

    def poll(self) -> None:
        while not self.results.empty():
            job, result, error = self.results.get()
            if error is None:
                job.done(result)
            else:
                job.failed(error)
