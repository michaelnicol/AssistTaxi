"""Logging for every process, and the small live-status files the GUI reads.

Every process (train, score, queue, gui) writes the same three streams:

* the console, at INFO, so a terminal or tmux pane shows what is happening;
* ``<folder>/<process>.log`` at DEBUG, the full trace of what the code did;
* ``<folder>/<process>.errors.log`` at WARNING, only what went wrong or was changed.

Both files rotate: a debug log is cut at ``DEBUG_BYTES`` and an error log at
``ERROR_BYTES``, keeping ``BACKUPS`` old copies each. One process therefore never
holds more than ``(DEBUG_BYTES + ERROR_BYTES) * (BACKUPS + 1)`` bytes of logs, no
matter how long it runs.

A training or scoring process only learns its run folder after it has built the
run config, so records from before that point are held in memory and written to
the files the moment :func:`attach` is called. Nothing logged early is lost.

Only one process writes a given file at a time: train and score write inside
their run folder, the queue runner holds a lock, and the GUI writes its own
files. Python's rotating handler is not safe for two writers on one file.
"""

import json
import logging
import logging.handlers
import os
import socket
import sys
import time
import warnings
from pathlib import Path

from transformers.utils import logging as hf_logging

DEBUG_BYTES = 5 * 1024 * 1024
ERROR_BYTES = 1024 * 1024
BACKUPS = 2
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that log every HTTP request or file lock at INFO or DEBUG. Their
# warnings still come through; their chatter would fill the debug log.
QUIET = (
    "httpx",
    "httpcore",
    "urllib3",
    "PIL",
    "matplotlib",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "asyncio",
    "torch",
)

log = logging.getLogger("spline.logs")
_early: logging.handlers.MemoryHandler | None = None


def setup(process: str, folder: Path | None = None) -> None:
    """Configure the root logger for this process.

    ``process`` names the log files, for example ``train`` or ``gui``. With a
    ``folder`` the files open now; without one, records wait in memory until
    :func:`attach` names the folder.
    """
    global _early
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    root.addHandler(console)
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Python warnings (DeprecationWarning and friends) become log records, so they
    # land in the error log instead of only on stderr.
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    _route_transformers()
    sys.excepthook = _log_uncaught
    if folder is None:
        # flushLevel above CRITICAL: the buffer is only emptied by attach().
        _early = logging.handlers.MemoryHandler(capacity=1_000_000, flushLevel=logging.CRITICAL + 10)
        _early.setLevel(logging.DEBUG)
        root.addHandler(_early)
    else:
        attach(folder, process)


def attach(folder: Path, process: str) -> None:
    """Open this process's rotating log files in ``folder`` and write any held records."""
    global _early
    folder.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    files = [
        _rotating(folder / f"{process}.log", logging.DEBUG, DEBUG_BYTES),
        _rotating(folder / f"{process}.errors.log", logging.WARNING, ERROR_BYTES),
    ]
    for handler in files:
        root.addHandler(handler)
    if _early is not None:
        # Replay what was logged before the folder was known, honoring each file's level.
        for record in _early.buffer:
            for handler in files:
                if record.levelno >= handler.level:
                    handler.handle(record)
        root.removeHandler(_early)
        _early.buffer.clear()
        _early = None
    log.debug("logging to %s (%s.log and %s.errors.log)", folder, process, process)


def _rotating(path: Path, level: int, max_bytes: int) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=BACKUPS, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    return handler


def _route_transformers() -> None:
    """Send Hugging Face logs through our handlers at WARNING instead of their own stderr handler."""
    hf_logging.set_verbosity_warning()
    hf_logging.disable_default_handler()
    hf_logging.enable_propagation()
    # Their progress bars write carriage returns many times a second and would bloat the logs.
    hf_logging.disable_progress_bar()


def _log_uncaught(kind, value, trace) -> None:
    """Write an uncaught exception to the error log before the process exits."""
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, value, trace)
        return
    logging.getLogger("spline").critical("uncaught exception", exc_info=(kind, value, trace))


def write_json(path: Path, data) -> None:
    """Replace ``path`` with ``data`` in one step, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def read_json(path: Path) -> dict | None:
    """The JSON in ``path``, or None if it is missing or mid-write."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


class Progress:
    """Live status of one long step (training, or scoring one split), for the GUI.

    The status lives in ``<run>/live/<stage>.json`` and is rewritten at most every
    ``every_s`` seconds, plus once at the start and once at the end. The ETA uses
    only the work this process did, so a resumed run is not judged by the speed of
    the process before it.
    """

    def __init__(self, folder: Path, stage: str, total: int, unit: str, every_s: float = 2.0) -> None:
        self.path = folder / "live" / f"{stage}.json"
        self.stage = stage
        self.total = total
        self.unit = unit
        self.every_s = every_s
        self.started = time.time()
        self.first_done: int | None = None
        self.last_write = 0.0
        self.extra: dict = {}
        # The latest count and ETA, for callers that log them.
        self.done = 0
        self.eta_s: float | None = None
        self._write("running", 0)

    def update(self, done: int, force: bool = False, **extra) -> None:
        """Record that ``done`` of ``total`` units are finished. Extra keys are shown as-is."""
        self.extra.update(extra)
        self.done = done
        if self.first_done is None:
            self.first_done = done
        now = time.time()
        if not force and now - self.last_write < self.every_s:
            return
        self.last_write = now
        self._write("running", done)

    def finish(self, state: str = "done", done: int | None = None) -> None:
        """Mark the step ``done`` or ``failed``. A failure keeps the last count reached."""
        if done is None:
            done = self.total if state == "done" else self.done
        self._write(state, done)

    def _write(self, state: str, done: int) -> None:
        now = time.time()
        elapsed = now - self.started
        progressed = done - (self.first_done or 0)
        rate = elapsed / progressed if progressed > 0 else None
        eta = rate * (self.total - done) if rate and state == "running" else None
        self.eta_s = eta
        write_json(
            self.path,
            {
                "stage": self.stage,
                "state": state,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "done": done,
                "total": self.total,
                "unit": self.unit,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
                "updated_unix": now,
                "elapsed_s": round(elapsed, 1),
                "seconds_per_unit": round(rate, 3) if rate else None,
                "eta_s": round(eta, 1) if eta is not None else None,
                "eta_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now + eta)) if eta is not None else None,
                **self.extra,
            },
        )


def is_alive(status: dict) -> bool:
    """Whether the process that wrote a live status file is still running on this machine."""
    if status.get("host") != socket.gethostname():
        # Another machine: trust the file only if it was updated in the last 5 minutes.
        return time.time() - float(status.get("updated_unix", 0)) < 300
    try:
        os.kill(int(status["pid"]), 0)
    except (OSError, KeyError, ValueError):
        return False
    return True
