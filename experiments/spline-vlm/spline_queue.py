#!/usr/bin/env python3
"""Run queued specs one at a time: train, score on tune, score on test (a base spec: test only).

Each line of output/queue.txt is the path of one spec file, relative to this
folder, for example ``specs/lora-vs-tiles/r16-alpha32-tiles13.json``. Lines
starting with # are skipped, and lines can be added while the runner works.

queue.txt is never rewritten. A finished line is appended to queue-done.txt and a
failed one to queue-failed.txt; either is then skipped. Delete the line from
queue-failed.txt to retry it. A killed runner restarts the same line, and
training resumes from its last checkpoint.

  tmux new-session -d -s spline-queue -- train-venv/bin/python spline_queue.py
  train-venv/bin/python spline_queue.py grid specs/grids/lora-vs-tiles.json

``grid`` writes one spec per combination to specs/<grid name>/ and appends their
paths to queue.txt. The runner's own log is output/logs/queue.log; each step
logs into its run folder. output/live/queue.json shows the queue to the GUI.
"""

import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from spline import logs, spec
from spline.runs import HERE, now
from spline_train import OUTPUT, build_config

QUEUE = OUTPUT / "queue.txt"

log = logging.getLogger("spline.queue")


def steps(method: str) -> tuple[tuple[str, ...], ...]:
    """The spline_train.py steps a spec runs. A base run is only scored on test."""
    match method:
        case "base":
            return (("score", "test"),)
        case "lora" | "full":
            return (("train",), ("score", "tune"), ("score", "test"))
        case other:
            raise ValueError(f"unknown model.method {other!r}")


def recorded(name: str) -> set[str]:
    """Queue lines already listed in queue-done.txt or queue-failed.txt."""
    path = OUTPUT / name
    found = set()
    if path.exists():
        for line in path.read_text().splitlines():
            _stamp, sep, entry = line.partition("  ")
            if sep and entry.strip():
                found.add(entry.strip())
    return found


def pending() -> list[str]:
    """Queue lines not yet finished or failed, in file order."""
    skip = recorded("queue-done.txt") | recorded("queue-failed.txt")
    lines = QUEUE.read_text().splitlines() if QUEUE.exists() else []
    out = []
    for line in (line.strip() for line in lines):
        if line and not line.startswith("#") and line not in skip and line not in out:
            out.append(line)
    return out


def finish(line: str, dest: str) -> None:
    with (OUTPUT / dest).open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {line}\n")


def status(state: str, current: str | None = None, step: str | None = None, run: str | None = None) -> None:
    """Rewrite output/live/queue.json for the GUI."""
    logs.write_json(
        OUTPUT / "live" / "queue.json",
        {
            "state": state,
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "updated_at": now(),
            "updated_unix": time.time(),
            "current": current,
            "step": step,
            "run": run,
            "pending": pending(),
        },
    )


def run_step(step: tuple[str, ...], line: str) -> int:
    """One spline_train.py step in its own process, so a crash or leak cannot reach the next run."""
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **os.environ}
    argv = [sys.executable, "-u", str(HERE / "spline_train.py"), step[0], line, *step[1:]]
    log.info("running %s", " ".join(argv[2:]))
    # The child logs to its own run folder and the console; its output is not copied here.
    return subprocess.run(argv, cwd=HERE, env=env, check=False).returncode


def work() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.error("another queue runner holds output/queue.lock")
            return
        logs.setup("queue", OUTPUT / "logs")
        _drain()


def _drain() -> None:
    """Run queue lines until none are left. The caller holds the queue lock."""
    try:
        while queue := pending():
            line = queue[0]
            try:
                run, _sha = spec.load(HERE / line)
                name = build_config(run)[0].name
            except Exception:
                log.exception("queue line %s is not a usable spec", line)
                finish(line, "queue-failed.txt")
                continue
            log.info("starting %s (run %s), %d left after it", line, name, len(queue) - 1)
            code = 0
            for step in steps(run.model.method):
                status("running", line, " ".join(step), name)
                code = run_step(step, line)
                if code:
                    log.error("%s %s exited with code %d; see output/runs/%s/logs/", " ".join(step), line, code, name)
                    finish(line, "queue-failed.txt")
                    break
            else:
                finish(line, "queue-done.txt")
            log.info("%s %s", "failed" if code else "finished", name)
        status("idle")
        log.info("queue empty")
    except BaseException:
        status("stopped")
        raise


def grid(path: Path) -> None:
    """Write one spec per grid combination and append each to queue.txt.

    Every spec is built into a run config first, so a grid that would make the
    same run twice, or asks for an impossible tile count, is refused before
    anything is queued.
    """
    logs.setup("grid", OUTPUT / "logs")
    data = json.loads(path.read_text())
    specs = spec.expand_grid(data)
    folder = HERE / "specs" / data["name"]
    names = {}
    for stem, body in specs:
        name = build_config(spec.parse(body, stem))[0].name
        if name in names:
            raise SystemExit(f"{stem} and {names[name]} make the same run {name}")
        names[name] = stem
    folder.mkdir(parents=True, exist_ok=True)
    queued = set(pending())
    with QUEUE.open("a", encoding="utf-8") as handle:
        for (stem, body), name in zip(specs, names, strict=True):
            dest = folder / f"{stem}.json"
            text = json.dumps(body, indent=2) + "\n"
            if dest.exists() and dest.read_text() != text:
                raise SystemExit(f"{dest} exists with other contents; rename the grid or delete it")
            dest.write_text(text)
            line = str(dest.relative_to(HERE))
            if line not in queued:
                handle.write(line + "\n")
            log.info("%s -> run %s", line, name)


if __name__ == "__main__":
    if sys.argv[1:2] == ["grid"] and len(sys.argv) == 3:
        grid(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        work()
    else:
        print("usage: spline_queue.py | spline_queue.py grid GRID.json", file=sys.stderr)
        sys.exit(2)
