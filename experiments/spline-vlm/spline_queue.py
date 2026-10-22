#!/usr/bin/env python3
"""Run queued trainings one at a time, each trained and then scored.

Each line of output/queue.txt holds one run's spline_train.py flags, for example

  --lora --data-percent 10 --tiles 29 --batch-size 1 --grad-accum 8

Lines starting with # are skipped. Lines can be added while the runner works.
A finished line moves to output/queue-done.txt, a failed one to queue-failed.txt.
A killed runner restarts the same line, which resumes from its checkpoint.

  tmux new-session -d -s spline-queue -- train-venv/bin/python spline_queue.py
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from spline.runs import append_error

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
QUEUE = OUTPUT / "queue.txt"
LOG = OUTPUT / "queue.log"


def next_line() -> str | None:
    lines = QUEUE.read_text().splitlines() if QUEUE.exists() else []
    return next((line.strip() for line in lines if line.strip() and not line.strip().startswith("#")), None)


def finish(line: str, dest: str) -> None:
    lines = QUEUE.read_text().splitlines()
    lines.pop(next(i for i, text in enumerate(lines) if text.strip() == line))
    QUEUE.write_text("".join(text + "\n" for text in lines))
    with (OUTPUT / dest).open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {line}\n")


def run(command: str, line: str) -> int:
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} {command} {line}\n")
        log.flush()
        argv = [sys.executable, "-u", str(HERE / "spline_train.py"), command, *shlex.split(line)]
        return subprocess.run(argv, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main() -> None:
    while (line := next_line()) is not None:
        print(time.strftime("%H:%M"), "starting", line, flush=True)
        for command in ("train", "score"):
            code = run(command, line)
            if code:
                append_error(OUTPUT, f"queue: {line}", f"{command} exited with code {code}. See output/queue.log.")
                finish(line, "queue-failed.txt")
                break
        else:
            finish(line, "queue-done.txt")
        print(time.strftime("%H:%M"), "finished" if not code else "failed", line, flush=True)
    print("queue empty")


if __name__ == "__main__":
    main()
