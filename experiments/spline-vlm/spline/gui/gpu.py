"""Reading the GPU: live use for the GPU page, and whether an inference job can start
without taking memory the trainer still needs."""

import logging
import subprocess
import time

from spline.logs import is_alive, read_json
from spline.runs import OUTPUT

log = logging.getLogger("spline.gui.gpu")

# Left free while a train or score process is alive, so its next allocation still fits.
TRAINING_CUSHION_MIB = 10 * 1024
IDLE_CUSHION_MIB = 2 * 1024
GPU_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.max.sm",
)


class NotYet:
    """The GPU does not have room yet. The page should try again later."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def estimate_mib(model_id: str, tiles: int | None) -> int:
    """A rough upper bound on what loading this model and answering one photo needs."""
    ident = model_id.lower()
    if "12b" in ident:
        weights = 26_000
    elif "8b" in ident:
        weights = 18_000
    elif "14b" in ident:
        weights = 30_000
    else:
        weights = 6_000
    return weights + 500 * max(1, tiles or 8)


def free_mib() -> int | None:
    """Free GPU memory in MiB, or None if nvidia-smi cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning("could not read free GPU memory: %s", error)
        return None
    line = out.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return int(line[0].strip())
    except ValueError:
        return None


def _number(text: str) -> float | None:
    """A reading as a float, or None for "-" and "[N/A]"."""
    try:
        return float(text)
    except ValueError:
        return None


def sample() -> dict | None:
    """One reading of the whole GPU and of each process on it, or None if nvidia-smi cannot be read.

    The GPU fields are keyed by their nvidia-smi names. ``processes`` maps pid to
    {"sm": percent of the GPU's compute, "mem_mib": memory held}, from ``nvidia-smi pmon``.
    """
    try:
        whole = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("could not read the GPU: %s", error)
        return None
    lines = whole.strip().splitlines()
    if not lines:
        return None
    reading = dict(zip(GPU_FIELDS, (_number(part.strip()) for part in lines[0].split(",")), strict=False))
    columns: list[str] = []
    processes = {}
    for line in pmon.splitlines():
        parts = line.split()
        if line.startswith("#"):
            # The first header line names the columns; the second gives their units.
            columns = columns or parts[1:]
            continue
        row = dict(zip(columns, parts, strict=False))
        if "pid" in row and row["pid"].isdigit():
            processes[int(row["pid"])] = {"sm": _number(row.get("sm", "-")), "mem_mib": _number(row.get("fb", "-"))}
    reading["time"] = time.time()
    reading["processes"] = processes
    return reading


def live_jobs() -> dict[int, tuple[str, str]]:
    """pid -> (run name, stage) for every train or score process from this project that is alive."""
    jobs = {}
    for path in (OUTPUT / "runs").glob("*/live/*.json"):
        status = read_json(path)
        if status and status.get("state") == "running" and is_alive(status):
            jobs[int(status["pid"])] = (path.parent.parent.name, status["stage"])
    return jobs


def job_running() -> bool:
    """Whether a training or scoring process from this project is still alive."""
    if live_jobs():
        return True
    queue = read_json(OUTPUT / "live" / "queue.json")
    return bool(queue and queue.get("state") == "running" and is_alive(queue))


def room_for(model_id: str, tiles: int | None, already_loaded: bool = False) -> tuple[bool, str]:
    """Whether this inference leaves the cushion a live training job needs.

    ``already_loaded`` means the weights are on the GPU, so only the photo's
    activations still have to fit.
    """
    free = free_mib()
    need = 1500 + 400 * max(1, tiles or 8) if already_loaded else estimate_mib(model_id, tiles)
    busy = job_running()
    cushion = TRAINING_CUSHION_MIB if busy else IDLE_CUSHION_MIB
    if free is None:
        return False, "Waiting: cannot read free GPU memory"
    if free < need + cushion:
        held = ", rest held for training" if busy else ""
        return False, f"Waiting: needs ~{need / 1024:.0f} GB, {free / 1024:.0f} GB free{held}"
    return True, ""
