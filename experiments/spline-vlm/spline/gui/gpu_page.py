"""GPU page: live use of the GPU, with the train or score process split out from everything else.

A background thread reads nvidia-smi every SAMPLE_S seconds, so a slow read never
freezes the window. The plots cover the last HISTORY_MIN minutes and are only
redrawn while the page is on screen.
"""

import collections
import logging
import math
import os
import threading
import time
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from spline.gui.gpu import live_jobs, sample
from spline.gui.names import labels_for
from spline.runs import OUTPUT, all_runs

log = logging.getLogger("spline.gui.gpu")

SAMPLE_S = 2.0
REFRESH_MS = 2000
HISTORY_MIN = 15
WHOLE, JOB, GUI, LIMIT = "#1d4ed8", "#c2410c", "#7c3aed", "#71717a"
STAGE_WORDS = {
    "train": "Training",
    "sweep": "Checking saved checkpoints",
    "tune": "Scoring tune photos",
    "test": "Scoring test photos",
}


def _series(rows: list[dict], value) -> list[float]:
    """value(row) for each row, with None as NaN so matplotlib leaves a gap."""
    out = []
    for row in rows:
        got = value(row)
        out.append(math.nan if got is None else got)
    return out


def _held(row: dict, pids, key: str) -> float | None:
    """The sum of ``key`` over the processes in ``pids`` that are on the GPU, or None if none are."""
    values = [row["processes"][pid][key] for pid in pids if pid in row["processes"]]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _gb(mib: float | None) -> float | None:
    return None if mib is None else mib / 1024


class GpuPage(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        self.app = app
        self.lock = threading.Lock()
        self.samples: collections.deque[dict] = collections.deque(maxlen=int(HISTORY_MIN * 60 / SAMPLE_S))
        self.friendly: dict[str, str] = {}
        self.job = ttk.Label(self, text="", font=("TkDefaultFont", 12, "bold"))
        self.job.pack(anchor="w")
        self.now = ttk.Label(self, text="Reading the GPU…", font=("TkFixedFont", 11))
        self.now.pack(anchor="w", pady=(2, 6))
        self.figure = Figure(figsize=(12, 7), layout="constrained")
        axes = self.figure.subplots(3, 2, sharex=True)
        (self.busy, self.memory), (self.power, self.heat), (self.clock, self.bandwidth) = axes
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        threading.Thread(target=self._sample_loop, name="gpu-sampler", daemon=True).start()
        self.tick()

    def _sample_loop(self) -> None:
        while True:
            try:
                reading = sample()
                if reading is not None:
                    reading["jobs"] = live_jobs()
                    with self.lock:
                        self.samples.append(reading)
            except Exception:
                log.exception("GPU sample failed")
            time.sleep(SAMPLE_S)

    def tick(self) -> None:
        try:
            if self.winfo_ismapped():
                self._draw()
        finally:
            self.after(REFRESH_MS, self.tick)

    def _run_label(self, run: str) -> str:
        if run not in self.friendly:
            self.friendly = labels_for(all_runs(OUTPUT))
        return self.friendly.get(run, run)

    def _draw(self) -> None:
        with self.lock:
            rows = list(self.samples)
        if not rows:
            return
        last = rows[-1]
        jobs = last["jobs"]
        gui = os.getpid()
        if jobs:
            self.job.configure(
                text="   ".join(
                    f"{STAGE_WORDS.get(stage, stage)}: {self._run_label(run)} (pid {pid})"
                    for pid, (run, stage) in jobs.items()
                )
            )
        else:
            self.job.configure(text="No job running")

        def show(key: str, scale: float = 1, digits: int = 0) -> str:
            value = last.get(key)
            return "?" if value is None else f"{value / scale:.{digits}f}"

        now = (
            f"GPU {show('utilization.gpu')}% busy    "
            f"memory {show('memory.used', 1024, 1)} of {show('memory.total', 1024)} GB    "
            f"power {show('power.draw')} of {show('power.limit')} W    "
            f"{show('temperature.gpu')} °C    SM clock {show('clocks.sm')} MHz"
        )
        job_sm, job_mem = _held(last, jobs, "sm"), _held(last, jobs, "mem_mib")
        if jobs and job_mem is not None:
            now += f"\njob: {job_mem / 1024:.1f} GB"
            if job_sm is not None:
                now += f", {job_sm:.0f}% compute"
        gui_mem = _held(last, [gui], "mem_mib")
        if gui_mem is not None:
            now += f"\nGUI: {gui_mem / 1024:.1f} GB"
        self.now.configure(text=now)

        x = [(row["time"] - last["time"]) / 60 for row in rows]

        def job(key, scale=lambda value: value):
            return _series(rows, lambda row: None if (v := _held(row, row["jobs"], key)) is None else scale(v))

        self._plot(
            self.busy,
            "Compute busy",
            "%",
            x,
            [
                (_series(rows, lambda row: row["utilization.gpu"]), WHOLE, "whole GPU", "-"),
                (job("sm"), JOB, "training or scoring process", "-"),
            ],
            top=100,
        )
        self._plot(
            self.memory,
            "Memory held",
            "GB",
            x,
            [
                (_series(rows, lambda row: _gb(row["memory.used"])), WHOLE, "whole GPU", "-"),
                (job("mem_mib", lambda mib: mib / 1024), JOB, "training or scoring process", "-"),
                (_series(rows, lambda row: _gb(_held(row, [gui], "mem_mib"))), GUI, "this window", "-"),
                (_series(rows, lambda row: _gb(row["memory.total"])), LIMIT, "card total", "--"),
            ],
        )
        self._plot(
            self.power,
            "Power",
            "W",
            x,
            [
                (_series(rows, lambda row: row["power.draw"]), WHOLE, "draw", "-"),
                (_series(rows, lambda row: row["power.limit"]), LIMIT, "limit", "--"),
            ],
        )
        self._plot(
            self.heat, "Temperature", "°C", x, [(_series(rows, lambda row: row["temperature.gpu"]), WHOLE, "", "-")]
        )
        self._plot(
            self.clock,
            "SM clock",
            "MHz",
            x,
            [
                (_series(rows, lambda row: row["clocks.sm"]), WHOLE, "now", "-"),
                (_series(rows, lambda row: row["clocks.max.sm"]), LIMIT, "max", "--"),
            ],
        )
        self._plot(
            self.bandwidth,
            "Memory bandwidth busy",
            "%",
            x,
            [(_series(rows, lambda row: row["utilization.memory"]), WHOLE, "", "-")],
            top=100,
        )
        for axes in (self.clock, self.bandwidth):
            axes.set_xlabel("minutes ago")
        self.canvas.draw_idle()

    @staticmethod
    def _plot(axes, title: str, unit: str, x: list[float], lines, top: float | None = None) -> None:
        axes.clear()
        for values, color, label, style in lines:
            axes.plot(x, values, style, color=color, label=label or None, linewidth=1.4)
        axes.set_title(title, loc="left", fontsize=10)
        axes.set_ylabel(unit)
        axes.set_xlim(-HISTORY_MIN, 0)
        axes.set_ylim(bottom=0, top=top)
        axes.grid(alpha=0.3)
        if sum(1 for _values, _color, label, _style in lines if label) > 1:
            axes.legend(loc="upper left", fontsize=8)
