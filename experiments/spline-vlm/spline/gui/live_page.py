"""Live page: the queue, each stage's progress and ETA, and loss and learning-rate curves.

Reads ``live/*.json`` and ``live/train.jsonl`` in the chosen run folder every
few seconds. The jsonl is read from the last byte offset seen, so a long run
costs the same to follow as a short one.
"""

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from spline.gui.names import HoverTip, labels_for
from spline.logs import is_alive, read_json
from spline.runs import OUTPUT, all_runs

REFRESH_MS = 3000
# Bytes read from the end of an error log for the panel at the bottom.
ERROR_TAIL = 4096
STAGES = ("train", "sweep", "tune", "test")
STAGE_LABELS = {"train": "Training", "sweep": "Check checkpoints", "tune": "Score tune", "test": "Score test"}


def ago(seconds: float | None) -> str:
    """A duration as "2 h 05 m", "4 m 10 s", or "12 s"."""
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} m"
    if seconds >= 60:
        return f"{seconds // 60} m {seconds % 60:02d} s"
    return f"{seconds} s"


def smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average, the same smoothing TensorBoard uses."""
    out, last = [], None
    for value in values:
        last = value if last is None else weight * last + (1 - weight) * value
        out.append(last)
    return out


class History:
    """The rows of one run's live/train.jsonl, read incrementally.

    A step can log a training row, a tune-loss row, and an end-of-run summary.
    Each kind is kept. Plots use the training and tune-loss rows; the summary's
    ``train_loss`` is an average over the run, so it is not drawn as a step.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.rows: dict[tuple[int, str], dict] = {}

    def read(self) -> bool:
        """Read new lines. True if anything changed."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return False
        if size < self.offset:
            self.offset, self.rows = 0, {}
        if size == self.offset:
            return False
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        # Only whole lines; a line being written is picked up next time.
        end = chunk.rfind(b"\n") + 1
        self.offset += end
        for line in chunk[:end].splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # One step can log a training row, a tune-loss row, a tune-cost row, and the end-of-run summary.
            if "eval_tune_cost_px" in row:
                kind = "cost"
            elif "eval_loss" in row:
                kind = "eval"
            else:
                kind = "summary" if "train_runtime" in row else "train"
            self.rows[(row["step"], kind)] = row
        return end > 0

    def series(self, key: str) -> tuple[list[int], list[float]]:
        points = sorted((step, row[key]) for (step, kind), row in self.rows.items() if kind != "summary" and key in row)
        return [step for step, _v in points], [value for _s, value in points]


class LivePage(ttk.Frame):
    """Pick a run (the queue's current run by default) and follow it."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        self.app = app
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Run").pack(side="left")
        self.run = tk.StringVar()
        self.label = tk.StringVar()
        self.friendly: dict[str, str] = {}
        self.choice = ttk.Combobox(top, textvariable=self.label, width=52, state="readonly")
        self.choice.pack(side="left", padx=4)
        self.choice.bind("<<ComboboxSelected>>", lambda _event: self._picked())
        self.tip = HoverTip(self.choice)
        self.choice.bind("<Enter>", self._hover, add="+")
        self.choice.bind("<Motion>", self._hover, add="+")
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Follow the queue", variable=self.follow).pack(side="left", padx=8)
        self.queue = ttk.Label(top, text="")
        self.queue.pack(side="right")

        bars = ttk.Frame(self)
        bars.pack(fill="x", pady=6)
        self.bars, self.texts = {}, {}
        for row, stage in enumerate(STAGES):
            ttk.Label(bars, text=STAGE_LABELS[stage]).grid(row=row, column=0, sticky="w")
            self.bars[stage] = ttk.Progressbar(bars, length=400, maximum=1.0)
            self.bars[stage].grid(row=row, column=1, padx=6, pady=1)
            self.texts[stage] = ttk.Label(bars, text="")
            self.texts[stage].grid(row=row, column=2, sticky="w")

        self.figure = Figure(figsize=(10, 6), layout="constrained")
        grid = self.figure.add_gridspec(2, 2)
        self.loss_axes = self.figure.add_subplot(grid[0, :])
        self.cost_axes = self.loss_axes.twinx()
        self.lr_axes = self.figure.add_subplot(grid[1, 0])
        self.grad_axes = self.figure.add_subplot(grid[1, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False).pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Label(self, text="Warnings and errors (end of the run's error logs)").pack(anchor="w")
        self.errors = tk.Text(self, height=6, wrap="none", font=("TkFixedFont", 9), foreground="#a00")
        self.errors.pack(fill="x")
        self.history: History | None = None
        self.names: list[str] = []
        self.tick()

    def select(self, name: str) -> None:
        """Show ``name`` and stop following the queue."""
        self.follow.set(False)
        self._use(name)
        self._switch()

    def _use(self, name: str) -> None:
        self.run.set(name)
        self.label.set(self.friendly.get(name, name))

    def _picked(self) -> None:
        real = next((name for name, label in self.friendly.items() if label == self.label.get()), "")
        if real:
            self.follow.set(False)
            self.run.set(real)
        self._switch()

    def _hover(self, event) -> None:
        self.tip.show(self.run.get(), event)

    def _switch(self) -> None:
        self.history = History(OUTPUT / "runs" / self.run.get() / "live" / "train.jsonl")
        self._draw()

    def tick(self) -> None:
        """Refresh the run list, the queue line, the bars, and the plots, then schedule the next tick."""
        try:
            self._refresh()
        finally:
            self.after(REFRESH_MS, self.tick)

    def _refresh(self) -> None:
        runs = all_runs(OUTPUT)
        names = [run["name"] for run in runs]
        if names != self.names:
            self.names = names
            self.friendly = labels_for(runs)
            self.choice.configure(values=[self.friendly[name] for name in names])
            if self.run.get() in self.friendly:
                self.label.set(self.friendly[self.run.get()])
        status = read_json(OUTPUT / "live" / "queue.json")
        current = None
        if status and status["state"] == "running" and is_alive(status):
            current = status["run"]
            self.queue.configure(
                text=f"Queue: {status['step']} {status['current']}, {len(status['pending']) - 1} waiting"
            )
        elif status:
            self.queue.configure(text=f"Queue: {status['state']}, {len(status['pending'])} waiting")
        else:
            self.queue.configure(text="Queue: not started")
        if self.follow.get():
            live = [run["name"] for run in runs if any(is_alive(s) for s in run["live"].values())]
            target = current or (live[0] if live else None) or (names[0] if names and not self.run.get() else None)
            if target and target != self.run.get():
                self._use(target)
                self._switch()
                return
        if self.run.get() and self.history is None:
            self._switch()
            return
        self._bars()
        if self.history and self.history.read():
            self._draw()
        self._errors()

    def _bars(self) -> None:
        folder = OUTPUT / "runs" / self.run.get()
        for stage in STAGES:
            status = read_json(folder / "live" / f"{stage}.json")
            if not status:
                self.bars[stage]["value"] = 0
                self.texts[stage].configure(text="not started")
                continue
            self.bars[stage]["value"] = status["done"] / max(1, status["total"])
            text = f"{status['done']} / {status['total']} {status['unit']}"
            if status["state"] == "running" and not is_alive(status):
                text += f"  stopped (process {status['pid']} is gone)"
            elif status["state"] == "running":
                text += f"  elapsed {ago(status['elapsed_s'])}"
                if status.get("eta_s") is not None:
                    finish = time.strftime("%a %H:%M", time.localtime(status["updated_unix"] + status["eta_s"]))
                    text += f", about {ago(status['eta_s'])} left, done around {finish} {time.strftime('%Z')}"
                if status.get("loss") is not None:
                    text += f"  loss {status['loss']:.4f}"
            else:
                text += f"  {status['state']} after {ago(status['elapsed_s'])}"
            self.texts[stage].configure(text=text)

    def _draw(self) -> None:
        for axes in (self.loss_axes, self.cost_axes, self.lr_axes, self.grad_axes):
            axes.clear()
        history = self.history
        if history:
            history.read()
            steps, loss = history.series("loss")
            if steps:
                self.loss_axes.plot(steps, loss, color="#9ab", linewidth=0.8, label="train loss")
                self.loss_axes.plot(steps, smooth(loss), color="#246", label="train loss, smoothed")
            steps, tune = history.series("eval_loss")
            if steps:
                self.loss_axes.plot(steps, tune, "o-", color="#c50", label="tune loss")
                best = min(range(len(tune)), key=tune.__getitem__)
                label = self.loss_axes.annotate(
                    f"best {tune[best]:.4f}", (steps[best], tune[best]), textcoords="offset points", xytext=(-40, 8)
                )
                # A label near the edge would otherwise make the layout shrink the axes to fit it.
                label.set_in_layout(False)
            steps, cost = history.series("eval_tune_cost_px")
            if steps:
                self.cost_axes.plot(steps, cost, "s--", color="#a0a", label="tune cost px (right axis)")
                best = min(range(len(cost)), key=cost.__getitem__)
                label = self.cost_axes.annotate(
                    f"best {cost[best]:.1f} px",
                    (steps[best], cost[best]),
                    textcoords="offset points",
                    xytext=(-40, -14),
                )
                label.set_in_layout(False)
            steps, lr = history.series("learning_rate")
            self.lr_axes.plot(steps, lr, color="#383")
            steps, grad = history.series("grad_norm")
            self.grad_axes.plot(steps, grad, color="#838", linewidth=0.8)
        title = self.friendly.get(self.run.get(), self.run.get()) or "no run selected"
        self.loss_axes.set(title=title, xlabel="update", ylabel="loss")
        self.cost_axes.set(ylabel="tune cost px")
        # clear() moves a twin axis back to the left side.
        self.cost_axes.yaxis.tick_right()
        self.cost_axes.yaxis.set_label_position("right")
        handles, names = self.loss_axes.get_legend_handles_labels()
        cost_handles, cost_names = self.cost_axes.get_legend_handles_labels()
        if handles or cost_handles:
            self.loss_axes.legend(handles + cost_handles, names + cost_names, loc="upper right")
        self.lr_axes.set(title="learning rate", xlabel="update")
        self.grad_axes.set(title="gradient norm (before clipping)", xlabel="update")
        self.canvas.draw_idle()

    def _errors(self) -> None:
        lines = []
        for path in sorted((OUTPUT / "runs" / self.run.get() / "logs").glob("*.errors.log")):
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - ERROR_TAIL))
                tail = handle.read().decode("utf-8", "replace").splitlines()[-8:]
            lines += [f"{path.name}: {line}" for line in tail]
        text = "\n".join(lines[-12:])
        if self.errors.get("1.0", "end-1c") != text:
            self.errors.delete("1.0", "end")
            self.errors.insert("1.0", text)
