"""Compare page: one metric against one setting across runs, and tune curves side by side.

Pick runs in the list (all by default). The left plot puts a score metric
against a setting, one colour per value of a second setting, for example tune
cost against tiles, coloured by LoRA rank. The right plot overlays the chosen
runs' tune loss (solid, left axis) and tune cost (dashed, right axis) while
training; base runs have no curve. A warning shows when the plotted scores used
different photos or decode settings, because then they are not comparable.
"""

import json
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from spline.gui.names import HoverTip, labels_for
from spline.runs import OUTPUT, all_runs


def _lora(run: dict, key: str):
    lora = run["config"]["model"]["lora"]
    return lora[key] if lora else None


def _train(run: dict, key: str):
    train = run["config"]["train"]
    return train[key] if train else None


# Settings a run can be placed by: name -> how to read it from a run summary.
SETTINGS = {
    "tiles": lambda run: run["config"]["image"]["tiles"],
    "LoRA r": lambda run: _lora(run, "r"),
    "LoRA alpha": lambda run: _lora(run, "alpha"),
    "method": lambda run: run["config"]["model"]["method"],
    "train %": lambda run: run["config"]["data"]["train_percent"],
    "learning rate": lambda run: _train(run, "learning_rate"),
    "updates": lambda run: _train(run, "updates"),
    "seed": lambda run: _train(run, "seed"),
}


def _score(split: str, *keys: str):
    def read(run: dict):
        value = run["scores"][split]
        for key in keys:
            value = value[key] if value else None
        return value

    return read


# Metrics: name -> (reader, which split's score it comes from, or None).
METRICS = {
    "tune cost median px": (_score("tune", "cost", "median_px"), "tune"),
    "tune cost mean px": (_score("tune", "cost", "mean_px"), "tune"),
    "tune ≤10px %": (_score("tune", "cost", "within_10_pct"), "tune"),
    "tune precision %": (_score("tune", "precision_pct"), "tune"),
    "tune recall %": (_score("tune", "recall_pct"), "tune"),
    "tune F %": (_score("tune", "f_pct"), "tune"),
    "tune readable %": (_score("tune", "readable_pct"), "tune"),
    "test cost median px": (_score("test", "cost", "median_px"), "test"),
    "test F %": (_score("test", "f_pct"), "test"),
    "best tune cost px": (lambda run: (run["train"] or {}).get("best_tune_cost_px"), None),
    "best tune loss": (lambda run: (run["train"] or {}).get("best_tune_loss"), None),
    "train minutes": (lambda run: round(run["train"]["runtime_s"] / 60, 1) if run["train"] else None, None),
}


class ComparePage(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        side = ttk.Frame(self)
        side.pack(side="left", fill="y")
        self.metric = tk.StringVar(value="tune cost median px")
        self.x = tk.StringVar(value="tiles")
        self.colour = tk.StringVar(value="LoRA r")
        for title, var, values in (
            ("Metric (y)", self.metric, list(METRICS)),
            ("Setting (x)", self.x, list(SETTINGS)),
            ("Colour by", self.colour, list(SETTINGS)),
        ):
            ttk.Label(side, text=title).pack(anchor="w")
            box = ttk.Combobox(side, textvariable=var, values=values, state="readonly", width=28)
            box.pack(fill="x", pady=(0, 4))
            box.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        ttk.Button(side, text="Reload runs", command=self.refresh).pack(fill="x", pady=4)
        ttk.Label(side, text="Runs (pick several)").pack(anchor="w")
        self.list = tk.Listbox(side, selectmode="extended", width=48, exportselection=False)
        self.list.pack(fill="both", expand=True)
        self.list.bind("<<ListboxSelect>>", lambda _event: self.draw())
        self.tip = HoverTip(self.list)
        self.list.bind("<Motion>", self._hover, add="+")
        self.warning = ttk.Label(side, text="", foreground="#b91c1c", wraplength=330)
        self.warning.pack(fill="x")
        self.figure = Figure(figsize=(11, 5), layout="constrained")
        self.scatter = self.figure.add_subplot(1, 2, 1)
        self.curves = self.figure.add_subplot(1, 2, 2)
        self.costs = self.curves.twinx()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False).pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.runs: list[dict] = []
        self.refresh()

    def refresh(self) -> None:
        self.runs = all_runs(OUTPUT)
        self.labels = labels_for(self.runs)
        self.list.delete(0, "end")
        for run in self.runs:
            self.list.insert("end", self.labels[run["name"]])
        self.list.select_set(0, "end")
        self.draw()

    def _hover(self, event) -> None:
        index = self.list.nearest(event.y)
        name = self.runs[index]["name"] if 0 <= index < len(self.runs) else ""
        self.tip.show(name, event)

    def chosen(self) -> list[dict]:
        return [self.runs[index] for index in self.list.curselection()]

    def draw(self) -> None:
        runs = self.chosen()
        read, split = METRICS[self.metric.get()]
        place, colour = SETTINGS[self.x.get()], SETTINGS[self.colour.get()]
        self.scatter.clear()
        groups: dict = {}
        for run in runs:
            y = read(run)
            if y is not None and place(run) is not None:
                groups.setdefault(colour(run), []).append((place(run), y, self.labels[run["name"]]))
        for value, points in sorted(groups.items(), key=lambda item: str(item[0])):
            points.sort()
            self.scatter.plot(
                [p[0] for p in points], [p[1] for p in points], "o-", label=f"{self.colour.get()} {value}"
            )
        self.scatter.set(xlabel=self.x.get(), ylabel=self.metric.get(), title=f"{len(runs)} runs chosen")
        if groups:
            self.scatter.legend()
        self.scatter.grid(alpha=0.3)
        self.curves.clear()
        self.costs.clear()
        for run in runs:
            if run["config"]["model"]["method"] == "base":
                continue
            history = (run["train"] or {}).get("log_history") or _live_history(run)
            points = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
            costs = [(row["step"], row["eval_tune_cost_px"]) for row in history if "eval_tune_cost_px" in row]
            if points:
                line = self.curves.plot(*zip(*points, strict=True), "o-", markersize=3, label=self.labels[run["name"]])[
                    0
                ]
                if costs:
                    self.costs.plot(*zip(*costs, strict=True), "s--", markersize=3, color=line.get_color())
        self.curves.set(xlabel="update", ylabel="tune loss (solid)", title="tune loss and tune cost while training")
        self.costs.set(ylabel="tune cost px (dashed)")
        # clear() moves a twin axis back to the left side.
        self.costs.yaxis.tick_right()
        self.costs.yaxis.set_label_position("right")
        if self.curves.get_legend_handles_labels()[0]:
            self.curves.legend(fontsize=7)
        self.curves.grid(alpha=0.3)
        self.canvas.draw_idle()
        self.warning.configure(text=_comparable(runs, split))


def _live_history(run: dict) -> list[dict]:
    """A run still training has no train.json yet; its live jsonl has the same entries."""
    path = run["folder"] / "live" / "train.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _comparable(runs: list[dict], split: str | None) -> str:
    """A warning if the chosen runs' scores on ``split`` cannot be compared, else empty."""
    if split is None:
        return ""
    scores = [run["scores"][split] for run in runs if run["scores"][split]]
    photos = {score["photos_sha256"] for score in scores}
    decodes = {json.dumps(score["decode"], sort_keys=True) for score in scores}
    splits = {run["config"]["data"]["split_sha256"] for run in runs}
    problems = []
    if len(photos) > 1:
        problems.append("different photos")
    if len(decodes) > 1:
        problems.append("different decode settings (batch size changes answers)")
    if len(splits) > 1:
        problems.append("different split.json files")
    return f"Not comparable: {', '.join(problems)}." if problems else ""
