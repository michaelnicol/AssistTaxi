"""Runs page: one row per run folder with its state and headline numbers, and the selected run's files."""

import json
import tkinter as tk
from tkinter import ttk

from spline.gui.names import HoverTip, labels_for
from spline.logs import is_alive
from spline.runs import OUTPUT, all_runs

COLUMNS = (
    ("state", "State", 90),
    ("tiles", "Tiles", 50),
    ("method", "Method", 90),
    ("data", "Train %", 60),
    ("updates", "Updates", 70),
    ("loss", "Tune loss", 80),
    ("tune", "Tune cost px", 85),
    ("test", "Test cost px", 85),
    ("within", "Test ≤10px %", 90),
    ("f", "Test F %", 70),
)


def state_of(run: dict) -> str:
    """One word for where a run is: training, scoring, failed, trained, scored, or new."""
    for stage, status in run["live"].items():
        if status.get("state") == "running":
            if is_alive(status):
                return {"train": "training", "sweep": "checking checkpoints"}.get(stage, f"scoring {stage}")
            return "stopped"
        if status.get("state") == "failed":
            return f"failed {stage}"
    if run["scores"]["test"]:
        return "scored"
    if run["train"]:
        return "trained"
    return "new"


def _px(score: dict | None, key: str = "median_px"):
    return score["cost"][key] if score else ""


class RunsPage(ttk.Frame):
    """Table of runs. Selecting one shows its spec, config, and train result on the right."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="Open in Live", command=lambda: self._open("live")).pack(side="left", padx=4)
        ttk.Button(bar, text="Open in Models", command=lambda: self._open("models")).pack(side="left")
        ttk.Button(bar, text="Open in Audit", command=lambda: self._open("audit")).pack(side="left", padx=4)
        self.count = ttk.Label(bar, text="")
        self.count.pack(side="right")
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(6, 0))
        table = ttk.Frame(panes)
        self.tree = ttk.Treeview(table, columns=[key for key, _t, _w in COLUMNS], selectmode="browse")
        self.tree.heading("#0", text="Run")
        self.tree.column("#0", width=380)
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="e")
        scroll = ttk.Scrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._show())
        self.tree.bind("<Double-1>", lambda _event: self._open("live"))
        self.tip = HoverTip(self.tree)
        self.tree.bind("<Motion>", self._hover, add="+")
        panes.add(table, weight=3)
        self.detail = tk.Text(panes, width=60, wrap="none", font=("TkFixedFont", 9))
        panes.add(self.detail, weight=2)
        self.runs: dict[str, dict] = {}
        self.refresh()

    def refresh(self) -> None:
        """Re-read every run folder and redraw the table, keeping the selection."""
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        self.runs = {run["name"]: run for run in all_runs(OUTPUT)}
        labels = labels_for(list(self.runs.values()))
        for name, run in self.runs.items():
            config, lora = run["config"], run["config"]["model"]["lora"]
            base = config["model"]["method"] == "base"
            trained = run["train"] or {}
            values = (
                state_of(run),
                config["image"]["tiles"] or "-",
                f"lora r{lora['r']} a{lora['alpha']}" if lora else config["model"]["method"],
                "-" if base else config["data"]["train_percent"],
                "-" if base else config["train"]["updates"],
                "-" if base else round(trained["best_tune_loss"], 4) if trained.get("best_tune_loss") else "",
                "-" if base else _px(run["scores"]["tune"]),
                _px(run["scores"]["test"]),
                _px(run["scores"]["test"], "within_10_pct"),
                run["scores"]["test"]["f_pct"] if run["scores"]["test"] else "",
            )
            self.tree.insert("", "end", iid=name, text=labels[name], values=values)
        if selected and selected[0] in self.runs:
            self.tree.selection_set(selected)
        self.count.configure(text=f"{len(self.runs)} runs in {OUTPUT / 'runs'}")

    def _hover(self, event) -> None:
        row = self.tree.identify_row(event.y)
        self.tip.show(row, event)

    def selected(self) -> dict | None:
        chosen = self.tree.selection()
        return self.runs.get(chosen[0]) if chosen else None

    def _show(self) -> None:
        run = self.selected()
        self.detail.delete("1.0", "end")
        if not run:
            return
        train = dict(run["train"] or {})
        # The full log history and environment are long; the Live and Compare pages plot the history.
        train.pop("log_history", None)
        train.pop("environment", None)
        parts = [
            f"# {run['folder']}",
            "# spec.json",
            json.dumps(run["spec"], indent=2),
            "# config.json",
            json.dumps(run["config"], indent=2),
            "# train.json (without log_history and environment)",
            json.dumps(train, indent=2) if run["train"] else "not trained",
        ]
        for split_name, score in run["scores"].items():
            if score:
                score = {key: value for key, value in score.items() if key != "environment"}
                parts += [f"# score-{split_name}.json", json.dumps(score, indent=2)]
        self.detail.insert("1.0", "\n".join(parts))

    def _open(self, page: str) -> None:
        run = self.selected()
        if run:
            self.app.show(page, run["name"])
