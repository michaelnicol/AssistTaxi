"""Models page: one plain-language description of each trained run, and of the base models."""

import tkinter as tk
from tkinter import ttk

from spline.config import MODELS
from spline.gui.names import HoverTip, friendly_name, labels_for
from spline.gui.runs_page import state_of
from spline.runs import OUTPUT, all_runs

BASES = {
    "smolvlm2-256m": ("SmolVLM2, 256 million parameters", "Tiled like the 2.2B."),
    "smolvlm2-500m": ("SmolVLM2, 500 million parameters", "Tiled like the 2.2B."),
    "smolvlm2-2.2b": ("SmolVLM2, 2.2 billion parameters", "Photos are cut into square tiles."),
    "qwen3-vl-2b": ("Qwen3-VL, 2 billion parameters", "No tiling; native resolution."),
    "qwen3-vl-8b": ("Qwen3-VL, 8 billion parameters", "No tiling; native resolution."),
    "gemma3-12b": ("Gemma 3, 12 billion parameters", "No tiling; 896 px square."),
}


def short_base(short: str) -> str:
    return BASES[short][0]


def _hours(seconds: float | None) -> str:
    if not seconds:
        return "an unknown time"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} m"
    return f"{max(1, seconds // 60)} m"


def _model_name(model_id: str) -> str:
    for short, (title, _body) in BASES.items():
        if MODELS.get(short) == model_id or short == model_id:
            return title
    return model_id


def describe(run: dict) -> str:
    """One fact per line, readable without opening the JSON."""
    config = run["config"]
    model, image, data, train = config["model"], config["image"], config["data"], config["train"]
    lora = model["lora"]
    base = model["method"] == "base"
    lines = [friendly_name(run), run["name"], f"State: {state_of(run)}", ""]
    if base:
        lines.append(f"Model: {_model_name(model['id'])}, untrained")
    elif lora:
        order = ["q_proj", "k_proj", "v_proj", "o_proj"]
        names = sorted(lora["target_suffixes"], key=lambda name: order.index(name) if name in order else 99)
        vision = "" if lora["skip_vision"] else " + vision"
        lines.append(f"Model: {_model_name(model['id'])}")
        lines.append(f"LoRA: rank {lora['r']}, alpha {lora['alpha']}, on {', '.join(names)}{vision}")
    else:
        lines.append(f"Model: {_model_name(model['id'])}, full fine-tune")
    lines.append(f"Tiles: {image['tiles'] or 'none'}")
    if not base:
        lines.append(f"Data: {data['train_percent']}% ({data['train_photos']:,} photos), {_passes(train)}")
        if train.get("init_from"):
            lines.append(f"Started from: {train['init_from']['run']}")
        lines.append(
            f"Update: {train['photos_per_update']} photos ({train['batch_size']} × {train['grad_accum']}), "
            f"lr {train['learning_rate']:g} {train['lr_scheduler']}, warmup {train['warmup_steps']}, "
            f"seed {train['seed']}"
        )
        every = train.get("save_every") or train["eval_steps"]
        lines.append(f"Checkpoints: every {every} updates, picked on {data['tune_cost_photos']} tune photos")
        trained = run["train"]
        if trained:
            lines.append(
                f"Kept: update {trained['best_step']}, tune cost {_num(trained['best_tune_cost_px'])} px, "
                f"tune loss {_num(trained['best_tune_loss'])}, took {_hours(trained.get('runtime_s'))}"
            )
            if trained["fallbacks"]:
                lines.append("Memory fallbacks: " + "; ".join(trained["fallbacks"]))
    lines.append("")
    for split, title in (("tune", "Tune"), ("test", "Test")):
        score = run["scores"][split]
        if not score:
            lines.append(f"{title}: not scored")
            continue
        cost = score["cost"]
        lines.append(
            f"{title} ({score['photos']:,} photos): cost median {_num(cost['median_px'])} px, "
            f"mean {_num(cost['mean_px'])} px; F {score['f_pct']}% "
            f"(P {score['precision_pct']}%, R {score['recall_pct']}%); "
            f"readable {score['readable_pct']}%, lanes {score['lanes_pred_mean']} vs {score['lanes_gold_mean']}"
        )
    lines.append("")
    lines.append(f"Prompt: {data['prompt']!r}")
    lines.append(_judge_words(config["judge"]))
    return "\n".join(lines)


def _judge_words(judge: dict) -> str:
    return (
        f"Cost: truncated Chamfer, step {judge['step_px']:g} px, cap {judge['tau_px']:g} px, "
        f"hit radius {judge['radius_px']:g} px"
    )


def _passes(train: dict) -> str:
    updates = train["updates"]
    if train["max_steps"] > 0:
        return f"{updates:,} updates"
    word = "pass" if train["epochs"] == 1 else "passes"
    return f"{train['epochs']:g} {word}, {updates:,} updates"


def _num(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value < 10 else f"{value:.1f}"
    return str(value)


class ModelsPage(ttk.Frame):
    """A list of runs and base models, with a paragraph about the one you pick."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        self.app = app
        side = ttk.Frame(self)
        side.pack(side="left", fill="y")
        bar = ttk.Frame(side)
        bar.pack(fill="x")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="Open in Audit", command=self._audit).pack(side="left", padx=4)
        self.list = tk.Listbox(side, width=46, exportselection=False)
        self.list.pack(fill="both", expand=True, pady=(6, 0))
        self.list.bind("<<ListboxSelect>>", lambda _event: self._show())
        self.tip = HoverTip(self.list)
        self.list.bind("<Motion>", self._hover, add="+")
        self.text = tk.Text(self, wrap="word", font=("TkDefaultFont", 12), padx=12, pady=8)
        self.text.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.text.tag_configure("title", font=("TkDefaultFont", 14, "bold"))
        self.runs: list[dict] = []
        self.rows: list[tuple[str, str]] = []
        self.refresh()

    def refresh(self) -> None:
        chosen = self.list.curselection()
        self.runs = all_runs(OUTPUT)
        labels = labels_for(self.runs)
        self.rows = [("head", "")]
        self.list.delete(0, "end")
        self.list.insert("end", "Base models")
        for short in MODELS:
            self.rows.append(("base", short))
            self.list.insert("end", "  " + short_base(short))
        self.rows.append(("head", ""))
        self.list.insert("end", "Runs")
        for run in self.runs:
            self.rows.append(("run", run["name"]))
            self.list.insert("end", "  " + labels[run["name"]])
        index = chosen[0] if chosen and chosen[0] < self.list.size() else self.list.size() - 1
        if index >= 0:
            self.list.selection_set(index)
            self.list.see(index)
        self._show()

    def select(self, name: str) -> None:
        self.refresh()
        for index, (kind, key) in enumerate(self.rows):
            if kind == "run" and key == name:
                self.list.selection_clear(0, "end")
                self.list.selection_set(index)
                self.list.see(index)
                self._show()
                return

    def _row(self) -> tuple[str, str]:
        chosen = self.list.curselection()
        if not chosen or chosen[0] >= len(self.rows):
            return ("head", "")
        return self.rows[chosen[0]]

    def _selected_run(self) -> dict | None:
        kind, key = self._row()
        if kind != "run":
            return None
        for run in self.runs:
            if run["name"] == key:
                return run
        return None

    def _hover(self, event) -> None:
        index = self.list.nearest(event.y)
        kind, key = self.rows[index] if index < len(self.rows) else ("head", "")
        self.tip.show(key if kind == "run" else "", event)

    def _show(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        run = self._selected_run()
        kind, key = self._row()
        if run:
            body = describe(run)
            self.text.insert("1.0", body)
            self.text.tag_add("title", "1.0", "1.end")
        elif kind == "base" and key in BASES:
            title, paragraph = BASES[key]
            self.text.insert("1.0", f"{title}\n\n{paragraph}\nUntrained hub weights.")
            self.text.tag_add("title", "1.0", "1.end")
        else:
            self.text.insert("1.0", "Pick a run.")
        self.text.configure(state="disabled")

    def _audit(self) -> None:
        run = self._selected_run()
        if run:
            self.app.show("audit", run["name"])
