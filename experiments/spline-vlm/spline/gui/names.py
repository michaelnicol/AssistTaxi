"""Short names for runs, and a hover tip that shows the folder name."""

import tkinter as tk

from spline.config import MODELS


def friendly_name(run: dict) -> str:
    """A readable label, for example ``SmolVLM2, rank 16, 5 tiles, 25%, 1 pass`` or ``SmolVLM2, untrained, 5 tiles``."""
    config = run["config"]
    model, image, data, train = config["model"], config["image"], config["data"], config["train"]
    lora = model["lora"]
    tiles = f"{image['tiles']} tiles" if image["tiles"] else "no tiling"
    if model["method"] == "base":
        return f"{short_name(model['id'])}, untrained, {tiles}"
    kind = f"rank {lora['r']}" if lora else "full model"
    if train["max_steps"] > 0:
        passes = f"{train['updates']} updates"
    elif train["epochs"] == 1:
        passes = "1 pass"
    else:
        passes = f"{train['epochs']:g} passes"
    if train.get("init_from"):
        passes = f"+{passes} on {train['init_from']['run'].rsplit('-', 1)[-1]}"
    return f"{short_name(model['id'])}, {kind}, {tiles}, {data['train_percent']}%, {passes}"


def labels_for(runs: list[dict]) -> dict[str, str]:
    """Real folder name to a unique readable label. A repeated label keeps the hash."""
    raw = [(run["name"], friendly_name(run)) for run in runs]
    counts: dict[str, int] = {}
    for _real, label in raw:
        counts[label] = counts.get(label, 0) + 1
    labels = {}
    for real, label in raw:
        if counts[label] > 1:
            label = f"{label} ({real.rsplit('-', 1)[-1]})"
        labels[real] = label
    return labels


def short_name(model_id: str) -> str:
    for short, full in MODELS.items():
        if model_id in (short, full):
            return {
                "smolvlm2-256m": "SmolVLM2-256M",
                "smolvlm2-500m": "SmolVLM2-500M",
                "smolvlm2-2.2b": "SmolVLM2",
                "qwen3-vl-2b": "Qwen3-VL-2B",
                "qwen3-vl-8b": "Qwen3-VL",
                "gemma3-12b": "Gemma 3",
            }.get(short, short)
    return model_id.rstrip("/").split("/")[-1]


class HoverTip:
    """A small label that follows the pointer while it is over ``widget``."""

    def __init__(self, widget: tk.Widget) -> None:
        self.widget = widget
        self.win: tk.Toplevel | None = None
        self.text = ""
        widget.bind("<Leave>", lambda _event: self.hide(), add="+")

    def show(self, text: str, event: tk.Event) -> None:
        if not text:
            self.hide()
            return
        if self.win is not None and text == self.text:
            self.win.wm_geometry(f"+{event.x_root + 14}+{event.y_root + 16}")
            return
        self.hide()
        self.text = text
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{event.x_root + 14}+{event.y_root + 16}")
        tk.Label(
            self.win, text=text, background="#fff8c4", relief="solid", borderwidth=1, font=("TkDefaultFont", 9)
        ).pack()

    def hide(self) -> None:
        if self.win is not None:
            self.win.destroy()
            self.win = None
            self.text = ""
