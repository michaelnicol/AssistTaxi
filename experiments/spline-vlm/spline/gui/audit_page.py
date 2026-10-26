"""Audit page: put any photo in front of any model and see the labels, the answer, and the cost.

The left image shows what the annotators drew (raw LLAMAS marker segments) and
the gold polylines simplified from them, which are what the model is trained to say.
The right image shows the predicted polylines over a faint band ``radius_px`` wide
on each side of the gold lanes. Each predicted point is green when it lies within
that radius of gold and red when it does not. Below, the cost, precision, and
recall come from the same function as scoring (:func:`spline.score.measure`),
with the judge settings of the chosen run.

A photo can come from any split of split.json, or from any file on disk. For a
LLAMAS photo the label is looked up in labels.zip even if split.json left it out.
The page warns when the chosen run trained on the photo, or used it to pick its
checkpoint, because the error on such a photo says little about new photos.

Choosing a model loads it, then answers the photo on screen. Changing the photo
asks again. A stored score answer is shown while that live answer is on its way.
The live answer is one photo at a time, which can differ slightly from a batched score.
"""

import gc
import logging
import random
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from spline.config import MODELS, PROMPT, DecodeSettings, JudgeSettings, from_dict
from spline.data import IMAGES, label_for, marker_segments, reference_size
from spline.gui.gpu import NotYet, room_for
from spline.gui.names import HoverTip, labels_for, short_name
from spline.gui.state import SPLITS, Job, answering_runs, run_photos, split, split_place, stored_answers
from spline.lanes import DEFAULT_JUDGE, densify, nearest, read_lanes, spline_answer
from spline.model import (
    PromptCollator,
    answer_batch,
    context_tokens,
    load_base,
    load_for_scoring,
    load_processor,
    precision,
    resolve_model,
    resolve_revision,
)
from spline.score import measure

log = logging.getLogger("spline.gui.audit")

NO_MODEL = "(no model: labels only)"
BASE = "base model: "
BASE_TILES = ("default", "1", "5", "7", "13", "16", "25", "29", "41")
GOLD, PREDICTED, MARKER = "#22c55e", "#f97316", "#ef4444"
NEAR, FAR = "#16a34a", "#dc2626"
# What each image can draw, in drawing order. Hit / miss dots cover the orange prediction line.
LAYERS = (
    ("zone", "Hit zone"),
    ("markers", "Annotator marks"),
    ("labels", "Labels"),
    ("predicted", "Prediction"),
    ("hits", "Hit / miss"),
)
SHOWN = {"left": {"markers", "labels"}, "right": {"zone", "labels", "predicted", "hits"}}
CUT = "\n[cut off at the context window]"


@dataclass
class Loaded:
    """A model ready to answer, and what it is."""

    key: str
    processor: object
    model: object
    context_tokens: int
    prompt: str


@dataclass
class Photo:
    """Everything known about the photo on screen."""

    path: str
    pixels: np.ndarray
    width: int
    height: int
    # (split, index, gold answer) in split.json, or None.
    place: tuple[str, int, str] | None
    # Raw LLAMAS label, or None for a photo with no label.
    label: dict | None
    # Gold answer text: from split.json, else fitted from the label, else None.
    gold: str | None


class AuditPage(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, padding=6)
        self.app = app
        self.loaded: Loaded | None = None
        self.loading_key: str | None = None
        self.photo: Photo | None = None
        self.answer: tuple[str, str] | None = None  # (raw text, where it came from)
        self.runs: dict[str, dict] = {}

        side = ttk.Frame(self, width=400)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._model_box(side)
        self._photo_box(side)
        self.card = tk.Canvas(side, width=380, height=168, highlightthickness=0, background="#f4f4f5")
        self.card.pack(fill="x", pady=4)
        self.bars = tk.Canvas(side, width=380, height=220, highlightthickness=0, background="white")
        self.bars.pack(fill="both", expand=True)

        main = ttk.Frame(self)
        main.pack(side="left", fill="both", expand=True, padx=(6, 0))
        phase = ttk.Frame(main)
        phase.pack(fill="x")
        self.phase_label = ttk.Label(phase, text="Pick a model.")
        self.phase_label.pack(anchor="w")
        self.phase_bar = ttk.Progressbar(phase, mode="indeterminate")
        self.phase_bar.pack(fill="x", pady=(2, 6))
        self._layer_boxes(main)
        self.figure = Figure(figsize=(12, 4), layout="constrained")
        self.left = self.figure.add_subplot(1, 2, 1)
        self.right = self.figure.add_subplot(1, 2, 2, sharex=self.left, sharey=self.left)
        self.canvas = FigureCanvasTkAgg(self.figure, main)
        NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False).pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.prompt_label = ttk.Label(main, text="", wraplength=1100, foreground="#52525b")
        self.prompt_label.pack(anchor="w")
        ttk.Label(main, text="Model answer (raw text)").pack(anchor="w")
        self.raw = tk.Text(main, height=7, wrap="none", font=("TkFixedFont", 9))
        self.raw.pack(fill="x")
        self._spinning = False
        self.refresh_models()
        self._phase("idle", "Pick a model.")

    # ----- controls -----

    def _model_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Model", padding=4)
        box.pack(fill="x")
        self.model = tk.StringVar(value=NO_MODEL)
        self.model_label = tk.StringVar(value=NO_MODEL)
        self.model_real: dict[str, str] = {NO_MODEL: NO_MODEL}
        self._gpu_wait: str | None = None
        self.model_choice = ttk.Combobox(box, textvariable=self.model_label, state="readonly", width=48)
        self.model_choice.pack(fill="x")
        self.model_choice.bind("<<ComboboxSelected>>", lambda _event: self._picked_model())
        self.tip = HoverTip(self.model_choice)
        self.model_choice.bind("<Enter>", self._hover_model, add="+")
        self.model_choice.bind("<Motion>", self._hover_model, add="+")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Base model tiles").pack(side="left")
        self.base_tiles = tk.StringVar(value="default")
        tiles = ttk.Combobox(row, textvariable=self.base_tiles, values=BASE_TILES, width=8, state="readonly")
        tiles.pack(side="left", padx=4)
        tiles.bind("<<ComboboxSelected>>", lambda _event: self._base_tiles_changed())
        ttk.Button(row, text="Reload list", command=self.refresh_models).pack(side="right")

    def _photo_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Photo", padding=4)
        box.pack(fill="x", pady=4)
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.split = tk.StringVar(value="test")
        ttk.Combobox(row, textvariable=self.split, values=SPLITS, width=6, state="readonly").pack(side="left")
        self.index = tk.IntVar(value=0)
        spin = ttk.Spinbox(row, from_=0, to=10**6, textvariable=self.index, width=8, command=self.show_split_photo)
        spin.pack(side="left", padx=4)
        spin.bind("<Return>", lambda _event: self.show_split_photo())
        ttk.Button(row, text="Go", command=self.show_split_photo, width=4).pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="◀ Prev", command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(row, text="Next ▶", command=lambda: self._step(1)).pack(side="left", padx=4)
        ttk.Button(row, text="Random", command=self._random).pack(side="left")
        ttk.Button(row, text="Open file…", command=self._open_file).pack(side="right")

    def _layer_boxes(self, parent) -> None:
        """One row of switches above each image, so the two images can show different things."""
        row = ttk.Frame(parent)
        row.pack(fill="x")
        self.layers: dict[str, dict[str, tk.BooleanVar]] = {}
        for side, title in (("left", "Left image"), ("right", "Right image")):
            box = ttk.LabelFrame(row, text=title, padding=4)
            box.pack(side="left", fill="x", expand=True, padx=(0, 6) if side == "left" else 0)
            self.layers[side] = {}
            for place, (key, text) in enumerate(LAYERS):
                var = tk.BooleanVar(value=key in SHOWN[side])
                ttk.Checkbutton(box, text=text, variable=var, command=self._draw).grid(
                    row=place // 3, column=place % 3, sticky="w", padx=(0, 10)
                )
                self.layers[side][key] = var

    # ----- models -----

    def refresh_models(self) -> None:
        """Fill the model list: no model, each base model, then every trained run."""
        found = answering_runs()
        self.runs = {run["name"]: run for run in found}
        labels = labels_for(found)
        self.model_real = {NO_MODEL: NO_MODEL}
        values = [NO_MODEL]
        for short in MODELS:
            label = f"{short_name(MODELS[short])}, not trained"
            self.model_real[label] = BASE + short
            values.append(label)
        for name, label in labels.items():
            self.model_real[label] = name
            values.append(label)
        self.model_choice.configure(values=values)
        self.model_label.set(
            next((label for label, real in self.model_real.items() if real == self.model.get()), NO_MODEL)
        )

    def select(self, name: str) -> None:
        """Pick a run from another page."""
        self.refresh_models()
        if name in self.runs:
            self.model.set(name)
            self.model_label.set(next(label for label, real in self.model_real.items() if real == name))
            self._model_changed()

    def _picked_model(self) -> None:
        self.model.set(self.model_real.get(self.model_label.get(), NO_MODEL))
        self._model_changed()

    def _hover_model(self, event) -> None:
        real = self.model.get()
        self.tip.show("" if real == NO_MODEL or real.startswith(BASE) else real, event)

    def _base_tiles_changed(self) -> None:
        if self.model.get().startswith(BASE):
            self._model_changed()

    def _model_changed(self) -> None:
        """Show any stored answer, then load this model and answer the photo on screen."""
        self._fill_answer()
        self._draw()
        if self.model.get() == NO_MODEL:
            self.loading_key = None
            self._phase("idle", "Labels only")
            return
        self.load_model()

    def _wanted_key(self) -> str:
        choice = self.model.get()
        return f"{choice} tiles={self.base_tiles.get()}" if choice.startswith(BASE) else choice

    def load_model(self) -> None:
        choice, key = self.model.get(), self._wanted_key()
        if choice == NO_MODEL:
            return
        if self.loaded and self.loaded.key == key:
            self._phase("answer", f"Answering with {self.model_label.get()}")
            if self.photo:
                self.ask()
            return
        if self.loading_key == key:
            return
        self.loading_key = key
        run = self.runs.get(choice)
        tiles = self.base_tiles.get()
        model_id, tile_count = self._demand()
        ready, reason = room_for(model_id, tile_count)
        if not ready:
            self.loading_key = None
            self._phase("wait", reason)
            self._retry(self.load_model)
            return

        def work() -> Loaded | NotYet:
            self._free()
            ready_now, why = room_for(model_id, tile_count)
            if not ready_now:
                return NotYet(why)
            if run:
                config = from_dict(run["config"])
                size = (config.image.reference_width, config.image.reference_height)
                processor = load_processor(config.model.id, config.model.revision, config.image.tiles, size)
                model = load_for_scoring(config, run["folder"] / "model", precision())
                window = context_tokens(config.model.id, config.model.revision)
                return Loaded(key, processor, model, window, config.data.prompt)
            data = split()
            if not data:
                raise RuntimeError("output/split.json does not exist yet; its photo size sets the tile count")
            base_id = resolve_model(choice.removeprefix(BASE))
            revision = resolve_revision(base_id, "main")
            processor = load_processor(
                base_id, revision, None if tiles == "default" else int(tiles), reference_size(data)
            )
            model = load_base(base_id, revision, precision())
            return Loaded(key, processor, model, context_tokens(base_id, revision), PROMPT)

        def done(loaded: Loaded | NotYet) -> None:
            if isinstance(loaded, NotYet):
                if self.loading_key == key:
                    self.loading_key = None
                if self._wanted_key() == key:
                    self._phase("wait", loaded.reason)
                    self._retry(self.load_model)
                return
            self._cancel_retry()
            if self.loading_key == key:
                self.loading_key = None
            if self._wanted_key() != key:
                return
            self.loaded = loaded
            self._phase("ready", self.model_label.get())
            if self.photo:
                self.ask()

        self._phase("load", f"Loading {self.model_label.get()}")
        self.app.worker.submit(Job(f"load {key}", work, done, self._failed))

    def _free(self) -> None:
        """Drop the loaded model before loading another, so two never share the GPU. Worker thread only."""
        self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _demand(self) -> tuple[str, int | None]:
        """The Hugging Face id and tile count the chosen model will load."""
        choice = self.model.get()
        run = self.runs.get(choice)
        if run:
            return run["config"]["model"]["id"], run["config"]["image"]["tiles"]
        tiles = self.base_tiles.get()
        return resolve_model(choice.removeprefix(BASE)), None if tiles == "default" else int(tiles)

    def _retry(self, again) -> None:
        """Try ``again`` in a few seconds. A newer wait replaces an older one."""
        self._cancel_retry()
        self._gpu_wait = self.after(5000, again)

    def _cancel_retry(self) -> None:
        if self._gpu_wait is not None:
            self.after_cancel(self._gpu_wait)
            self._gpu_wait = None

    def _failed(self, error: BaseException) -> None:
        self._phase("fail", f"Failed: {type(error).__name__}: {error}")

    def ask(self) -> None:
        """Answer the photo on screen with the loaded model."""
        if not self.photo:
            return
        if not self.loaded or self.loaded.key != self._wanted_key():
            return
        loaded, photo = self.loaded, self.photo
        path = photo.path
        model_id, tile_count = self._demand()
        ready, reason = room_for(model_id, tile_count, already_loaded=True)
        if not ready:
            self._phase("wait", reason)
            self._retry(self.ask)
            return

        def work() -> str | NotYet:
            ready_now, why = room_for(model_id, tile_count, already_loaded=True)
            if not ready_now:
                return NotYet(why)
            decode = DecodeSettings(
                context_tokens=loaded.context_tokens, do_sample=False, num_beams=1, batch_size=1, precision=precision()
            )
            _rows, inputs = PromptCollator(loaded.processor, loaded.prompt)([[path, "", photo.width, photo.height]])
            sizes = [(photo.width, photo.height)]
            text, cut = answer_batch(loaded.processor, loaded.model, inputs, decode, sizes)[0]
            return text + (CUT if cut else "")

        def done(text: str | NotYet) -> None:
            if isinstance(text, NotYet):
                if self._wanted_key() == loaded.key:
                    self._phase("wait", text.reason)
                    self._retry(self.ask)
                return
            self._cancel_retry()
            if self.photo and self.photo.path == path and self._wanted_key() == loaded.key:
                self.answer = (text.removesuffix(CUT), f"asked now: {loaded.key}")
                self.raw.delete("1.0", "end")
                self.raw.insert("1.0", text)
                self._phase("ready", self.model_label.get())
                self._draw()

        self._phase("answer", f"Answering {Path(path).name}")
        self.app.worker.submit(Job(f"answer {Path(path).name}", work, done, self._failed))

    # ----- photos -----

    def _step(self, delta: int) -> None:
        self.index.set(max(0, self.index.get() + delta))
        self.show_split_photo()

    def _random(self) -> None:
        data = split()
        if data:
            self.index.set(random.randrange(len(data[self.split.get()])))
            self.show_split_photo()

    def show_split_photo(self) -> None:
        data = split()
        if not data:
            self._phase("fail", "No output/split.json yet")
            return
        rows = data[self.split.get()]
        self.index.set(min(max(0, self.index.get()), len(rows) - 1))
        self.show_photo(rows[self.index.get()][0])

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=IMAGES, filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*")]
        )
        if path:
            self.show_photo(path)

    def show_photo(self, path: str) -> None:
        """Load a photo and its label, then ask the model that is already loaded."""
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        height, width = pixels.shape[:2]
        place = split_place(path)
        label = label_for(path)
        gold = place[2] if place else (spline_answer(label, width, height, DEFAULT_JUDGE.rdp_px) if label else None)
        self.photo = Photo(path, pixels, width, height, place, label, gold)
        log.debug("photo %s place %s label %s", path, place and place[:2], bool(label))
        self._fill_answer()
        self._draw()
        if self.loaded and self.loaded.key == self._wanted_key():
            self.ask()
        elif self.model.get() != NO_MODEL:
            self._phase("load", "Loading model…")

    def _fill_answer(self) -> None:
        """The chosen run's stored answer for this photo, if scoring answered it."""
        self.answer = None
        self.raw.delete("1.0", "end")
        run = self.runs.get(self.model.get())
        if not self.photo or not run:
            return
        stored = stored_answers(run["folder"]).get(self.photo.path)
        if stored:
            batch = stored["header"].get("batch_size")
            self.answer = (stored["raw"], f"stored in {stored['file']} (batch {batch})")
            self.raw.insert("1.0", stored["raw"] + (CUT if stored["truncated"] else ""))

    # ----- drawing -----

    def _membership(self) -> list[tuple[str, str]]:
        """Lines for the notes box, each with a tag: "warn", "caution", or ""."""
        photo = self.photo
        lines = [(photo.path, "")]
        if photo.label:
            lines.append((f"label: {photo.label['_label_file']}", ""))
        else:
            lines.append(("No label for this photo", "caution"))
        if photo.place:
            lines.append((f"split.json: {photo.place[0]} #{photo.place[1]}", ""))
        elif photo.label:
            lines.append(("Not in split.json: no usable lane", ""))
        run = self.runs.get(self.model.get())
        if run and run["config"]["model"]["method"] == "base":
            lines.append(("Untrained: never saw a photo", ""))
        elif run:
            used = run_photos(run["folder"])
            if photo.path in used["train"]:
                lines.append(("TRAINED ON this photo", "warn"))
            elif photo.path in used["tune_cost"]:
                lines.append(("Tune-cost photo: picked the checkpoint", "caution"))
            elif photo.path in used["tune_loss"]:
                lines.append(("Tune-loss photo: not trained on", "caution"))
            elif photo.place and photo.place[0] == "train":
                lines.append(("Train split, past this run's %: not trained on", ""))
            elif photo.place:
                lines.append((f"Held out ({photo.place[0]})", ""))
        elif self.model.get().startswith(BASE):
            lines.append(("Untrained: never saw a photo", ""))
        if self.answer:
            lines.append((f"answer: {self.answer[1]}", ""))
        return lines

    def _draw(self) -> None:
        for axes in (self.left, self.right):
            axes.clear()
            axes.set_axis_off()
        photo = self.photo
        self.prompt_label.configure(text=f"prompt: {self._prompt()!r}" if self.model.get() != NO_MODEL else "")
        if not photo:
            self.canvas.draw_idle()
            self._paint_card()
            self._paint_bars()
            return
        width, height = photo.width, photo.height
        judge = self._judge()
        for axes in (self.left, self.right):
            axes.imshow(photo.pixels)
        gold = read_lanes(photo.gold, width, height) if photo.gold else None
        guess = read_lanes(self.answer[0], width, height) if self.answer else None
        band = None
        for side, axes in (("left", self.left), ("right", self.right)):
            on = {key for key, var in self.layers[side].items() if var.get()}
            if "zone" in on and gold:
                band = band if band is not None else self._band(gold, width, height, judge.radius_px)
                axes.imshow(band, zorder=2)
            if "markers" in on and photo.label:
                for lane in marker_segments(photo.label):
                    for x1, y1, x2, y2 in lane:
                        axes.plot([x1, x2], [y1, y2], color=MARKER, linewidth=1.5, zorder=4)
            if "labels" in on and gold:
                with_guess = "predicted" in on or "hits" in on
                self._lanes(axes, gold, GOLD, "--" if with_guess else "-", None if with_guess else "L")
            if "predicted" in on and guess:
                self._lanes(axes, guess, PREDICTED, "-", "P")
            if "hits" in on and guess and gold:
                self._overlap_points(axes, guess, gold, judge)
            axes.set_title(self._title(on, guess), fontsize=10)
            self._legend(axes, on, judge)
        self.left.set_xlim(0, width)
        self.left.set_ylim(height, 0)
        self.canvas.draw_idle()
        self._paint_card()
        self._paint_bars()

    def _phase(self, phase: str, text: str) -> None:
        """Show what the model is doing, as a moving bar above the photos."""
        self.phase_label.configure(text=text)
        spinning = phase in ("wait", "load", "answer")
        if spinning and not self._spinning:
            self.phase_bar.configure(mode="indeterminate")
            self.phase_bar.start(12)
            self._spinning = True
        elif not spinning and self._spinning:
            self.phase_bar.stop()
            self._spinning = False
        self._paint_card()

    def _banner(self) -> tuple[str, str]:
        if not self.photo:
            return "#e4e4e7", "Choose a photo"
        for text, tag in self._membership():
            if tag == "warn":
                return "#fecaca", text
            if tag == "caution":
                return "#fde68a", text
        if self.photo.place:
            return "#bbf7d0", f"{self.photo.place[0]} #{self.photo.place[1]}"
        return "#e4e4e7", "Not in the shared split"

    def _paint_card(self) -> None:
        """Tiles, rank, and how much of the split this run trained on."""
        canvas = self.card
        canvas.delete("all")
        run = self.runs.get(self.model.get())
        tiles = percent = 0
        rank = None
        if run:
            tiles = run["config"]["image"]["tiles"] or 0
            lora = run["config"]["model"]["lora"]
            if run["config"]["model"]["method"] != "base":
                rank = lora["r"] if lora else 0
            percent = run["config"]["data"]["train_percent"]
        elif self.model.get().startswith(BASE) and self.base_tiles.get() != "default":
            tiles = int(self.base_tiles.get())
        color, banner = self._banner()
        canvas.create_rectangle(0, 0, 400, 34, fill=color, outline="")
        canvas.create_text(8, 17, anchor="w", text=banner[:64], font=("TkDefaultFont", 9, "bold"))
        canvas.create_text(8, 54, anchor="w", text="Tiles", fill="#52525b")
        count = max(tiles, 1)
        for index in range(count):
            x = 52 + (index % 10) * 18
            y = 44 + (index // 10) * 18
            canvas.create_rectangle(x, y, x + 14, y + 14, fill="#93c5fd", outline="#1d4ed8")
        canvas.create_text(8, 108, anchor="w", text="Data", fill="#52525b")
        canvas.create_rectangle(52, 98, 280, 116, outline="#a1a1aa")
        if percent:
            canvas.create_rectangle(53, 99, 53 + 226 * percent / 100, 115, fill="#4ade80", outline="")
            canvas.create_text(288, 107, anchor="w", text=f"{percent}% of the split")
        else:
            canvas.create_text(288, 107, anchor="w", text="not trained")
        rank_text = "base weights" if rank is None else ("full model" if rank == 0 else f"LoRA rank {rank}")
        canvas.create_text(8, 146, anchor="w", text=rank_text, font=("TkDefaultFont", 12, "bold"))

    def _title(self, on: set[str], guess) -> str:
        if not {"predicted", "hits"} & on:
            return "Labels" if {"labels", "markers"} & on else "Photo"
        if not self.answer:
            return "Prediction (waiting)"
        return "Prediction" if guess else "Prediction (unreadable)"

    @staticmethod
    def _legend(axes, on: set[str], judge: JudgeSettings) -> None:
        dashed = "--" if {"predicted", "hits"} & on else "-"
        items = {
            "zone": Patch(color=GOLD, alpha=0.25, label=f"Within {judge.radius_px:g} px"),
            "markers": Line2D([], [], color=MARKER, label="Annotator marks"),
            "labels": Line2D([], [], color=GOLD, linestyle=dashed, label="Label"),
            "predicted": Line2D([], [], color=PREDICTED, label="Prediction"),
        }
        handles = [items[key] for key, _text in LAYERS if key in on and key in items]
        if "hits" in on:
            handles += [
                Line2D([], [], color=NEAR, marker="o", linestyle="", label="Hit"),
                Line2D([], [], color=FAR, marker="o", linestyle="", label="Miss"),
            ]
        if handles:
            axes.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    def _judge(self) -> JudgeSettings:
        """The chosen run's judge settings, or the defaults for a model with no run."""
        run = self.runs.get(self.model.get())
        return from_dict(run["config"]).judge if run else DEFAULT_JUDGE

    def _prompt(self) -> str:
        run = self.runs.get(self.model.get())
        return run["config"]["data"]["prompt"] if run else PROMPT

    def _paint_bars(self) -> None:
        """Cost against the cap, then precision and recall, all from measure()."""
        canvas = self.bars
        canvas.delete("all")
        if not self.photo or not self.photo.gold or not self.answer:
            canvas.create_text(8, 16, anchor="w", text="No answer yet", fill="#71717a")
            return
        judge = self._judge()
        one = measure(self.photo.gold, self.answer[0], self.photo.width, self.photo.height, judge)
        cost_fill = "#22c55e" if one.cost_px <= 10 else "#f59e0b" if one.cost_px <= 25 else "#ef4444"
        rows = [
            ("cost", one.cost_px / judge.tau_px, f"{one.cost_px:.1f} of {judge.tau_px:g} px", cost_fill),
            ("prec", one.prec, f"{100 * one.prec:.0f}% within {judge.radius_px:g} px", "#3b82f6"),
            ("recall", one.rec, f"{100 * one.rec:.0f}% of gold covered", "#3b82f6"),
        ]
        for index, (name, share, text, fill) in enumerate(rows):
            y = 6 + index * 26
            canvas.create_text(4, y + 9, anchor="w", text=name, font=("TkDefaultFont", 9))
            canvas.create_rectangle(52, y, 220, y + 18, fill="#f4f4f5", outline="")
            canvas.create_rectangle(52, y, 52 + 168 * min(share, 1.0), y + 18, fill=fill, outline="")
            canvas.create_text(228, y + 9, anchor="w", text=text)
        lanes = f"{one.lanes_pred} predicted lanes, {one.lanes_gold} gold" if one.readable else "answer unreadable"
        canvas.create_text(4, 6 + 3 * 26 + 9, anchor="w", text=lanes, fill="#52525b")

    @staticmethod
    def _band(gold, width: int, height: int, radius_px: float) -> np.ndarray:
        """An RGBA overlay: every pixel within about radius_px of a gold lane, tinted faintly."""
        band = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(band)
        for lane in gold:
            draw.line([tuple(point) for point in lane.tolist()], fill=(34, 197, 94, 60), width=round(2 * radius_px))
        return np.asarray(band)

    @staticmethod
    def _overlap_points(axes, guess, gold, judge: JudgeSettings) -> None:
        """Predicted sample points, green within radius_px of gold and red beyond it."""
        points = np.concatenate([densify(lane, judge.step_px) for lane in guess])
        others = np.concatenate([densify(lane, judge.step_px) for lane in gold])
        near = nearest(points, others) <= judge.radius_px
        axes.scatter(*points[near].T, s=3, color=NEAR, zorder=6)
        axes.scatter(*points[~near].T, s=3, color=FAR, zorder=6)

    @staticmethod
    def _lanes(axes, lanes, color, style, prefix) -> None:
        for number, lane in enumerate(lanes):
            axes.plot(lane[:, 0], lane[:, 1], style, color=color, linewidth=2, zorder=5)
            if prefix:
                x, y = lane[-1]
                axes.annotate(f"{prefix}{number}", (x, y), color=color, fontsize=8, weight="bold")
