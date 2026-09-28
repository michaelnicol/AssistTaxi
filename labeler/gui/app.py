"""Side-by-side window. Caller: __main__. Owns widgets. Must not open frames or write labels."""

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from labeler.compose.session import Session
from labeler.io.log import DEFAULT_LOG, Log
from labeler.methods.alina import AlinaParams
from labeler.methods.cbem import CbemParams
from labeler.methods.cdlem import CdlemParams

ROOT = Path("/home/michaelnicol/github/AssistTaxi")
DEFAULT_FRAMES = ROOT / "AssistTaxi" / "main" / "raw_data" / "vid_5"
DEFAULT_LABELS = ROOT / "AssistTaxi" / "main" / "annotations" / "vid_5"
DEFAULT_OUT = ROOT / "labeler_out"
PANE_W, PANE_H = 640, 360
STAGE_W, STAGE_H = 1280, 150

HINTS = {
    "existing": "Existing: right pane is the repo x y file, rasterized. Nothing is written into the clone.",
    "alina": "ALINA: click bottom-left, top-left, top-right, bottom-right once. Play reuses that trapezoid.",
    "cdlem": "CDLEM: click the contour on this frame. Play reuses that polygon.",
    "cbem": "CBEM: click a ground-truth contour. Save writes under labeler_out/cbem.",
}


def fit_photo(image: np.ndarray, max_w: int, max_h: int):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(max_w / width, max_h / height)
    disp_w = max(1, int(width * scale))
    disp_h = max(1, int(height * scale))
    disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(disp)), disp_w, disp_h, width, height


def paint_left(image: np.ndarray, mask: np.ndarray, roi: list[tuple[int, int]], closed: bool) -> np.ndarray:
    painted = image.copy()
    if len(roi) >= 2:
        pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(painted, [pts], closed and len(roi) >= 3, (0, 255, 0), 2)
    painted[mask > 0] = (0, 0, 255)
    return painted


class LabelerApp:
    def __init__(
        self,
        root: tk.Tk,
        frames: Path | None = None,
        labels: Path | None = None,
        out: Path | None = None,
        log_path: Path | None = None,
    ):
        self.root = root
        self.frames: list[Path] = []
        self.index = 0
        self.roi: list[tuple[int, int]] = []
        self.coords = np.zeros((0, 2), dtype=np.int32)
        self.playing = False
        self._slider_quiet = False
        self._debounce = None
        self.full_w = 1
        self.full_h = 1
        self.disp_w = 1
        self.disp_h = 1
        self._left_photo = None
        self._right_photo = None
        self._stage_photo = None
        self._stages: dict = {}

        self.frames_var = tk.StringVar(value=str(frames or DEFAULT_FRAMES))
        self.labels_var = tk.StringVar(value=str(labels or DEFAULT_LABELS))
        self.out_var = tk.StringVar(value=str(out or DEFAULT_OUT))
        self.method = tk.StringVar(value="existing")
        self.stage_var = tk.StringVar()
        self.status = tk.StringVar()
        self.hint = tk.StringVar(value=HINTS["existing"])

        self.peak = tk.IntVar(value=50)
        self.radius = tk.IntVar(value=15)
        self.min_white = tk.IntVar(value=200)
        self.ignore_left = tk.IntVar(value=300)
        self.h_lo = tk.IntVar(value=0)
        self.s_lo = tk.IntVar(value=70)
        self.v_lo = tk.IntVar(value=170)
        self.h_hi = tk.IntVar(value=255)
        self.s_hi = tk.IntVar(value=255)
        self.v_hi = tk.IntVar(value=255)
        self.sigma = tk.DoubleVar(value=0.33)
        self.blur = tk.IntVar(value=5)
        self.hough_t = tk.IntVar(value=80)
        self.min_len = tk.IntVar(value=30)
        self.max_gap = tk.IntVar(value=10)
        self.epsilon = tk.DoubleVar(value=5.0)

        self.log = Log(log_path or DEFAULT_LOG)
        self._build()
        self.reload()
        self.log.write("INFO", "boot.ok", f"frames={len(self.frames)}")

    def _session(self) -> Session:
        return Session(Path(self.frames_var.get()), Path(self.labels_var.get()), Path(self.out_var.get()))

    def _build(self) -> None:
        paths = ttk.Frame(self.root)
        paths.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._path_row(paths, 0, "Frames", self.frames_var)
        self._path_row(paths, 1, "Labels", self.labels_var)
        self._path_row(paths, 2, "Output", self.out_var)
        ttk.Button(paths, text="Load", command=self.reload).grid(row=0, column=2, rowspan=3, padx=6)

        bar = ttk.Frame(self.root)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        for value, label in (("existing", "Existing"), ("alina", "ALINA"), ("cdlem", "CDLEM"), ("cbem", "CBEM")):
            ttk.Radiobutton(bar, text=label, value=value, variable=self.method, command=self._method_changed).pack(side="left")
        ttk.Button(bar, text="Paper 150", command=self._preset_paper).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Code 50", command=self._preset_code).pack(side="left")
        ttk.Button(bar, text="Clear ROI", command=self.clear_roi).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Save frame", command=self.save_current).pack(side="left")
        ttk.Label(bar, textvariable=self.hint).pack(side="left", padx=8)

        self.left = ttk.Label(self.root)
        self.right = ttk.Label(self.root)
        self.left.grid(row=2, column=0, padx=4, pady=4)
        self.right.grid(row=2, column=1, padx=4, pady=4)
        self.left.bind("<Button-1>", self.on_click)

        self.stage_row = ttk.Frame(self.root)
        self.stage_row.grid(row=3, column=0, columnspan=2, sticky="ew")
        stage_row = self.stage_row
        ttk.Label(stage_row, text="Stage").pack(side="left")
        self.stage_box = ttk.Combobox(stage_row, textvariable=self.stage_var, state="readonly", width=16)
        self.stage_box.pack(side="left")
        self.stage_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw_stage())
        self.stage = ttk.Label(stage_row)
        self.stage.pack(side="left", padx=6)

        controls = ttk.Frame(self.root)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(controls, text="Step", command=lambda: self.step(1)).pack(side="left")
        self.slider = ttk.Scale(controls, from_=0, to=0, orient="horizontal", command=self.on_slider, length=520)
        self.slider.pack(side="left", padx=8)
        ttk.Label(controls, textvariable=self.status).pack(side="left")

        params = ttk.Frame(self.root)
        params.grid(row=5, column=0, columnspan=2, sticky="ew")
        alina = ttk.LabelFrame(params, text="ALINA")
        alina.grid(row=0, column=0, sticky="nw", padx=4)
        for label, var, start, end in (
            ("Peak", self.peak, 0, 400),
            ("Radius", self.radius, 1, 40),
            ("Min white", self.min_white, 0, 800),
            ("Ignore left", self.ignore_left, 0, 600),
            ("H lo", self.h_lo, 0, 255),
            ("S lo", self.s_lo, 0, 255),
            ("V lo", self.v_lo, 0, 255),
            ("H hi", self.h_hi, 0, 255),
            ("S hi", self.s_hi, 0, 255),
            ("V hi", self.v_hi, 0, 255),
        ):
            self._scale(alina, label, var, start, end)
        classic = ttk.LabelFrame(params, text="CDLEM / CBEM")
        classic.grid(row=0, column=1, sticky="nw", padx=4)
        self._scale(classic, "Sigma", self.sigma, 0.05, 1.0, resolution=0.01)
        self._scale(classic, "Blur", self.blur, 1, 15)
        self._scale(classic, "Hough T", self.hough_t, 1, 200)
        self._scale(classic, "Min len", self.min_len, 1, 200)
        self._scale(classic, "Max gap", self.max_gap, 0, 80)
        self._scale(classic, "RDP eps", self.epsilon, 0.5, 20, resolution=0.5)

        self.root.bind("<Left>", lambda _event: self.step(-1))
        self.root.bind("<Right>", lambda _event: self.step(1))
        self.root.bind("<space>", self._space)

    def _path_row(self, parent, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, width=8).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=88).grid(row=row, column=1, sticky="ew")

    def _scale(self, parent, label: str, variable, start, end, resolution=1) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label, width=10).pack(side="left")
        tk.Scale(
            row,
            from_=start,
            to=end,
            resolution=resolution,
            variable=variable,
            orient=tk.HORIZONTAL,
            length=160,
            showvalue=False,
            command=self._params_moved,
        ).pack(side="left")
        ttk.Label(row, textvariable=variable, width=6).pack(side="left")

    def _params_moved(self, _value: str) -> None:
        if self._debounce is not None:
            self.root.after_cancel(self._debounce)
        self._debounce = self.root.after(250, self.show)

    def _method_changed(self) -> None:
        self.hint.set(HINTS[self.method.get()])
        self.roi.clear()
        self.show()

    def _preset_paper(self) -> None:
        self.peak.set(150)
        self.sigma.set(0.33)
        self.show()

    def _preset_code(self) -> None:
        self.peak.set(50)
        self.radius.set(15)
        self.min_white.set(200)
        self.ignore_left.set(300)
        self.h_lo.set(0)
        self.s_lo.set(70)
        self.v_lo.set(170)
        self.h_hi.set(255)
        self.s_hi.set(255)
        self.v_hi.set(255)
        self.show()

    def _space(self, _event) -> str:
        self.toggle_play()
        return "break"

    def alina_params(self) -> AlinaParams:
        return AlinaParams(
            peak_pixel_threshold=self.peak.get(),
            min_white_pixels=self.min_white.get(),
            circular_threshold=max(1, self.radius.get()),
            yellow_lower=(self.h_lo.get(), self.s_lo.get(), self.v_lo.get()),
            yellow_upper=(self.h_hi.get(), self.s_hi.get(), self.v_hi.get()),
            mask_ignore_left_columns=self.ignore_left.get(),
        )

    def cdlem_params(self) -> CdlemParams:
        return CdlemParams(
            sigma=float(self.sigma.get()),
            blur_ksize=self.blur.get(),
            hough_threshold=self.hough_t.get(),
            min_line_length=self.min_len.get(),
            max_line_gap=self.max_gap.get(),
            epsilon=float(self.epsilon.get()),
        )

    def cbem_params(self) -> CbemParams:
        return CbemParams(sigma=float(self.sigma.get()), blur_ksize=self.blur.get())

    def reload(self) -> None:
        self.playing = False
        self.play_btn.configure(text="Play")
        self.frames = self._session().list_frames()
        self.index = 0
        self.roi.clear()
        self.slider.configure(to=max(len(self.frames) - 1, 0))
        self._slider_quiet = True
        self.slider.set(0)
        self._slider_quiet = False
        self.show()

    def clear_roi(self) -> None:
        self.roi.clear()
        self.show()

    def on_slider(self, value: str) -> None:
        if self._slider_quiet:
            return
        self.playing = False
        self.play_btn.configure(text="Play")
        self.index = int(float(value))
        self.show()

    def step(self, delta: int) -> None:
        if not self.frames:
            return
        self.playing = False
        self.play_btn.configure(text="Play")
        self.index = min(max(self.index + delta, 0), len(self.frames) - 1)
        self._slider_quiet = True
        self.slider.set(self.index)
        self._slider_quiet = False
        self.show()

    def toggle_play(self) -> None:
        if not self.frames:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_btn.configure(text="Play")
            return
        self.index += 1
        self._slider_quiet = True
        self.slider.set(self.index)
        self._slider_quiet = False
        self.show()
        self.root.after(20, self._tick)

    def on_click(self, event) -> None:
        if self.method.get() == "existing" or not self.frames:
            return
        if self.method.get() == "alina" and len(self.roi) >= 4:
            return
        self.roi.append((int(event.x * self.full_w / self.disp_w), int(event.y * self.full_h / self.disp_h)))
        self.show()

    def show(self) -> None:
        self._debounce = None
        if not self.frames:
            self.status.set("no frames")
            return
        path = self.frames[self.index]
        started = time.perf_counter()
        try:
            result = self._session().show(
                path,
                self.method.get(),
                self.roi,
                self.alina_params(),
                self.cdlem_params(),
                self.cbem_params(),
            )
        except OSError as exc:
            self._fail("frame.unreadable", f"path={path} detail={exc}")
            return
        except Exception as exc:
            self._fail("detect.failed", f"method={self.method.get()} frame={path.name} detail={type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.coords = result.coords
        closed = self.method.get() == "alina" or len(self.roi) >= 3
        left = paint_left(result.image, result.mask, self.roi, closed)
        self._left_photo, self.disp_w, self.disp_h, self.full_w, self.full_h = fit_photo(left, PANE_W, PANE_H)
        self.left.configure(image=self._left_photo)
        self._right_photo, _, _, _, _ = fit_photo(result.mask, PANE_W, PANE_H)
        self.right.configure(image=self._right_photo)
        self._set_stages(result.stages)
        recall = "n/a" if result.recall is None else f"{result.recall:.3f}"
        self.status.set(
            f"{path.name}  {self.index + 1}/{len(self.frames)}  {elapsed_ms:.0f} ms  {len(result.coords)} px  {result.note}  recall {recall}"
        )

    def _set_stages(self, stages: dict) -> None:
        names = list(stages)
        self.stage_box.configure(values=names)
        self._stages = stages
        if not names:
            self.stage_var.set("")
            self.stage.configure(image="")
            self._stage_photo = None
            self.stage_row.grid_remove()
            return
        self.stage_row.grid(row=3, column=0, columnspan=2, sticky="ew")
        if self.stage_var.get() not in stages:
            self.stage_var.set(names[0])
        self._redraw_stage()

    def _redraw_stage(self) -> None:
        image = self._stages.get(self.stage_var.get())
        if image is None:
            return
        self._stage_photo, _, _, _, _ = fit_photo(image, STAGE_W, STAGE_H)
        self.stage.configure(image=self._stage_photo)

    def save_current(self) -> None:
        if not self.frames:
            return
        try:
            path = self._session().save(self.frames[self.index].stem, self.coords, self.method.get())
        except ValueError as exc:
            self.log.write("ERROR", "label.refused", f"path={self.out_var.get()} detail={exc}")
            self.status.set(f"ERROR label.refused — grep {self.log.path}")
            return
        self.status.set(f"wrote {path}")

    def _fail(self, code: str, fields: str) -> None:
        self.log.write("ERROR", code, fields)
        self.status.set(f"ERROR {code} — grep {self.log.path}")


def main() -> None:
    root = tk.Tk()
    root.title("AssistTaxi labeler")
    LabelerApp(root)
    root.mainloop()
