"""The GUI window: a notebook with the Runs, Models, Live, GPU, Audit, and Compare pages."""

import logging
import tkinter as tk
from tkinter import ttk

from spline.gui.audit_page import AuditPage
from spline.gui.compare_page import ComparePage
from spline.gui.gpu_page import GpuPage
from spline.gui.live_page import LivePage
from spline.gui.models_page import ModelsPage
from spline.gui.runs_page import RunsPage
from spline.gui.state import Worker

log = logging.getLogger("spline.gui")

POLL_MS = 200


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Spline runs")
        root.geometry("1600x950")
        # Tk swallows callback errors by printing them; send them to the logs instead.
        root.report_callback_exception = lambda kind, value, trace: log.error(
            "GUI callback failed", exc_info=(kind, value, trace)
        )
        self.worker = Worker()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.pages = {
            "runs": RunsPage(self.notebook, self),
            "models": ModelsPage(self.notebook, self),
            "live": LivePage(self.notebook, self),
            "gpu": GpuPage(self.notebook, self),
            "audit": AuditPage(self.notebook, self),
            "compare": ComparePage(self.notebook, self),
        }
        for key, title in (
            ("runs", "Runs"),
            ("models", "Models"),
            ("live", "Live training"),
            ("gpu", "GPU"),
            ("audit", "Audit"),
            ("compare", "Compare"),
        ):
            self.notebook.add(self.pages[key], text=title)
        self._poll()

    def show(self, page: str, run: str | None = None) -> None:
        """Switch to ``page``, optionally with ``run`` selected on it."""
        if run is not None and hasattr(self.pages[page], "select"):
            self.pages[page].select(run)
        self.notebook.select(self.pages[page])

    def _poll(self) -> None:
        self.worker.poll()
        self.root.after(POLL_MS, self._poll)
