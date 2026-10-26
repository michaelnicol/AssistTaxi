#!/usr/bin/env python3
"""Watch, audit, and compare spline runs.

  train-venv/bin/python spline_gui.py                 # opens on the Runs page
  train-venv/bin/python spline_gui.py audit           # or: runs, models, live, compare

Logs go to output/logs/gui.log and gui.errors.log.
"""

import sys
import tkinter as tk

from spline import logs
from spline.gui.app import App
from spline.runs import OUTPUT


def main(argv: list[str]) -> int:
    logs.setup("gui", OUTPUT / "logs")
    root = tk.Tk()
    app = App(root)
    if argv and argv[0] in app.pages:
        app.show(argv[0])
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
