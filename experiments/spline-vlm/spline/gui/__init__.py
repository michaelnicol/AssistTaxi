"""The Tk GUI started by ``spline_gui.py``: one window, one notebook tab per page.

* :mod:`spline.gui.runs_page`: every run folder, its state and headline numbers.
* :mod:`spline.gui.live_page`: live progress, loss and learning-rate curves, ETAs.
* :mod:`spline.gui.audit_page`: any photo against any run or the base model.
* :mod:`spline.gui.compare_page`: metrics and tune-loss curves across runs.

Pages only read run folders; they never write into them. Slow work (loading a
model, answering a photo, reading split.json) runs on worker threads, and results
come back to the Tk thread through :class:`spline.gui.state.Worker`.
"""
