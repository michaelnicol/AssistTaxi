import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk

from labeler.gui.app import LabelerApp
from labeler.io.labels import write_xy


class GuiTest(unittest.TestCase):
    def test_panes_show_existing_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            frames = root_dir / "frames"
            labels = root_dir / "labels"
            out = root_dir / "out"
            frames.mkdir()
            labels.mkdir()
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.imwrite(str(frames / "00001.jpg"), image)
            write_xy(labels / "00001.txt", np.array([[10, 12], [11, 12]], dtype=np.int32))
            root = tk.Tk()
            root.withdraw()
            app = LabelerApp(root, frames, labels, out, log_path=root_dir / "log" / "labeler.log")
            self.assertGreater(app._left_photo.width(), 0)
            self.assertGreater(app._right_photo.width(), 0)
            self.assertEqual(len(app.coords), 2)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
