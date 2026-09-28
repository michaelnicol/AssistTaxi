import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from labeler.compose.session import Session
from labeler.io.labels import write_xy
from labeler.methods.alina import AlinaParams
from labeler.methods.cbem import CbemParams
from labeler.methods.cdlem import CdlemParams


class SessionTest(unittest.TestCase):
    def test_existing_show_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            labels = root / "labels"
            out = root / "out"
            frames.mkdir()
            labels.mkdir()
            cv2.imwrite(str(frames / "00001.jpg"), np.zeros((40, 50, 3), dtype=np.uint8))
            write_xy(labels / "00001.txt", np.array([[3, 4]], dtype=np.int32))
            session = Session(frames, labels, out)
            result = session.show(
                session.list_frames()[0],
                "existing",
                [],
                AlinaParams(),
                CdlemParams(),
                CbemParams(),
            )
            self.assertEqual(result.coords.tolist(), [[3, 4]])
            self.assertEqual(result.recall, 1.0)
            saved = session.save("00001", result.coords, "existing")
            self.assertEqual(saved, out / "00001.txt")
            self.assertTrue(saved.is_file())


if __name__ == "__main__":
    unittest.main()
