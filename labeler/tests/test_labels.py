import tempfile
import unittest
from pathlib import Path

import numpy as np

from labeler.io.guard import assert_output_allowed
from labeler.io.labels import format_xy, parse_xy, read_xy, write_xy

REPO_TXT = Path("/home/michaelnicol/github/AssistTaxi/AssistTaxi/main/annotations/vid_5/00001.txt")


class LabelIoTest(unittest.TestCase):
    def test_round_trip(self):
        coords = np.array([[898, 694], [899, 694]], dtype=np.int32)
        self.assertEqual(parse_xy(format_xy(coords)).tolist(), coords.tolist())

    def test_repo_sample_is_xy(self):
        coords = read_xy(REPO_TXT)
        self.assertEqual(coords[0].tolist(), [898, 694])
        self.assertEqual(coords.shape[1], 2)

    def test_refuses_clone_write(self):
        target = Path("/home/michaelnicol/github/AssistTaxi/AssistTaxi/main/annotations/vid_5/not-written.txt")
        with self.assertRaises(ValueError):
            write_xy(target, np.array([[1, 2]], dtype=np.int32))
        self.assertFalse(target.exists())

    def test_writes_outside_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "00001.txt"
            assert_output_allowed(target)
            write_xy(target, np.array([[4, 5]], dtype=np.int32))
            self.assertEqual(read_xy(target).tolist(), [[4, 5]])


if __name__ == "__main__":
    unittest.main()
