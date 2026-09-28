import tempfile
import unittest
from pathlib import Path

from labeler.io.log import Log


class LogTest(unittest.TestCase):
    def test_grep_line_and_rotate(self):
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "labeler.log"
            log = Log(active)
            log.write("DEBUG", "frame.unreadable", "path=skip")
            log.write("ERROR", "frame.unreadable", "path=/tmp/a.jpg detail=not\na jpeg")
            body = active.read_text()
            self.assertNotIn("DEBUG", body)
            self.assertIn(" ERROR frame.unreadable labeler ", body)
            self.assertIn("detail=not a jpeg", body)
            self.assertNotIn("\n a jpeg", body)

            rotate_dir = Path(tmp) / "rotate"
            rotating = Log(rotate_dir / "labeler.log", max_file_bytes=64)
            rotating.write("ERROR", "frame.unreadable", "detail=" + ("a" * 80))
            rotating.write("INFO", "boot.ok", "frames=1")
            rotated = list(rotate_dir.glob("labeler-*.log"))
            self.assertEqual(len(rotated), 1)
            self.assertIn("frame.unreadable", rotated[0].read_text())
            self.assertIn("boot.ok", (rotate_dir / "labeler.log").read_text())


if __name__ == "__main__":
    unittest.main()
