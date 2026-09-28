"""Process log in the SkyscrapeBackend line format. Caller: GUI. Owns labeler.log. Must not write the dataset or secrets."""

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from labeler.io.guard import assert_output_allowed

LEVELS = ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL")
DEFAULT_LOG = Path("/home/michaelnicol/github/AssistTaxi/labeler_out/log/labeler.log")


def _rank(level: str) -> int:
    return LEVELS.index(level)


def _one_line(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


class Log:
    def __init__(self, path: Path, floor: str = "INFO", max_file_bytes: int = 8 * 1024 * 1024):
        self.path = Path(path)
        self.component = self.path.stem or "log"
        self.floor = floor
        self.max_file_bytes = max_file_bytes
        self._lock = threading.Lock()
        self._reported = False

    def write(self, level: str, code: str, fields: str = "") -> None:
        """Caller: GUI. Owns one append. Must not throw."""
        with self._lock:
            if _rank(level) < _rank(self.floor):
                return
            line = self._line(level, code, fields)
            try:
                assert_output_allowed(self.path)
                self._append(line)
                self._maybe_rotate()
            except (OSError, ValueError) as exc:
                self._unavailable(exc)
                return
            if _rank(level) >= _rank("ERROR"):
                sys.stderr.write(line + "\n")

    def _line(self, level: str, code: str, fields: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        line = f"{stamp} {level} {_one_line(code)} {self.component}"
        clean = _one_line(fields).strip()
        if clean:
            line = f"{line} {clean}"
        return line

    def _append(self, line: str) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        os.chmod(parent, 0o700)
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _maybe_rotate(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size < self.max_file_bytes:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = self.path.with_name(f"{self.component}-{stamp}.log")
        n = 2
        while dest.exists():
            dest = self.path.with_name(f"{self.component}-{stamp}-{n}.log")
            n += 1
        self.path.rename(dest)

    def _unavailable(self, exc: Exception) -> None:
        if self._reported:
            return
        self._reported = True
        sys.stderr.write(self._line("CRITICAL", "log.unavailable", f"detail={exc}") + "\n")
