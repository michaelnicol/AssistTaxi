"""Clone-path check. Caller: labels and log. Owns the write ban. Must not read or write."""

from pathlib import Path

CLONE_ROOTS = (
    Path("/home/michaelnicol/github/AssistTaxi/AssistTaxi"),
    Path("/home/michaelnicol/github/AssistTaxi/AssistTaxiDataset"),
)


def assert_output_allowed(path: Path) -> None:
    resolved = Path(path).expanduser().resolve()
    for root in CLONE_ROOTS:
        banned = root.resolve()
        if resolved == banned or banned in resolved.parents:
            raise ValueError(f"refusing to write inside {banned}")
