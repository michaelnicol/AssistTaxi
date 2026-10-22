#!/usr/bin/env python3
"""Train a small vision-language model to trace lane lines, then score it.

  train-venv/bin/python spline_train.py train --lora --data-percent 10
  train-venv/bin/python spline_train.py score

--data-percent 10 keeps the first 10% of the saved train, tune, and test lists
and writes output/dataset-10.json. Later runs and other models reuse that file.
--batch-size is how many photos the graphics card handles at once.
SPLINE_TILES=16 (or --tiles 16) cuts each photo into more tiles. Each mix of
model, LoRA, data percent, and tiles gets its own output/runs/<name>/ folder.
"""

import argparse
import os
from pathlib import Path

from spline.fit import train
from spline.model import current_tiles, load_processor, resolve_model
from spline.runs import run_name
from spline.score import score

OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "score"))
    parser.add_argument("--model", default=os.environ.get("SPLINE_MODEL", "smolvlm2-2.2b"))
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--tiles",
        type=int,
        default=int(os.environ.get("SPLINE_TILES") or 0) or None,
        help="tiles per photo (SmolVLM: 1, 5, 7, 13, 16, 25, 29, 41); default is the model's own",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data-percent", type=int, default=100, help="percent of each split; saved and reused")
    parser.add_argument("--batch-size", type=int, default=2, help="photos per graphics-card step")
    parser.add_argument(
        "--grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="trade speed for memory; turn on for full fine-tuning or bigger models",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None, help="default 1e-4 with --lora, 1e-5 without")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=50, help="updates between restart checkpoints")
    parser.add_argument("--workers", type=int, default=4, help="CPU processes that prepare training batches")
    parser.add_argument("--score-batch", type=int, default=16, help="test photos answered at once")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.lora else 1e-5
    args.model = resolve_model(args.model)
    processor = load_processor(args.model, args.tiles)
    args.tiles = current_tiles(processor)
    args.run = run_name(args.model, args.lora, args.data_percent, args.tiles)
    if args.command == "train":
        train(args, processor)
    else:
        score(args)


if __name__ == "__main__":
    main()
