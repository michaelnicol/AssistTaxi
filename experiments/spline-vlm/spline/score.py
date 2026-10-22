"""Ask the trained model about the test photos and measure the lanes in pixels."""

import json

import numpy as np
from tqdm import tqdm

from spline.data import load_dataset
from spline.lanes import lane_errors, read_lanes
from spline.model import answer_photos, load_for_scoring
from spline.runs import run_dir


def score(args) -> None:
    folder = run_dir(args.output, args.run)
    saved = folder / "model"
    run = json.loads((saved / "run.json").read_text())
    percent = int(run.get("data_percent", args.data_percent))
    rows = load_dataset(args.output, percent)["test"]
    processor, model = load_for_scoring(saved, run)
    dest = folder / "test.jsonl"
    done = {json.loads(line)["image"] for line in dest.read_text().splitlines()} if dest.exists() else set()
    pending = [row for row in rows if row[0] not in done]
    size = max(1, args.score_batch)
    with dest.open("a", encoding="utf-8") as handle, tqdm(total=len(pending), desc="Answering test photos") as bar:
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            raws = answer_photos(processor, model, [path for path, _gold in chunk])
            for (path, gold), raw in zip(chunk, raws):
                handle.write(json.dumps({"image": path, "gold": gold, "raw": raw}) + "\n")
            handle.flush()
            bar.update(len(chunk))

    wanted = {row[0] for row in rows}
    results = [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
    results = [row for row in results if row["image"] in wanted]
    errors, readable, right_count = [], 0, 0
    for row in results:
        gold = read_lanes(row["gold"])
        guess = read_lanes(row["raw"])
        readable += guess is not None
        right_count += guess is not None and len(guess) == len(gold)
        errors += lane_errors(gold, guess or [])
    errors = np.array(errors)
    print(f"\n{len(results)} test photos, run {args.run}")
    print(f"  readable answers       {100 * readable / len(results):.1f}%")
    print(f"  right number of lanes  {100 * right_count / len(results):.1f}%")
    print(f"  lane error (px)        median {np.median(errors):.1f}, mean {errors.mean():.1f}")
    print(f"  lanes within 10 px     {100 * (errors <= 10).mean():.1f}%")
    print(f"  lanes within 25 px     {100 * (errors <= 25).mean():.1f}%")
