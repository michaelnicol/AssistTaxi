"""Answer the first N tune or test photos with a run's weights and measure the lanes in pixels.

Answers go to ``score-<split>.jsonl``: a header line holding every setting that
shapes them (prompt, judge, decode), then one ``{image, width, height, gold, raw,
truncated}`` line per photo. A stopped
score resumes after the last whole batch, because an answer depends on which
photos share its batch. The summary goes to ``score-<split>.json``
(:class:`spline.config.ScoreResult`). Live progress is in ``live/<split>.json``.
"""

import json
import logging
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
import torch

from spline.config import SCHEMA_VERSION, CostStats, DecodeSettings, JudgeSettings, RunConfig, ScoreResult
from spline.data import first, read_split
from spline.lanes import overlap, read_lanes
from spline.logs import Progress
from spline.model import PromptCollator, answer_batch, load_for_scoring, load_processor
from spline.runs import dump, environment, file_sha256, now, text_sha256, weights_identity, write_once

log = logging.getLogger("spline.score")


@dataclass
class PhotoScore:
    """How one answer compares with its gold answer."""

    # Every line of the answer parsed as a lane inside the photo.
    readable: bool
    # Lanes in the answer (0 if unreadable) and in the gold text.
    lanes_pred: int
    lanes_gold: int
    # Cost L in px, and the shares of predicted and gold points within radius_px (see overlap()).
    cost_px: float
    prec: float
    rec: float


def measure(gold_text: str, raw_text: str, width: int, height: int, judge: JudgeSettings) -> PhotoScore:
    """The per-photo numbers that make up a score. Used by scoring, checkpoint choice, and the GUI audit page."""
    gold = read_lanes(gold_text, width, height) or []
    guess = read_lanes(raw_text, width, height) or []
    cost_px, prec, rec = overlap(guess, gold, judge)
    return PhotoScore(
        readable=bool(guess), lanes_pred=len(guess), lanes_gold=len(gold), cost_px=cost_px, prec=prec, rec=rec
    )


def f_score(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def log_setup(config: RunConfig) -> None:
    """The first INFO lines of every train and score process: the exact prompt and judge."""
    log.info("prompt: %s", json.dumps(config.data.prompt))
    log.info("judge: %s", json.dumps(asdict(config.judge)))


def stats(costs: list[float]) -> CostStats:
    """Median, mean, and share within 10 and 25 px, over the per-photo costs of a split."""
    if not costs:
        return CostStats(photos=0, median_px=None, mean_px=None, within_10_pct=None, within_25_pct=None)
    values = np.array(costs)
    return CostStats(
        photos=len(values),
        median_px=round(float(np.median(values)), 2),
        mean_px=round(float(values.mean()), 2),
        within_10_pct=round(100 * float((values <= 10).mean()), 2),
        within_25_pct=round(100 * float((values <= 25).mean()), 2),
    )


def _kept_answers(dest: Path, header: dict, rows: list, batch_size: int) -> int:
    """Photos already answered in whole batches. A partly written batch is cut off.

    Answers depend on which photos share a batch, so a resumed run must use the
    same batch boundaries as one that was never stopped.
    """
    if not dest.exists():
        dest.write_text(json.dumps(header) + "\n")
        return 0
    lines = dest.read_text().splitlines()
    if not lines or json.loads(lines[0]) != header:
        raise RuntimeError(f"{dest} was made with other score settings; delete it to score again")
    answered = []
    for line in lines[1:]:
        try:
            answered.append(json.loads(line)["image"])
        except (json.JSONDecodeError, KeyError):
            break
    kept = len(answered) // batch_size * batch_size
    if len(answered) == len(rows):
        kept = len(rows)
    if answered[:kept] != [row[0] for row in rows[:kept]]:
        raise RuntimeError(f"{dest} answers photos out of order; delete it to score again")
    if kept != len(lines) - 1:
        log.info("%s: dropping %d answers from a partly written batch", dest.name, len(lines) - 1 - kept)
        dest.write_text("\n".join(lines[: kept + 1]) + "\n")
    return kept


def _earlier_fallback(dest: Path, header: dict, decode: DecodeSettings) -> DecodeSettings:
    """The smaller batch an earlier out-of-memory fallback chose, so a resume keeps it."""
    if not dest.exists():
        return decode
    lines = dest.read_text().splitlines()
    if not lines:
        return decode
    saved = json.loads(lines[0])
    smaller = saved.get("batch_size", decode.batch_size)
    if smaller < decode.batch_size and {**saved, "batch_size": decode.batch_size} == header:
        log.warning("resuming %s at batch %d, set by an earlier out-of-memory fallback", dest.name, smaller)
        return replace(decode, batch_size=smaller)
    return decode


def _answer(load, rows, decode, header, dest, progress: Progress, prompt: str) -> None:
    """Append answers to dest in fixed batches, resuming after the last whole batch."""
    kept = _kept_answers(dest, header, rows, decode.batch_size)
    progress.update(kept, force=True, batch_size=decode.batch_size)
    pending = rows[kept:]
    if not pending:
        return
    log.info("answering %d %s photos at batch %d (%d already done)", len(pending), dest.stem, decode.batch_size, kept)
    processor, model = load()
    prepare = PromptCollator(processor, prompt)
    done = kept
    with dest.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), decode.batch_size):
            chunk, inputs = prepare(pending[start : start + decode.batch_size])
            answers = answer_batch(processor, model, inputs, decode, [(row[2], row[3]) for row in chunk])
            handle.write(
                "".join(
                    json.dumps(
                        {"image": path, "width": width, "height": height, "gold": gold, "raw": raw, "truncated": cut}
                    )
                    + "\n"
                    for (path, gold, width, height), (raw, cut) in zip(chunk, answers, strict=True)
                )
            )
            handle.flush()
            done += len(chunk)
            progress.update(done)
            log.debug("answered %d of %d", done, len(rows))


def answer_all(load, rows, decode, header_for, dest, progress, prompt: str, fallbacks: list, name: str):
    """Answer every row into dest, resumable. Returns the decode settings that answered them.

    A resume keeps a smaller batch that an earlier out-of-memory fallback chose. Out
    of memory now halves the batch, and every photo is answered again at the new size.
    Each change is appended to ``fallbacks``.
    """
    asked = decode.batch_size
    decode = _earlier_fallback(dest, header_for(decode), decode)
    if decode.batch_size != asked:
        fallbacks.append(f"batch {asked} ran out of GPU memory earlier; used batch {decode.batch_size}")
    while True:
        try:
            _answer(load, rows, decode, header_for(decode), dest, progress, prompt)
            return decode
        except torch.OutOfMemoryError:
            if decode.batch_size == 1:
                raise
        # Answers depend on the batch, so every photo is answered again at the new size.
        smaller = replace(decode, batch_size=decode.batch_size // 2)
        message = (
            f"out of GPU memory at score batch {decode.batch_size}: answering all "
            f"{name} photos again at batch {smaller.batch_size}"
        )
        log.warning(message)
        fallbacks.append(message)
        dest.unlink()
        torch.cuda.empty_cache()
        decode = smaller


def answers(dest: Path, rows: list) -> list[dict]:
    """The answer lines in dest for exactly these rows, refused if any is missing."""
    wanted = {row[0] for row in rows}
    results = [json.loads(line) for line in dest.read_text().splitlines()[1:] if line.strip()]
    results = [row for row in results if row["image"] in wanted]
    if len(results) != len(rows):
        raise RuntimeError(f"{dest} has {len(results)} of {len(rows)} answers")
    return results


def score(config: RunConfig, folder: Path, split_name: str, photos: int, decode: DecodeSettings) -> None:
    """Score the run in ``folder`` on the first ``photos`` photos of ``split_name``.

    Refuses weights or a split.json whose hash differs from the run's records. If
    the GPU runs out of memory, the batch is halved and every photo is answered
    again at the new size, and the change is recorded in ``fallbacks``.
    """
    started = now()
    log_setup(config)
    result_path = folder / f"score-{split_name}.json"
    _weights_file, weights_sha = weights_identity(config, folder)
    if result_path.exists():
        if json.loads(result_path.read_text())["weights_sha256"] != weights_sha:
            raise RuntimeError(f"{result_path} scored other weights; delete it and {result_path.stem}.jsonl")
        log.info("%s already exists; delete it and %s.jsonl to score again", result_path, result_path.stem)
        return
    split_file = folder / config.data.split_file
    if file_sha256(split_file) != config.data.split_sha256:
        raise RuntimeError(f"{split_file} does not match the hash in config.json")
    rows = first(read_split(split_file)[split_name], photos, f"score {split_name}")
    paths_sha = text_sha256(json.dumps([row[0] for row in rows]))

    def header_for(settings: DecodeSettings) -> dict:
        return {
            "config_sha256": config.config_sha256,
            "split": split_name,
            "photos_sha256": paths_sha,
            "prompt": config.data.prompt,
            "judge": asdict(config.judge),
            **asdict(settings),
        }

    @cache
    def load():
        size = (config.image.reference_width, config.image.reference_height)
        processor = load_processor(config.model.id, config.model.revision, config.image.tiles, size)
        return processor, load_for_scoring(config, folder / "model", decode.precision)

    dest = folder / f"score-{split_name}.jsonl"
    progress = Progress(folder, split_name, len(rows), "photos")
    fallbacks: list[str] = []
    try:
        decode = answer_all(load, rows, decode, header_for, dest, progress, config.data.prompt, fallbacks, split_name)
    except BaseException:
        progress.finish("failed")
        raise

    results = answers(dest, rows)
    scores = [measure(row["gold"], row["raw"], row["width"], row["height"], config.judge) for row in results]
    n = len(results)
    prec = float(np.mean([one.prec for one in scores]))
    rec = float(np.mean([one.rec for one in scores]))
    result = ScoreResult(
        schema_version=SCHEMA_VERSION,
        run=config.name,
        prompt=config.data.prompt,
        judge=config.judge,
        config_sha256=config.config_sha256,
        weights_sha256=weights_sha,
        split=split_name,
        photos=n,
        photos_sha256=paths_sha,
        decode=decode,
        readable_pct=round(100 * sum(one.readable for one in scores) / n, 2),
        truncated_pct=round(100 * sum(row["truncated"] for row in results) / n, 2),
        cost=stats([one.cost_px for one in scores]),
        precision_pct=round(100 * prec, 2),
        recall_pct=round(100 * rec, 2),
        f_pct=round(100 * f_score(prec, rec), 2),
        lanes_pred_mean=round(float(np.mean([one.lanes_pred for one in scores])), 2),
        lanes_gold_mean=round(float(np.mean([one.lanes_gold for one in scores])), 2),
        predictions_file=dest.name,
        predictions_sha256=file_sha256(dest),
        started_at=started,
        finished_at=now(),
        environment=environment(),
        fallbacks=fallbacks,
    )
    write_once(result_path, dump(asdict(result)))
    progress.finish("done")
    c = result.cost
    log.info(
        "%d %s photos, run %s: readable %s%%, cut off %s%%, cost median %s px mean %s px, "
        "within 10 px %s%%, within 25 px %s%%, precision %s%%, recall %s%%, F %s%%",
        n,
        split_name,
        config.name,
        result.readable_pct,
        result.truncated_pct,
        c.median_px,
        c.mean_px,
        c.within_10_pct,
        c.within_25_pct,
        result.precision_pct,
        result.recall_pct,
        result.f_pct,
    )
