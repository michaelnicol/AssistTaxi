"""Fine-tune on the training photos and keep the checkpoint with the lowest tune cost.

At each check the token loss is measured on the ``tune_loss`` rows, and the
``tune_cost`` rows are answered greedily and judged with :func:`spline.score.measure`.
Their mean cost in pixels picks the checkpoint.

With ``train.save_every`` the tune_cost rows are not answered during training.
A checkpoint is saved every ``save_every`` updates and at the last update, and
after training each one answers them, newest first (see :func:`_sweep`). Its
state is in ``sweep/``: ``trained.json`` once training is done, ``sweep.json``
with the order and each result, and one answer file per checkpoint.

While it runs, training writes two live files the GUI reads:

* ``live/train.json``: the latest step, loss, learning rate, and ETA (see
  :class:`spline.logs.Progress`).
* ``live/train.jsonl``: one line per Trainer log entry, for the loss and
  learning-rate plots. A resumed run appends to it, so a step can appear twice;
  readers keep the last line for each step.

The weights are only ever shaped by ``config.json``. Speed settings (workers,
gradient checkpointing) and any automatic fallback are recorded in ``train.json``.
"""

import copy
import gc
import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from functools import cache, partial
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from spline.config import SCHEMA_VERSION, DecodeSettings, RunConfig, TrainResult
from spline.logs import Progress, read_json, write_json
from spline.model import PromptCollator, SplineCollator, answer_batch, load_for_scoring, load_for_training
from spline.runs import OUTPUT, dump, environment, now, run_dir, weights_digest, write_once
from spline.score import answer_all, answers, f_score, log_setup, measure

log = logging.getLogger("spline.fit")

# Seconds between INFO lines about training loss. Every step still goes to the debug log.
INFO_EVERY_S = 60.0


class CheckLastUpdate(TrainerCallback):
    """Measure tune loss and save on the final update too, so it can be the one picked."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class LiveStatus(TrainerCallback):
    """Copies each Trainer log entry to the live files and the logs."""

    def __init__(self, folder: Path, total: int) -> None:
        self.progress = Progress(folder, "train", total, "updates")
        self.history = folder / "live" / "train.jsonl"
        self.history.parent.mkdir(parents=True, exist_ok=True)
        self.last_info = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        entry = {"step": int(state.global_step), "time": round(time.time(), 1), **(logs or {})}
        with self.history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        log.debug("step %d %s", state.global_step, logs)
        if "eval_tune_cost_px" in entry:
            log.info(
                "update %d: tune cost %.2f px, precision %.3f, recall %.3f",
                state.global_step,
                entry["eval_tune_cost_px"],
                entry["eval_tune_prec"],
                entry["eval_tune_rec"],
            )
            self.progress.update(state.global_step, force=True, eval_tune_cost_px=entry["eval_tune_cost_px"])
            return
        if "eval_loss" in entry:
            log.info("update %d: tune loss %.4f", state.global_step, entry["eval_loss"])
            self.progress.update(state.global_step, force=True, eval_loss=entry["eval_loss"])
            return
        extra = {key: entry[key] for key in ("loss", "learning_rate", "grad_norm", "epoch") if key in entry}
        self.progress.update(state.global_step, **extra)
        if time.time() - self.last_info >= INFO_EVERY_S:
            self.last_info = time.time()
            log.info(
                "update %d of %d: loss %s, learning rate %s, eta %s s",
                state.global_step,
                state.max_steps,
                entry.get("loss"),
                entry.get("learning_rate"),
                None if self.progress.eta_s is None else round(self.progress.eta_s),
            )


def worker_count(asked: int) -> int:
    """DataLoader processes to use: the spec's number, but one fewer than the cores here at most."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if asked >= cores:
        log.warning("%d workers asked, %d CPU cores here: using %d", asked, cores, cores - 1)
        return max(0, cores - 1)
    return asked


class SplineTrainer(Trainer):
    """A Trainer whose evaluate() also answers the tune_cost rows and adds their mean cost.

    The Trainer picks the best checkpoint from the metrics evaluate() returns, so
    ``eval_tune_cost_px`` here is what ``metric_for_best_model`` selects on.
    """

    def __init__(self, *args, config: RunConfig, tune_cost: list, decode: DecodeSettings, fallbacks: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.spline_config = config
        self.tune_cost = tune_cost
        self.decode = decode
        self.fallbacks = fallbacks
        # Its own copy, because answering pads on the left and training batches must pad on the right.
        self.answer_processor = copy.deepcopy(kwargs["data_collator"].processor)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.spline_config.train.save_every is not None:
            # The checkpoint sweep after training answers the tune_cost photos instead.
            return metrics
        started = time.monotonic()
        while True:
            try:
                scores = self._tune_scores()
                break
            except torch.OutOfMemoryError:
                if self.decode.batch_size == 1:
                    raise
            torch.cuda.empty_cache()
            message = (
                f"out of GPU memory answering tune_cost photos at batch {self.decode.batch_size}: "
                f"using batch {self.decode.batch_size // 2} for this and every later check"
            )
            log.warning(message)
            self.fallbacks.append(message)
            self.decode = replace(self.decode, batch_size=self.decode.batch_size // 2)
        prec = float(np.mean([one.prec for one in scores]))
        rec = float(np.mean([one.rec for one in scores]))
        cost = {
            f"{metric_key_prefix}_tune_cost_px": float(np.mean([one.cost_px for one in scores])),
            f"{metric_key_prefix}_tune_prec": prec,
            f"{metric_key_prefix}_tune_rec": rec,
            f"{metric_key_prefix}_tune_f": f_score(prec, rec),
            f"{metric_key_prefix}_tune_readable": float(np.mean([one.readable for one in scores])),
            f"{metric_key_prefix}_tune_cost_s": round(time.monotonic() - started, 1),
        }
        self.log(cost)
        return {**metrics, **cost}

    def _tune_scores(self) -> list:
        config, decode = self.spline_config, self.decode
        prepare = PromptCollator(self.answer_processor, config.data.prompt)
        model = self.model
        model.eval()
        scores = []
        with self.accelerator.autocast():
            for start in range(0, len(self.tune_cost), decode.batch_size):
                chunk, inputs = prepare(self.tune_cost[start : start + decode.batch_size])
                sizes = [(row[2], row[3]) for row in chunk]
                answers = answer_batch(self.answer_processor, model, inputs, decode, sizes)
                scores += [
                    measure(gold, raw, width, height, config.judge)
                    for (_path, gold, width, height), (raw, _cut) in zip(chunk, answers, strict=True)
                ]
                longest = max((raw for raw, _cut in answers), key=len)
                log.debug(
                    "tune cost: answered %d of %d, %d cut off, longest %d chars: %r",
                    len(scores),
                    len(self.tune_cost),
                    sum(cut for _raw, cut in answers),
                    len(longest),
                    longest[:160],
                )
        return scores


def _trainer(config, model, data, processor, folder, workers, checkpointing, live, decode, fallbacks) -> Trainer:
    spec = config.train
    sweep = spec.save_every is not None
    return SplineTrainer(
        config=config,
        tune_cost=data["tune_cost"],
        decode=decode,
        fallbacks=fallbacks,
        model=model,
        args=TrainingArguments(
            output_dir=str(folder / "checkpoints"),
            seed=spec.seed,
            data_seed=spec.seed,
            max_steps=spec.updates,
            per_device_train_batch_size=spec.batch_size,
            per_device_eval_batch_size=spec.eval_batch_size,
            gradient_accumulation_steps=spec.grad_accum,
            learning_rate=spec.learning_rate,
            lr_scheduler_type=spec.lr_scheduler,
            warmup_steps=spec.warmup_steps,
            optim=spec.optimizer,
            adam_beta1=spec.adam_beta1,
            adam_beta2=spec.adam_beta2,
            adam_epsilon=spec.adam_epsilon,
            weight_decay=spec.weight_decay,
            max_grad_norm=spec.max_grad_norm,
            eval_strategy="steps",
            eval_steps=spec.eval_steps,
            save_strategy="steps",
            save_steps=spec.save_every if sweep else spec.eval_steps,
            save_total_limit=None if sweep else 2,
            load_best_model_at_end=not sweep,
            metric_for_best_model=None if sweep else "eval_tune_cost_px",
            greater_is_better=False,
            prediction_loss_only=True,
            logging_steps=1,
            # The Trainer's progress bar rewrites the console many times a second; LiveStatus replaces it.
            disable_tqdm=True,
            bf16=config.model.precision == "bf16",
            fp16=config.model.precision == "fp16",
            gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
            remove_unused_columns=False,
            dataloader_num_workers=workers,
            dataloader_persistent_workers=workers > 0,
            # Pinned memory speeds host-to-GPU copies and does nothing without CUDA.
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=data["train"],
        eval_dataset=data["tune_loss"],
        data_collator=SplineCollator(processor, config.data.prompt),
        callbacks=[CheckLastUpdate(), live],
    )


def train(
    config: RunConfig,
    folder: Path,
    processor,
    decode: DecodeSettings,
    workers: int,
    checkpointing: bool,
    spec_file: str,
    spec_sha256: str,
) -> None:
    """Train the run in ``folder``, resuming from its last checkpoint, and write train.json.

    If the GPU runs out of memory without gradient checkpointing, training
    restarts from the last checkpoint with it on. That changes speed and memory,
    not the math, and is recorded in ``fallbacks``. With ``train.save_every``,
    the checkpoint sweep follows, and a rerun after training goes straight to it.
    """
    result_path = folder / "train.json"
    if result_path.exists():
        log.info("run %s is already trained", config.name)
        return
    log_setup(config)
    started = now()
    clock = time.monotonic()
    data = json.loads((folder / "data.json").read_text())
    spec = config.train
    workers = worker_count(workers)
    start = _start(config)
    log.info(
        "training %s from %s: %d photos, %s tiles, %d updates, every %d updates tune loss on %d photos; "
        "tune cost on %d photos %s, %d workers",
        config.name,
        spec.init_from["run"] if spec.init_from else "the hub weights",
        len(data["train"]),
        config.image.tiles,
        spec.updates,
        spec.eval_steps,
        len(data["tune_loss"]),
        len(data["tune_cost"]),
        f"for a checkpoint every {spec.save_every} updates after training"
        if spec.save_every
        else f"every {spec.eval_steps} updates",
        workers,
    )
    saved = folder / "model"
    trained_path = folder / "sweep" / "trained.json"
    if spec.save_every and trained_path.exists():
        log.info("training finished in an earlier process; going on with the checkpoint sweep")
        trained = read_json(trained_path)
    else:
        trainer, trained = _fit(config, data, processor, folder, workers, checkpointing, decode, start)
    if spec.save_every:
        if not trained_path.exists():
            write_json(trained_path, trained)
            # Frees the training model before each checkpoint is loaded for answering.
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        sweep = _sweep(config, folder, processor, decode, data["tune_cost"])
        best = min(sweep["results"], key=lambda row: row["tune_cost_px"])
        best_step, best_cost = best["step"], best["tune_cost_px"]
        shutil.copytree(
            folder / "checkpoints" / f"checkpoint-{best_step}",
            saved,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("trainer_state.json"),
        )
        history = trained["log_history"] + [_sweep_entry(row) for row in sorted(sweep["results"], key=_step)]
        fallbacks = trained["fallbacks"] + sweep["fallbacks"]
    else:
        trainer.model.save_pretrained(saved)
        best = trainer.state.best_model_checkpoint
        best_step = int(Path(best).name.split("-")[-1]) if best else None
        best_cost = trainer.state.best_metric
        history, fallbacks, sweep = trained["log_history"], trained["fallbacks"], {"results": []}
    weights_file, weights_sha = weights_digest(saved)
    losses = {entry["step"]: entry["eval_loss"] for entry in history if "eval_loss" in entry}
    result = TrainResult(
        schema_version=SCHEMA_VERSION,
        run=config.name,
        prompt=config.data.prompt,
        judge=config.judge,
        config_sha256=config.config_sha256,
        trainable_params=trained["trainable_params"],
        total_params=trained["total_params"],
        updates=trained["updates"],
        best_step=best_step,
        best_tune_cost_px=best_cost,
        best_tune_loss=losses.get(best_step),
        train_loss=trained["train_loss"],
        runtime_s=round(time.monotonic() - clock, 1),
        log_history=history,
        weights_file=weights_file,
        weights_sha256=weights_sha,
        resumed_from=trained["resumed_from"],
        grad_checkpointing=trained["grad_checkpointing"],
        workers=workers,
        fallbacks=fallbacks,
        spec_file=spec_file,
        spec_sha256=spec_sha256,
        started_at=started,
        finished_at=now(),
        environment=environment(),
        sweep=sweep["results"],
    )
    write_once(result_path, dump(asdict(result)))
    shutil.rmtree(folder / "checkpoints", ignore_errors=True)
    log.info(
        "saved %s: update %s with tune cost %s px, tune loss %s",
        saved,
        result.best_step,
        result.best_tune_cost_px,
        result.best_tune_loss,
    )


def _start(config: RunConfig) -> Path | None:
    """The saved weights this run starts from, checked against the hash in config.json."""
    source = config.train.init_from
    if source is None:
        return None
    saved = run_dir(OUTPUT, source["run"]) / "model"
    if weights_digest(saved)[1] != source["weights_sha256"]:
        raise RuntimeError(f"the weights in {saved} do not match the hash config.json recorded for them")
    return saved


def _fit(config, data, processor, folder, workers, checkpointing, decode, start) -> tuple[Trainer, dict]:
    """Run the Trainer to the last update. Returns it and what train.json records about training."""
    spec = config.train
    checkpoints = folder / "checkpoints"
    live = LiveStatus(folder, spec.updates)
    fallbacks: list[str] = []
    resume = None
    try:
        while True:
            # Before the model loads, so LoRA's random start is the same for every run with this seed.
            set_seed(spec.seed)
            model = load_for_training(config, start)
            trainer = _trainer(config, model, data, processor, folder, workers, checkpointing, live, decode, fallbacks)
            resume = get_last_checkpoint(str(checkpoints)) if checkpoints.is_dir() else None
            if resume:
                log.info("resuming from %s", resume)
            try:
                output = trainer.train(resume_from_checkpoint=resume)
                break
            except torch.OutOfMemoryError:
                if checkpointing:
                    raise RuntimeError(
                        "out of GPU memory even with gradient checkpointing. Halve train.batch_size and "
                        "double train.grad_accum in the spec; that is a new run."
                    ) from None
            message = "out of GPU memory: restarting from the last checkpoint with gradient checkpointing"
            log.warning(message)
            fallbacks.append(message)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            checkpointing = True
    except BaseException:
        live.progress.finish("failed")
        raise
    if trainer.state.global_step != spec.updates:
        raise RuntimeError(f"made {trainer.state.global_step} updates, config says {spec.updates}")
    live.progress.finish("done")
    return trainer, {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "updates": int(trainer.state.global_step),
        "train_loss": output.metrics.get("train_loss"),
        "log_history": trainer.state.log_history,
        "resumed_from": Path(resume).name if resume else None,
        "grad_checkpointing": checkpointing,
        "fallbacks": fallbacks,
    }


# Trainer files a checkpoint only needs to resume training. The sweep deletes them.
RESUME_ONLY = ("optimizer.pt", "scheduler.pt", "scaler.pt", "training_args.bin")


class _Offset:
    """The sweep's Progress, seen from one checkpoint, so per-file counts add up across checkpoints."""

    def __init__(self, progress: Progress, base: int) -> None:
        self.progress = progress
        self.base = base

    def update(self, done: int, force: bool = False, **extra) -> None:
        self.progress.update(self.base + done, force, **extra)


def _step(row: dict) -> int:
    return row["step"]


def _sweep_entry(row: dict) -> dict:
    """A sweep result as a log_history entry, under the names an inline check uses."""
    return {
        "step": row["step"],
        "eval_tune_cost_px": row["tune_cost_px"],
        "eval_tune_prec": row["tune_prec"],
        "eval_tune_rec": row["tune_rec"],
        "eval_tune_f": f_score(row["tune_prec"], row["tune_rec"]),
        "eval_tune_readable": row["tune_readable"],
        "eval_tune_cost_s": row["seconds"],
    }


def _load_checkpoint(config: RunConfig, saved: Path, processor, precision: str):
    return processor, load_for_scoring(config, saved, precision)


def _sweep_header(config: RunConfig, step: int, settings: DecodeSettings) -> dict:
    return {
        "config_sha256": config.config_sha256,
        "checkpoint": step,
        "prompt": config.data.prompt,
        "judge": asdict(config.judge),
        **asdict(settings),
    }


def _sweep(config: RunConfig, folder: Path, processor, decode: DecodeSettings, rows: list) -> dict:
    """Answer the tune_cost rows with every saved checkpoint, newest first, keeping only the best so far.

    A checkpoint is kept when its mean cost is lower than every newer one's, and
    the one it beat is deleted; otherwise it is deleted itself. ``sweep/sweep.json``
    records the order and each result before any folder is deleted, so a stopped
    sweep resumes at the checkpoint it was on, after that checkpoint's last whole batch.
    Returns that record: {order, results, fallbacks}.
    """
    checkpoints = folder / "checkpoints"
    state_path = folder / "sweep" / "sweep.json"
    for path in checkpoints.glob("checkpoint-*/*"):
        if path.name in RESUME_ONLY or path.name.startswith("rng_state"):
            path.unlink()
    state = read_json(state_path)
    if state is None:
        order = sorted((int(path.name.split("-")[-1]) for path in checkpoints.glob("checkpoint-*")), reverse=True)
        if not order or order[0] != config.train.updates:
            raise RuntimeError(f"{checkpoints} has no checkpoint of the last update, {config.train.updates}")
        state = {"order": order, "results": [], "fallbacks": []}
        write_json(state_path, state)
    log.info("checkpoint sweep, newest first: %s", ", ".join(str(step) for step in state["order"]))
    # Its own copy, because answering pads on the left.
    processor = copy.deepcopy(processor)
    history = folder / "live" / "train.jsonl"
    progress = Progress(folder, "sweep", len(state["order"]) * len(rows), "photos")
    try:
        for index, step in enumerate(state["order"]):
            if step not in {row["step"] for row in state["results"]}:
                folder_now = checkpoints / f"checkpoint-{step}"
                dest = folder / "sweep" / f"{folder_now.name}.jsonl"
                progress.update(index * len(rows), force=True, checkpoint=step)
                load = cache(partial(_load_checkpoint, config, folder_now, processor, decode.precision))
                clock = time.monotonic()
                decode = answer_all(
                    load,
                    rows,
                    decode,
                    partial(_sweep_header, config, step),
                    dest,
                    _Offset(progress, index * len(rows)),
                    config.data.prompt,
                    state["fallbacks"],
                    folder_now.name,
                )
                load.cache_clear()
                gc.collect()
                torch.cuda.empty_cache()
                scores = [
                    measure(one["gold"], one["raw"], one["width"], one["height"], config.judge)
                    for one in answers(dest, rows)
                ]
                prior = min((row["tune_cost_px"] for row in state["results"]), default=math.inf)
                row = {
                    "step": step,
                    "tune_cost_px": float(np.mean([one.cost_px for one in scores])),
                    "tune_prec": float(np.mean([one.prec for one in scores])),
                    "tune_rec": float(np.mean([one.rec for one in scores])),
                    "tune_readable": float(np.mean([one.readable for one in scores])),
                    "seconds": round(time.monotonic() - clock, 1),
                }
                row["better"] = row["tune_cost_px"] < prior
                state["results"].append(row)
                write_json(state_path, state)
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"time": round(time.time(), 1), **_sweep_entry(row)}) + "\n")
                log.info(
                    "checkpoint %d: tune cost %.2f px, precision %.3f, recall %.3f: %s",
                    step,
                    row["tune_cost_px"],
                    row["tune_prec"],
                    row["tune_rec"],
                    "best so far, kept" if row["better"] else f"not better than {prior:.2f} px, deleted",
                )
            best = min(state["results"], key=lambda row: row["tune_cost_px"])
            for row in state["results"]:
                if row["step"] != best["step"]:
                    shutil.rmtree(checkpoints / f"checkpoint-{row['step']}", ignore_errors=True)
    except BaseException:
        progress.finish("failed")
        raise
    progress.finish("done")
    return state
