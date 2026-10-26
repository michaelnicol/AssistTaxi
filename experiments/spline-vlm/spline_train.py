#!/usr/bin/env python3
"""Train a vision-language model to trace lane lines, score it, and compare runs.

Every run is described by one JSON spec file (format in spline/spec.py and README.md)::

  train-venv/bin/python spline_train.py train specs/smolvlm2-r16-t13.json
  train-venv/bin/python spline_train.py score specs/smolvlm2-r16-t13.json tune
  train-venv/bin/python spline_train.py score specs/smolvlm2-r16-t13.json test
  train-venv/bin/python spline_train.py report

The spec names the run: its training settings are resolved into config.json,
whose hash ends the run folder's name. Logs go to output/runs/<name>/logs/.
"""

import json
import logging
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from spline import logs, spec
from spline.config import (
    LABEL_FORMAT,
    PROMPT,
    SCHEMA_VERSION,
    DataSettings,
    DecodeSettings,
    LoraSettings,
    ModelSettings,
    RunConfig,
    TrainSettings,
    finish,
    plain,
)
from spline.data import first, make_split, read_split, reference_size, train_count
from spline.fit import train
from spline.lanes import DEFAULT_JUDGE
from spline.logs import read_json
from spline.model import (
    context_tokens,
    image_settings,
    load_processor,
    precision,
    resolve_model,
    resolve_revision,
)
from spline.runs import OUTPUT, all_runs, dump, file_sha256, run_dir, write_once
from spline.score import score

USAGE = "usage: spline_train.py train SPEC | score SPEC tune|test | report"
# Fixed for every run. They are still written into config.json so a run records them.
OPTIMIZER = {"optimizer": "adamw_torch_fused", "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8}

log = logging.getLogger("spline.train")


def _lora(run: spec.RunSpec) -> LoraSettings:
    return LoraSettings(
        r=run.model.lora.r,
        alpha=run.model.lora.alpha,
        dropout=run.model.lora.dropout,
        target_suffixes=sorted(run.model.lora.target_suffixes),
        skip_vision=not run.model.lora.vision,
        bias="none",
    )


def _init_from(name: str | None, model: ModelSettings) -> dict | None:
    """The run a spec starts from, checked to have saved weights for this same model."""
    if name is None:
        return None
    folder = run_dir(OUTPUT, name)
    trained, built = read_json(folder / "train.json"), read_json(folder / "config.json")
    if trained is None or built is None:
        raise RuntimeError(f"train.init_from: {folder} is not a trained run")
    ours = asdict(model)
    for key in ("id", "revision", "method", "lora"):
        if built["model"][key] != ours[key]:
            raise RuntimeError(
                f"train.init_from: {name} has model.{key} {built['model'][key]!r}, this spec {ours[key]!r}"
            )
    return {"run": name, "weights_sha256": trained["weights_sha256"]}


def _train_settings(t: spec.TrainSpec, photos: int, model: ModelSettings) -> TrainSettings:
    if t.max_steps > 0:
        updates = t.max_steps
    else:
        updates = math.ceil(t.epochs * math.ceil(math.ceil(photos / t.batch_size) / t.grad_accum))
    return TrainSettings(
        seed=t.seed,
        epochs=t.epochs,
        max_steps=t.max_steps,
        updates=updates,
        learning_rate=t.learning_rate,
        lr_scheduler=t.lr_scheduler,
        warmup_fraction=t.warmup_fraction,
        warmup_steps=max(1, int(updates * t.warmup_fraction)),
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        batch_size=t.batch_size,
        grad_accum=t.grad_accum,
        photos_per_update=t.batch_size * t.grad_accum,
        eval_batch_size=t.eval_batch_size,
        eval_steps=t.eval_steps,
        select_checkpoint="lowest_tune_cost",
        save_every=t.save_every,
        init_from=_init_from(t.init_from, model),
        **OPTIMIZER,
    )


def build_config(run: spec.RunSpec) -> tuple[RunConfig, object]:
    """Every setting that can change the trained weights, plus the processor those settings made.

    Branch names become commits and tile counts are measured, so config.json
    holds what was used, not what was asked for. A base run records zero photos
    and no train settings.
    """
    model_id = resolve_model(run.model.id)
    revision = resolve_revision(model_id, run.model.revision)
    split_file = make_split(OUTPUT)
    split = read_split(split_file)
    if split["rdp_px"] != DEFAULT_JUDGE.rdp_px:
        raise RuntimeError(
            f"{split_file} was made with rdp_px {split['rdp_px']}, the judge uses {DEFAULT_JUDGE.rdp_px}"
        )
    size = reference_size(split)
    processor = load_processor(model_id, revision, run.image.tiles, size)
    first(split["tune"], run.score.tune_photos, "score.tune_photos")
    first(split["test"], run.score.test_photos, "score.test_photos")
    lora = _lora(run) if run.model.method == "lora" else None
    model = ModelSettings(id=model_id, revision=revision, method=run.model.method, lora=lora, precision=precision())
    match run.model.method:
        case "base":
            photos, tune_loss, tune_cost, train_settings = 0, 0, 0, None
        case "lora" | "full":
            photos = train_count(split, run.data.train_percent)
            tune_loss = len(first(split["tune"], run.data.tune_loss_photos, "data.tune_loss_photos"))
            tune_cost = len(first(split["tune"], run.data.tune_cost_photos, "data.tune_cost_photos"))
            train_settings = _train_settings(run.train, photos, model)
        case other:
            raise ValueError(f"unknown model.method {other!r}")
    config = RunConfig(
        schema_version=SCHEMA_VERSION,
        name="",
        config_sha256="",
        model=model,
        image=image_settings(processor, size),
        data=DataSettings(
            split_file=split_file.name,
            split_sha256=file_sha256(split_file),
            train_percent=run.data.train_percent if run.data else 0,
            train_photos=photos,
            tune_loss_photos=tune_loss,
            tune_cost_photos=tune_cost,
            prompt=PROMPT,
            label_format=LABEL_FORMAT,
        ),
        judge=DEFAULT_JUDGE,
        train=train_settings,
    )
    config = finish(config)
    log.debug("config %s", json.dumps(plain(config)))
    return config, processor


def decode_settings(config: RunConfig, run: spec.RunSpec) -> DecodeSettings:
    """Greedy answers with the whole context window, the same for scoring and for checkpoint choice."""
    return DecodeSettings(
        context_tokens=context_tokens(config.model.id, config.model.revision),
        do_sample=False,
        num_beams=1,
        batch_size=run.score.batch_size,
        precision=precision(),
    )


def prepare(config: RunConfig, run: spec.RunSpec, folder: Path) -> None:
    """Write config.json, spec.json, a byte copy of split.json, and data.json with the exact photos used.

    spec.json is write-once like the rest: a second spec that makes the same run
    but asks for other speed or score settings is refused, so a run folder has
    exactly one spec behind it. A base run's data.json has empty lists.
    """
    write_once(folder / "config.json", dump(plain(config)))
    try:
        write_once(folder / "spec.json", spec.normalized(run))
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}. Another spec already made run {config.name} with other train speed or score "
            "settings; use that spec, or change a training setting to make a new run."
        ) from None
    copy = folder / config.data.split_file
    if not copy.exists():
        shutil.copyfile(OUTPUT / config.data.split_file, copy)
    if file_sha256(copy) != config.data.split_sha256:
        raise RuntimeError(f"{copy} does not match the hash in config.json")
    split = read_split(copy)
    data = {
        "schema_version": SCHEMA_VERSION,
        "split_sha256": config.data.split_sha256,
        "train": split["train"][: config.data.train_photos],
        "tune_loss": split["tune"][: config.data.tune_loss_photos],
        "tune_cost": split["tune"][: config.data.tune_cost_photos],
    }
    write_once(folder / "data.json", json.dumps(data) + "\n")


def trained_config(folder: Path, built: RunConfig, run: spec.RunSpec) -> RunConfig:
    """The built config, checked against the one the run was trained with."""
    path = folder / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"{folder} has no config.json; train this run first")
    if json.loads(path.read_text()) != plain(built):
        raise RuntimeError(f"{path} differs from what this spec builds now (did the code or the model change?)")
    if (folder / "spec.json").read_text() != spec.normalized(run):
        raise RuntimeError(f"{folder / 'spec.json'} differs from this spec; score with the spec that trained it")
    return built


def _report_row(run: dict) -> str:
    lora = run["config"]["model"]["lora"] or {}
    loss = (run["train"] or {}).get("best_tune_loss")
    tune = run["scores"]["tune"]["cost"]["median_px"] if run["scores"]["tune"] else None
    test = run["scores"]["test"] or {}
    cost = test.get("cost") or {}
    return (
        f"{run['name']:58} {run['config']['image']['tiles'] or '-':>5} {lora.get('r', '-'):>4} "
        f"{lora.get('alpha', '-'):>4} {loss if loss is None else round(loss, 4)!s:>9} {tune!s:>8} "
        f"{cost.get('median_px')!s:>8} {cost.get('within_10_pct')!s:>10} {test.get('f_pct')!s:>7}"
    )


def report(output: Path) -> None:
    """Runs ranked by median tune cost. Test numbers are shown, but only tune should pick a winner.

    Base runs have no tune score, so they come after the ranked runs.
    """
    runs = all_runs(output)
    for split in ("tune", "test"):
        scored = [run["scores"][split] for run in runs if run["scores"][split]]
        sets = {score["photos_sha256"] for score in scored}
        decodes = {json.dumps(score["decode"], sort_keys=True) for score in scored}
        if len(sets) > 1 or len(decodes) > 1:
            log.warning("%s scores use different photos or decode settings, so they are not comparable", split)

    def key(run):
        tune = run["scores"]["tune"]
        return tune["cost"]["median_px"] if tune and tune["cost"]["median_px"] is not None else math.inf

    print(
        f"{'run':58} {'tiles':>5} {'r':>4} {'a':>4} {'tune loss':>9} {'tune px':>8} {'test px':>8} "
        f"{'test 10px%':>10} {'test F%':>7}"
    )
    trained = [run for run in runs if run["config"]["model"]["method"] != "base"]
    for run in sorted(trained, key=key):
        print(_report_row(run))
    baselines = [run for run in runs if run["config"]["model"]["method"] == "base"]
    if baselines:
        print("baselines (test only)")
        for run in baselines:
            print(_report_row(run))


def main(argv: list[str]) -> int:
    if argv[:1] == ["report"] and len(argv) == 1:
        logs.setup("report", OUTPUT / "logs")
        report(OUTPUT)
        return 0
    if not (argv[:1] == ["train"] and len(argv) == 2) and not (
        argv[:1] == ["score"] and len(argv) == 3 and argv[2] in ("tune", "test")
    ):
        print(USAGE, file=sys.stderr)
        return 2
    command, spec_path = argv[0], Path(argv[1])
    process = "train" if command == "train" else f"score-{argv[2]}"
    logs.setup(process)
    attached = False
    try:
        run, spec_sha = spec.load(spec_path)
        config, processor = build_config(run)
        folder = run_dir(OUTPUT, config.name)
        logs.attach(folder / "logs", process)
        attached = True
        log.info("run %s from %s", config.name, spec_path)
        base = config.model.method == "base"
        decode = decode_settings(config, run)
        if command == "train":
            if base:
                raise RuntimeError(f"base runs are not trained; use: spline_train.py score {spec_path} test")
            prepare(config, run, folder)
            train(
                config,
                folder,
                processor,
                decode,
                run.train.workers,
                run.train.grad_checkpointing,
                str(spec_path),
                spec_sha,
            )
        else:
            split = argv[2]
            if base:
                if split != "test":
                    raise RuntimeError("base runs are scored on test only, so they never pick a winner")
                prepare(config, run, folder)
            photos = run.score.tune_photos if split == "tune" else run.score.test_photos
            score(trained_config(folder, config, run), folder, split, photos, decode)
    except Exception:
        if not attached:
            # The run folder is unknown, so the held records go to the shared log folder.
            logs.attach(OUTPUT / "logs", process)
        log.exception("%s %s failed", command, spec_path)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
