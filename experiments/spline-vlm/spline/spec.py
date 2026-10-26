"""The run spec: one JSON file that says everything a run should do.

A run is started with ``spline_train.py train specs/<file>.json``. Every key must
be present and no unknown key is allowed, so the file alone says how the run was
made. The only exceptions are two optional train keys, which are null when left
out. The run folder keeps a normalized copy as ``spec.json``.

* ``"save_every": N`` saves a checkpoint every N updates and at the last update.
  After training, each checkpoint answers the tune_cost photos, newest first. One
  that beats every newer one is kept, and the rest are deleted. Tune loss is still
  logged every eval_steps; no tune photos are answered during training.
* ``"init_from": "<run folder name>"`` starts from that run's saved weights instead
  of a fresh adapter. Its model, revision, method, and LoRA settings must match.
  ``epochs`` then counts the extra passes, with a new learning-rate schedule.

Structure (see each dataclass for meaning and limits)::

    {
      "model": {"id": "smolvlm2-2.2b", "revision": "main", "method": "lora",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                         "target_suffixes": ["q_proj", "k_proj", "v_proj", "o_proj"],
                         "vision": false}},
      "image": {"tiles": 13},
      "data":  {"train_percent": 10, "tune_loss_photos": 256, "tune_cost_photos": 100},
      "train": {"seed": 0, "epochs": 1.0, "max_steps": -1, "learning_rate": 0.0001,
                "lr_scheduler": "cosine", "warmup_fraction": 0.03, "weight_decay": 0.0,
                "max_grad_norm": 1.0, "batch_size": 2, "grad_accum": 4,
                "eval_batch_size": 2, "eval_steps": 100,
                "grad_checkpointing": false, "workers": 4},
      "score": {"tune_photos": 500, "test_photos": 1000, "batch_size": 32}
    }

A baseline that is never trained sets ``"method": "base"`` with ``"lora"``,
``"data"``, and ``"train"`` all null. It is scored on the test photos only, with
the hub weights at the resolved commit.

A grid file (``spline_queue.py grid specs/grids/<file>.json``) expands one base
spec into many. ``vary`` maps a dotted path to the values to try. Paths joined
by commas change together, so ``"model.lora.r,model.lora.alpha": [[8, 16], [16, 32]]``
keeps alpha at twice r. Every combination of the ``vary`` entries becomes one spec::

    {"name": "lora-vs-tiles", "base": {...a full spec...},
     "vary": {"model.lora.r,model.lora.alpha": [[8, 16], [16, 32]], "image.tiles": [5, 13]}}
"""

import copy
import hashlib
import itertools
import json
import logging
import types
import typing
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path

from spline.config import plain

log = logging.getLogger("spline.spec")


@dataclass
class LoraSpec:
    r: int  # adapter rank, at least 1; SmolVLM2 text attention has 393,216 * r adapter weights
    alpha: int  # the adapter output is scaled by alpha / r; written out, never derived
    dropout: float  # dropout on the adapter input while training, 0 <= dropout < 1
    target_suffixes: list[str]  # module names ending in one of these get an adapter
    vision: bool  # also put adapters on the vision encoder's modules


@dataclass
class ModelSpec:
    id: str  # Hugging Face id, or a short name from spline.config.MODELS
    revision: str  # branch, tag, or commit; the run records the commit it resolved to
    method: str  # "lora" trains an adapter; "full" trains every weight; "base" trains nothing
    lora: LoraSpec | None  # required with "lora", null otherwise


@dataclass
class ImageSpec:
    tiles: int | None  # tiles per photo (SmolVLM2: 1 5 7 13 16 25 29 41); null keeps the model's own


@dataclass
class DataSpec:
    train_percent: int  # first percent (1-100) of the train list; tune and test never shrink
    tune_loss_photos: int  # first N tune photos whose token loss is logged at each check
    tune_cost_photos: int  # first N tune photos answered at each check; their mean cost picks the checkpoint


@dataclass
class TrainSpec:
    seed: int  # seeds LoRA init, dropout, and the order of training photos
    epochs: float  # passes over the training photos; ignored when max_steps > 0
    max_steps: int  # weight updates to make; -1 means use epochs
    learning_rate: float
    lr_scheduler: str  # Hugging Face scheduler name, for example "cosine" or "linear"
    warmup_fraction: float  # share of updates spent raising the learning rate from 0
    weight_decay: float
    max_grad_norm: float  # gradients are clipped to this norm
    batch_size: int  # photos per GPU step
    grad_accum: int  # GPU steps per weight update; batch_size * grad_accum photos per update
    eval_batch_size: int  # tune photos per loss step
    eval_steps: int  # updates between tune-loss checks; the last update is always checked
    grad_checkpointing: bool  # start with activation recomputation; same math, less memory, slower
    workers: int  # CPU processes preparing batches; lowered on machines with fewer cores
    # Optional. Save a checkpoint every N updates (and at the last one) and pick the best
    # after training, newest first, instead of answering tune_cost photos during training.
    save_every: int | None = None
    # Optional. A run folder name under output/runs: start from its saved weights.
    init_from: str | None = None


@dataclass
class ScoreSpec:
    tune_photos: int  # first N tune photos answered to compare runs
    test_photos: int  # first N test photos answered once, for the final report
    batch_size: int  # photos answered at once; changes the answers, so keep it equal across runs


@dataclass
class RunSpec:
    model: ModelSpec
    image: ImageSpec
    data: DataSpec | None  # null exactly when model.method is "base"
    train: TrainSpec | None  # null exactly when model.method is "base"
    score: ScoreSpec


class SpecError(ValueError):
    """A spec or grid file that does not match the structure above."""


def _build(cls, data, where: str):
    """Turn parsed JSON into ``cls``, checking keys and types on the way down."""
    hints = typing.get_type_hints(cls)
    if not isinstance(data, dict):
        raise SpecError(f"{where} must be an object")
    names = {field.name for field in fields(cls)}
    optional = {field.name for field in fields(cls) if field.default is not MISSING}
    missing = names - optional - data.keys()
    unknown = data.keys() - names
    if missing:
        raise SpecError(f"{where} is missing {sorted(missing)}")
    if unknown:
        raise SpecError(f"{where} has unknown keys {sorted(unknown)}")
    return cls(**{name: _value(hints[name], data[name], f"{where}.{name}") for name in names & data.keys()})


def _value(kind, value, where: str):
    """Check one value against its annotation and convert nested objects."""
    options = typing.get_args(kind) if isinstance(kind, types.UnionType) else (kind,)
    if value is None:
        if type(None) in options:
            return None
        raise SpecError(f"{where} cannot be null")
    for option in options:
        if option is type(None):
            continue
        if is_dataclass(option):
            return _build(option, value, where)
        if typing.get_origin(option) is list:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SpecError(f"{where} must be a list of strings")
            return list(value)
        if option is float and isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if option is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if option in (str, bool) and type(value) is option:
            return value
    raise SpecError(f"{where} must be {kind}, got {value!r}")


def _check(spec: RunSpec) -> None:
    """Limits the type check cannot express."""
    model, data, train, score = spec.model, spec.data, spec.train, spec.score
    if model.method not in ("lora", "full", "base"):
        raise SpecError('model.method must be "lora", "full", or "base"')
    if (model.method == "lora") != (model.lora is not None):
        raise SpecError('model.lora must be set with "lora" and null otherwise')
    trained = model.method != "base"
    if trained != (data is not None) or trained != (train is not None):
        raise SpecError('data and train must be null with "base" and set otherwise')
    if model.lora:
        if model.lora.r < 1 or model.lora.alpha < 1:
            raise SpecError("model.lora.r and model.lora.alpha must be at least 1")
        if not 0 <= model.lora.dropout < 1:
            raise SpecError("model.lora.dropout must be in [0, 1)")
        if not model.lora.target_suffixes:
            raise SpecError("model.lora.target_suffixes is empty")
    if spec.image.tiles is not None and spec.image.tiles < 1:
        raise SpecError("image.tiles must be at least 1 or null")
    positive = {
        "score.tune_photos": score.tune_photos,
        "score.test_photos": score.test_photos,
        "score.batch_size": score.batch_size,
    }
    if trained:
        positive |= {
            "data.tune_loss_photos": data.tune_loss_photos,
            "data.tune_cost_photos": data.tune_cost_photos,
            "train.batch_size": train.batch_size,
            "train.grad_accum": train.grad_accum,
            "train.eval_batch_size": train.eval_batch_size,
            "train.eval_steps": train.eval_steps,
        }
    for name, value in positive.items():
        if value < 1:
            raise SpecError(f"{name} must be at least 1")
    if not trained:
        return
    if not 1 <= data.train_percent <= 100:
        raise SpecError("data.train_percent must be 1 to 100")
    if train.max_steps < 1 and train.epochs <= 0:
        raise SpecError("set train.max_steps above 0 or train.epochs above 0")
    if train.learning_rate <= 0 or not 0 <= train.warmup_fraction < 1 or train.workers < 0:
        raise SpecError("train.learning_rate > 0, 0 <= train.warmup_fraction < 1, train.workers >= 0")
    if train.save_every is not None and train.save_every < 1:
        raise SpecError("train.save_every must be at least 1 or left out")
    if train.init_from is not None and (not train.init_from or "/" in train.init_from):
        raise SpecError("train.init_from must be a run folder name under output/runs")


def parse(data: dict, where: str = "spec") -> RunSpec:
    """A checked RunSpec from parsed JSON."""
    spec = _build(RunSpec, data, where)
    _check(spec)
    return spec


def load(path: Path) -> tuple[RunSpec, str]:
    """Read a spec file. Returns the spec and the sha256 of the file's bytes."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SpecError(f"{path} is not valid JSON: {error}") from error
    spec = parse(data, str(path))
    digest = hashlib.sha256(raw).hexdigest()
    log.info("spec %s (sha256 %s)", path, digest[:12])
    log.debug("spec contents: %s", json.dumps(asdict(spec)))
    return spec, digest


def normalized(spec: RunSpec) -> str:
    """The spec as the run folder stores it: sorted keys, two-space indent."""
    return json.dumps(plain(spec), indent=2, sort_keys=True) + "\n"


def _set(data: dict, path: str, value) -> None:
    """Set ``a.b.c`` inside nested dicts; the path must already exist in the base spec."""
    *parents, last = path.split(".")
    node = data
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise SpecError(f"grid path {path}: {key} is not an object in the base spec")
        node = node[key]
    if last not in node:
        raise SpecError(f"grid path {path}: {last} is not in the base spec")
    node[last] = value


def expand_grid(grid: dict) -> list[tuple[str, dict]]:
    """Every spec a grid file describes, as (file stem, spec JSON), in a fixed order."""
    for key in ("name", "base", "vary"):
        if key not in grid:
            raise SpecError(f"grid file needs {key!r}")
    if set(grid) - {"name", "base", "vary"}:
        raise SpecError(f"grid file has unknown keys {sorted(set(grid) - {'name', 'base', 'vary'})}")
    axes = []
    for key, values in grid["vary"].items():
        paths = key.split(",")
        if not isinstance(values, list) or not values:
            raise SpecError(f"grid vary {key} needs a non-empty list")
        rows = [value if len(paths) > 1 else [value] for value in values]
        if any(not isinstance(row, list) or len(row) != len(paths) for row in rows):
            raise SpecError(f"grid vary {key}: each value needs {len(paths)} items")
        axes.append([(paths, row) for row in rows])
    specs = []
    for combo in itertools.product(*axes):
        data = copy.deepcopy(grid["base"])
        parts = []
        for paths, row in combo:
            for path, value in zip(paths, row, strict=True):
                _set(data, path, value)
                parts.append(f"{path.split('.')[-1]}{value}")
        parse(data, f"grid {grid['name']} {parts}")
        specs.append(("-".join(parts), data))
    return specs
