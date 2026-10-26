"""What a run folder records: the exact structure of every JSON file in it.

Each dataclass here is written with ``dataclasses.asdict`` (or :func:`plain`,
which leaves out optional fields that are null), so its field list is the file format. The input side, the JSON spec a run is started from, lives in
``spline/spec.py``.

``RunConfig`` (``config.json``) holds everything that can change the trained
weights, resolved and measured: the commit the model id pointed at, the tile
count the processor actually produced, the number of updates the epochs work out
to. Its sha256 is the last part of the run name, so two runs with one name were
trained the same way, and changing any of these values makes a new run.

Speed-only settings (``workers``, ``grad_checkpointing``) and scoring settings are
not part of that hash. Training records what it used in ``train.json``, and each
score records its decode settings in ``score-<split>.json``.
"""

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass

# Bumped whenever a field below changes meaning. Files from another version are refused.
SCHEMA_VERSION = 4
# The same text for every photo, phase, and run. It never states the photo size;
# the model has to read the scale from the image. The example lanes are made up.
PROMPT = (
    "List every lane line in this road photo. Write each lane as x y points in whole pixels from top to "
    "bottom, using as many points as the lane needs. Separate lanes with a newline character, one lane per "
    "line, and output nothing else. Example with two lanes:\n"
    "526 326 421 502 134 710\n"
    "664 326 778 502 1072 710"
)
LABEL_FORMAT = (
    "one lane per newline-separated line: x1 y1 x2 y2 ... xk yk, k >= 2, whole pixels of that photo, "
    "points from top to bottom; lanes sorted by the x of their lowest point"
)
# Short names a spec may use for model.id. Any other value is used as a Hugging Face id.
MODELS = {
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
}


@dataclass
class LoraSettings:
    # Adapter rank. SmolVLM2 text attention gives 393,216 * r trainable weights.
    r: int
    # The adapter output is scaled by alpha / r.
    alpha: int
    # Dropout on the adapter input while training.
    dropout: float
    # Module names ending in one of these get an adapter, sorted.
    target_suffixes: list[str]
    # True: modules with "vision" or "visual" in their name get no adapter.
    skip_vision: bool
    # peft bias mode. "none" trains no biases.
    bias: str


@dataclass
class ModelSettings:
    # Full Hugging Face id, after short names are resolved.
    id: str
    # Commit hash of the weights, resolved from the local cache or the hub.
    revision: str
    # "lora" trains an adapter, "full" trains every weight, "base" trains nothing
    # and scores the hub weights at `revision`.
    method: str
    # Null unless method is "lora".
    lora: LoraSettings | None
    # "bf16" or "fp16" mixed precision, or "fp32" without CUDA. Under fp16 the
    # weights that train are kept in fp32, because fp16 cannot hold small updates.
    precision: str


@dataclass
class ImageSettings:
    # The most common photo size in split.json. Each photo's own size is in its row.
    reference_width: int
    reference_height: int
    # Tiles the processor cuts a photo of the reference size into, measured on a
    # blank photo. Null if the model does not tile.
    tiles: int | None
    # The processor's size["longest_edge"] that gives those tiles.
    longest_edge: int | None
    # Side of one tile in pixels.
    tile_edge: int | None


@dataclass
class DataSettings:
    # "split.json", copied byte for byte into the run folder.
    split_file: str
    # sha256 of that file. Scoring refuses a split.json with another hash.
    split_sha256: str
    # Percent of the train list asked for, and the photos that gave. 0 for a base run.
    train_percent: int
    train_photos: int
    # First N tune photos whose token loss is logged at each check. 0 for a base run.
    tune_loss_photos: int
    # First N tune photos answered at each check. Their mean cost picks the checkpoint. 0 for a base run.
    tune_cost_photos: int
    # The user text sent with every photo, and what the answer text means.
    prompt: str
    label_format: str


@dataclass
class JudgeSettings:
    # Tolerance for simplifying gold lanes. It changes the gold text, so split.json records it too.
    rdp_px: float
    # Spacing of the points each lane is sampled into before distances are taken.
    step_px: float
    # A point overlaps the other side when it is at most this far from it (precision and recall).
    radius_px: float
    # Cap on any one point's distance in the cost. Also the cost of an unreadable or empty answer.
    tau_px: float
    # The distance and cost, in words.
    distance: str
    cost: str
    # One sentence per number: why it has this value.
    why: dict[str, str]


@dataclass
class TrainSettings:
    seed: int
    # As given in the spec.
    epochs: float
    max_steps: int
    # Weight updates the run makes: max_steps, or
    # ceil(epochs * ceil(ceil(photos / batch_size) / grad_accum)).
    updates: int
    learning_rate: float
    # Hugging Face scheduler name.
    lr_scheduler: str
    # Share of updates spent warming up, and the step count it gave: max(1, int(updates * fraction)).
    warmup_fraction: float
    warmup_steps: int
    # Hugging Face optimizer name and its constants.
    optimizer: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    # Gradients are clipped to this norm.
    max_grad_norm: float
    batch_size: int
    grad_accum: int
    # batch_size * grad_accum.
    photos_per_update: int
    # Tune photos per loss step. The reported loss is the mean over those batches.
    eval_batch_size: int
    # Updates between tune checks and checkpoints. The final update is always checked.
    eval_steps: int
    # "lowest_tune_cost": the saved model is the checked update with the lowest mean tune cost.
    select_checkpoint: str
    # Null: tune cost is checked during training at every eval_steps. A number: a checkpoint
    # is saved every save_every updates and at the last one, and after training each is
    # answered on the tune_cost photos, newest first; only the best so far is kept.
    save_every: int | None = None
    # Null: training starts from the hub weights (and a fresh adapter). Otherwise
    # {"run": name, "weights_sha256": hash}: it starts from that run's saved weights.
    init_from: dict | None = None


@dataclass
class RunConfig:
    schema_version: int
    # <model>-<lora-rRaA or full>-d<percent>-t<tiles>-<first 8 of config_sha256>
    name: str
    # sha256 of this object as sorted JSON, with name and config_sha256 left out.
    config_sha256: str
    model: ModelSettings
    image: ImageSettings
    data: DataSettings
    judge: JudgeSettings
    # Null for a base run.
    train: TrainSettings | None


@dataclass
class DecodeSettings:
    # The model's whole context window. Each answer may write at most 8192 new tokens
    # (SmolVLM2's window), or less if the prompt leaves fewer. An answer that fills
    # that budget with no stop token counts in truncated_pct.
    context_tokens: int
    # False: greedy decoding, the same answer every time for the same batch.
    do_sample: bool
    num_beams: int
    # Photos answered at once, left padded. Changes the answers, so compare runs at one size.
    batch_size: int
    # Weight dtype while answering. A LoRA adapter is merged into the base weights first.
    precision: str


@dataclass
class CostStats:
    # Photos measured.
    photos: int
    # Per-photo cost L in pixels (see JudgeSettings.cost). Null when there were none.
    median_px: float | None
    mean_px: float | None
    # Percent of photos with cost at most 10 px and at most 25 px.
    within_10_pct: float | None
    within_25_pct: float | None


@dataclass
class ScoreResult:
    schema_version: int
    run: str
    # The exact text sent with every photo, and how the answers were judged.
    prompt: str
    judge: JudgeSettings
    # Must equal config.json.
    config_sha256: str
    # Must equal train.json, or the hub commit for a base run. Proves which weights answered.
    weights_sha256: str
    # "tune" or "test", and the first N photos of it in split.json.
    split: str
    photos: int
    # sha256 of the JSON list of scored photo paths. Equal across runs means the same photos.
    photos_sha256: str
    decode: DecodeSettings
    # Answers whose every line is an even count of at least 4 whole numbers inside the photo.
    readable_pct: float
    # Answers that filled the whole context window.
    truncated_pct: float
    # Per-photo cost L: all predicted lanes pooled against all gold lanes pooled.
    cost: CostStats
    # Mean over photos of the share of predicted points within radius_px of gold, and of
    # gold points within radius_px of a prediction; F is their harmonic mean.
    precision_pct: float
    recall_pct: float
    f_pct: float
    # Mean lanes per photo in the answers and in the gold text.
    lanes_pred_mean: float
    lanes_gold_mean: float
    # The jsonl with one {image, width, height, gold, raw, truncated} line per photo, and its hash.
    predictions_file: str
    predictions_sha256: str
    started_at: str
    finished_at: str
    # See environment() in spline/runs.py.
    environment: dict
    # Each automatic change made to fit this machine, for example a smaller batch.
    fallbacks: list[str] = field(default_factory=list)


@dataclass
class TrainResult:
    schema_version: int
    run: str
    # The exact text sent with every photo, and the judge that picked the checkpoint.
    prompt: str
    judge: JudgeSettings
    config_sha256: str
    trainable_params: int
    total_params: int
    # Updates actually made.
    updates: int
    # The checked update whose weights were saved, its mean tune cost, and its tune loss.
    best_step: int | None
    best_tune_cost_px: float | None
    best_tune_loss: float | None
    # Hugging Face mean training loss over the run.
    train_loss: float | None
    # Wall time of the process that finished. A resumed run counts only from the resume.
    runtime_s: float
    # Every Trainer log entry: loss, grad_norm, learning_rate, epoch, and at each check
    # eval_loss, eval_tune_cost_px, eval_tune_prec, and eval_tune_rec.
    log_history: list = field(default_factory=list)
    # Path inside the run folder, and its sha256 (for a full model, the hash of its file list).
    weights_file: str = ""
    weights_sha256: str = ""
    # Checkpoint folder the run restarted from, if any.
    resumed_from: str | None = None
    # Whether activations were recomputed. Same math, less memory.
    grad_checkpointing: bool = False
    # DataLoader processes actually used.
    workers: int = 0
    # Each automatic change made to fit this machine.
    fallbacks: list[str] = field(default_factory=list)
    # The spec file the run was started from, and the sha256 of its bytes. spec.json
    # in the run folder is the same spec, normalized.
    spec_file: str = ""
    spec_sha256: str = ""
    started_at: str = ""
    finished_at: str = ""
    environment: dict = field(default_factory=dict)
    # With train.save_every: one {step, tune_cost_px, tune_prec, tune_rec, better} per
    # checkpoint, newest first. "better" means it beat every newer one and was kept.
    sweep: list = field(default_factory=list)


def plain(obj):
    """``asdict``, leaving out optional fields that are still null.

    Fields added later default to null, so a run made before they existed keeps
    the same config.json, spec.json, and config hash.
    """
    if is_dataclass(obj):
        return {
            one.name: plain(getattr(obj, one.name))
            for one in fields(obj)
            if not (one.default is None and getattr(obj, one.name) is None)
        }
    if isinstance(obj, list):
        return [plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: plain(value) for key, value in obj.items()}
    return obj


def config_hash(config: RunConfig) -> str:
    """sha256 of the config with its own name and hash left out."""
    body = plain(config)
    body.pop("name")
    body.pop("config_sha256")
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def run_name(config: RunConfig) -> str:
    """A readable name with the hash on the end, for example ``...-lora-r16a32-d10-t13-9dc7ae4d``."""
    short = config.model.id.rstrip("/").split("/")[-1].lower()
    lora = config.model.lora
    tiles = f"-t{config.image.tiles}" if config.image.tiles else ""
    match config.model.method:
        case "lora":
            method = f"lora-r{lora.r}a{lora.alpha}-d{config.data.train_percent}"
        case "full":
            method = f"full-d{config.data.train_percent}"
        case "base":
            method = "base"
        case other:
            raise ValueError(f"unknown model.method {other!r}")
    return f"{short}-{method}{tiles}-{config.config_sha256[:8]}"


def finish(config: RunConfig) -> RunConfig:
    """Fill in the hash and the name once every other field is set."""
    config.config_sha256 = config_hash(config)
    config.name = run_name(config)
    return config


def from_dict(data: dict) -> RunConfig:
    """Rebuild a RunConfig from a config.json, refusing other schema versions."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config.json is schema {data.get('schema_version')}, this code reads {SCHEMA_VERSION}")
    model = dict(data["model"])
    model["lora"] = LoraSettings(**model["lora"]) if model["lora"] else None
    return RunConfig(
        schema_version=data["schema_version"],
        name=data["name"],
        config_sha256=data["config_sha256"],
        model=ModelSettings(**model),
        image=ImageSettings(**data["image"]),
        data=DataSettings(**data["data"]),
        judge=JudgeSettings(**data["judge"]),
        train=TrainSettings(**data["train"]) if data["train"] else None,
    )
