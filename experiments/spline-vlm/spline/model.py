"""Photos to model tensors, tile control, and loading weights for training or answering.

Every model the project uses goes through this file, so the rules for precision,
devices, tiles, prompt masking, and generation live in one place:

* :func:`precision` picks bf16, fp16, or fp32 for this machine and logs why.
* :func:`load_processor` sets how many tiles each photo is cut into.
* :class:`SplineCollator` builds training batches whose loss covers only the answer.
* :func:`load_for_training` and :func:`load_for_scoring` load a run's weights.
* :class:`PromptCollator` and :func:`answer_batch` turn photos into answer text.
"""

import logging
from dataclasses import replace
from functools import cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from spline import PINNED_GPU
from spline.config import (
    MODELS,
    DecodeSettings,
    ImageSettings,
    LoraSettings,
    RunConfig,
)
from spline.lanes import readable_prefix

log = logging.getLogger("spline.model")
# SmolVLM2's whole window. A 128k-context model still stops here, so a runaway cannot fill it.
MAX_NEW_TOKENS = 8192


def resolve_model(name: str) -> str:
    """A short name from MODELS becomes its Hugging Face id; anything else is used as given."""
    return MODELS.get(name.strip(), name.strip())


def resolve_revision(model_id: str, revision: str) -> str:
    """The commit hash behind a branch or tag.

    The local Hugging Face cache is asked first, so a machine without network
    still resolves a model it has downloaded. Only a cache miss goes online.
    """
    try:
        path = snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception as error:  # noqa: BLE001  any cache miss falls through to the hub
        log.info("%s@%s is not cached (%s); asking the hub", model_id, revision, type(error).__name__)
        path = snapshot_download(model_id, revision=revision)
    commit = Path(path).name
    log.debug("%s@%s resolved to commit %s", model_id, revision, commit)
    return commit


def load_image(path: str, width: int, height: int) -> Image.Image:
    """An RGB copy of a photo, refused if its size is not the one recorded in the split.

    The file is closed before returning, so handles do not pile up.
    """
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError(f"{path} is {image.size[0]}x{image.size[1]}, split.json recorded {width}x{height}")
        return image.convert("RGB")


def chat(image: Image.Image, prompt: str, answer: str | None = None) -> list[dict]:
    """The chat messages for one photo: the prompt, and the answer when training."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def device() -> str:
    """Where weights and inputs go: the GPU when CUDA is present, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def precision() -> str:
    """bf16 on cards that run it natively, fp16 on older CUDA cards, fp32 without CUDA.

    Cached so the choice and its log line happen once per process.
    """
    if not torch.cuda.is_available():
        log.warning("no CUDA GPU found: running on CPU in fp32, which is very slow")
        return "fp32"
    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            "more than one GPU is visible, and the Trainer would split batches across "
            "them. Set CUDA_VISIBLE_DEVICES to one GPU."
        )
    name = torch.cuda.get_device_name(0)
    if PINNED_GPU:
        log.info("CUDA_VISIBLE_DEVICES was unset: using GPU 0 (%s) only", name)
    # The default check counts emulated bf16, which runs about 10x slower on Turing cards.
    if torch.cuda.is_bf16_supported(including_emulation=False):
        log.info("%s runs bf16 natively: using bf16 mixed precision", name)
        return "bf16"
    log.warning("%s has no native bf16: using fp16 mixed precision with fp32 trainable weights", name)
    return "fp16"


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _template_kwargs(processor) -> dict:
    """Video settings some processors want even for still images; passing them silences a warning."""
    video = getattr(processor, "video_processor", None)
    if video is None:
        return {}
    return {"num_frames": video.num_frames, "fps": video.fps}


def _tiles_for_edge(image_processor, edge: int | None, size: tuple[int, int]) -> int:
    """Tiles the processor makes from a blank photo of ``size`` (width, height) at a given long side."""
    photo = Image.new("RGB", size)
    resize = {"size": {"longest_edge": edge}} if edge else {}
    pixels = image_processor([[photo]], return_tensors="pt", **resize)["pixel_values"]
    return int(pixels.shape[1])


def _tile_edge(processor) -> int | None:
    image_processor = getattr(processor, "image_processor", None)
    return (getattr(image_processor, "max_image_size", None) or {}).get("longest_edge")


def tile_options(processor, size: tuple[int, int]) -> dict[int, int]:
    """Tiles per photo of ``size`` mapped to the long-side length that produces them.

    SmolVLM2 resizes a photo so its long side is a multiple of the tile side
    (384 px) and cuts it into tiles plus one whole-photo thumbnail. Different
    multiples can give the same count; the smallest one wins. Empty if the model
    does not tile.
    """
    tile_side = _tile_edge(processor)
    if not tile_side:
        return {}
    options: dict[int, int] = {}
    for multiple in range(1, 9):
        edge = tile_side * multiple
        options.setdefault(_tiles_for_edge(processor.image_processor, edge, size), edge)
    return options


def load_processor(model_id: str, revision: str, tiles: int | None, size: tuple[int, int]):
    """The model's processor, set to cut a photo of ``size`` into ``tiles`` tiles (None keeps its own)."""
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if tiles is None:
        return processor
    options = tile_options(processor, size)
    if not options:
        raise ValueError("this model does not split photos into tiles, so image.tiles must be null")
    if tiles not in options:
        raise ValueError(
            f"{tiles} tiles is not possible for {size[0]}x{size[1]} photos. Choose one of {sorted(options)}"
        )
    image_processor = processor.image_processor
    image_processor.size = replace(image_processor.size, longest_edge=options[tiles])
    log.debug("tiles=%d: longest_edge=%d", tiles, options[tiles])
    return processor


def image_settings(processor, size: tuple[int, int]) -> ImageSettings:
    """Measured from the processor itself on a blank photo of ``size``, so the record is what the model saw."""
    width, height = size
    if not tile_options(processor, size):
        return ImageSettings(
            reference_width=width, reference_height=height, tiles=None, longest_edge=None, tile_edge=None
        )
    image_processor = processor.image_processor
    return ImageSettings(
        reference_width=width,
        reference_height=height,
        tiles=_tiles_for_edge(image_processor, None, size),
        longest_edge=int(image_processor.size.longest_edge),
        tile_edge=int(_tile_edge(processor)),
    )


def encode(processor, texts: list[str], images: list[Image.Image], padding: bool):
    """Tokens and pixels for a batch."""
    # One image list per text. A flat list is read as several images for a single sample.
    return processor(text=texts, images=[[image] for image in images], padding=padding, return_tensors="pt")


class SplineCollator:
    """Builds a training batch whose loss covers only the answer tokens.

    The prompt is encoded twice, alone and with the answer, and every token up to
    the prompt's length is masked with -100 so the model is not trained to repeat
    the prompt. Padding is on the right so those prompt lengths line up. A class,
    not a closure, so DataLoader worker processes can pickle it.
    """

    def __init__(self, processor, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt
        self.quiet = _template_kwargs(processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"

    def __call__(self, batch: list[list]) -> dict:
        images, full_texts, prompt_texts = [], [], []
        for path, answer, width, height in batch:
            image = load_image(path, width, height)
            images.append(image)
            full_texts.append(
                self.processor.apply_chat_template(
                    chat(image, self.prompt, answer), tokenize=False, processor_kwargs=self.quiet
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    chat(image, self.prompt), tokenize=False, add_generation_prompt=True, processor_kwargs=self.quiet
                )
            )
        full = encode(self.processor, full_texts, images, padding=len(batch) > 1)
        prompt = encode(self.processor, prompt_texts, images, padding=True)
        labels = full["input_ids"].clone()
        prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()
        full_lens = full["attention_mask"].sum(dim=1).tolist()
        for row, (prompt_len, full_len) in enumerate(zip(prompt_lens, full_lens, strict=True)):
            if prompt_len >= full_len:
                raise ValueError(f"the prompt is not shorter than the spline answer for {batch[row][0]}")
            labels[row, : int(prompt_len)] = -100
            labels[row, int(full_len) :] = -100
        out = dict(full.items())
        out["labels"] = labels
        return out


def lora_targets(model, lora: LoraSettings) -> list[str]:
    """Full names of the modules that get an adapter."""
    names = []
    for name, _module in model.named_modules():
        if name.split(".")[-1] not in lora.target_suffixes:
            continue
        if lora.skip_vision and ("vision" in name or "visual" in name):
            continue
        names.append(name)
    if not names:
        raise RuntimeError(f"no modules end in {lora.target_suffixes}")
    log.debug("LoRA on %d modules, first %s, last %s", len(names), names[0], names[-1])
    return names


def load_for_training(config: RunConfig, start: Path | None = None):
    """The base model with the run's adapter added (or every weight trainable for "full").

    ``start`` is another run's saved weights to continue from: its adapter, or its full model.
    """
    spec = config.model
    # fp16 mixed precision cannot step fp16 weights, so weights that train must be fp32.
    full_dtype = torch.float32 if spec.precision == "fp16" else _dtype(spec.precision)
    dtype = _dtype(spec.precision) if spec.lora else full_dtype
    if start and not spec.lora:
        log.info("loading %s in %s for training", start, dtype)
        model = AutoModelForMultimodalLM.from_pretrained(start, dtype=dtype)
    else:
        log.info("loading %s@%s in %s for training", spec.id, spec.revision[:8], dtype)
        model = AutoModelForMultimodalLM.from_pretrained(spec.id, revision=spec.revision, dtype=dtype)
    if spec.lora and start:
        log.info("continuing from the adapter in %s", start)
        model = PeftModel.from_pretrained(model, start, is_trainable=True)
    elif spec.lora:
        lora = spec.lora
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora.r,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=lora_targets(model, lora),
                bias=lora.bias,
                task_type="CAUSAL_LM",
            ),
        )
    if spec.lora:
        # Gradients must flow back through the frozen embeddings to reach the adapters.
        model.enable_input_require_grads()
        if spec.precision == "fp16":
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
    model.config.use_cache = False
    return model


def load_base(model_id: str, revision: str, score_precision: str):
    """Hub weights at a commit with no training, ready to answer photos."""
    log.info("loading base %s@%s in %s for answering", model_id, revision[:8], score_precision)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, dtype=_dtype(score_precision))
    model.config.use_cache = True
    return model.to(device()).eval()


def load_for_scoring(config: RunConfig, saved: Path, score_precision: str):
    """A run's weights, ready to answer photos. A base run uses the hub weights at its commit."""
    spec = config.model
    dtype = _dtype(score_precision)
    match spec.method:
        case "base":
            return load_base(spec.id, spec.revision, score_precision)
        case "lora":
            log.info("loading %s from %s in %s for answering", config.name, saved, dtype)
            base = AutoModelForMultimodalLM.from_pretrained(spec.id, revision=spec.revision, dtype=dtype)
            # Folding the adapter into the base weights removes the extra LoRA matmuls from every token.
            model = PeftModel.from_pretrained(base, saved).merge_and_unload()
        case "full":
            log.info("loading %s from %s in %s for answering", config.name, saved, dtype)
            model = AutoModelForMultimodalLM.from_pretrained(saved, dtype=dtype)
        case other:
            raise ValueError(f"unknown model.method {other!r}")
    model.config.use_cache = True
    model.to(device()).eval()
    return model


def _stop_ids(model, processor) -> set[int]:
    """Token ids that end an answer. An answer without one was cut off."""
    ids = model.generation_config.eos_token_id
    found = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if processor.tokenizer.eos_token_id is not None:
        found.add(processor.tokenizer.eos_token_id)
    return found


class PromptCollator:
    """Encodes a batch of [path, gold, width, height] rows for generation, left padded."""

    def __init__(self, processor, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt
        self.quiet = _template_kwargs(processor)
        # Generation appends to the right end, so a padded batch must be padded on the left.
        processor.tokenizer.padding_side = "left"

    def __call__(self, batch: list[list]) -> tuple[list[list], dict]:
        images = [load_image(path, width, height) for path, _gold, width, height in batch]
        prompts = [
            self.processor.apply_chat_template(
                chat(image, self.prompt), tokenize=False, add_generation_prompt=True, processor_kwargs=self.quiet
            )
            for image in images
        ]
        return batch, dict(encode(self.processor, prompts, images, padding=len(batch) > 1))


def context_tokens(model_id: str, revision: str) -> int:
    """The model's whole context window in tokens, read from its config without loading weights."""
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    text = getattr(config, "text_config", None) or config
    return int(text.max_position_embeddings)


class StopUnreadable(StoppingCriteria):
    """Ends an answer as soon as it can no longer be read as lanes. read_lanes would cost it tau_px anyway."""

    def __init__(self, processor, start: int, sizes: list[tuple[int, int]]) -> None:
        self.processor = processor
        self.start = start
        self.sizes = sizes
        self.stopped = [False] * len(sizes)

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        texts = self.processor.batch_decode(input_ids[:, self.start :], skip_special_tokens=True)
        done = [not readable_prefix(text, *size) for text, size in zip(texts, self.sizes, strict=True)]
        self.stopped = [was or now for was, now in zip(self.stopped, done, strict=True)]
        return torch.tensor(done, device=input_ids.device)


def answer_batch(
    processor, model, inputs: dict, decode: DecodeSettings, sizes: list[tuple[int, int]]
) -> list[tuple[str, bool]]:
    """Each photo's answer text, and whether it filled the answer budget with no stop token.

    ``sizes`` is each photo's (width, height). An answer stops early once it is unreadable.
    The budget is ``MAX_NEW_TOKENS`` (8192, SmolVLM2's window), or whatever the context
    window still has if that is smaller.
    """
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    # Left padding gives every row the same prompt length, so one limit fits the batch.
    start = inputs["input_ids"].shape[1]
    room = min(decode.context_tokens - start, MAX_NEW_TOKENS)
    if room < 1:
        raise ValueError(f"the prompt fills the {decode.context_tokens}-token context window")
    unreadable = StopUnreadable(processor, start, sizes)
    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            max_new_tokens=room,
            do_sample=decode.do_sample,
            num_beams=decode.num_beams,
            # Training sets config.use_cache off; answering mid-training still wants the cache.
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([unreadable]),
        )
    new = tokens[:, start:]
    stops = _stop_ids(model, processor)
    texts = processor.batch_decode(new, skip_special_tokens=True)
    cut = [
        not early and not any(int(token) in stops for token in row)
        for row, early in zip(new, unreadable.stopped, strict=True)
    ]
    return [(text.strip(), truncated) for text, truncated in zip(texts, cut, strict=True)]
