"""Turn a photo and its spline text into model tensors, and load the weights."""

from dataclasses import replace

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from spline.lanes import HEIGHT, WIDTH

PROMPT = "Generate a list of splines for all road and taxi lines in this image"
MODELS = {
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
}
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def resolve_model(name: str) -> str:
    return MODELS.get(name.strip(), name.strip())


def chat(image: Image.Image, answer: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def native_bf16() -> bool:
    # The default check counts emulated bf16, which runs about 10x slower on Turing cards.
    return torch.cuda.is_bf16_supported(including_emulation=False)


def pick_dtype(full_training: bool) -> torch.dtype:
    if native_bf16():
        return torch.bfloat16
    # Turing cards: fp16 mixed precision cannot step fp16 weights, so trained weights stay fp32.
    return torch.float32 if full_training else torch.float16


def _template_kwargs(processor) -> dict:
    video = getattr(processor, "video_processor", None)
    if video is None:
        return {}
    return {"num_frames": video.num_frames, "fps": video.fps}


def _tiles_for_edge(image_processor, edge: int | None) -> int:
    photo = Image.new("RGB", (WIDTH, HEIGHT))
    size = {"size": {"longest_edge": edge}} if edge else {}
    return int(image_processor([[photo]], return_tensors="pt", **size)["pixel_values"].shape[1])


def tile_options(processor) -> dict[int, int]:
    """Tiles per LLAMAS photo -> long-side length that produces them. Empty if the model does not tile."""
    image_processor = getattr(processor, "image_processor", None)
    tile_side = (getattr(image_processor, "max_image_size", None) or {}).get("longest_edge")
    if not tile_side:
        return {}
    options: dict[int, int] = {}
    for multiple in range(1, 9):
        options.setdefault(_tiles_for_edge(image_processor, tile_side * multiple), tile_side * multiple)
    return options


def current_tiles(processor) -> int | None:
    image_processor = getattr(processor, "image_processor", None)
    if not tile_options(processor):
        return None
    return _tiles_for_edge(image_processor, None)


def set_tiles(processor, tiles: int) -> None:
    options = tile_options(processor)
    if not options:
        raise ValueError("this model does not split photos into tiles, so SPLINE_TILES cannot be used")
    if tiles not in options:
        raise ValueError(f"{tiles} tiles is not possible for {WIDTH}x{HEIGHT} photos. Choose one of {sorted(options)}")
    image_processor = processor.image_processor
    image_processor.size = replace(image_processor.size, longest_edge=options[tiles])


def load_processor(model_id: str, tiles: int | None):
    processor = AutoProcessor.from_pretrained(model_id)
    if tiles:
        set_tiles(processor, tiles)
    return processor


def encode(processor, texts: list[str], images: list[Image.Image], padding: bool):
    # One image list per text. A flat list is read as several images for a single sample.
    return processor(text=texts, images=[[image] for image in images], padding=padding, return_tensors="pt")


class SplineCollator:
    """Builds a training batch. A class, not a closure, so DataLoader worker processes can pickle it."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.quiet = _template_kwargs(processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"

    def __call__(self, batch: list[list[str]]) -> dict:
        images, full_texts, prompt_texts = [], [], []
        for path, answer in batch:
            image = Image.open(path).convert("RGB")
            images.append(image)
            full_texts.append(
                self.processor.apply_chat_template(chat(image, answer), tokenize=False, processor_kwargs=self.quiet)
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    chat(image), tokenize=False, add_generation_prompt=True, processor_kwargs=self.quiet
                )
            )
        full = encode(self.processor, full_texts, images, padding=len(batch) > 1)
        prompt = encode(self.processor, prompt_texts, images, padding=True)
        labels = full["input_ids"].clone()
        prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()
        full_lens = full["attention_mask"].sum(dim=1).tolist()
        for row, (prompt_len, full_len) in enumerate(zip(prompt_lens, full_lens)):
            if prompt_len >= full_len:
                raise ValueError("the prompt is not shorter than the spline answer")
            labels[row, : int(prompt_len)] = -100
            labels[row, int(full_len) :] = -100
        out = dict(full.items())
        out["labels"] = labels
        return out


def make_collate(processor) -> SplineCollator:
    return SplineCollator(processor)


def lora_targets(model) -> list[str]:
    names = []
    for name, _module in model.named_modules():
        if name.split(".")[-1] not in LORA_SUFFIXES:
            continue
        if "vision" in name or "visual" in name:
            continue
        names.append(name)
    if not names:
        raise RuntimeError("no attention projections found for LoRA")
    return names


def load_for_training(processor, model_id: str, lora: bool):
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=pick_dtype(not lora))
    if lora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=lora_targets(model),
                task_type="CAUSAL_LM",
            ),
        )
        model.enable_input_require_grads()
        if not native_bf16():
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
    model.config.use_cache = False
    return model


def load_for_scoring(saved, run: dict):
    processor = AutoProcessor.from_pretrained(saved)
    if run["lora"]:
        base = AutoModelForMultimodalLM.from_pretrained(run["model"], dtype=pick_dtype(False))
        # Folding the adapter into the base weights removes the extra LoRA matmuls from every generated token.
        model = PeftModel.from_pretrained(base, saved).merge_and_unload()
    else:
        model = AutoModelForMultimodalLM.from_pretrained(saved, dtype=pick_dtype(False))
    model.config.use_cache = True
    model.to("cuda").eval()
    return processor, model


def answer_photos(processor, model, paths: list[str]) -> list[str]:
    images = [Image.open(path).convert("RGB") for path in paths]
    quiet = _template_kwargs(processor)
    prompts = [
        processor.apply_chat_template(chat(image), tokenize=False, add_generation_prompt=True, processor_kwargs=quiet)
        for image in images
    ]
    # Generation appends to the right end, so a padded batch must be padded on the left.
    processor.tokenizer.padding_side = "left"
    inputs = encode(processor, prompts, images, padding=len(paths) > 1).to("cuda")
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    texts = processor.batch_decode(tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return [text.strip() for text in texts]
