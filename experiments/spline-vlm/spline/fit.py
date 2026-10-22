"""Fine-tune on the training photos and keep one restart checkpoint."""

import json
import math
import shutil
import traceback

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from spline.data import load_dataset
from spline.model import load_for_training, make_collate, native_bf16
from spline.runs import append_error, run_dir


def train(args, processor) -> None:
    try:
        _train(args, processor)
    except Exception:
        append_error(args.output, args.run, traceback.format_exc())
        raise


def _train(args, processor) -> None:
    folder = run_dir(args.output, args.run)
    if (folder / "model" / "run.json").exists():
        print(f"run {args.run} is already trained; delete {folder / 'model'} to train it again")
        return
    rows = load_dataset(args.output, args.data_percent)["train"]
    photos_per_update = args.batch_size * args.grad_accum
    print(
        f"training run {args.run}: {len(rows)} photos ({args.data_percent}% of each split), "
        f"{args.tiles} tiles per photo, {args.batch_size} at a time"
    )
    model = load_for_training(processor, args.model, args.lora)
    updates = args.max_steps if args.max_steps > 0 else math.ceil(len(rows) / photos_per_update * args.epochs)
    checkpoints = folder / "checkpoints"
    bf16 = native_bf16()
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(checkpoints),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=max(1, int(updates * 0.03)),
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=1,
            bf16=bf16,
            fp16=not bf16,
            gradient_checkpointing=args.grad_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_checkpointing else None,
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            dataloader_persistent_workers=args.workers > 0,
            report_to="none",
        ),
        train_dataset=rows,
        data_collator=make_collate(processor),
    )
    resume = get_last_checkpoint(str(checkpoints)) if checkpoints.is_dir() else None
    if resume:
        print("resuming from", resume)
    trainer.train(resume_from_checkpoint=resume)
    saved = folder / "model"
    trainer.model.save_pretrained(saved)
    processor.save_pretrained(saved)
    run = {"model": args.model, "lora": args.lora, "data_percent": args.data_percent, "tiles": args.tiles}
    (saved / "run.json").write_text(json.dumps(run) + "\n")
    shutil.rmtree(checkpoints, ignore_errors=True)
    (folder / "test.jsonl").unlink(missing_ok=True)
    print("saved", saved)
