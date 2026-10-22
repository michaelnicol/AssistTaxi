# Spline experiment

This folder trains a vision-language model to list road lines as splines, then scores those lines in pixels on photos the model never trained on. The older judge loop is in `mothball/`.

The model is one switch. `--model` or `SPLINE_MODEL` takes a short name or any Hugging Face id. `--lora` or `SPLINE_LORA=1` trains an adapter. Leave that off to train every weight. `--tiles` or `SPLINE_TILES` sets how many 384 px tiles each photo is cut into. SmolVLM2 uses 13 by default, and 1, 5, 7, 16, 25, 29, and 41 also work for LLAMAS photos. Any other number stops with the list of valid counts.

Each mix of model, LoRA or full, data percent, and tiles is one run in `output/runs/<name>/`, for example `smolvlm2-2.2b-instruct-lora-d10-t16`. That folder holds the restart checkpoint, the saved `model/` with its `run.json`, and the scored `test.jsonl`.

| Short name | Weights on this machine |
|---|---|
| `smolvlm2-2.2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `phi4-multimodal` | `microsoft/Phi-4-multimodal-instruct` |
| `gemma3-12b` | `google/gemma-3-12b-it` |
| `internvl3-14b` | `OpenGVLab/InternVL3-14B` |

The default is `smolvlm2-2.2b` with every weight trained. The LLAMAS zips stay at `/home/michaelnicol/github/LLAMAS/`.

Labeled photos are shuffled with seed 0 and split 80% training, 10% tuning, 10% testing. The lists are written to `output/splits/`. Training updates weights on the training photos and measures loss on the tuning photos. The judge later sees only the tuning photos. The test photos are not used. `--fresh` writes the split again.

## Layout

| Path | What it is |
|---|---|
| `spline_train.py` | Command line: `train` or `score` |
| `spline_queue.py` | Runs `output/queue.txt` one line at a time |
| `spline/lanes.py` | Lane text and pixel error |
| `spline/data.py` | Full split and the saved percent subset |
| `spline/model.py` | Photos to tensors, and loading weights |
| `spline/fit.py` | The training loop |
| `spline/score.py` | Test-set measurement |
| `spline/runs.py` | Run names, run folders, and `errors.md` |
| `requirements-train.txt` | Packages for the student model |
| `train-venv/` | Environment that trains the student model |
| `output/` | `split.json`, `dataset-<percent>.json`, `errors.md`, and `runs/` |
| `mothball/` | The earlier judge loop, kept but not used |

## Models

- Student: `--model` / `SPLINE_MODEL`. Weights, or the adapter with `--lora`, go to `output/runs/<name>/model`.
- Judge: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`

The prompt is `Generate a list of splines for all road and taxi lines in this image`. Each answer line is one lane, `y_top y_bottom x1 x2 x3 x4`: the cubic through `x1`..`x4` at four evenly spaced heights from `y_top` to `y_bottom`, all as fractions of the image to three decimals (about 1 px). Only the part of a lane inside the photo is kept. Lanes are left to right by where they meet the bottom. Polynomial coefficients were dropped because the gold coefficients reached 10^7 and rounding moved lanes by hundreds of pixels.

## Run

From this directory:

```bash
train-venv/bin/python spline_train.py train --lora --data-percent 10
train-venv/bin/python spline_train.py score --lora --data-percent 10
SPLINE_TILES=16 train-venv/bin/python spline_train.py train --lora --data-percent 10
SPLINE_TILES=16 train-venv/bin/python spline_train.py score --lora --data-percent 10
```

`score` takes the same model, LoRA, percent, and tiles settings as `train` so it finds the same run folder.

To run several in a row, put one line of flags per run in `output/queue.txt` and start `spline_queue.py` in tmux. It trains and scores each line top to bottom, logs to `output/queue.log`, and moves lines to `queue-done.txt` or `queue-failed.txt`. Lines can be added while it runs. If it is killed, start it again and the current line resumes from its checkpoint. A run whose model already exists is only scored. On the 48 GB card, 29 and 41 tiles need `--batch-size 1 --grad-accum 8`. `--data-percent 10` keeps the first 10% of the training, tuning, and test lists and writes `output/dataset-10.json`. The same file is reused for every model. A checkpoint is kept every `--save-steps` updates, and running the same command again resumes from it.

Speed settings, measured on the Quadro RTX 8000 with SmolVLM2 and LoRA:

| Flag | Default | Why |
|---|---|---|
| `--batch-size` | 2 | Photos per GPU step. Time per photo is the same from 2 to 16, so a bigger batch does not help. |
| `--grad-accum` | 4 | Steps per weight update, so each update still sees 8 photos. |
| `--grad-checkpointing` | off | Off is 27% faster at batch 2 and peaks at 26.5 GB. Turn it on for full fine-tuning or bigger models, or they run out of memory. |
| `--workers` | 4 | CPU processes that prepare the next batches while the GPU trains. |
| `--score-batch` | 16 | Test photos answered at once. |

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation. The card is Turing, which has no hardware bf16, so the code uses fp16. PyTorch's default bf16 check counts slow emulation as support, and trusting it made training about 7 times slower.

Color frames must be extracted to `/home/michaelnicol/github/LLAMAS/color_images` before training. `labels.zip` is already next to that path.
