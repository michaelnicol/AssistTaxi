# Spline experiment

This folder trains a vision-language model to list road lines as polylines in whole pixels, then scores them on photos the model never trained on by how well all predicted lines overlap all gold lines. It is built for a grid search over LoRA settings and tile counts, so every run is trained, picked, and scored the same way, and every setting that can change a result is written to JSON in the run folder. An untrained base model can go through the same scoring as a baseline. Runs from before 2026-10-23 used older formats and are in `mothball/` (the cubic-spline runs are in `mothball/runs-2026-10-23-v3/`).

## Protocol

LLAMAS labels from `/home/michaelnicol/github/LLAMAS/labels.zip` with a photo in `color_images/` are shuffled with seed 0 and cut 80% train, 10% tune, 10% test into `output/split.json`. That file is made once and never rewritten. Every run copies it byte for byte, and its sha256 is part of the run's config.

| Split | Used for | Same for every run |
|---|---|---|
| train | The first `data.train_percent` of the list updates the weights | Nested: 10% is a prefix of 25%, and of 50% |
| tune | The first `data.tune_cost_photos` (100) are answered at each check, and their mean cost picks the saved checkpoint. The first `data.tune_loss_photos` (256) give the token loss logged beside it. The first `score.tune_photos` (500) are answered to compare runs | Yes |
| test | The first `score.test_photos` (1000) are answered once, after training. Report them, but do not pick a run with them. A base run is scored on these only | Yes |

A grid is fair only when the runs read one `output/split.json`. The shuffle happens once, with seed 0, and a later run reuses the file instead of drawing again. Each run records `data.split_sha256` in `config.json` and the exact rows in `data.json`. Two runs trained on the same photos when those hashes match and `data.train_percent` matches. Their tune and test scores use the same photos when the `photos_sha256` in each `score-<split>.json` matches, and that includes base runs.

A smaller percent is the start of a larger one, so a 10% run's training photos are a prefix of a 25% run's. Tune and test do not shrink with the train percent. `train.epochs` does not draw new photos either. A second epoch shuffles that same train list again, using `train.seed`.

Every `train.eval_steps` updates, and on the last update, the run measures tune loss and answers the tune-cost photos. The saved model is the checkpoint with the lowest mean tune cost. To choose a grid winner, use `report`, which ranks runs by their median tune cost.

With `train.save_every`, no tune photos are answered while training, which makes training much faster. A checkpoint is saved every `save_every` updates and on the last update. After training, each checkpoint answers the tune-cost photos, newest first. A checkpoint is kept only if its mean cost is lower than every newer checkpoint's; the rest are deleted as the sweep goes. A killed sweep resumes at the checkpoint it was on. With `train.init_from`, a run starts from another run's saved weights, so `"epochs": 2` there means two more passes with a fresh learning-rate schedule. The source run keeps its own folder and scores.

### Prompt and answer format

Every photo in every phase (training, tune, test, and base runs) gets the same prompt, `PROMPT` in `spline/config.py`. It never states the photo size, so the model has to read the scale from the image. The run records it in `config.json` (`data.prompt`, part of the hash), `train.json`, `score-<split>.json`, the jsonl header, and the first INFO line of every train and score log:

```text
List every lane line in this road photo. Write each lane as x y points in whole pixels from top to bottom, using as many points as the lane needs. Separate lanes with a newline character, one lane per line, and output nothing else. Example with two lanes:
526 326 421 502 134 710
664 326 778 502 1072 710
```

Each answer line is one lane, `x1 y1 x2 y2 ... xk yk`, with k of 2 or more, in whole pixels of that photo (0 ≤ x < width, 0 ≤ y < height). There is no limit on points per lane, lanes per answer, or answer length other than the model's context window: each answer may use every token the prompt leaves free, up to 8192 new tokens (SmolVLM2's whole window). A model with a longer context still stops there, so a runaway cannot fill 128k tokens. Generation also stops an answer early once its text can no longer be read as lanes (a character other than digits, spaces, and newlines; a number past the photo; or a finished line with an odd count or fewer than 4 numbers), since such an answer already costs τ. Gold lanes come from the LLAMAS marker endpoints inside the photo. Endpoints on one pixel row are averaged, the points are ordered top to bottom, and Ramer-Douglas-Peucker keeps only the points needed to stay within `judge.rdp_px` (2 px) of the rest, so a straight lane has 2 points and a curve has more. Gold lanes are sorted by the x of their lowest point. Each photo's width and height are read from its file and saved in its `split.json` row.

### Cost

Each lane is sampled every `judge.step_px` (2 px) along its segments. All predicted lanes are pooled into one point set P and all gold lanes into one set G, so lanes are never matched one to one. With d(p, S) the distance from p to the nearest point of S and the cap τ = `judge.tau_px` (50 px), the cost of one photo is

```text
L = ½ · ( mean over p in P of min(d(p, G), τ)  +  mean over g in G of min(d(g, P), τ) )
```

The first term grows with extra or misplaced lines, and the second with missed lines. L = 0 is a perfect overlap. An answer with any line that is not an even count of at least 4 whole numbers inside the photo, or with no lanes, costs L = τ. Precision is the share of predicted points within `judge.radius_px` (15 px, half the 30 px lane width of the CULane and LLAMAS metric) of gold, recall is the share of gold points within that radius of a prediction, and F is their harmonic mean. A score reports the median and mean of L over photos, the mean precision and recall, and F of those means. The judge settings and a sentence on why each has its value are in `config.json` under `judge`, so a run says exactly how it was judged.

Seeds fix the LoRA start, dropout, and photo order. GPU kernels are not fully deterministic, so two trainings with one config match in the first updates and then differ in about the fourth digit of the loss.

Scoring answers photos in fixed batches of `score.batch_size`. Every prompt has the same length, but the batch size still changes the greedy answers: 128 tune photos answered at batch 16 and at batch 32 gave the same text for only 38 of them. `report` warns when compared runs used different decode settings. A resumed score drops any partly written batch, so its batches match a run that was never stopped.

## Run

From this directory:

Every run starts from one JSON spec file. There are no other flags:

```bash
train-venv/bin/python spline_train.py train specs/smolvlm2-r16-t13.json
train-venv/bin/python spline_train.py score specs/smolvlm2-r16-t13.json tune
train-venv/bin/python spline_train.py score specs/smolvlm2-r16-t13.json test
train-venv/bin/python spline_train.py report
train-venv/bin/python spline_train.py score specs/base/smolvlm2-t5.json test   # baseline, no training
```

`score` takes the spec that trained the run. It stops if the spec, `config.json`, `split.json`, or weights do not match what the run recorded. `specs/smoke.json` is a 6-update run on 1% of the data that checks the whole pipeline. A base spec (`"method": "base"`) is never trained: `score SPEC test` makes its run folder and answers the test photos with the hub weights at the resolved commit, using the same split, prompt, decode, and judge as trained runs. `train` refuses a base spec, and the queue only scores it on test.

A grid file expands one base spec into many. Each spec goes to `specs/<grid name>/`, and its path is appended to `output/queue.txt`. The queue then trains each one and scores it on tune and then on test:

```bash
train-venv/bin/python spline_queue.py grid specs/grids/lora-vs-tiles.json
tmux new-session -d -s spline-queue -- train-venv/bin/python spline_queue.py
```

Each grid is built into run configs before anything is queued, so a grid that makes the same run twice, or asks for an impossible tile count, is refused. `queue.txt` is never rewritten. Finished lines go to `queue-done.txt` and failed ones to `queue-failed.txt`, and a line in either is skipped. Delete it from `queue-failed.txt` to retry it. A killed runner resumes the same line from its last checkpoint. On the 48 GB card, 29 and 41 tiles need `"batch_size": 1, "grad_accum": 8`, which still makes 8 photos per update.

## Spec

Every key is required except `train.save_every` and `train.init_from`, and an unknown key is an error, so the file alone says how a run was made. Left out, those two are null, and the run is made the same way it was before they existed. `spline/spec.py` holds the dataclasses and checks. Settings that change the weights are resolved into `config.json` and hashed into the run name. `train.workers`, `train.grad_checkpointing`, and `score.*` do not change the weights, so they are not in that hash. They are still fixed per run: the run folder keeps the spec as `spec.json`, and a second spec that makes the same run with other values is refused.

| Key | Example | Meaning |
|---|---|---|
| `model.id` | `smolvlm2-2.2b` | Short name from `spline/config.py` `MODELS` (`smolvlm2-256m`, `smolvlm2-500m`, `smolvlm2-2.2b`, `qwen3-vl-2b`, `qwen3-vl-8b`, `gemma3-12b`), or a Hugging Face id |
| `model.revision` | `main` | Branch, tag, or commit. Stored as the resolved commit |
| `model.method` | `lora` | `lora` trains an adapter, `full` trains every weight, `base` trains nothing and is scored on test only. With `base`, `data` and `train` are null |
| `model.lora` | object | Null unless `lora`. `r`, `alpha`, `dropout`, `target_suffixes`, `vision` |
| `model.lora.r`, `.alpha` | 16, 32 | Rank, and the scale alpha / r. SmolVLM2 has 393,216 × r adapter weights |
| `model.lora.target_suffixes` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Module name endings that get an adapter |
| `model.lora.vision` | false | Also put adapters on the vision encoder |
| `image.tiles` | 13 | 384 px tiles per photo. SmolVLM2 accepts 1, 5, 7, 13, 16, 25, 29, and 41. Null keeps the model's own |
| `data.train_percent` | 10 | Percent of the train list |
| `data.tune_loss_photos` | 256 | First tune photos whose token loss is logged at each check |
| `data.tune_cost_photos` | 100 | First tune photos answered at each check. Their mean cost picks the checkpoint |
| `train.seed` | 0 | LoRA start, dropout, photo order |
| `train.epochs`, `train.max_steps` | 1.0, -1 | `max_steps` wins when above 0 |
| `train.learning_rate`, `train.lr_scheduler`, `train.warmup_fraction` | 1e-4, `cosine`, 0.03 | Use about 1e-5 for `full` |
| `train.weight_decay`, `train.max_grad_norm` | 0.0, 1.0 | |
| `train.batch_size`, `train.grad_accum`, `train.eval_batch_size` | 2, 4, 2 | Photos per GPU step, steps per update, tune photos per loss step |
| `train.eval_steps` | 100 | Updates between tune checks and checkpoints |
| `train.grad_checkpointing`, `train.workers` | false, 4 | Speed and memory only |
| `train.save_every` | 200 | Optional. Save a checkpoint every N updates and on the last one, then pick among them after training, newest first. Tune loss is still logged every `eval_steps` |
| `train.init_from` | `smolvlm2-2.2b-instruct-lora-r16a32-d25-t5-3f25856d` | Optional. Run folder whose saved weights this run starts from. Its model, revision, method, and LoRA settings must match |
| `score.tune_photos`, `score.test_photos` | 500, 1000 | First N photos answered |
| `score.batch_size` | 32 | Photos answered at once, also for the tune cost while training. The batch changes the answers, so keep it the same across runs you compare |

A grid file has `name`, `base` (a full spec), and `vary`, which maps a dotted key to the values to try. Keys joined by a comma change together, which keeps alpha at twice r. Every combination of the `vary` entries becomes one spec:

```json
{"name": "lora-vs-tiles", "base": {"...": "a full spec"},
 "vary": {"model.lora.r,model.lora.alpha": [[8, 16], [16, 32], [32, 64]], "image.tiles": [5, 13, 16]}}
```

## Logs

Every process writes the console at INFO, `<process>.log` at DEBUG, and `<process>.errors.log` at WARNING. The debug log rotates at 5 MB and the error log at 1 MB, with 2 old copies each, so one process never keeps more than 18 MB of logs. Hugging Face progress bars are off and HTTP chatter is muted.

| Process | Folder |
|---|---|
| `train`, `score-tune`, `score-test` | `output/runs/<name>/logs/` |
| `queue`, `grid`, `report`, `gui`, and any step that fails before its run is known | `output/logs/` |

## GUI

```bash
train-venv/bin/python spline_gui.py          # or: spline_gui.py models | live | audit | compare
```

| Page | What it shows |
|---|---|
| Runs | Every run folder: state (training, scoring, stopped, failed, trained, scored), tiles, method, tune loss, tune and test cost, test F. Base runs show "-" for training columns. The selected run's files are on the right |
| Models | A plain-language description of the selected run: what the adapter is, which photos it saw, how the checkpoint was picked, the tune and test scores, the prompt, and the judge settings in words. Base runs and base models are listed too |
| Live training | The queue, a progress bar per stage with ETA and finish time, train and tune loss with tune cost on a second axis, learning rate, gradient norm, and the end of the run's error logs. It follows the queue's current run unless you pick one |
| Audit | Pick a model and a photo. The model loads and answers on its own. The left image shows the annotated LLAMAS markers and the gold polylines. The right image shows the predicted polylines over dashed gold and a faint band `radius_px` wide around it, with each predicted point green inside the band and red outside. Bars show the cost, precision, and recall from `measure`. The prompt is shown above the answer. It warns in red when the run trained on the photo |
| Compare | One metric against one setting, coloured by another (for example tune cost against tiles, by LoRA rank), and the chosen runs' tune loss and tune cost curves. It warns when the scores used different photos or decode settings |

A photo the run already scored shows that stored answer while the live answer is on its way. Loading a model while a run trains or scores shares the GPU and can run out of memory.

## Run folder

Each run is `output/runs/<name>/`, for example `smolvlm2-2.2b-instruct-lora-r16a32-d10-t13-9dc7ae4d`, or `smolvlm2-2.2b-instruct-base-t5-<hash8>` for a base run. The last part is the first 8 characters of `config_sha256`, so any change to a setting makes a new folder. The dataclasses in `spline/config.py` are the code for these structures. Each field below matches one there. Files are written once. Writing a file again with different content is an error.

| File | Written by | What it is |
|---|---|---|
| `spec.json` | train (or score for a base run), first | The spec it was started from, with sorted keys |
| `config.json` | train (or score for a base run), first | `RunConfig` |
| `split.json` | train (or score for a base run), first | Byte copy of `output/split.json` |
| `data.json` | train (or score for a base run), first | The exact train, tune-loss, and tune-cost rows. Empty lists for a base run |
| `checkpoints/` | train | Restart and best checkpoints, deleted when training ends |
| `sweep/` | train with `save_every` | `trained.json` once training is done, `sweep.json` with the checkpoint order and each result, and `checkpoint-<step>.jsonl` answers. Kept after the run |
| `model/` | train | Adapter or full weights from the best checkpoint |
| `train.json` | train, at the end | `TrainResult` |
| `score-<split>.jsonl` | score | A header line, then one answer per photo |
| `score-<split>.json` | score, at the end | `ScoreResult` |
| `live/<stage>.json` | train, score | Live status for the GUI: `state` (running, done, failed), `pid`, `host`, `done`, `total`, `unit`, `elapsed_s`, `eta_s`, `eta_at`, and the latest loss. Rewritten at most every 2 s |
| `live/train.jsonl` | train | One line per Trainer log entry. A resumed run appends, so readers keep the last line per step |
| `logs/` | train, score | Rotating logs, see [Logs](#logs) |

`output/live/queue.json` holds the queue runner's state, current spec and step, and waiting lines.

### `split.json`

`split.json` has its own version, separate from the run schema, so runs share one photo list across schema changes.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 3 |
| `seed` | int | Shuffle seed, 0 |
| `fractions` | [float, float, float] | `[0.8, 0.1, 0.1]` |
| `labels_zip`, `labels_sha256` | str | Label archive and its hash |
| `images_dir` | str | Where the photos are |
| `rdp_px` | float | Simplification tolerance the gold text was made with. Must equal `judge.rdp_px` |
| `label_files` | int | Label files in the zip |
| `skipped_no_lanes`, `skipped_no_photo` | int | Labels left out, and why |
| `sizes` | {"WxH": int} | Photos of each size. The most common one is the reference size for tiling |
| `odd_size` | [str] | Every photo whose size is not the most common one. They are kept, and judged at their own size |
| `train`, `tune`, `test` | [[path, answer, width, height], ...] | Absolute photo path, gold answer text, and the size read from the photo file. Opening a photo whose size changed since is an error |

### `data.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 4 |
| `split_sha256` | str | Hash of the `split.json` it came from |
| `train` | [[path, answer, width, height], ...] | Every photo trained on, in split order. The Trainer shuffles them with `train.seed` |
| `tune_loss` | [[path, answer, width, height], ...] | Photos whose token loss is logged at each check |
| `tune_cost` | [[path, answer, width, height], ...] | Photos answered at each check. Their mean cost picks the checkpoint |

### `config.json` (`RunConfig`)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 4 |
| `name` | str | `<model>-<lora-rRaA-d<percent>, full-d<percent>, or base>-t<tiles>-<hash8>` |
| `config_sha256` | str | sha256 of this JSON with sorted keys, without `name` and `config_sha256` |
| `model.id` | str | Hugging Face id |
| `model.revision` | str | Commit hash of the weights |
| `model.method` | str | `lora`, `full`, or `base` |
| `model.lora` | object or null | `r`, `alpha`, `dropout`, `target_suffixes` (sorted), `skip_vision`, `bias` |
| `model.precision` | str | `bf16` or `fp16` mixed precision, or `fp32` without CUDA. Under fp16, trained weights stay fp32 |
| `image.reference_width`, `image.reference_height` | int | The most common photo size in `split.json`, used to measure tiles |
| `image.tiles` | int or null | Tiles per photo of the reference size, measured from the processor |
| `image.longest_edge` | int or null | Processor `size.longest_edge` that gives those tiles |
| `image.tile_edge` | int or null | Side of one tile, 384 |
| `data.split_file`, `data.split_sha256` | str | `split.json` and its hash |
| `data.train_percent`, `data.train_photos` | int | Percent asked for, and photos it gave. 0 for a base run |
| `data.tune_loss_photos`, `data.tune_cost_photos` | int | Tune photos for the logged loss and for the cost that picks the checkpoint. 0 for a base run |
| `data.prompt`, `data.label_format` | str | Text sent with each photo, and what the answer means |
| `judge.rdp_px`, `judge.step_px`, `judge.radius_px`, `judge.tau_px` | float | 2, 2, 15, 50: gold simplification, sampling step, overlap radius, and distance cap (see [Cost](#cost)) |
| `judge.distance`, `judge.cost` | str | The distance and the cost, in words |
| `judge.why` | {str: str} | One sentence per number on why it has that value |
| `train` | object or null | Null for a base run |
| `train.seed` | int | |
| `train.epochs`, `train.max_steps` | float, int | As given |
| `train.updates` | int | Updates made: `max_steps`, or ceil(epochs × ceil(ceil(photos / batch) / accum)) |
| `train.learning_rate`, `train.lr_scheduler` | float, str | |
| `train.warmup_fraction`, `train.warmup_steps` | float, int | `warmup_steps` = max(1, int(updates × fraction)) |
| `train.optimizer`, `train.adam_beta1`, `train.adam_beta2`, `train.adam_epsilon` | str, float | `adamw_torch_fused`, 0.9, 0.999, 1e-8 |
| `train.weight_decay`, `train.max_grad_norm` | float | |
| `train.batch_size`, `train.grad_accum`, `train.photos_per_update` | int | |
| `train.eval_batch_size`, `train.eval_steps` | int | |
| `train.select_checkpoint` | str | `lowest_tune_cost` |
| `train.save_every` | int | Only present when set |
| `train.init_from` | object | Only present when set: `run` and the `weights_sha256` of its saved weights, checked before training |

### `train.json` (`TrainResult`)

A base run has no `train.json`.

| Field | Type | Meaning |
|---|---|---|
| `schema_version`, `run` | | Which run |
| `prompt`, `judge` | str, object | The prompt sent with every photo, and the judge that picked the checkpoint. Same as `config.json` |
| `config_sha256` | str | |
| `trainable_params`, `total_params` | int | |
| `updates` | int | Updates made |
| `best_step`, `best_tune_cost_px`, `best_tune_loss` | int, float, float | The checkpoint saved in `model/`, its mean tune cost, and its tune loss |
| `train_loss` | float | Mean training loss over the run |
| `runtime_s` | float | Wall time of the process that finished. A resumed run counts only from the resume |
| `log_history` | list | Every Trainer log: `loss`, `grad_norm`, `learning_rate`, `epoch`, and at each check `eval_loss`, `eval_tune_cost_px`, `eval_tune_prec`, `eval_tune_rec`, `eval_tune_f`, `eval_tune_readable`, and `eval_tune_cost_s` |
| `weights_file`, `weights_sha256` | str | `model/adapter_model.safetensors` and its hash. For a full model, `model` and the hash of its file list |
| `resumed_from` | str or null | Checkpoint folder it restarted from |
| `grad_checkpointing` | bool | Whether activations were recomputed. Same math, less memory |
| `workers` | int | DataLoader processes used |
| `fallbacks` | [str] | Every automatic change made to fit this machine. Empty on the Quadro at default settings |
| `spec_file`, `spec_sha256` | str | The spec path given to `train`, and the sha256 of its bytes |
| `sweep` | list | With `save_every`: `step`, `tune_cost_px`, `tune_prec`, `tune_rec`, `tune_readable`, `seconds`, and `better` per checkpoint, newest first. Empty otherwise |
| `started_at`, `finished_at` | str | ISO time with UTC offset |
| `environment` | object | See below |

### `score-<split>.json` (`ScoreResult`)

| Field | Type | Meaning |
|---|---|---|
| `schema_version`, `run` | | Which run |
| `prompt`, `judge` | str, object | The prompt sent with every photo, and how the answers were judged. Same as `config.json` |
| `config_sha256` | str | |
| `weights_sha256` | str | Must equal `train.json`, or the hub commit for a base run. Scoring stops if the weights changed |
| `split`, `photos` | str, int | `tune` or `test`, and the first N photos of it |
| `photos_sha256` | str | Hash of the scored photo paths. Equal across runs, base runs included, means the same photos |
| `decode` | object | `context_tokens` (the model's window; each answer may use all of it after the prompt), `do_sample` (false), `num_beams` (1), `batch_size`, `precision` |
| `readable_pct` | float | Answers whose every line is an even count of at least 4 whole numbers inside the photo |
| `truncated_pct` | float | Answers that filled the whole context window |
| `cost` | object | Per-photo cost L: `photos`, `median_px`, `mean_px`, `within_10_pct`, `within_25_pct` |
| `precision_pct`, `recall_pct`, `f_pct` | float | Mean per-photo precision and recall within `judge.radius_px`, and F of those means |
| `lanes_pred_mean`, `lanes_gold_mean` | float | Mean lanes per answer and per gold text |
| `predictions_file`, `predictions_sha256` | str | The jsonl and its hash |
| `started_at`, `finished_at` | str | |
| `environment` | object | See below |
| `fallbacks` | [str] | For example a smaller batch after running out of GPU memory |

The first line of `score-<split>.jsonl` is `{config_sha256, split, photos_sha256, prompt, judge, ...decode}`. Each later line is `{image, width, height, gold, raw, truncated}`. A rerun with other settings stops instead of mixing answers.

### `environment`

`python`, `packages` (torch, transformers, peft, accelerate, huggingface-hub, pillow, numpy, safetensors), `cuda`, `cudnn`, `gpu` (name, driver, memory), `host`, `platform`, `argv`, and `code_sha256` with the hash of every `.py` file here. Compare `code_sha256` across runs to see whether the code changed between them.

## Layout

| Path | What it is |
|---|---|
| `spline_train.py` | `train`, `score`, `report`, and the config builder |
| `spline_queue.py` | Runs `output/queue.txt` line by line. `grid` writes specs and adds lines |
| `spline_gui.py` | The GUI |
| `specs/` | Spec files, and grid files in `specs/grids/` |
| `spline/spec.py` | The spec format, its checks, and grid expansion |
| `spline/config.py` | The dataclasses for every JSON a run folder holds |
| `spline/logs.py` | Rotating logs and live status files |
| `spline/data.py` | `split.json`, which photos a run uses, and raw LLAMAS labels for any photo |
| `spline/model.py` | Tiles, photos to tensors, LoRA, and loading weights |
| `spline/fit.py` | Training loop, and the tune cost that picks the checkpoint |
| `spline/score.py` | Answering photos, `measure` (the per-photo cost), and aggregation |
| `spline/lanes.py` | Lane text, gold polylines from LLAMAS markers, and the pooled overlap cost |
| `spline/runs.py` | Run folders, hashes, write-once files, environment, run summaries |
| `spline/gui/` | One file per GUI page, and `state.py` for what they share |
| `requirements-train.txt` | Pinned packages in `train-venv/` |

The card is a Turing Quadro RTX 8000 with no hardware bf16, so the code uses fp16. PyTorch's default bf16 check counts slow emulation as support, and trusting it made training about 7 times slower. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation. The queue sets it for you.

## GPU use

Measured on 2026-10-22 with 13 tiles unless noted:

| What | Result | So |
|---|---|---|
| Training at batch 2, 4 workers | 97% GPU busy, 231 W of 260 W, 29.5 GB, 0.95 s per photo | The card is the limit. A bare training loop with no Trainer is 0.85 s, and that gap includes photo length |
| Training at batch 4 or 8 | Out of memory without `train.grad_checkpointing` | Batch 2 is the largest that fits. Checkpointing fits more but is slower per photo |
| One training step on the GPU | 30% attention, the rest fp16 matmul. A plain matmul peaks at 90 TFLOPS | Turing has no FlashAttention 2, so PyTorch SDPA's memory-efficient kernel is already the fastest |
| `torch.compile` on the vision encoder | 367 ms to 362 ms per 26 tiles | Not used |
| Scoring at batch 16, 32, 64 | 1.14, 0.90, 1.37 s per photo | 32 is the default. At 64, one long answer holds 63 finished ones |
| Scoring at batch 32, 41 tiles | 34.9 GB peak, 2.2 s per photo | 32 fits every tile count |
| Static KV cache without compile | 5 to 20 times slower per token | Not used |
| Preparing score batches in a second process | 0.90 vs 0.89 s per photo | Not used. Preparation is about 2.5 s of each 29 s batch |

Running two jobs on the card at once, such as scoring one run while the next trains, does not help. Training already keeps the GPU busy, so the two would take turns. At 29 or 41 tiles, the two together also need more than 48 GB.

## Other machines

Some defaults are tuned for this Quadro. Each has a fallback that logs a `WARNING` line and is written to the run's JSON.

| Tuned for this card | Elsewhere |
|---|---|
| fp16, because Turing has no native bf16 | bf16 on cards that run it natively, such as Ampere and newer. fp32 on CPU with no CUDA. Precision is in the config hash, so these are separate runs |
| One GPU | `CUDA_VISIBLE_DEVICES` defaults to `0`. If you list more than one GPU, the run stops, because the Trainer would split each batch across them and change the math |
| Training batch 2 fits in 48 GB | If memory runs out, training restarts from its last checkpoint with gradient checkpointing, which is the same math with less memory. If that also runs out, it stops and suggests `train.batch_size` 1 with twice the `train.grad_accum`, which is a new run |
| Score batch 32 fits in 48 GB | If memory runs out, the batch is halved and that split is answered again from the start, because answers depend on the batch. A resume keeps the smaller batch. `report` warns when compared runs used different batches |
| 4 data workers | Lowered to one less than the number of CPU cores when the machine has fewer |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from the queue | Only set when you have not set it yourself |
