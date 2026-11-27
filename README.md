# Dojo 24

LoRA training for Krea 2, MOSS-Image and MiniMax-H3 Image.

A LoRA trainer for exactly three models, fitted to one card: the RTX 3090
(24 GB) in this machine. The training itself is done by
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), pinned in
`engine/ai-toolkit`; this app replaces its UI with one that only knows these
models and plans every job against the card's memory. The page follows
ai-toolkit's own look (sidebar, gray cards, queue table).

Start it with **Start Dojo 24.bat** (opens http://127.0.0.1:7870).

## What it does for a 24 GB card

| | Krea 2 (12B) | MOSS-Image (5B) | MiniMax-H3 Image |
|---|---|---|---|
| Transformer | 8-bit (`qfloat8`); 24.5 GB of bf16 does not fit | full bf16, no quantization | int8 as shipped (19.5 GB), 10 to 25% of layers kept in system RAM |
| Text encoder | Qwen3-VL-4B, unloaded after captions are cached | Qwen3.5-9B int8, unloaded after caching | Qwen3-VL-32B nvfp4, unloaded after caching |
| Resolutions | 512 + 1024 (768 is skipped) | 768 + 1024 | 1024 + 1536, up to 2048 |
| Batch at top resolution | 1 | 2 (4 is tight) | 1 |

The GPU Fit page in the app shows the same thing live, with the reason for
every default and which batch sizes fit at each resolution.

- The memory strip shows one cell per gigabyte: what other apps hold now and
  what the run will take. A run is refused if it cannot fit, and a queued run
  waits until enough memory is free (for example after ComfyUI unloads).
- The first figures are estimates. After a finished run the measured peak is
  stored in `measured.json` and used for that recipe from then on.
- If dedicated memory sits pinned at the ceiling during a run, the app warns
  that the driver is paging into system RAM.
- Base weights are read from the Hugging Face cache and from ComfyUI's models
  folder; anything the engine has to download goes into the ComfyUI folder.
- A finished LoRA is copied to `ComfyUI/models/loras/KR2` or `loras/MOSS`
  (never overwriting; a `_v2` suffix is added instead).

## Folders

- `app/` the web app (FastAPI + one HTML page)
- `engine/` ai-toolkit, its Python 3.12 and venv (rebuild with `setup.bat`)
- `jobs/<id>/` config.yaml, train.log and state for each run
- `output/<name>/` checkpoints and preview images
- `settings.json` ComfyUI paths, port, LoRA folders, memory reserve

Closing the toolkit window does not stop a run; reopen it to see progress.
Stopping a run keeps its checkpoints, and Resume continues from the latest.
