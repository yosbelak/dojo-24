# Dojo 24

LoRA training for **Krea 2**, **MOSS-Image** and **MiniMax-H3 Image**, planned
for one card: a 24 GB GPU such as the RTX 3090.

Dojo 24 is a native Windows app written in Rust, drawn with
[rust-glass](https://crates.io/crates/rust-glass) (Liquid Glass style). The
training itself runs in [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit),
pinned to one commit in `engine\`. Dojo 24 knows only these three models and plans
every job against the memory on your card before it starts.

## Set up

1. Install [Rust](https://rustup.rs), [uv](https://docs.astral.sh/uv/) and git.
2. Run **setup.bat**. It clones ai-toolkit at the pinned commit, installs a private
   Python 3.12 with PyTorch for CUDA 13.0, and builds the app.
3. Open **Start Dojo 24.bat**.

Base weights are read from your ComfyUI models folder and the Hugging Face cache.
Anything the engine has to download goes into the ComfyUI folder, so nothing is
stored twice. Paths live in `settings.json`, created on first start.

## What it does for a 24 GB card

| | Krea 2 (12B) | MOSS-Image (5B) | MiniMax-H3 Image |
|---|---|---|---|
| Transformer | 8-bit (`qfloat8`); 24.5 GB of bf16 does not fit | full bf16, no quantization | int8 as shipped (19.5 GB), 15 to 25% of layers kept in system RAM |
| Text encoder | Qwen3-VL-4B, unloaded after captions are cached | Qwen3.5-9B int8, unloaded after caching | Qwen3-VL-32B nvfp4, unloaded after caching |
| Resolutions | 512 + 1024 (768 is skipped) | 768 + 1024 | 1024 + 1536, up to 2048 |
| Batch at top resolution | 1 | 2 (4 is tight) | 1 |

- **Memory plan:** the New job page shows the card one gigabyte per cell: what other
  apps hold now and what the job will take. A job that cannot fit is refused.
- **Queue:** a queued job waits until enough memory is free, for example after
  ComfyUI unloads its models (there is a button for that), and then starts by itself.
  The queue can be paused.
- **GPU fit:** every default, whether it is the engine author's choice or chosen for
  this card, the reason, and which batch sizes fit at each resolution.
- **Jobs:** live step, speed, time left, loss and GPU memory charts, preview images,
  and the engine log. Stop keeps the checkpoints; Resume continues from the latest.
  Closing the window does not stop a job.
- **Delivery:** a finished LoRA is copied to `ComfyUI\models\loras\KR2`, `loras\MOSS`
  or `loras\H3`. A taken name gets a `_v2` suffix instead of being overwritten.

The memory figures are estimates from the model sizes. What the driver reports
during a run is shown on the GPU fit page, but PyTorch keeps freed memory reserved,
so that number overstates what a job needs.

## Folders

- `src\` the app
- `engine\` ai-toolkit, its Python and venv (rebuilt by `setup.bat`, not in git)
- `jobs\<id>\` `config.json`, `train.log` and state for each job
- `output\<name>\` checkpoints and preview images

## Tests

```bash
cargo test
```

The window test runs off-screen on the GPU and never starts training:

```bash
DOJO_TEST_DATASET=<folder with a few images> cargo test --release ui_ -- --ignored --nocapture
```
