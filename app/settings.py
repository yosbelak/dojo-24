"""Paths and user settings. Edit settings.json (created on first run) to change them."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engine" / "ai-toolkit"
ENGINE_PY = ROOT / "engine" / "venv" / "Scripts" / "python.exe"
JOBS_DIR = ROOT / "jobs"
OUTPUT_DIR = ROOT / "output"
SETTINGS_FILE = ROOT / "settings.json"
MEASURED_FILE = ROOT / "measured.json"

DEFAULTS = {
    # ComfyUI's models folder. The engine reads base weights from it and puts
    # anything it has to download there, so nothing is stored twice.
    "comfy_models": r"C:\ComfyUI-Easy-Install\ComfyUI\models",
    "comfy_url": "http://127.0.0.1:8188",
    "host": "127.0.0.1",
    "port": 7870,
    # where finished LoRAs are copied, relative to comfy_models
    "lora_dirs": {"krea2": "loras/KR2", "moss_image": "loras/MOSS", "minimax_h3_image": "loras/H3"},
    # GB kept free for Windows and the desktop. Past this the driver spills
    # into shared system memory and training slows to a crawl without failing.
    "vram_reserve_gb": 1.0,
}


def load() -> dict:
    data = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            dirs = {**DEFAULTS["lora_dirs"], **saved.get("lora_dirs", {})}
            data.update(saved)
            data["lora_dirs"] = dirs
        except Exception:
            pass
    else:
        SETTINGS_FILE.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    return data


def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def hf_cached_file(repo_id: str, filename: str):
    """Path of a file already in the Hugging Face cache, or None."""
    snaps = hf_cache() / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), reverse=True):
        candidate = snap / filename
        if candidate.exists():
            return candidate
    return None
