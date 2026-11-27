"""The two models this toolkit trains, and how each one is fitted to the card.

Everything here is derived from the engine's own loaders (ai-toolkit
extensions_built_in/diffusion_models/{krea2,moss_image}) and its UI defaults.
The memory figures are estimates until a run on this machine replaces them
(see measured.json).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from . import settings, sniff

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

FAMILIES = {
    "krea2": {
        "label": "Krea 2",
        "params": "12B single-stream DiT, Qwen3-VL-4B text encoder",
        # resident transformer weights on the GPU, GB
        "weights_gb": {"bf16": 24.5, "qfloat8": 12.6, "convrot8": 12.9},
        # activations + gradients with checkpointing, GB per image in the batch
        "act_gb": {512: 1.6, 768: 2.8, 1024: 4.4},
        "overhead_gb": 1.6,
        "timestep_type": "linear",
        "sample": {"guidance_scale": 4, "sample_steps": 25},
    },
    "moss_image": {
        "label": "MOSS-Image",
        "params": "5B single-stream DiT, Qwen3.5-9B text encoder",
        "weights_gb": {"bf16": 10.3, "convrot8": 5.2},
        "act_gb": {512: 0.9, 768: 1.6, 1024: 2.4},
        "overhead_gb": 1.2,
        "timestep_type": "shift",
        "sample": {"guidance_scale": 3, "sample_steps": 30},
    },
    "minimax_h3_image": {
        "label": "MiniMax-H3 Image",
        "params": "Pruned H3 DiT without the audio stream, 6-step distilled, Qwen3-VL-32B text encoder",
        # the shipped int8 file is 19.5 GB
        "weights_gb": {"convrot8": 19.6},
        # 32x compression: 2048 px here is the token count of 1024 px on Krea 2
        "act_gb": {1024: 1.6, 1536: 2.9, 2048: 4.6},
        "overhead_gb": 1.8,
        "timestep_type": "shift",
        "sample": {"guidance_scale": 1, "sample_steps": 6},
        "sample_size": 2048,
        "rank": 16,
        # the DiT cannot sit on a 24 GB card together with its activations, so a
        # share of its layers is kept in system RAM (the engine's layer offloading)
        "offload": True,
    },
}

H3_REPO = "Comfy-Org/MiniMax-H3-Image"
H3_ADAPTER = "ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors"
H3_FILES = {
    "dit": ("diffusion_models", ["minimax_h3_image_pruned_int8_convrot.safetensors"], "the 19.5 GB transformer"),
    "te": ("text_encoders", ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"], "the 14.6 GB text encoder"),
    "vae": ("vae", ["minimax_h3_image_vae_fp16.safetensors"], "the 1.1 GB image VAE"),
}

GOALS = {
    "character": {
        "label": "Character",
        "hint": "One person or creature, kept consistent across poses.",
        "rank": 32,
        "lr": 1e-4,
        "steps_per_image": 75,
        "min_steps": 1500,
        "max_steps": 3000,
    },
    "style": {
        "label": "Style",
        "hint": "A look: palette, rendering, linework.",
        "rank": 16,
        "lr": 5e-5,
        "steps_per_image": 50,
        "min_steps": 1500,
        "max_steps": 4000,
    },
    "concept": {
        "label": "Concept",
        "hint": "An object, outfit, pose or action.",
        "rank": 16,
        "lr": 1e-4,
        "steps_per_image": 60,
        "min_steps": 1250,
        "max_steps": 3000,
    },
}

# Krea 2 was trained at 256/512/1024 and community runs at 768 are reported
# to go wrong, so 768 is offered for MOSS-Image only.
DETAIL = {
    "fast": {"label": "Fast", "krea2": [512], "moss_image": [512, 768], "minimax_h3_image": [1024]},
    "balanced": {"label": "Balanced", "krea2": [512, 1024], "moss_image": [768, 1024], "minimax_h3_image": [1024, 1536]},
    "max": {"label": "Max detail", "krea2": [1024], "moss_image": [1024], "minimax_h3_image": [1536, 2048]},
}


_SNIFF_CACHE: Dict[tuple, dict] = {}


def _sniff_cached(path: str, mtime_ns: int, size: int) -> dict:
    key = (path, mtime_ns, size)
    if key not in _SNIFF_CACHE:
        _SNIFF_CACHE[key] = sniff.sniff(path)
    return _SNIFF_CACHE[key]


def _find(root: Path, names: List[str]) -> Optional[Path]:
    if not root.is_dir():
        return None
    wanted = set(names)
    for d, _, files in os.walk(root):
        for f in files:
            if f in wanted:
                return Path(d) / f
    return None


def _glob_local(root: Path, must: List[str], avoid: List[str] = ()) -> List[Path]:
    hits = []
    if root.is_dir():
        for d, _, files in os.walk(root):
            for f in files:
                low = f.lower()
                if f.endswith(".safetensors") and all(m in low for m in must) and not any(a in low for a in avoid):
                    hits.append(Path(d) / f)
    return sorted(hits)


def catalog(cfg: dict) -> dict:
    """What can be trained on right now, and what each choice would download."""
    models = Path(cfg["comfy_models"])
    dm, te_dir, vae_dir = models / "diffusion_models", models / "text_encoders", models / "vae"

    bases = []

    raw = settings.hf_cached_file("krea/Krea-2-Raw", "raw.safetensors")
    bases.append({
        "id": "krea2:raw", "family": "krea2", "label": "Krea 2 Raw",
        "note": "The undistilled base. LoRAs trained here carry over to Krea 2 Turbo and to finetunes.",
        "ready": raw is not None,
        "download": None if raw else "24.5 GB from krea/Krea-2-Raw (needs a Hugging Face login that accepted the licence)",
        "recommended": True,
    })
    moss_base = _find(dm, ["moss_image_base_bf16.safetensors", "moss_image_base_int8_convrot_attn.safetensors"])
    bases.append({
        "id": "moss_image:base", "family": "moss_image", "label": "MOSS-Image Base",
        "note": "The vendor's training base. LoRAs carry over to MOSS-Image SFT and to finetunes.",
        "ready": moss_base is not None,
        "download": None if moss_base else "10.3 GB (moss_image_base_bf16.safetensors) into ComfyUI's diffusion_models",
        "recommended": True,
    })
    missing = [what for folder, names, what in H3_FILES.values() if _find(models / folder, names) is None]
    if _find(models / "loras", [H3_ADAPTER.split("/")[-1]]) is None:
        missing.append("ostris' training adapter LoRA")
    core_ready = all(_find(models / folder, names) is not None for folder, names, _ in H3_FILES.values())
    bases.append({
        "id": "minimax_h3_image:base", "family": "minimax_h3_image", "label": "MiniMax-H3-Image",
        "note": "The 6-step image model, trained through a de-distillation adapter so the LoRA does not undo the distillation.",
        "ready": core_ready,
        "download": ("; ".join(missing) + ", into ComfyUI's models folder") if missing else None,
        "recommended": True,
    })

    # Checkpoints already in ComfyUI. Only ones the engine can open as-is are
    # offered: no model.diffusion_model. key prefix, and for Krea 2 the
    # quantization markers must be stored as tensors (its loader does not read
    # the header-style layout).
    local = []
    for d, dirs, files in os.walk(dm) if dm.is_dir() else []:
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for f in sorted(files):
            p = Path(d) / f
            if p.suffix != ".safetensors" or p.stat().st_size < (2 << 30):
                continue
            if f.startswith(("moss_image_base_", "moss_image_sft_", "minimax_h3_")):
                continue  # the official files, listed above
            info = _sniff_cached(str(p), p.stat().st_mtime_ns, p.stat().st_size)
            if info["family"] not in FAMILIES:
                continue
            reason = None
            if info.get("prefixed"):
                reason = "keys carry a ComfyUI checkpoint prefix"
            elif info["family"] == "krea2" and info["quant_layout"] == "header":
                reason = "quantized in a layout the Krea 2 loader does not read"
            elif info["precision"] not in ("int8", "bf16", "fp16", "fp32"):
                reason = f"{info['precision']} weights"
            local.append({
                "id": "file:" + str(p), "family": info["family"], "label": p.stem,
                "precision": info["precision"], "size_gb": info["size_gb"],
                "ready": reason is None, "blocked": reason,
                "note": "Your own checkpoint. Untested with the engine; the LoRA is tied to this finetune.",
            })

    # MOSS text encoder and VAE: use what ComfyUI already has before downloading
    te_official = _find(te_dir, ["qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_bf16.safetensors"])
    te_local = _glob_local(te_dir, ["qwen3.5", "9b"])
    text_encoders = []
    if te_official:
        text_encoders.append({"id": "official", "label": te_official.name, "download": None})
    for p in te_local:
        if te_official and p == te_official:
            continue
        text_encoders.append({
            "id": str(p), "label": p.name, "download": None,
            "note": "Already in ComfyUI. Train with the encoder you generate with.",
        })
    if not te_official:
        text_encoders.append({
            "id": "official", "label": "qwen3.5_9b_int8_convrot (official)",
            "download": "11.5 GB into ComfyUI's text_encoders",
        })

    vae = _find(vae_dir, ["flux2-vae.safetensors"])
    vae_alt = None
    if vae is None:
        for p in _glob_local(vae_dir, ["flux2", "vae"], avoid=["fp8", "hdr", "small"]):
            vae_alt = p
            break

    return {
        "bases": bases,
        "local": local,
        "moss_text_encoders": text_encoders,
        "moss_vae": {
            "path": str(vae or vae_alt) if (vae or vae_alt) else None,
            "download": None if (vae or vae_alt) else "0.3 GB (flux2-vae.safetensors)",
        },
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def inspect_dataset(path: str) -> dict:
    root = Path(path)
    if not root.is_dir():
        return {"ok": False, "error": "That folder does not exist."}
    folders, total, captioned, sample_files = [], 0, 0, []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith((".", "_")))
        images = sorted(f for f in files if Path(f).suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        caps = sum(1 for f in images if (Path(d) / (Path(f).stem + ".txt")).exists())
        folders.append({"path": d, "rel": os.path.relpath(d, root), "images": len(images), "captions": caps})
        total += len(images)
        captioned += caps
        sample_files += [str(Path(d) / f) for f in images]
    if total == 0:
        return {"ok": False, "error": "No images found in that folder or its subfolders."}

    small, sizes = 0, []
    try:
        from PIL import Image

        step = max(1, len(sample_files) // 60)
        for f in sample_files[::step][:60]:
            try:
                with Image.open(f) as im:
                    sizes.append(im.size)
            except Exception:
                pass
        small = sum(1 for w, h in sizes if min(w, h) < 768)
    except Exception:
        pass
    step = max(1, len(sample_files) // 10)
    return {
        "ok": True, "root": str(root), "images": total, "captioned": captioned,
        "folders": folders, "checked": len(sizes), "small": small,
        "min_side": min((min(s) for s in sizes), default=None),
        "previews": sample_files[::step][:10],
    }


def suggest_steps(goal: str, images: int, batch: int = 1) -> int:
    g = GOALS[goal]
    steps = max(g["min_steps"], min(g["max_steps"], images * g["steps_per_image"]))
    steps = int(round(steps / max(1, batch) / 250.0)) * 250
    return max(500, steps)


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------

def _measured() -> dict:
    try:
        return json.loads(settings.MEASURED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def measured_key(family: str, precision: str, res: int, batch: int) -> str:
    return f"{family}|{precision}|{res}|{batch}"


def record_measurement(key: str, peak_gb: float, baseline_gb: float):
    data = _measured()
    data[key] = {"peak_gb": round(peak_gb, 2), "baseline_gb": round(baseline_gb, 2)}
    settings.MEASURED_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def budget(family: str, precision: str, res: int, batch: int, total_gb: float, reserve_gb: float) -> dict:
    fam = FAMILIES[family]
    weights = fam["weights_gb"].get(precision, max(fam["weights_gb"].values()))
    act = fam["act_gb"][res] * batch
    over = fam["overhead_gb"]
    usable = total_gb - reserve_gb
    offload = 0.0
    if fam.get("offload"):
        # keep just enough layers in system RAM to leave 1 GB of slack; each
        # offloaded layer is copied to the card and back every step
        spare = usable - 1.0 - act - over
        if spare < weights:
            offload = min(0.9, max(0.0, 1 - spare / weights))
            offload = round(-(-offload // 0.05) * 0.05, 2)  # round up to 5%
    resident = weights * (1 - offload)
    need = resident + act + over
    out = {
        "weights_gb": round(resident, 1), "activations_gb": round(act, 1),
        "overhead_gb": round(over, 1), "need_gb": round(need, 1),
        "usable_gb": round(usable, 1), "total_gb": round(total_gb, 1),
        "offload": offload, "ram_gb": round(weights * offload, 1),
        # estimates get half a GB of margin; a measured figure is taken as is
        "fits": need + 0.5 <= usable and offload <= 0.6, "source": "estimate",
    }
    hit = _measured().get(measured_key(family, precision, res, batch))
    if hit:
        own = max(0.0, hit["peak_gb"] - hit["baseline_gb"])
        scale = own / need if need else 1
        out.update({
            "need_gb": round(own, 1), "fits": own <= usable, "source": "measured",
            "weights_gb": round(resident * scale, 1), "activations_gb": round(act * scale, 1),
        })
        out["overhead_gb"] = round(max(0.0, own - out["weights_gb"] - out["activations_gb"]), 1)
    return out


def pick_precision(family: str, base_id: str, local_precision: Optional[str], total_gb: float, reserve_gb: float,
                   res: int, batch: int) -> str:
    """Highest precision that fits. A pre-quantized int8 file stays int8."""
    if local_precision == "int8":
        return "convrot8"
    fam = FAMILIES[family]
    if family == "minimax_h3_image":
        return "convrot8"  # trained on the int8 file as shipped
    order = ["bf16", "qfloat8"] if family == "krea2" else ["bf16", "convrot8"]
    for p in order:
        need = fam["weights_gb"][p] + fam["act_gb"][res] * batch + fam["overhead_gb"]
        if need + 0.5 <= total_gb - reserve_gb:
            return p
    return order[-1]


# ---------------------------------------------------------------------------
# Why each setting, for this card
# ---------------------------------------------------------------------------

def explain(family: str, precision: str, bud: dict, resolutions: List[int], batch: int, rank: int,
            total_gb: float) -> List[dict]:
    """[{setting, value, why, basis}]. basis 'engine' = the engine author's own
    default for this model; 'card' = chosen here for the memory on this card."""
    fam = FAMILIES[family]
    card = f"{round(total_gb)} GB"
    res_text = " + ".join(map(str, resolutions)) + " px"
    rows = []
    if family == "krea2":
        rows.append({
            "setting": "Transformer weights", "value": "8-bit (qfloat8)", "basis": "card",
            "why": f"Krea 2 is 24.5 GB in bf16, more than the whole card. At 8-bit it is {fam['weights_gb']['qfloat8']} GB, "
                   "which leaves room to train at 1024 px with everything on the GPU. qfloat8 is the engine's default 8-bit type for this model.",
        })
        rows.append({
            "setting": "Text encoder", "value": "Qwen3-VL-4B, 8-bit, then unloaded", "basis": "card",
            "why": "Captions are encoded once before training and cached, so the encoder's memory goes back to the transformer.",
        })
        rows.append({
            "setting": "Resolutions", "value": res_text, "basis": "card",
            "why": "Krea 2 was trained at 256, 512 and 1024, and community runs at 768 are reported to go wrong, so 768 is left out. "
                   f"1024 px is the largest size where batch 1 fits a {card} card without offloading.",
        })
        rows.append({
            "setting": "Timesteps", "value": "linear", "basis": "engine",
            "why": "The engine author's default for Krea 2; the model applies its own resolution-dependent shift on top.",
        })
    elif family == "moss_image":
        if precision == "bf16":
            rows.append({
                "setting": "Transformer weights", "value": "full bf16, not quantized", "basis": "card",
                "why": f"The 5B transformer is {fam['weights_gb']['bf16']} GB in bf16, so a {card} card can train it at full precision. "
                       "The engine's default is 8-bit because it targets smaller cards; you do not need that trade-off.",
            })
        else:
            rows.append({
                "setting": "Transformer weights", "value": "8-bit (convrot8)", "basis": "card",
                "why": "This checkpoint is already stored as int8, so it is trained as it is.",
            })
        rows.append({
            "setting": "Text encoder", "value": "Qwen3.5-9B, int8, then unloaded", "basis": "card",
            "why": "Captions are cached before training. The vendor says to keep this encoder frozen, and it is.",
        })
        rows.append({
            "setting": "Resolutions", "value": res_text, "basis": "card",
            "why": "The vendor recommends 1 to 2 megapixel output, and there is memory to train at 1024 px with batch 2.",
        })
        rows.append({
            "setting": "Layer 22", "value": "left untouched", "basis": "engine",
            "why": "MOSS-Image skips block 22 on the negative branch when sampling. Keeping the LoRA off that block preserves the effect.",
        })
        rows.append({"setting": "Timesteps", "value": "shift", "basis": "engine", "why": "The engine author's default for MOSS-Image."})
    else:
        rows.append({
            "setting": "Transformer weights", "value": "int8 (convrot8), as shipped", "basis": "engine",
            "why": "The 19.5 GB file in your ComfyUI folder is used directly, with no re-quantizing.",
        })
        if bud["offload"]:
            rows.append({
                "setting": "Layer offloading", "value": f"{round(bud['offload'] * 100)}% of layers in system RAM", "basis": "card",
                "why": f"19.6 GB of weights plus training memory does not fit in {card}. Keeping {bud['ram_gb']} GB of layers in system RAM "
                       "is the smallest share that fits; more would only slow training down. The engine marks this feature experimental.",
            })
        else:
            rows.append({
                "setting": "Layer offloading", "value": "off", "basis": "card",
                "why": "Everything fits on the card at this resolution, so no layers are streamed from RAM.",
            })
        rows.append({
            "setting": "Text encoder", "value": "Qwen3-VL-32B, nvfp4, then unloaded", "basis": "card",
            "why": "At 14.6 GB it cannot share the card with the transformer. The transformer waits in system RAM while captions are cached.",
        })
        rows.append({
            "setting": "Resolutions", "value": res_text, "basis": "card",
            "why": "The model is built for 4 megapixels and up. Training lower is faster and lighter but learns less fine detail; "
                   "2048 px is the practical ceiling on this card.",
        })
        rows.append({
            "setting": "Training adapter", "value": "on", "basis": "engine",
            "why": "The model is step- and guidance-distilled. The adapter stands in for the lost steps during training and is switched off for previews.",
        })
    source = "measured on this card" if bud["source"] == "measured" else "an estimate until the first run finishes"
    rows.append({
        "setting": "Batch size", "value": str(batch), "basis": "card",
        "why": f"This recipe is planned at {bud['need_gb']} GB of the {bud['usable_gb']} GB that can be used ({source}).",
    })
    rows.append({
        "setting": "Gradient checkpointing", "value": "on", "basis": "engine",
        "why": "Recomputes activations during the backward pass instead of storing them. Slower per step, far less memory.",
    })
    rows.append({
        "setting": "Optimizer", "value": "AdamW 8-bit", "basis": "engine",
        "why": "The engine's default; its optimizer state takes a quarter of the memory of plain AdamW.",
    })
    rows.append({
        "setting": "Rank", "value": str(rank), "basis": "engine",
        "why": "Capacity of the LoRA. It changes file size and how much detail can be learned, not whether the run fits.",
    })
    return rows


# ---------------------------------------------------------------------------
# Plan -> engine config
# ---------------------------------------------------------------------------

# preview prompts when none are given; {t} is the trigger word
DEFAULT_PROMPTS = {
    "character": [
        "{t}, portrait, soft window light, looking at the camera",
        "{t}, full body, walking down a rainy street at night, neon reflections",
        "{t}, sitting in a cafe reading a book, candid photo",
        "{t}, standing on a mountain trail at sunrise, wide shot",
    ],
    "style": [
        "a lighthouse on a cliff at dusk, {t}",
        "a cat asleep on a windowsill full of plants, {t}",
        "a crowded night market seen from above, {t}",
        "an old bicycle leaning on a brick wall, {t}",
    ],
    "concept": [
        "{t}, on a wooden table, soft daylight",
        "{t}, outdoors in a park, wide shot",
        "{t}, close-up, studio lighting, plain backdrop",
        "{t}, in a busy city street at night",
    ],
}


def build_plan(form: dict, cfg: dict, gpu_total_gb: float) -> dict:
    """Validate the form and return {config, budget, downloads, warnings, ...}."""
    cat = catalog(cfg)
    base_id = form.get("base") or "krea2:raw"
    entry = next((b for b in cat["bases"] + cat["local"] if b["id"] == base_id), None)
    if entry is None:
        raise ValueError("Pick a base model.")
    if entry.get("blocked"):
        raise ValueError(f"{entry['label']} cannot be trained on: {entry['blocked']}.")
    family = entry["family"]
    fam = FAMILIES[family]

    name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (form.get("name") or "").strip())
    if not name:
        raise ValueError("Give the LoRA a name.")
    trigger = (form.get("trigger") or "").strip()

    ds = inspect_dataset(form.get("dataset") or "")
    if not ds["ok"]:
        raise ValueError(ds["error"])

    goal = form.get("goal") if form.get("goal") in GOALS else "character"
    detail = form.get("detail") if form.get("detail") in DETAIL else "balanced"
    resolutions = DETAIL[detail][family]
    top_res = max(resolutions)
    batch = max(1, min(4, int(form.get("batch") or 1)))
    reserve = float(cfg["vram_reserve_gb"])

    precision = pick_precision(family, base_id, entry.get("precision"), gpu_total_gb, reserve, top_res, batch)
    bud = budget(family, precision, top_res, batch, gpu_total_gb, reserve)
    if not bud["fits"]:
        raise ValueError(
            f"Batch {batch} at {top_res} px needs about {bud['need_gb']} GB"
            + (f" even with {round(bud['offload'] * 100)}% of the model in system RAM" if bud["offload"] else "")
            + f", and this card has {bud['usable_gb']} GB to give. Lower the batch size or the detail level."
        )

    g = GOALS[goal]
    rank = int(form.get("rank") or fam.get("rank") or g["rank"])
    lr = float(form.get("lr") or g["lr"])
    steps = int(form.get("steps") or suggest_steps(goal, ds["images"], batch))
    save_every = 250 if steps >= 1000 else max(50, steps // 4 // 50 * 50)
    sample_every = int(form.get("sample_every") or save_every)
    previews = bool(form.get("previews", True))

    downloads, warnings = [], []
    uncaptioned = ds["images"] - ds["captioned"]
    if uncaptioned and not trigger:
        # allowed: those images train against an empty prompt
        warnings.append(
            f"{uncaptioned} of {ds['images']} images have no caption file and there is no trigger word, so they "
            "train with an empty caption. The LoRA then applies whenever it is loaded, with nothing to call it by."
        )
    if entry.get("download"):
        downloads.append(f"{entry['label']}: {entry['download']}")

    model: Dict = {"arch": family, "low_vram": True, "model_kwargs": {}}
    sample = {
        "sampler": "flowmatch", "sample_every": sample_every,
        "width": fam.get("sample_size", 1024), "height": fam.get("sample_size", 1024),
        "neg": "", "seed": 42, "walk_seed": True, **fam["sample"],
    }
    network_kwargs: Dict = {"ignore_if_contains": []}

    if base_id.startswith("file:"):
        model["name_or_path"] = base_id[5:]
    elif base_id == "krea2:raw":
        model["name_or_path"] = "krea/Krea-2-Raw"
    elif base_id == "moss_image:base":
        model["name_or_path"] = "OpenMOSS-Team/MOSS-Image-Base"
    elif base_id == "minimax_h3_image:base":
        model["name_or_path"] = H3_REPO

    model["quantize"] = precision != "bf16"
    if model["quantize"]:
        model["qtype"] = precision

    if family == "krea2":
        # Qwen3-VL-4B, quantized; it is unloaded once the captions are cached
        model["quantize_te"] = True
        model["qtype_te"] = "qfloat8"
        for repo, fname, what in (
            ("Qwen/Qwen3-VL-4B-Instruct", "config.json", "Qwen3-VL-4B text encoder, 8.3 GB"),
            ("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors", "Qwen-Image VAE, 0.25 GB"),
        ):
            if settings.hf_cached_file(repo, fname) is None:
                downloads.append(what)
    elif family == "minimax_h3_image":
        # both files are used exactly as ComfyUI ships them
        model["quantize_te"] = True
        model["qtype_te"] = "nvfp4"
        model["assistant_lora_path"] = H3_ADAPTER
        network_kwargs["ignore_if_contains"] = ["adaln_proj"]
        if bud["offload"] > 0:
            model["layer_offloading"] = True
            model["layer_offloading_transformer_percent"] = bud["offload"]
            # the text encoder gets the card to itself while captions are cached
            model["layer_offloading_text_encoder_percent"] = 0.0
    else:
        model["quantize_te"] = True
        model["qtype_te"] = "convrot8"
        model["model_kwargs"]["uncond_skip_layers"] = [22]
        # leave the block the negative branch skips to the base model
        network_kwargs["ignore_if_contains"] = ["layers.22."]
        te_choice = form.get("text_encoder") or cat["moss_text_encoders"][0]["id"]
        te_entry = next((t for t in cat["moss_text_encoders"] if t["id"] == te_choice), cat["moss_text_encoders"][0])
        if te_entry["id"] != "official":
            model["te_name_or_path"] = te_entry["id"]
        if te_entry.get("download"):
            downloads.append(f"Text encoder: {te_entry['download']}")
        if cat["moss_vae"]["path"]:
            if Path(cat["moss_vae"]["path"]).name != "flux2-vae.safetensors":
                model["vae_path"] = cat["moss_vae"]["path"]
        else:
            downloads.append(f"VAE: {cat['moss_vae']['download']}")

    if ds["small"] and top_res >= 1024:
        warnings.append(
            f"{ds['small']} of {ds['checked']} checked images are under 768 px on their short side; "
            f"they will be upscaled for the {top_res} px bucket."
        )
    if bud["offload"]:
        warnings.append(
            f"{round(bud['offload'] * 100)}% of the transformer ({bud['ram_gb']} GB) stays in system RAM and is "
            "streamed to the card every step. That is what makes this model fit, and it makes each step slower."
        )
    if ds["images"] < 8:
        warnings.append("Fewer than 8 images. Expect an overfit LoRA; 15 to 40 varied images work better.")

    datasets = [
        {
            "folder_path": f["path"],
            "caption_ext": "txt",
            "default_caption": trigger,
            # captions are encoded once and the text encoder is unloaded, so
            # there is no per-step dropout
            "caption_dropout_rate": 0,
            "cache_latents_to_disk": True,
            "resolution": list(resolutions),
            "num_repeats": 1,
        }
        for f in ds["folders"]
    ]

    prompts = [p.strip() for p in (form.get("prompts") or "").splitlines() if p.strip()]
    if not prompts:
        fallback = {"character": "a person", "style": "in this style", "concept": "the subject"}[goal]
        prompts = [p.format(t=trigger or fallback) for p in DEFAULT_PROMPTS[goal]]
    sample["samples"] = [{"prompt": p} for p in prompts[:6]]

    process = {
        "type": "sd_trainer",
        "training_folder": str(settings.OUTPUT_DIR),
        "device": "cuda:0",
        "performance_log_every": 250,
        "network": {"type": "lora", "linear": rank, "linear_alpha": rank, "network_kwargs": network_kwargs},
        "save": {"dtype": "bf16", "save_every": save_every, "max_step_saves_to_keep": 6},
        "datasets": datasets,
        "train": {
            "batch_size": batch,
            "steps": steps,
            "gradient_accumulation": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "optimizer": "adamw8bit",
            "optimizer_params": {"weight_decay": 1e-4},
            "lr": lr,
            "timestep_type": fam["timestep_type"],
            "content_or_style": "balanced",
            "dtype": "bf16",
            "cache_text_embeddings": True,
            "unload_text_encoder": False,
            "ema_config": {"use_ema": False},
            "skip_first_sample": True,
            "disable_sampling": not previews,
            "loss_type": "mse",
        },
        "model": model,
        "sample": sample,
    }
    if trigger:
        process["trigger_word"] = trigger

    config = {
        "job": "extension",
        "config": {"name": name, "process": [process]},
        "meta": {"name": "[name]", "version": "1.0"},
    }
    return {
        "name": name, "family": family, "base": entry["label"], "base_id": base_id,
        "goal": goal, "detail": detail, "precision": precision, "resolutions": resolutions,
        "batch": batch, "rank": rank, "lr": lr, "steps": steps, "save_every": save_every,
        "images": ds["images"], "captioned": ds["captioned"], "trigger": trigger,
        "budget": bud, "downloads": downloads, "warnings": warnings,
        "why": explain(family, precision, bud, resolutions, batch, rank, gpu_total_gb),
        "measured_key": measured_key(family, precision, top_res, batch),
        "config": config,
    }


def fit_table(cfg: dict, total_gb: float) -> List[dict]:
    """Per model: what fits on this card at each detail level and batch size."""
    reserve = float(cfg["vram_reserve_gb"])
    out = []
    for family, fam in FAMILIES.items():
        rows = []
        for key, d in DETAIL.items():
            res = max(d[family])
            cells = []
            for batch in (1, 2, 4):
                precision = pick_precision(family, "", None, total_gb, reserve, res, batch)
                b = budget(family, precision, res, batch, total_gb, reserve)
                cells.append({
                    "batch": batch, "precision": precision, "need_gb": b["need_gb"], "fits": b["fits"],
                    "offload": b["offload"], "source": b["source"],
                })
            rows.append({"detail": d["label"], "resolutions": d[family], "cells": cells})
        res = max(DETAIL["balanced"][family])
        precision = pick_precision(family, "", None, total_gb, reserve, res, 1)
        bud = budget(family, precision, res, 1, total_gb, reserve)
        rank = fam.get("rank") or GOALS["character"]["rank"]
        out.append({
            "family": family, "label": fam["label"], "params": fam["params"], "rows": rows,
            "why": explain(family, precision, bud, DETAIL["balanced"][family], 1, rank, total_gb),
        })
    return out
