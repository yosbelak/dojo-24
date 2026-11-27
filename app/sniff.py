"""Read a .safetensors header (no tensor data) and work out what the file is.

Stdlib only, so it runs before the training environment exists. Used to list
which of the checkpoints already sitting in the ComfyUI models folder can be
trained on, and with which quantization the engine has to be asked for.
"""

import json
import os
import struct
from typing import Dict, Optional

MAX_HEADER = 256 << 20


def read_header(path: str) -> Dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER:
            raise ValueError("implausible safetensors header")
        return json.loads(f.read(n))


def _shape(header: Dict, key: str) -> Optional[list]:
    entry = header.get(key)
    return entry.get("shape") if isinstance(entry, dict) else None


def _strip(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def sniff(path: str) -> Dict:
    """-> {family, precision, quant_layout, trainable, note, size_gb}"""
    info = {
        "path": path,
        "name": os.path.basename(path),
        "size_gb": round(os.path.getsize(path) / (1 << 30), 2),
        "family": "unknown",
        "precision": "unknown",
        "quant_layout": None,
        "trainable": False,
        "note": "",
    }
    try:
        header = read_header(path)
    except Exception as e:  # unreadable / truncated download
        info["note"] = f"unreadable: {e}"
        return info

    meta = header.pop("__metadata__", None) or {}
    keys = {_strip(k): v for k, v in header.items()}
    prefixed = any(k.startswith(("model.diffusion_model.", "diffusion_model.")) for k in header)

    # ---- which model ---------------------------------------------------
    # Krea 2: SingleStreamDiT, 6144 wide, blocks.N.*, a text fusion stack
    # MOSS-Image: 42 x layers.N.*, 3072 wide, fused qkv projections
    has_blocks = any(k.startswith("blocks.") for k in keys)
    has_layers = any(k.startswith("layers.") for k in keys)
    has_txtfusion = any(k.startswith("txtfusion.") for k in keys)
    widths = set()
    for k, v in keys.items():
        if isinstance(v, dict) and len(v.get("shape") or []) == 2:
            widths.add(v["shape"][-1])
            if len(widths) > 64:
                break
    if has_blocks and has_txtfusion:
        info["family"] = "krea2"
    elif has_layers and any(".proj.qkv." in k or ".qkv." in k for k in keys):
        info["family"] = "moss_image"
    elif has_layers and 3072 in widths and not has_blocks:
        info["family"] = "moss_image"
    elif any(k.startswith(("lora_", "transformer.")) or ".lora_A." in k or ".lora_down." in k for k in header):
        info["family"] = "lora"

    # ---- how it is stored ----------------------------------------------
    dtypes: Dict[str, int] = {}
    for v in keys.values():
        if isinstance(v, dict) and "dtype" in v:
            start, end = v.get("data_offsets", [0, 0])
            dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + (end - start)
    dominant = max(dtypes, key=dtypes.get) if dtypes else "?"
    marker_tensors = any(k.endswith(".comfy_quant") for k in keys)
    header_markers = "_quantization_metadata" in meta
    scaled_fp8 = "scaled_fp8" in keys

    quant_format = None
    if header_markers:
        try:
            layers = json.loads(meta["_quantization_metadata"]).get("layers", {})
            formats = {str(c.get("format")) for c in layers.values() if isinstance(c, dict)}
            quant_format = "/".join(sorted(formats)) or None
        except Exception:
            quant_format = "?"
        info["quant_layout"] = "header"
    elif marker_tensors:
        info["quant_layout"] = "markers"
    elif scaled_fp8:
        info["quant_layout"] = "scaled_fp8"

    if dominant in ("I8", "U8"):
        info["precision"] = "int8"
    elif dominant == "F8_E4M3":
        info["precision"] = "fp8"
    elif dominant == "BF16":
        info["precision"] = "bf16"
    elif dominant == "F16":
        info["precision"] = "fp16"
    elif dominant == "F32":
        info["precision"] = "fp32"
    else:
        info["precision"] = dominant.lower()
    if quant_format:
        info["quant_format"] = quant_format

    info["prefixed"] = prefixed
    info["tensors"] = len(keys)
    return info


if __name__ == "__main__":
    import sys

    for root in sys.argv[1:]:
        paths = []
        if os.path.isfile(root):
            paths = [root]
        else:
            for d, _, files in os.walk(root):
                paths += [os.path.join(d, f) for f in files if f.endswith(".safetensors")]
        for p in sorted(paths):
            i = sniff(p)
            print(
                f"{i['size_gb']:>6} GB  {i['family']:<10} {i['precision']:<5} "
                f"layout={i['quant_layout']} fmt={i.get('quant_format')} "
                f"prefixed={i.get('prefixed')} n={i.get('tensors')}  {i['name']}  {i['note']}"
            )
