"""GPU state (nvidia-smi) and the ComfyUI instance that shares the card."""

import json
import subprocess
import urllib.request
from typing import Optional

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def gpu() -> Optional[dict]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=NO_WINDOW,
        ).stdout.strip().splitlines()[0]
        name, total, used, temp, util = [p.strip() for p in out.split(",")]
        total_gb, used_gb = int(total) / 1024, int(used) / 1024
        return {
            "name": name.replace("NVIDIA GeForce ", ""),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(total_gb - used_gb, 2),
            "temp_c": int(temp) if temp.isdigit() else None,
            "util": int(util) if util.isdigit() else None,
        }
    except Exception:
        return None


def _get(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def comfy_status(base_url: str) -> dict:
    """Is ComfyUI up, and is it in the middle of a job?"""
    try:
        queue = _get(base_url.rstrip("/") + "/queue")
    except Exception:
        return {"running": False, "busy": False, "pending": 0}
    return {
        "running": True,
        "busy": len(queue.get("queue_running", [])) > 0,
        "pending": len(queue.get("queue_pending", [])),
    }


def comfy_free(base_url: str) -> bool:
    """Ask ComfyUI to unload its models. It reloads them on its next prompt."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False
