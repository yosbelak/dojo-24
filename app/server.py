"""Local web app. Run with: engine\\venv\\Scripts\\python.exe -m app.server"""

import io
import os
import string
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from . import hardware, recipes, settings
from .jobs import Manager

cfg = settings.load()
app = FastAPI(title="Dojo 24", docs_url=None, redoc_url=None)
manager: Manager = None  # created on startup
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def _startup():
    global manager
    manager = Manager()


@app.middleware("http")
async def local_only(request: Request, call_next):
    # This app can list folders and read images, so refuse requests that reach
    # it under any other host name (DNS rebinding from a web page).
    host = (request.headers.get("host") or "").split(":")[0].lower()
    allowed = {"127.0.0.1", "localhost", cfg["host"].lower()}
    if cfg["host"] in ("127.0.0.1", "localhost") and host not in allowed:
        return PlainTextResponse("Forbidden", status_code=403)
    return await call_next(request)


def fail(e: Exception):
    raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/system")
def system():
    gpu = hardware.gpu()
    return {
        "gpu": gpu,
        "reserve_gb": cfg["vram_reserve_gb"],
        "comfy": hardware.comfy_status(cfg["comfy_url"]),
        "engine_ready": settings.ENGINE_PY.exists() and (settings.ENGINE_DIR / "run.py").exists(),
        "comfy_models": cfg["comfy_models"],
        "lora_dirs": cfg["lora_dirs"],
    }


@app.get("/api/catalog")
def catalog():
    return {
        **recipes.catalog(cfg),
        "families": {k: {"label": v["label"], "params": v["params"]} for k, v in recipes.FAMILIES.items()},
        "goals": recipes.GOALS,
        "detail": recipes.DETAIL,
    }


@app.get("/api/fit")
def fit():
    gpu = hardware.gpu()
    total = gpu["total_gb"] if gpu else 24.0
    return {"gpu": gpu, "reserve_gb": cfg["vram_reserve_gb"], "models": recipes.fit_table(cfg, total)}


@app.post("/api/dataset")
async def dataset(request: Request):
    body = await request.json()
    info = recipes.inspect_dataset(body.get("path") or "")
    if info.get("ok"):
        info["suggested_steps"] = {
            g: recipes.suggest_steps(g, info["images"], int(body.get("batch") or 1)) for g in recipes.GOALS
        }
    return info


@app.get("/api/thumb")
def thumb(path: str):
    p = Path(path)
    if p.suffix.lower() not in recipes.IMAGE_EXTS or not p.is_file():
        raise HTTPException(404)
    from PIL import Image

    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((220, 220))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=600"})


@app.get("/api/browse")
def browse(path: str = ""):
    if not path:
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        return {"path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in drives]}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, "Folder not found")
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.is_dir() and not child.name.startswith((".", "$")):
                dirs.append({"name": child.name, "path": str(child)})
    except PermissionError:
        pass
    try:
        images = sum(1 for c in p.iterdir() if c.suffix.lower() in recipes.IMAGE_EXTS)
    except PermissionError:
        images = 0
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "images": images}


def _plan(body: dict) -> dict:
    gpu = hardware.gpu()
    if gpu is None:
        raise ValueError("No NVIDIA GPU found (nvidia-smi did not answer).")
    plan = recipes.build_plan(body, cfg, gpu["total_gb"])
    plan["gpu"] = gpu
    return plan


@app.post("/api/plan")
async def plan(request: Request):
    try:
        p = _plan(await request.json())
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
    import yaml

    p["yaml"] = yaml.safe_dump(p.pop("config"), sort_keys=False, allow_unicode=True)
    return {"ok": True, **p}


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    try:
        job = manager.create(_plan(body), copy_to_comfy=bool(body.get("copy_to_comfy", True)))
    except Exception as e:
        fail(e)
    return job.view()


@app.get("/api/jobs")
def jobs():
    return manager.list()


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in manager.jobs:
        raise HTTPException(404)
    with manager.lock:
        return manager.jobs[jid].view(full=True)


@app.post("/api/jobs/{jid}/{action}")
async def job_action(jid: str, action: str, request: Request):
    if jid not in manager.jobs:
        raise HTTPException(404)
    try:
        if action == "stop":
            manager.stop(jid)
        elif action == "resume":
            manager.resume(jid)
        elif action == "delete":
            body = await request.json()
            manager.delete(jid, with_output=bool(body.get("with_output")))
        else:
            raise HTTPException(404)
    except ValueError as e:
        fail(e)
    return {"ok": True}


@app.get("/api/jobs/{jid}/sample/{name}")
def sample(jid: str, name: str):
    if jid not in manager.jobs or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    path = manager.jobs[jid].out_dir() / "samples" / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


@app.get("/api/jobs/{jid}/config")
def job_config(jid: str):
    if jid not in manager.jobs:
        raise HTTPException(404)
    return PlainTextResponse((manager.jobs[jid].folder / "config.yaml").read_text(encoding="utf-8"))


@app.post("/api/comfy/free")
async def comfy_free(request: Request):
    body = await request.json()
    status = hardware.comfy_status(cfg["comfy_url"])
    if not status["running"]:
        return {"ok": False, "error": "ComfyUI is not running, so something else is holding the memory."}
    if status["busy"] and not body.get("force"):
        return {"ok": False, "busy": True, "error": "ComfyUI is generating right now."}
    return {"ok": hardware.comfy_free(cfg["comfy_url"])}


def main():
    url = f"http://{cfg['host']}:{cfg['port']}"
    print(f"\n  Dojo 24  ->  {url}\n  Close this window to shut Dojo 24 down. A run in progress keeps going.\n")
    if "--open" in sys.argv:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]), log_level="warning")


if __name__ == "__main__":
    main()
