"""Training runs: one at a time on the card, the rest queued.

The engine writes straight to jobs/<id>/train.log and this module tails that
file, so a run survives the toolkit window being closed and is picked up again
when it reopens.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import hardware, recipes, settings

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

PROGRESS = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+), *([\d.]+)(s/it|it/s)")
LOSS = re.compile(r"loss: ([0-9.eE+-]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ACTIVE = ("queued", "waiting", "starting", "running")


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=8, creationflags=NO_WINDOW,
        ).stdout
        return f'"{pid}"' in out
    except Exception:
        return False


class Job:
    def __init__(self, folder: Path):
        self.folder = folder
        self.meta: dict = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        self.proc: Optional[subprocess.Popen] = None
        self.offset = 0
        self.carry = ""
        self.step = 0
        self.total = self.meta.get("steps", 0)
        self.speed = ""
        self.eta = ""
        self.phase = ""
        self.loss: List[List[float]] = []
        self.vram: List[float] = []
        self.peak_gb = self.meta.get("peak_gb", 0.0)
        self.tail: List[str] = []
        self.spill_ticks = 0
        self.train_started = False

    # -- persistence -------------------------------------------------------
    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def status(self) -> str:
        return self.meta["status"]

    def save(self):
        tmp = self.folder / "job.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        os.replace(tmp, self.folder / "job.json")

    def set(self, status: str, message: str = ""):
        self.meta["status"] = status
        self.meta["message"] = message
        self.save()

    # -- log tailing -------------------------------------------------------
    def read_log(self):
        log = self.folder / "train.log"
        if not log.exists():
            return
        with open(log, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        if not chunk:
            return
        self.offset += len(chunk)
        text = self.carry + chunk.decode("utf-8", errors="replace")
        parts = re.split(r"[\r\n]+", text)
        self.carry = parts.pop()
        for line in parts:
            line = ANSI.sub("", line).strip()
            if not line:
                continue
            m = PROGRESS.search(line)
            lm = LOSS.search(line)
            if m and lm and int(m.group(2)) == self.total:
                step = int(m.group(1))
                self.train_started = True
                self.phase = "Training"
                rate = float(m.group(5))
                self.speed = f"{rate:.2f} {m.group(6)}"
                self.eta = m.group(4)
                if step != self.step:
                    self.step = step
                    try:
                        self.loss.append([step, float(lm.group(1))])
                    except ValueError:
                        pass
                continue
            if m:
                # some other progress bar (caching latents, quantizing, sampling)
                label = line.split(":")[0].strip()[:60]
                self.phase = f"{label} {m.group(1)}/{m.group(2)}"
                continue
            self.tail.append(line[:400])
            if len(self.tail) > 200:
                self.tail = self.tail[-200:]
            if len(line) < 120 and not line.startswith(("warnings.warn", "File ")):
                self.phase = line

    # -- views -------------------------------------------------------------
    def out_dir(self) -> Path:
        return settings.OUTPUT_DIR / self.meta["name"]

    def samples(self) -> List[str]:
        folder = self.out_dir() / "samples"
        if not folder.is_dir():
            return []
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".webp"))]
        return sorted(files, reverse=True)[:24]

    def checkpoints(self) -> List[str]:
        d = self.out_dir()
        if not d.is_dir():
            return []
        return sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))

    def view(self, full: bool = False) -> dict:
        loss = self.loss
        if len(loss) > 300:
            k = len(loss) // 300 + 1
            loss = [
                [chunk[-1][0], sum(p[1] for p in chunk) / len(chunk)]
                for chunk in (loss[i:i + k] for i in range(0, len(loss), k))
            ]
        data = {
            **{k: v for k, v in self.meta.items() if k != "plan"},
            "step": self.step, "total": self.total, "speed": self.speed, "eta": self.eta,
            "phase": self.phase, "peak_gb": round(self.peak_gb, 2),
            "last_loss": self.loss[-1][1] if self.loss else None,
            "summary": {
                k: self.meta["plan"].get(k)
                for k in ("base", "family", "goal", "precision", "resolutions", "batch", "rank", "lr", "images", "budget")
            },
        }
        if full:
            data.update({
                "loss": loss, "vram": self.vram[-300:], "samples": self.samples(),
                "checkpoints": self.checkpoints(), "log": self.tail[-60:],
            })
        return data


class Manager:
    def __init__(self):
        self.cfg = settings.load()
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.RLock()
        settings.JOBS_DIR.mkdir(exist_ok=True)
        settings.OUTPUT_DIR.mkdir(exist_ok=True)
        for folder in sorted(settings.JOBS_DIR.iterdir()):
            if (folder / "job.json").exists():
                try:
                    job = Job(folder)
                except Exception:
                    continue
                if job.status in ("starting", "running"):
                    if _pid_alive(job.meta.get("pid")):
                        job.read_log()  # rebuild the curve from the log
                    else:
                        job.read_log()
                        job.set("interrupted", "The toolkit was closed while this was running. Resume to continue from the last checkpoint.")
                elif job.status == "waiting":
                    job.set("queued")
                else:
                    job.read_log()
                self.jobs[job.id] = job
        threading.Thread(target=self._loop, daemon=True).start()

    # -- public ------------------------------------------------------------
    def create(self, plan: dict, copy_to_comfy: bool = True) -> Job:
        with self.lock:
            if any(j.meta["name"] == plan["name"] and j.status in ACTIVE for j in self.jobs.values()):
                raise ValueError(f"A run named {plan['name']} is already queued or running.")
            jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            folder = settings.JOBS_DIR / jid
            folder.mkdir(parents=True)
            (folder / "config.yaml").write_text(
                yaml.safe_dump(plan["config"], sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            meta = {
                "id": jid, "name": plan["name"], "status": "queued", "message": "",
                "created": time.time(), "steps": plan["steps"], "copy_to_comfy": copy_to_comfy,
                "plan": {k: v for k, v in plan.items() if k != "config"},
            }
            (folder / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            job = Job(folder)
            self.jobs[jid] = job
            return job

    def stop(self, jid: str):
        with self.lock:
            job = self.jobs[jid]
            if job.status in ("queued", "waiting"):
                job.set("stopped", "Removed from the queue.")
                return
            if job.status in ("starting", "running"):
                self._kill(job)
                job.set("stopped", "Stopped. Saved checkpoints are kept; resume continues from the latest one.")

    def resume(self, jid: str):
        with self.lock:
            job = self.jobs[jid]
            if job.status in ACTIVE:
                return
            if job.status == "done":
                raise ValueError("This run already finished.")
            job.set("queued")

    def delete(self, jid: str, with_output: bool = False):
        with self.lock:
            job = self.jobs[jid]
            if job.status in ("starting", "running"):
                raise ValueError("Stop the run before deleting it.")
            if with_output:
                others = any(j.id != jid and j.meta["name"] == job.meta["name"] for j in self.jobs.values())
                if not others:
                    shutil.rmtree(job.out_dir(), ignore_errors=True)
            shutil.rmtree(job.folder, ignore_errors=True)
            del self.jobs[jid]

    def list(self) -> List[dict]:
        with self.lock:
            return [j.view() for j in sorted(self.jobs.values(), key=lambda j: j.meta["created"], reverse=True)]

    # -- internals ---------------------------------------------------------
    def _kill(self, job: Job):
        pid = job.meta.get("pid")
        if pid and _pid_alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        job.proc = None

    def _start(self, job: Job, gpu: dict):
        log = open(job.folder / "train.log", "ab")
        log.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
        log.flush()
        env = dict(os.environ)
        env.update({
            "MODELS_PATH": self.cfg["comfy_models"],
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        })
        job.proc = subprocess.Popen(
            [str(settings.ENGINE_PY), "run.py", str(job.folder / "config.yaml")],
            cwd=str(settings.ENGINE_DIR), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=NO_WINDOW | NEW_GROUP,
        )
        log.close()
        job.meta["pid"] = job.proc.pid
        job.meta["started"] = time.time()
        job.meta["baseline_gb"] = gpu["used_gb"]
        job.offset = (job.folder / "train.log").stat().st_size
        job.carry, job.step, job.loss, job.vram, job.tail = "", 0, [], [], []
        job.peak_gb, job.spill_ticks, job.train_started = 0.0, 0, False
        job.phase = "Starting the engine"
        job.set("running")

    def _finish(self, job: Job, code: Optional[int]):
        job.read_log()
        job.meta["finished"] = time.time()
        job.meta["peak_gb"] = round(job.peak_gb, 2)
        final = job.out_dir() / f"{job.meta['name']}.safetensors"
        if job.step >= job.total > 0 and final.exists():
            copied = ""
            if job.meta.get("copy_to_comfy"):
                copied = self._deliver(job, final)
            job.meta["lora"] = str(final)
            if job.train_started and job.peak_gb:
                recipes.record_measurement(
                    job.meta["plan"]["measured_key"], job.peak_gb, job.meta.get("baseline_gb", 0.0)
                )
            job.set("done", copied or f"Saved to {final}")
            return
        reason = next(
            (l for l in reversed(job.tail) if "Error" in l or "error" in l or "Exception" in l or "CUDA" in l), ""
        )
        if "out of memory" in " ".join(job.tail[-40:]).lower():
            reason = "The GPU ran out of memory. Free VRAM in other apps, or lower the batch size or detail level."
        job.set("failed", reason or f"The engine exited with code {code}. Open the log for details.")

    def _deliver(self, job: Job, final: Path) -> str:
        family = job.meta["plan"]["family"]
        target_dir = Path(self.cfg["comfy_models"]) / self.cfg["lora_dirs"].get(family, "loras")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / final.name
            n = 2
            while target.exists():
                target = target_dir / f"{final.stem}_v{n}{final.suffix}"
                n += 1
            shutil.copy2(final, target)
            job.meta["delivered"] = str(target)
            return f"Copied to {target}"
        except Exception as e:
            return f"Saved to {final} (copy to ComfyUI failed: {e})"

    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception as e:  # keep the scheduler alive
                print("scheduler error:", e)
            time.sleep(2)

    def _tick(self):
        gpu = hardware.gpu()
        with self.lock:
            running = [j for j in self.jobs.values() if j.status in ("starting", "running")]
            for job in running:
                job.read_log()
                if gpu:
                    job.vram.append(gpu["used_gb"])
                    if len(job.vram) > 1800:
                        job.vram = job.vram[-1800:]
                    job.peak_gb = max(job.peak_gb, gpu["used_gb"])
                    # dedicated memory pinned at the ceiling means the driver
                    # is paging into shared system memory
                    job.spill_ticks = job.spill_ticks + 1 if gpu["free_gb"] < 0.3 else 0
                    job.meta["spilling"] = job.spill_ticks >= 15
                code = job.proc.poll() if job.proc is not None else (None if _pid_alive(job.meta.get("pid")) else -1)
                if code is not None:
                    self._finish(job, code)
            if running or not gpu:
                return
            queued = sorted(
                (j for j in self.jobs.values() if j.status in ("queued", "waiting")),
                key=lambda j: j.meta["created"],
            )
            if not queued:
                return
            job = queued[0]
            if not settings.ENGINE_PY.exists():
                job.set("failed", "The training engine is not installed. Run setup.bat.")
                return
            need = job.meta["plan"]["budget"]["need_gb"] + 0.5
            if gpu["free_gb"] < need:
                job.set(
                    "waiting",
                    f"Needs {need:.1f} GB free, the card has {gpu['free_gb']:.1f} GB. "
                    "Starts by itself once other apps release memory.",
                )
                return
            self._start(job, gpu)
