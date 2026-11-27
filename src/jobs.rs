//! Training jobs: one at a time on the card, the rest queued.
//!
//! The engine writes straight to `jobs\<id>\train.log` and this module tails that file, so
//! a run survives the window being closed and is picked up again when Dojo 24 reopens.
//! `job.json` keeps the format of the earlier Python version, so its jobs still load.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::hardware::{Gpu, Monitor, NO_WINDOW};
use crate::recipes::{self, Plan};
use crate::settings::{Paths, Settings};

const NEW_PROCESS_GROUP: u32 = 0x0000_0200;
pub const ACTIVE: [&str; 4] = ["queued", "waiting", "starting", "running"];

static PROGRESS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+), *([\d.]+)(s/it|it/s)").unwrap());
static LOSS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"loss: ([0-9.eE+-]+)").unwrap());
static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").unwrap());

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Meta {
    pub id: String,
    pub name: String,
    pub status: String,
    pub message: String,
    pub created: f64,
    pub steps: u32,
    pub copy_to_comfy: bool,
    pub plan: Plan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished: Option<f64>,
    pub baseline_gb: f32,
    pub peak_gb: f32,
    pub spilling: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivered: Option<String>,
}

pub struct Job {
    pub folder: PathBuf,
    pub meta: Meta,
    child: Option<Child>,
    offset: u64,
    carry: String,
    pub step: u32,
    pub total: u32,
    pub speed: String,
    pub eta: String,
    pub phase: String,
    pub loss: Vec<(u32, f32)>,
    pub vram: Vec<f32>,
    pub tail: Vec<String>,
    spill_ticks: u32,
    train_started: bool,
}

impl Job {
    fn load(folder: &Path) -> Option<Self> {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(folder.join("job.json")).ok()?).ok()?;
        Some(Self {
            folder: folder.to_path_buf(),
            total: meta.steps,
            meta,
            child: None,
            offset: 0,
            carry: String::new(),
            step: 0,
            speed: String::new(),
            eta: String::new(),
            phase: String::new(),
            loss: Vec::new(),
            vram: Vec::new(),
            tail: Vec::new(),
            spill_ticks: 0,
            train_started: false,
        })
    }

    pub fn id(&self) -> &str {
        &self.meta.id
    }

    pub fn status(&self) -> &str {
        &self.meta.status
    }

    pub fn active(&self) -> bool {
        ACTIVE.contains(&self.meta.status.as_str())
    }

    fn save(&self) {
        let tmp = self.folder.join("job.json.tmp");
        if let Ok(text) = serde_json::to_string_pretty(&self.meta)
            && std::fs::write(&tmp, text).is_ok()
        {
            let _ = std::fs::rename(&tmp, self.folder.join("job.json"));
        }
    }

    fn set(&mut self, status: &str, message: impl Into<String>) {
        self.meta.status = status.into();
        self.meta.message = message.into();
        self.save();
    }

    /// The config this job runs: config.json, or config.yaml from the Python version.
    pub fn config_path(&self) -> PathBuf {
        let json = self.folder.join("config.json");
        if json.exists() { json } else { self.folder.join("config.yaml") }
    }

    pub fn out_dir(&self, paths: &Paths) -> PathBuf {
        paths.output.join(&self.meta.name)
    }

    /// Reads what the engine wrote since the last call.
    fn read_log(&mut self) {
        let Ok(mut f) = std::fs::File::open(self.folder.join("train.log")) else { return };
        if f.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() || buf.is_empty() {
            return;
        }
        self.offset += buf.len() as u64;
        let text = format!("{}{}", self.carry, String::from_utf8_lossy(&buf));
        let mut parts: Vec<&str> = text.split(['\r', '\n']).collect();
        self.carry = parts.pop().unwrap_or("").to_string();
        for raw in parts {
            let line = ANSI.replace_all(raw, "");
            let line = line.trim();
            if !line.is_empty() {
                self.parse_line(line);
            }
        }
    }

    fn parse_line(&mut self, line: &str) {
        let progress = PROGRESS.captures(line);
        let loss = LOSS.captures(line);
        if let (Some(m), Some(l)) = (&progress, &loss)
            && m[2].parse::<u32>().ok() == Some(self.total)
        {
            let step: u32 = m[1].parse().unwrap_or(0);
            self.train_started = true;
            self.phase = "Training".into();
            self.speed = format!("{:.2} {}", m[5].parse::<f32>().unwrap_or(0.0), &m[6]);
            self.eta = m[4].to_string();
            if step != self.step {
                self.step = step;
                if let Ok(v) = l[1].parse::<f32>() {
                    self.loss.push((step, v));
                }
            }
            return;
        }
        if let Some(m) = progress {
            // another progress bar: caching latents, quantizing, sampling, downloading
            let label: String = line.split(':').next().unwrap_or("").trim().chars().take(60).collect();
            self.phase = format!("{label} {}/{}", &m[1], &m[2]);
            return;
        }
        self.tail.push(line.chars().take(400).collect());
        if self.tail.len() > 300 {
            self.tail.drain(..self.tail.len() - 300);
        }
        if line.len() < 120 && !line.starts_with("warnings.warn") && !line.starts_with("File ") {
            self.phase = line.to_string();
        }
    }

    pub fn samples(&self, paths: &Paths) -> Vec<PathBuf> {
        let dir = self.out_dir(paths).join("samples");
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| recipes::is_image(p)).collect())
            .unwrap_or_default();
        files.sort();
        files.reverse();
        files.truncate(24);
        files
    }

    pub fn checkpoints(&self, paths: &Paths) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(self.out_dir(paths))
            .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().map(String::from)).filter(|n| n.ends_with(".safetensors")).collect())
            .unwrap_or_default();
        out.sort();
        out
    }

    pub fn last_loss(&self) -> Option<f32> {
        self.loss.last().map(|l| l.1)
    }

}

fn pid_alive(pid: Option<u32>) -> bool {
    let Some(pid) = pid else { return false };
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .creation_flags(NO_WINDOW)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")))
        .unwrap_or(false)
}

fn kill_tree(pid: u32) {
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).creation_flags(NO_WINDOW).output();
}

pub struct Manager {
    pub paths: Paths,
    pub cfg: Settings,
    pub jobs: Vec<Job>,
    /// While paused nothing new starts; a running job keeps going.
    pub paused: bool,
}

pub type Shared = Arc<Mutex<Manager>>;

impl Manager {
    pub fn new(paths: Paths, cfg: Settings) -> Self {
        let _ = std::fs::create_dir_all(&paths.jobs);
        let _ = std::fs::create_dir_all(&paths.output);
        let mut jobs = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&paths.jobs) {
            let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.join("job.json").is_file()).collect();
            dirs.sort();
            for d in dirs {
                let Some(mut job) = Job::load(&d) else { continue };
                job.read_log(); // rebuilds the loss curve from the log
                match job.status() {
                    "starting" | "running" if !pid_alive(job.meta.pid) => {
                        job.set("interrupted", "Dojo 24 was closed while this was running. Resume continues from the last checkpoint.")
                    }
                    "waiting" => job.set("queued", ""),
                    _ => {}
                }
                jobs.push(job);
            }
        }
        Self { paths, cfg, jobs, paused: false }
    }

    /// Starts the background loop that launches queued jobs and follows the running one.
    pub fn spawn(self, monitor: Monitor) -> Shared {
        let shared = Arc::new(Mutex::new(self));
        let s = shared.clone();
        std::thread::spawn(move || {
            loop {
                let gpu = monitor.gpu();
                s.lock().unwrap().tick(gpu.as_ref());
                std::thread::sleep(Duration::from_secs(2));
            }
        });
        shared
    }

    pub fn get(&self, id: &str) -> Option<&Job> {
        self.jobs.iter().find(|j| j.id() == id)
    }

    fn get_mut(&mut self, id: &str) -> Option<&mut Job> {
        self.jobs.iter_mut().find(|j| j.id() == id)
    }

    /// Newest first.
    pub fn ordered(&self) -> Vec<&Job> {
        let mut v: Vec<&Job> = self.jobs.iter().collect();
        v.sort_by(|a, b| b.meta.created.total_cmp(&a.meta.created));
        v
    }

    pub fn create(&mut self, plan: Plan, copy_to_comfy: bool) -> Result<String, String> {
        if self.jobs.iter().any(|j| j.meta.name == plan.name && j.active()) {
            return Err(format!("A job named {} is already queued or running.", plan.name));
        }
        let secs = now();
        let stamp = {
            // local-time-free id: seconds since epoch plus a short random-ish suffix
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
            format!("{:.0}-{:04x}", secs, nanos & 0xffff)
        };
        let folder = self.paths.jobs.join(&stamp);
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        std::fs::write(folder.join("config.json"), serde_json::to_string_pretty(&plan.config).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let meta = Meta {
            id: stamp.clone(),
            name: plan.name.clone(),
            status: "queued".into(),
            created: secs,
            steps: plan.steps,
            copy_to_comfy,
            plan,
            ..Default::default()
        };
        let job = Job { total: meta.steps, ..Job::load_from(folder, meta) };
        job.save();
        self.jobs.push(job);
        Ok(stamp)
    }

    pub fn stop(&mut self, id: &str) {
        let Some(job) = self.get_mut(id) else { return };
        match job.status() {
            "queued" | "waiting" => job.set("stopped", "Removed from the queue."),
            "starting" | "running" => {
                if let Some(pid) = job.meta.pid {
                    kill_tree(pid);
                }
                job.child = None;
                job.set("stopped", "Stopped. Saved checkpoints are kept; Resume continues from the latest one.");
            }
            _ => {}
        }
    }

    pub fn resume(&mut self, id: &str) -> Result<(), String> {
        let job = self.get_mut(id).ok_or("No such job.")?;
        if job.status() == "done" {
            return Err("This job already finished.".into());
        }
        if !job.active() {
            job.set("queued", "");
        }
        Ok(())
    }

    pub fn delete(&mut self, id: &str, with_output: bool) -> Result<(), String> {
        let i = self.jobs.iter().position(|j| j.id() == id).ok_or("No such job.")?;
        if matches!(self.jobs[i].status(), "starting" | "running") {
            return Err("Stop the job before deleting it.".into());
        }
        if with_output {
            let name = self.jobs[i].meta.name.clone();
            if !self.jobs.iter().any(|j| j.id() != id && j.meta.name == name) {
                let _ = std::fs::remove_dir_all(self.jobs[i].out_dir(&self.paths));
            }
        }
        let job = self.jobs.remove(i);
        let _ = std::fs::remove_dir_all(&job.folder);
        Ok(())
    }

    fn start(&mut self, idx: usize, gpu: &Gpu) {
        let paths = &self.paths;
        let comfy_models = self.cfg.comfy_models.clone();
        let job = &mut self.jobs[idx];
        let log_path = job.folder.join("train.log");
        let mut log = match OpenOptions::new().create(true).append(true).open(&log_path) {
            Ok(f) => f,
            Err(e) => return job.set("failed", format!("Could not open the log file: {e}")),
        };
        let _ = writeln!(log, "\n===== run started at {:.0} =====", now());
        let Ok(err) = log.try_clone() else { return job.set("failed", "Could not open the log file.") };
        let child = Command::new(&paths.engine_py)
            .arg("run.py")
            .arg(job.config_path())
            .current_dir(&paths.engine_dir)
            .stdin(Stdio::null())
            .stdout(log)
            .stderr(err)
            .env("MODELS_PATH", comfy_models)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            .env("NO_ALBUMENTATIONS_UPDATE", "1")
            .creation_flags(NO_WINDOW | NEW_PROCESS_GROUP)
            .spawn();
        match child {
            Ok(c) => {
                job.meta.pid = Some(c.id());
                job.meta.started = Some(now());
                job.meta.baseline_gb = gpu.used_gb;
                job.offset = std::fs::metadata(&log_path).map_or(0, |m| m.len());
                job.child = Some(c);
                job.carry.clear();
                job.step = 0;
                job.loss.clear();
                job.vram.clear();
                job.tail.clear();
                job.meta.peak_gb = 0.0;
                job.spill_ticks = 0;
                job.train_started = false;
                job.phase = "Starting the engine".into();
                job.set("running", "");
            }
            Err(e) => job.set("failed", format!("Could not start the engine: {e}")),
        }
    }

    fn finish(&mut self, idx: usize, code: Option<i32>) {
        let paths = &self.paths;
        let cfg = &self.cfg;
        let job = &mut self.jobs[idx];
        job.read_log();
        job.child = None;
        job.meta.finished = Some(now());
        let final_file = job.out_dir(paths).join(format!("{}.safetensors", job.meta.name));
        if job.total > 0 && job.step >= job.total && final_file.exists() {
            let mut msg = format!("Saved to {}", final_file.display());
            if job.meta.copy_to_comfy {
                msg = deliver(cfg, &job.meta.plan.family, &final_file).map(|t| {
                    job.meta.delivered = Some(t.display().to_string());
                    format!("Copied to {}", t.display())
                })
                .unwrap_or_else(|e| format!("Saved to {} (copy to ComfyUI failed: {e})", final_file.display()));
            }
            job.meta.lora = Some(final_file.display().to_string());
            if job.train_started && job.meta.peak_gb > 0.0 {
                recipes::record_measurement(paths, &job.meta.plan.measured_key, job.meta.peak_gb, job.meta.baseline_gb);
            }
            job.set("done", msg);
            return;
        }
        let recent = job.tail.iter().rev().take(40).cloned().collect::<Vec<_>>().join(" ").to_lowercase();
        let reason = if recent.contains("out of memory") {
            "The GPU ran out of memory. Free VRAM in other apps, or lower the batch size or resolution.".to_string()
        } else {
            job.tail
                .iter()
                .rev()
                .find(|l| l.contains("Error") || l.contains("error") || l.contains("Exception") || l.contains("CUDA"))
                .cloned()
                .unwrap_or_else(|| format!("The engine exited with code {}. Open the log for details.", code.map_or("?".into(), |c| c.to_string())))
        };
        job.set("failed", reason);
    }

    pub fn tick(&mut self, gpu: Option<&Gpu>) {
        let mut ended = Vec::new();
        for (i, job) in self.jobs.iter_mut().enumerate() {
            if !matches!(job.status(), "starting" | "running") {
                continue;
            }
            job.read_log();
            if let Some(g) = gpu {
                job.vram.push(g.used_gb);
                if job.vram.len() > 1800 {
                    job.vram.drain(..job.vram.len() - 1800);
                }
                job.meta.peak_gb = job.meta.peak_gb.max(g.used_gb);
                // dedicated memory pinned at the ceiling: the driver is paging into system RAM
                job.spill_ticks = if g.free_gb < 0.3 { job.spill_ticks + 1 } else { 0 };
                job.meta.spilling = job.spill_ticks >= 15;
            }
            let code = match &mut job.child {
                Some(c) => match c.try_wait() {
                    Ok(Some(st)) => Some(st.code()),
                    Ok(None) => None,
                    Err(_) => Some(None),
                },
                None => (!pid_alive(job.meta.pid)).then_some(None),
            };
            if let Some(code) = code {
                ended.push((i, code));
            }
        }
        for (i, code) in ended {
            self.finish(i, code);
        }
        let Some(gpu) = gpu else { return };
        if self.paused || self.jobs.iter().any(|j| matches!(j.status(), "starting" | "running")) {
            return;
        }
        let Some(idx) = (0..self.jobs.len())
            .filter(|&i| matches!(self.jobs[i].status(), "queued" | "waiting"))
            .min_by(|&a, &b| self.jobs[a].meta.created.total_cmp(&self.jobs[b].meta.created))
        else {
            return;
        };
        if !self.paths.engine_ready() {
            return self.jobs[idx].set("failed", "The training engine is not installed. Run setup.bat.");
        }
        let need = self.jobs[idx].meta.plan.budget.need_gb + 0.5;
        if gpu.free_gb < need {
            let msg = format!("Needs {need:.1} GB free, the card has {:.1} GB. Starts by itself once other apps release memory.", gpu.free_gb);
            if self.jobs[idx].meta.message != msg {
                self.jobs[idx].set("waiting", msg);
            }
            return;
        }
        self.start(idx, gpu);
    }
}

impl Job {
    fn load_from(folder: PathBuf, meta: Meta) -> Self {
        Self {
            folder,
            total: meta.steps,
            meta,
            child: None,
            offset: 0,
            carry: String::new(),
            step: 0,
            speed: String::new(),
            eta: String::new(),
            phase: String::new(),
            loss: Vec::new(),
            vram: Vec::new(),
            tail: Vec::new(),
            spill_ticks: 0,
            train_started: false,
        }
    }
}

/// Copies the LoRA into ComfyUI without overwriting: a taken name gets `_v2`, `_v3`, ...
fn deliver(cfg: &Settings, family: &str, file: &Path) -> Result<PathBuf, String> {
    let rel = cfg.lora_dirs.get(family).cloned().unwrap_or_else(|| "loras".into());
    let dir = PathBuf::from(&cfg.comfy_models).join(rel);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("lora");
    let mut target = dir.join(format!("{stem}.safetensors"));
    let mut n = 2;
    while target.exists() {
        target = dir.join(format!("{stem}_v{n}.safetensors"));
        n += 1;
    }
    std::fs::copy(file, &target).map_err(|e| e.to_string())?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_engine_progress_lines() {
        let dir = std::env::temp_dir().join(format!("dojo-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut job = Job::load_from(dir.clone(), Meta { steps: 500, ..Default::default() });
        let log = "Loading Krea 2 model\nQuantizing (qfloat8)\n\
            Caching latents to disk:  50%|█████     | 4/8 [00:01<00:01,  3.10it/s]\r\
            shape_test:  12%|█▏        | 60/500 [01:02<07:40,  1.05s/it, lr: 1.0e-04 loss: 3.210e-01]\r\
            shape_test:  12%|█▏        | 61/500 [01:03<07:39,  1.05s/it, lr: 1.0e-04 loss: 2.950e-01]\r\
            shape_test:  20%|██        | 100/500 [01:40<06:10,  1.08it/s, lr: 1.0e-04 loss: 4.1e-01]\n\
            Saved checkpoint\n";
        std::fs::write(dir.join("train.log"), log).unwrap();
        job.read_log();
        assert_eq!(job.step, 100);
        assert_eq!(job.loss.len(), 3);
        assert_eq!(job.speed, "1.08 it/s");
        assert_eq!(job.eta, "06:10");
        assert_eq!(job.phase, "Saved checkpoint");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn reads_the_python_versions_job_files() {
        let Ok(root) = std::env::var("DOJO_ROOT") else { return };
        let m = Manager::new(Paths::new(PathBuf::from(root)), Settings::default());
        for j in &m.jobs {
            assert!(!j.meta.name.is_empty(), "{:?}", j.folder);
            assert!(j.meta.plan.budget.need_gb > 0.0);
        }
    }
}
