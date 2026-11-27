//! Paths and user settings. `settings.json` next to the app is created on first run.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// ComfyUI's models folder. The engine reads base weights from it and puts
    /// anything it downloads there, so nothing is stored twice.
    pub comfy_models: String,
    pub comfy_url: String,
    /// Where finished LoRAs are copied, relative to `comfy_models`, per model family.
    pub lora_dirs: BTreeMap<String, String>,
    /// GB kept free for Windows and the desktop. Past this the driver pages into
    /// shared system memory and training slows to a crawl without failing.
    pub vram_reserve_gb: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let lora_dirs = [("krea2", "loras/KR2"), ("moss_image", "loras/MOSS"), ("minimax_h3_image", "loras/H3")]
            .into_iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        Self {
            comfy_models: r"C:\ComfyUI-Easy-Install\ComfyUI\models".into(),
            comfy_url: "http://127.0.0.1:8188".into(),
            lora_dirs,
            vram_reserve_gb: 1.0,
        }
    }
}

/// The folder holding `engine\`, `jobs\` and `output\`. Found by walking up from the exe
/// (so `target\release\dojo-24.exe` works), else the current directory. `DOJO_ROOT` overrides.
pub fn root() -> PathBuf {
    if let Some(r) = std::env::var_os("DOJO_ROOT") {
        return PathBuf::from(r);
    }
    let start = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    for base in [start, std::env::current_dir().ok()].into_iter().flatten() {
        for dir in base.ancestors() {
            if dir.join("engine").is_dir() || dir.join("Cargo.toml").is_file() {
                return dir.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[derive(Clone)]
pub struct Paths {
    #[allow(dead_code)]
    pub root: PathBuf,
    pub engine_dir: PathBuf,
    pub engine_py: PathBuf,
    pub jobs: PathBuf,
    pub output: PathBuf,
    pub settings: PathBuf,
    pub measured: PathBuf,
}

impl Paths {
    pub fn new(root: PathBuf) -> Self {
        Self {
            engine_dir: root.join("engine").join("ai-toolkit"),
            engine_py: root.join("engine").join("venv").join("Scripts").join("python.exe"),
            jobs: root.join("jobs"),
            output: root.join("output"),
            settings: root.join("settings.json"),
            measured: root.join("measured.json"),
            root,
        }
    }

    pub fn engine_ready(&self) -> bool {
        self.engine_py.is_file() && self.engine_dir.join("run.py").is_file()
    }
}

pub fn load(paths: &Paths) -> Settings {
    match std::fs::read_to_string(&paths.settings) {
        Ok(text) => {
            let mut s: Settings = serde_json::from_str(&text).unwrap_or_default();
            // a settings file from an older version gains the newer model folders
            for (k, v) in Settings::default().lora_dirs {
                s.lora_dirs.entry(k).or_insert(v);
            }
            s
        }
        Err(_) => {
            let s = Settings::default();
            let _ = std::fs::write(&paths.settings, serde_json::to_string_pretty(&s).unwrap_or_default());
            s
        }
    }
}

/// The Hugging Face hub cache folder.
pub fn hf_cache() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).unwrap_or_default();
        PathBuf::from(home).join(".cache").join("huggingface")
    });
    base.join("hub")
}

/// A file already in the Hugging Face cache, or None.
pub fn hf_cached_file(repo: &str, file: &str) -> Option<PathBuf> {
    let snaps = hf_cache().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).collect();
    dirs.sort();
    dirs.into_iter().rev().map(|d| d.join(file)).find(|p| p.exists())
}
