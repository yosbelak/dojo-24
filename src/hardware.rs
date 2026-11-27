//! GPU state from nvidia-smi and the ComfyUI instance sharing the card. Both are read on
//! background threads so the window never waits on them.

use std::os::windows::process::CommandExt;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    pub name: String,
    pub total_gb: f32,
    pub used_gb: f32,
    pub free_gb: f32,
    pub temp_c: Option<u32>,
    #[allow(dead_code)]
    pub util: Option<u32>,
}

pub fn read_gpu() -> Option<Gpu> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"])
        .creation_flags(NO_WINDOW)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?;
    let p: Vec<&str> = line.split(',').map(str::trim).collect();
    if p.len() < 5 {
        return None;
    }
    let total = p[1].parse::<f32>().ok()? / 1024.0;
    let used = p[2].parse::<f32>().ok()? / 1024.0;
    Some(Gpu {
        name: p[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", ""),
        total_gb: total,
        used_gb: used,
        free_gb: total - used,
        temp_c: p[3].parse().ok(),
        util: p[4].parse().ok(),
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Comfy {
    pub running: bool,
    pub busy: bool,
    #[allow(dead_code)]
    pub pending: usize,
}

pub fn read_comfy(base: &str) -> Comfy {
    let url = format!("{}/queue", base.trim_end_matches('/'));
    let agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(2))).build().new_agent();
    let Ok(mut resp) = agent.get(&url).call() else { return Comfy::default() };
    let Ok(v) = resp.body_mut().read_json::<serde_json::Value>() else { return Comfy::default() };
    let len = |k: &str| v.get(k).and_then(|x| x.as_array()).map_or(0, Vec::len);
    Comfy { running: true, busy: len("queue_running") > 0, pending: len("queue_pending") }
}

/// Ask ComfyUI to unload its models. It loads them again on its next prompt.
pub fn comfy_free(base: &str) -> bool {
    let url = format!("{}/free", base.trim_end_matches('/'));
    let agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(10))).build().new_agent();
    agent.post(&url).send_json(serde_json::json!({"unload_models": true, "free_memory": true})).is_ok()
}

/// Latest GPU and ComfyUI readings, refreshed every two seconds.
#[derive(Clone, Default)]
pub struct Monitor {
    pub gpu: Arc<Mutex<Option<Gpu>>>,
    pub comfy: Arc<Mutex<Comfy>>,
}

impl Monitor {
    pub fn start(comfy_url: String) -> Self {
        let m = Self::default();
        let (g, c) = (m.gpu.clone(), m.comfy.clone());
        *g.lock().unwrap() = read_gpu();
        std::thread::spawn(move || {
            loop {
                let reading = read_gpu();
                *g.lock().unwrap() = reading;
                let cs = read_comfy(&comfy_url);
                *c.lock().unwrap() = cs;
                std::thread::sleep(Duration::from_secs(2));
            }
        });
        m
    }

    pub fn gpu(&self) -> Option<Gpu> {
        self.gpu.lock().unwrap().clone()
    }

    pub fn comfy(&self) -> Comfy {
        *self.comfy.lock().unwrap()
    }
}
