//! Renders the README screenshots off-screen from demo data (nothing trains):
//!
//!   $env:DOJO_README_SHOTS = "docs\screenshots"
//!   cargo test --release readme_ -- --ignored --nocapture
//!
//! The dataset and sample images are painted procedurally, so no real photos end up in git.

use crate::app::{DojoApp, Page};
use crate::jobs::Meta;
use crate::recipes::{self, Form, Plan};
use crate::settings::{self, Paths};
use rust_glass::accessibility::Role;
use rust_glass::headless::Headless;
use rust_glass::input::Event;
use rust_glass::prelude::*;
use std::path::{Path, PathBuf};

/// A painted landscape: gradient sky, sun or moon with a glow, ridges fading into haze and
/// a still lake reflecting it all. Deterministic for a seed and palette.
pub fn landscape(seed: u32, palette: usize, size: u32) -> image::RgbImage {
    type P = ([f32; 3], [f32; 3], [f32; 3], [f32; 3]);
    let pals: [P; 4] = [
        ([0.98, 0.62, 0.32], [0.33, 0.18, 0.42], [1.0, 0.86, 0.55], [0.12, 0.08, 0.16]), // golden hour
        ([0.95, 0.55, 0.62], [0.16, 0.20, 0.45], [1.0, 0.80, 0.72], [0.08, 0.09, 0.20]), // dusk
        ([0.62, 0.82, 0.95], [0.98, 0.78, 0.62], [1.0, 0.95, 0.82], [0.20, 0.25, 0.36]), // dawn
        ([0.10, 0.16, 0.36], [0.03, 0.04, 0.12], [0.90, 0.94, 1.0], [0.02, 0.03, 0.07]), // moonlight
    ];
    let (sky_lo, sky_hi, sun_c, ink) = pals[palette % 4];
    let f = |a: f32| a.clamp(0.0, 1.0);
    let rnd = |k: u32| (seed.wrapping_mul(2_654_435_761).wrapping_add(k.wrapping_mul(40_503)) % 10_000) as f32 / 10_000.0;
    let horizon = 0.62 + rnd(1) * 0.08;
    let (sun_x, sun_y, sun_r) = (0.25 + rnd(2) * 0.5, horizon - 0.12 - rnd(3) * 0.18, 0.045 + rnd(4) * 0.03);
    let ridges: Vec<(f32, f32, f32, f32)> =
        (0..4).map(|i| (horizon - 0.20 + i as f32 * 0.06 + rnd(10 + i) * 0.04, 2.0 + rnd(20 + i) * 5.0, rnd(30 + i) * 6.28, 0.03 + i as f32 * 0.012)).collect();
    let ridge_y = |x: f32, i: usize| {
        let (base, fq, ph, amp) = ridges[i];
        base - amp * ((x * fq + ph).sin() * 0.6 + (x * fq * 2.7 + ph * 1.7).sin() * 0.3 + (x * fq * 7.1 + ph).sin() * 0.1)
    };
    let mut img = image::RgbImage::new(size, size);
    for py in 0..size {
        for px in 0..size {
            let (x, mut y) = (px as f32 / size as f32, py as f32 / size as f32);
            let water = y > horizon;
            if water {
                y = horizon - (y - horizon) * 0.9 + (x * 90.0 + y * 400.0).sin() * 0.002;
            }
            let t = f(y / horizon);
            let mut c = [0.0f32; 3];
            for k in 0..3 {
                c[k] = sky_hi[k] + (sky_lo[k] - sky_hi[k]) * t.powf(1.6);
            }
            let d = ((x - sun_x).powi(2) + (y - sun_y).powi(2)).sqrt();
            let glow = (-d / (sun_r * 3.5)).exp() * 0.55;
            for k in 0..3 {
                c[k] += sun_c[k] * glow;
                if d < sun_r {
                    c[k] = sun_c[k];
                }
            }
            for i in 0..4 {
                if y > ridge_y(x, i) {
                    let depth = (i as f32 + 1.0) / 4.0;
                    for k in 0..3 {
                        let haze = sky_lo[k] * (1.0 - depth) + ink[k] * depth;
                        c[k] = haze * 0.85 + c[k] * 0.15 * (1.0 - depth);
                    }
                }
            }
            if water {
                for k in 0..3 {
                    c[k] = c[k] * 0.72 + ink[k] * 0.28;
                }
            }
            img.put_pixel(px, py, image::Rgb([(f(c[0]) * 255.0) as u8, (f(c[1]) * 255.0) as u8, (f(c[2]) * 255.0) as u8]));
        }
    }
    img
}

#[allow(clippy::too_many_arguments)]
fn write_job(root: &Path, id: &str, plan: Plan, status: &str, message: &str, created: f64, pid: Option<u32>, log: &str) {
    let dir = root.join("jobs").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&plan.config).unwrap()).unwrap();
    std::fs::write(dir.join("train.log"), log).unwrap();
    let meta = Meta {
        id: id.into(),
        name: plan.name.clone(),
        status: status.into(),
        message: message.into(),
        created,
        steps: plan.steps,
        copy_to_comfy: true,
        pid,
        baseline_gb: 3.1,
        plan,
        ..Default::default()
    };
    std::fs::write(dir.join("job.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
}

/// A throwaway Dojo 24 folder with a dataset and three jobs, one of them mid-training.
fn build_demo(root: &Path) {
    let _ = std::fs::remove_dir_all(root);
    let ds_dir = root.join("datasets").join("golden_hour");
    std::fs::create_dir_all(&ds_dir).unwrap();
    for i in 0..18u32 {
        let p = ds_dir.join(format!("scene_{i:02}.jpg"));
        landscape(i * 7 + 3, (i % 4) as usize, 1024).save(&p).unwrap();
        std::fs::write(p.with_extension("txt"), "gldnhr style, a painted landscape with layered hills and a still lake").unwrap();
    }
    let paths = Paths::new(root.to_path_buf());
    let cfg = settings::load(&Paths::new(settings::root()));
    let cat = recipes::catalog(&cfg);
    let ds = recipes::inspect_dataset(&ds_dir.display().to_string()).unwrap();
    let mk = |base: &str, name: &str, goal: &str, detail: &str| {
        let f = Form { base: base.into(), name: name.into(), trigger: "gldnhr".into(), goal: goal.into(), detail: detail.into(), batch: 1, previews: true, ..Default::default() };
        recipes::build_plan(&f, &cat, &ds, &paths, &cfg, 24.0).unwrap()
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();

    // training: loss falling with noise, a checkpoint and samples every 250 steps
    let train = mk("moss_image:base", "golden_hour_style", "style", "balanced");
    let total = train.steps;
    let at = (total as f32 * 0.67) as u32;
    let mut log = String::from(
        "Loading MOSS-Image model\n - transformer weights: moss_image_base_bf16.safetensors\n\
         Caching latents to disk: 100%|##########| 18/18 [00:09<00:00,  1.95it/s]\n\
         Caching text embeddings to disk: 100%|##########| 18/18 [00:03<00:00,  5.40it/s]\n",
    );
    for s in 1..=at {
        let base = 0.62 * (-(s as f32) / 600.0).exp() + 0.36;
        let noise = ((s as f32 * 12.9898).sin() * 43_758.547).fract() - 0.5;
        let loss = base + noise * 0.16;
        let left = (total - s) as f32 * 2.91;
        log.push_str(&format!(
            "golden_hour_style: {:3}%|          | {s}/{total} [1:05:12<{:02}:{:02}, 2.91s/it, lr: 5.0e-05 loss: {loss:.3e}]\r",
            s * 100 / total,
            (left / 60.0) as u32,
            (left % 60.0) as u32
        ));
        if s % 250 == 0 {
            log.push_str(&format!(
                "\nSaving at step {s}\nSaved checkpoint to golden_hour_style_{s:09}.safetensors\nGenerating Samples: 100%|##########| 4/4 [00:41<00:00, 10.2s/it]\n"
            ));
        }
    }
    // the test process's own pid keeps the job "running" when the manager loads it
    write_job(root, "demo-1", train.clone(), "running", "", now - 100.0, Some(std::process::id()), &log);
    let out = root.join("output").join("golden_hour_style");
    std::fs::create_dir_all(out.join("samples")).unwrap();
    for (k, step) in (250..=at).step_by(250).enumerate() {
        for j in 0..4u32 {
            landscape(500 + k as u32 * 11 + j * 3, j as usize, 512).save(out.join("samples").join(format!("{step:09}_{j}.jpg"))).unwrap();
        }
        std::fs::write(out.join(format!("golden_hour_style_{step:09}.safetensors")), b"demo").unwrap();
    }

    let queued = mk("krea2:raw", "lantern_keeper", "character", "balanced");
    write_job(root, "demo-2", queued, "queued", "", now - 50.0, None, "");
    let done = mk("minimax_h3_image:base", "neon_harbor", "style", "fast");
    let done_log = format!("neon_harbor: 100%|##########| {0}/{0} [2:10:44<00:00,  5.23s/it, lr: 5.0e-05 loss: 4.102e-01]\n", done.steps);
    write_job(root, "demo-0", done, "done", "Copied to ComfyUI\\models\\loras\\H3\\neon_harbor.safetensors", now - 9000.0, None, &done_log);
}

struct Shots {
    h: Headless,
    app: DojoApp,
    out: PathBuf,
    size: (u32, u32),
}

impl Shots {
    fn step(&mut self, events: &[Event]) {
        self.h.step(&mut self.app, 1.0 / 60.0, events);
    }

    fn settle(&mut self) {
        self.h.settle(&mut self.app, 3.0);
        self.step(&[]);
    }

    fn click(&mut self, label: &str) {
        let node = self.h.ctx.access_nodes().iter().find(|n| n.label == label && !matches!(n.role, Role::Label | Role::Heading)).map(|n| n.rect.center());
        let p = node.unwrap_or_else(|| panic!("no widget labelled {label:?}"));
        self.step(&[Event::PointerMoved(p)]);
        self.step(&[Event::PointerDown]);
        self.step(&[Event::PointerUp]);
        // park the pointer where it highlights nothing
        self.step(&[Event::PointerMoved(vec2(5.0, 5.0))]);
        self.settle();
    }

    fn until(&mut self, what: &str, mut done: impl FnMut(&DojoApp) -> bool) {
        let t = std::time::Instant::now();
        while !done(&self.app) {
            assert!(t.elapsed().as_secs() < 60, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(30));
            self.step(&[]);
        }
    }

    fn snap(&mut self, name: &str) {
        // thumbnails decode on a thread: give them time to arrive
        for _ in 0..120 {
            std::thread::sleep(std::time::Duration::from_millis(15));
            self.step(&[]);
        }
        self.settle();
        let rgba = image::RgbaImage::from_raw(self.size.0, self.size.1, self.h.pixels()).unwrap();
        let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
        let file = std::fs::File::create(self.out.join(format!("{name}.jpg"))).unwrap();
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 90).encode_image(&rgb).unwrap();
    }
}

#[test]
#[ignore]
fn readme_screenshots() {
    let Some(out) = std::env::var_os("DOJO_README_SHOTS").map(PathBuf::from) else { return };
    std::fs::create_dir_all(&out).unwrap();
    let real = Paths::new(settings::root());
    let root = real.root.join("demo");
    build_demo(&root);
    let mut paths = Paths::new(root.clone());
    paths.engine_dir = real.engine_dir.clone();
    paths.engine_py = real.engine_py.clone();
    let cfg = settings::load(&real);
    let ppp = 1.5;
    let size = ((1440.0 * ppp) as u32, (900.0 * ppp) as u32);
    let mut h = Headless::new(size.0, size.1, ppp).expect("headless GPU");
    let mut app = DojoApp::new(paths, cfg, false);
    app.setup(&mut h.ctx);
    h.ctx.set_accessibility(true);
    app.wait_for_catalog();
    {
        let mut m = app.manager.lock().unwrap();
        for j in m.jobs.iter_mut().filter(|j| j.meta.status == "running") {
            // load, caching, then steady training with a spike at each sample render
            j.vram = (0..600)
                .map(|i| {
                    let t = i as f32;
                    let base = if t < 40.0 { 3.1 + t * 0.33 } else { 16.4 + (t * 0.7).sin() * 0.25 };
                    base + if i % 120 > 108 { 1.7 } else { 0.0 }
                })
                .collect();
            j.meta.peak_gb = 18.2;
        }
    }
    let mut s = Shots { h, app, out, size };
    s.settle();

    s.app.page = Page::Job("demo-1".into());
    s.snap("job");
    s.click("Queue");
    s.snap("queue");
    s.click("GPU fit");
    s.snap("fit");
    s.click("New job");
    s.app.dataset = root.join("datasets").join("golden_hour").display().to_string();
    s.app.name = "golden_hour_style".into();
    s.app.trigger = "gldnhr".into();
    s.app.goal = 1;
    s.app.family = 1;
    if let Some(cat) = s.app.catalog() {
        s.app.te = cat.moss_text_encoders.iter().position(|t| t.id == "official").unwrap_or(0);
    }
    s.until("a plan", |a| a.plan.as_ref().is_some_and(|p| p.as_ref().is_ok_and(|p| p.family == "moss_image")));
    s.snap("new_job");
    s.app.family = 2;
    s.until("an H3 plan", |a| a.plan.as_ref().is_some_and(|p| p.as_ref().is_ok_and(|p| p.family == "minimax_h3_image")));
    s.snap("new_job_h3");
}
