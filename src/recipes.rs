//! The three models Dojo 24 trains and how each is fitted to the card.
//!
//! Everything here comes from the engine's own loaders (ai-toolkit
//! extensions_built_in/diffusion_models/{krea2,moss_image,minimax_h3}) and its UI
//! defaults. Memory figures are estimates; the driver's peak from earlier runs on this
//! card is shown next to them (see measured.json).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::settings::{self, Paths, Settings};
use crate::sniff;

pub const IMAGE_EXTS: [&str; 5] = ["jpg", "jpeg", "png", "webp", "bmp"];

// ------------------------------------------------------------------ models

pub struct Family {
    pub id: &'static str,
    pub label: &'static str,
    pub params: &'static str,
    /// Resident transformer weights on the GPU by precision, GB.
    pub weights_gb: &'static [(&'static str, f32)],
    /// Activations + gradients with checkpointing, GB per image in the batch, by resolution.
    pub act_gb: &'static [(u32, f32)],
    pub overhead_gb: f32,
    pub timestep_type: &'static str,
    pub guidance: f32,
    pub sample_steps: u32,
    pub sample_size: u32,
    /// A rank that overrides the goal's (the engine author's default for this model).
    pub rank: Option<u32>,
    /// The DiT only fits with a share of its layers in system RAM.
    pub offload: bool,
}

pub const FAMILIES: [Family; 3] = [
    Family {
        id: "krea2",
        label: "Krea 2",
        params: "12B single-stream DiT, Qwen3-VL-4B text encoder",
        weights_gb: &[("bf16", 24.5), ("qfloat8", 12.6), ("convrot8", 12.9)],
        act_gb: &[(512, 1.6), (768, 2.8), (1024, 4.4)],
        overhead_gb: 1.6,
        timestep_type: "linear",
        guidance: 4.0,
        sample_steps: 25,
        sample_size: 1024,
        rank: None,
        offload: false,
    },
    Family {
        id: "moss_image",
        label: "MOSS-Image",
        params: "5B single-stream DiT, Qwen3.5-9B text encoder",
        weights_gb: &[("bf16", 10.3), ("convrot8", 5.2)],
        act_gb: &[(512, 0.9), (768, 1.6), (1024, 2.4)],
        overhead_gb: 1.2,
        timestep_type: "shift",
        guidance: 3.0,
        sample_steps: 30,
        sample_size: 1024,
        rank: None,
        offload: false,
    },
    Family {
        id: "minimax_h3_image",
        label: "MiniMax-H3 Image",
        params: "Pruned H3 DiT without the audio stream, 6-step distilled, Qwen3-VL-32B text encoder",
        weights_gb: &[("convrot8", 19.6)],
        // 32x compression: 2048 px here is the token count of 1024 px on Krea 2
        act_gb: &[(1024, 1.6), (1536, 2.9), (2048, 4.6)],
        overhead_gb: 1.8,
        timestep_type: "shift",
        guidance: 1.0,
        sample_steps: 6,
        sample_size: 2048,
        rank: Some(16),
        offload: true,
    },
];

pub fn family(id: &str) -> &'static Family {
    FAMILIES.iter().find(|f| f.id == id).unwrap_or(&FAMILIES[0])
}

impl Family {
    fn weights(&self, precision: &str) -> f32 {
        self.weights_gb.iter().find(|(p, _)| *p == precision).map(|w| w.1).unwrap_or_else(|| self.weights_gb.iter().map(|w| w.1).fold(0.0, f32::max))
    }
    fn act(&self, res: u32) -> f32 {
        self.act_gb.iter().find(|(r, _)| *r == res).map(|a| a.1).unwrap_or(self.act_gb[self.act_gb.len() - 1].1)
    }
}

pub const H3_REPO: &str = "Comfy-Org/MiniMax-H3-Image";
pub const H3_ADAPTER: &str = "ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors";
const H3_FILES: [(&str, &str, &str); 3] = [
    ("diffusion_models", "minimax_h3_image_pruned_int8_convrot.safetensors", "the 19.5 GB transformer"),
    ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "the 14.6 GB text encoder"),
    ("vae", "minimax_h3_image_vae_fp16.safetensors", "the 1.1 GB image VAE"),
];

pub struct Goal {
    pub id: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    pub rank: u32,
    pub lr: f64,
    steps_per_image: u32,
    min_steps: u32,
    max_steps: u32,
}

pub const GOALS: [Goal; 3] = [
    Goal { id: "character", label: "Character", hint: "One person or creature, kept consistent across poses.", rank: 32, lr: 1e-4, steps_per_image: 75, min_steps: 1500, max_steps: 3000 },
    Goal { id: "style", label: "Style", hint: "A look: palette, rendering, linework.", rank: 16, lr: 5e-5, steps_per_image: 50, min_steps: 1500, max_steps: 4000 },
    Goal { id: "concept", label: "Concept", hint: "An object, outfit, pose or action.", rank: 16, lr: 1e-4, steps_per_image: 60, min_steps: 1250, max_steps: 3000 },
];

pub fn goal(id: &str) -> &'static Goal {
    GOALS.iter().find(|g| g.id == id).unwrap_or(&GOALS[0])
}

/// Resolution presets. Krea 2 was trained at 256/512/1024 and community runs at 768 are
/// reported to go wrong, so 768 is offered for MOSS-Image only.
pub struct Detail {
    pub id: &'static str,
    pub label: &'static str,
    res: [&'static [u32]; 3],
}

pub const DETAILS: [Detail; 3] = [
    Detail { id: "fast", label: "Fast", res: [&[512], &[512, 768], &[1024]] },
    Detail { id: "balanced", label: "Balanced", res: [&[512, 1024], &[768, 1024], &[1024, 1536]] },
    Detail { id: "max", label: "Max detail", res: [&[1024], &[1024], &[1536, 2048]] },
];

impl Detail {
    pub fn resolutions(&self, family: &str) -> &'static [u32] {
        let i = FAMILIES.iter().position(|f| f.id == family).unwrap_or(0);
        self.res[i]
    }
}

pub fn detail(id: &str) -> &'static Detail {
    DETAILS.iter().find(|d| d.id == id).unwrap_or(&DETAILS[1])
}

// ------------------------------------------------------------------ catalog

#[derive(Clone, Debug)]
#[allow(dead_code)] // shown in later versions; kept so the catalog describes each file fully
pub struct Base {
    pub id: String,
    pub family: &'static str,
    pub label: String,
    pub note: String,
    pub ready: bool,
    /// What the first run has to fetch, or None.
    pub download: Option<String>,
    pub recommended: bool,
    /// For your own checkpoints: why the engine cannot load it.
    pub blocked: Option<String>,
    pub precision: Option<String>,
    pub size_gb: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct TextEncoder {
    /// "official" or a file path.
    pub id: String,
    pub label: String,
    pub download: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub bases: Vec<Base>,
    pub local: Vec<Base>,
    pub moss_text_encoders: Vec<TextEncoder>,
    pub moss_vae: Option<PathBuf>,
}

impl Catalog {
    pub fn all(&self) -> impl Iterator<Item = &Base> {
        self.bases.iter().chain(self.local.iter())
    }
    pub fn get(&self, id: &str) -> Option<&Base> {
        self.all().find(|b| b.id == id)
    }
}

fn find(root: &Path, names: &[&str]) -> Option<PathBuf> {
    walk_files(root).into_iter().find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| names.contains(&n)))
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !name.starts_with('.') {
                    stack.push(p);
                }
            } else {
                out.push(p);
            }
        }
    }
    out
}

static SNIFF_CACHE: Mutex<Option<HashMap<(PathBuf, u64, std::time::SystemTime), sniff::Info>>> = Mutex::new(None);

fn sniff_cached(p: &Path) -> Option<sniff::Info> {
    let m = std::fs::metadata(p).ok()?;
    let key = (p.to_path_buf(), m.len(), m.modified().ok()?);
    let mut guard = SNIFF_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    Some(cache.entry(key).or_insert_with(|| sniff::sniff(p)).clone())
}

/// What can be trained on right now, and what each choice would download.
pub fn catalog(cfg: &Settings) -> Catalog {
    let models = PathBuf::from(&cfg.comfy_models);
    let (dm, te_dir, vae_dir) = (models.join("diffusion_models"), models.join("text_encoders"), models.join("vae"));
    let mut bases = Vec::new();

    let raw = settings::hf_cached_file("krea/Krea-2-Raw", "raw.safetensors");
    bases.push(Base {
        id: "krea2:raw".into(),
        family: "krea2",
        label: "Krea 2 Raw".into(),
        note: "The undistilled base. LoRAs trained here carry over to Krea 2 Turbo and to finetunes.".into(),
        ready: raw.is_some(),
        download: raw.is_none().then(|| "24.5 GB from krea/Krea-2-Raw (needs a Hugging Face login that accepted the licence)".into()),
        recommended: true,
        blocked: None,
        precision: None,
        size_gb: None,
    });

    let moss = find(&dm, &["moss_image_base_bf16.safetensors", "moss_image_base_int8_convrot_attn.safetensors"]);
    bases.push(Base {
        id: "moss_image:base".into(),
        family: "moss_image",
        label: "MOSS-Image Base".into(),
        note: "The vendor's training base. LoRAs carry over to MOSS-Image SFT and to finetunes.".into(),
        ready: moss.is_some(),
        download: moss.is_none().then(|| "10.3 GB (moss_image_base_bf16.safetensors) into ComfyUI's diffusion_models".into()),
        recommended: true,
        blocked: None,
        precision: None,
        size_gb: None,
    });

    let mut missing: Vec<String> = H3_FILES.iter().filter(|(dir, f, _)| find(&models.join(dir), &[f]).is_none()).map(|(_, _, what)| what.to_string()).collect();
    let core_ready = missing.is_empty();
    if find(&models.join("loras"), &[H3_ADAPTER.rsplit('/').next().unwrap()]).is_none() {
        missing.push("ostris' training adapter LoRA (0.3 GB)".into());
    }
    bases.push(Base {
        id: "minimax_h3_image:base".into(),
        family: "minimax_h3_image",
        label: "MiniMax-H3-Image".into(),
        note: "The 6-step image model, trained through a de-distillation adapter so the LoRA does not undo the distillation.".into(),
        ready: core_ready,
        download: (!missing.is_empty()).then(|| format!("{}, into ComfyUI's models folder", missing.join("; "))),
        recommended: true,
        blocked: None,
        precision: None,
        size_gb: None,
    });

    // Checkpoints already in ComfyUI. Only files the engine can open as they are can be picked.
    let mut local = Vec::new();
    for p in walk_files(&dm) {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        if !name.ends_with(".safetensors") || std::fs::metadata(&p).map_or(true, |m| m.len() < (2u64 << 30)) {
            continue;
        }
        if name.starts_with("moss_image_base_") || name.starts_with("moss_image_sft_") || name.starts_with("minimax_h3_") {
            continue;
        }
        let Some(info) = sniff_cached(&p) else { continue };
        if !matches!(info.family, "krea2" | "moss_image") {
            continue;
        }
        let blocked = sniff::blocked_reason(&info);
        local.push(Base {
            id: format!("file:{}", p.display()),
            family: info.family,
            label: p.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string(),
            note: "Your own checkpoint. Untested with the engine, and the LoRA is tied to this finetune.".into(),
            ready: blocked.is_none(),
            download: None,
            recommended: false,
            blocked,
            precision: Some(info.precision.clone()),
            size_gb: Some(info.size_gb),
        });
    }

    // MOSS text encoder and VAE: use what ComfyUI already has before downloading.
    let official = find(&te_dir, &["qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_bf16.safetensors"]);
    let mut tes = Vec::new();
    if let Some(o) = &official {
        tes.push(TextEncoder { id: "official".into(), label: file_name(o), download: None });
    }
    for p in walk_files(&te_dir) {
        let low = file_name(&p).to_lowercase();
        if low.ends_with(".safetensors") && low.contains("qwen3.5") && low.contains("9b") && Some(&p) != official.as_ref() {
            tes.push(TextEncoder { id: p.display().to_string(), label: file_name(&p), download: None });
        }
    }
    if official.is_none() {
        tes.push(TextEncoder { id: "official".into(), label: "qwen3.5_9b_int8_convrot (official)".into(), download: Some("11.5 GB into ComfyUI's text_encoders".into()) });
    }
    let vae = find(&vae_dir, &["flux2-vae.safetensors"]).or_else(|| {
        walk_files(&vae_dir).into_iter().find(|p| {
            let l = file_name(p).to_lowercase();
            l.ends_with(".safetensors") && l.contains("flux2") && l.contains("vae") && !["fp8", "hdr", "small"].iter().any(|a| l.contains(a))
        })
    });

    Catalog { bases, local, moss_text_encoders: tes, moss_vae: vae }
}

fn file_name(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()
}

// ------------------------------------------------------------------ dataset

#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct DatasetFolder {
    pub path: PathBuf,
    pub images: usize,
    pub captions: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub root: PathBuf,
    pub images: usize,
    pub captioned: usize,
    pub folders: Vec<DatasetFolder>,
    /// How many images had their size checked, and how many of those are under 768 px.
    pub checked: usize,
    pub small: usize,
    pub previews: Vec<PathBuf>,
}

pub fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
}

pub fn inspect_dataset(path: &str) -> Result<Dataset, String> {
    let root = PathBuf::from(path.trim().trim_matches('"'));
    if !root.is_dir() {
        return Err("That folder does not exist.".into());
    }
    let mut ds = Dataset { root: root.clone(), ..Default::default() };
    let mut all = Vec::new();
    let mut stack = vec![root];
    let mut dirs = Vec::new();
    while let Some(d) = stack.pop() {
        dirs.push(d.clone());
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut subs: Vec<PathBuf> = rd
                .flatten()
                .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with(['.', '_']))
                .map(|e| e.path())
                .collect();
            subs.sort();
            stack.extend(subs.into_iter().rev());
        }
    }
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut imgs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_file() && is_image(p)).collect();
        if imgs.is_empty() {
            continue;
        }
        imgs.sort();
        let caps = imgs.iter().filter(|p| p.with_extension("txt").is_file()).count();
        ds.images += imgs.len();
        ds.captioned += caps;
        ds.folders.push(DatasetFolder { path: d, images: imgs.len(), captions: caps });
        all.extend(imgs);
    }
    if ds.images == 0 {
        return Err("No images found in that folder or its subfolders.".into());
    }
    let step = (all.len() / 60).max(1);
    for p in all.iter().step_by(step).take(60) {
        if let Ok((w, h)) = image::image_dimensions(p) {
            ds.checked += 1;
            if w.min(h) < 768 {
                ds.small += 1;
            }
        }
    }
    let step = (all.len() / 10).max(1);
    ds.previews = all.iter().step_by(step).take(10).cloned().collect();
    Ok(ds)
}

pub fn suggest_steps(goal_id: &str, images: usize, batch: u32) -> u32 {
    let g = goal(goal_id);
    let steps = (images as u32 * g.steps_per_image).clamp(g.min_steps, g.max_steps);
    let steps = ((steps as f32 / batch.max(1) as f32 / 250.0).round() as u32) * 250;
    steps.max(500)
}

// ------------------------------------------------------------------ memory budget

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Budget {
    pub weights_gb: f32,
    pub activations_gb: f32,
    pub overhead_gb: f32,
    pub need_gb: f32,
    pub usable_gb: f32,
    pub total_gb: f32,
    /// Share of the transformer kept in system RAM (layer offloading).
    #[serde(default)]
    pub offload: f32,
    #[serde(default)]
    pub ram_gb: f32,
    pub fits: bool,
}

fn r1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

pub fn budget(family_id: &str, precision: &str, res: u32, batch: u32, total_gb: f32, reserve_gb: f32) -> Budget {
    let fam = family(family_id);
    let weights = fam.weights(precision);
    let act = fam.act(res) * batch as f32;
    let over = fam.overhead_gb;
    let usable = total_gb - reserve_gb;
    let mut offload = 0.0f32;
    if fam.offload {
        // Keep just enough layers in system RAM to leave 1 GB of slack; every offloaded
        // layer is copied to the card and back on each step.
        let spare = usable - 1.0 - act - over;
        if spare < weights {
            offload = (1.0 - spare / weights).clamp(0.0, 0.9);
            offload = ((offload / 0.05 - 1e-4).ceil() * 0.05 * 100.0).round() / 100.0;
        }
    }
    let resident = weights * (1.0 - offload);
    let need = resident + act + over;
    Budget {
        weights_gb: r1(resident),
        activations_gb: r1(act),
        overhead_gb: r1(over),
        need_gb: r1(need),
        usable_gb: r1(usable),
        total_gb: r1(total_gb),
        offload,
        ram_gb: r1(weights * offload),
        // half a GB of margin on top of an estimate
        fits: need + 0.5 <= usable && offload <= 0.6,
    }
}

/// Highest precision that fits. A pre-quantized int8 file stays int8; H3 trains on its shipped int8 file.
pub fn pick_precision(family_id: &str, local_precision: Option<&str>, total_gb: f32, reserve_gb: f32, res: u32, batch: u32) -> &'static str {
    if family_id == "minimax_h3_image" || local_precision == Some("int8") {
        return "convrot8";
    }
    let fam = family(family_id);
    let order: &[&'static str] = if family_id == "krea2" { &["bf16", "qfloat8"] } else { &["bf16", "convrot8"] };
    for p in order {
        let need = fam.weights(p) + fam.act(res) * batch as f32 + fam.overhead_gb;
        if need + 0.5 <= total_gb - reserve_gb {
            return p;
        }
    }
    order[order.len() - 1]
}

pub fn measured_key(family: &str, precision: &str, res: u32, batch: u32) -> String {
    format!("{family}|{precision}|{res}|{batch}")
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Measured {
    pub peak_gb: f32,
    pub baseline_gb: f32,
}

pub fn measured(paths: &Paths) -> HashMap<String, Measured> {
    std::fs::read_to_string(&paths.measured).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

pub fn record_measurement(paths: &Paths, key: &str, peak_gb: f32, baseline_gb: f32) {
    let mut m = measured(paths);
    m.insert(key.to_string(), Measured { peak_gb: (peak_gb * 100.0).round() / 100.0, baseline_gb: (baseline_gb * 100.0).round() / 100.0 });
    if let Ok(text) = serde_json::to_string_pretty(&m) {
        let _ = std::fs::write(&paths.measured, text);
    }
}

// ------------------------------------------------------------------ why each setting

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Why {
    pub setting: String,
    pub value: String,
    pub why: String,
    /// true: chosen for this card's memory. false: the engine author's own default.
    pub card: bool,
}

fn why(setting: &str, value: impl Into<String>, card: bool, text: impl Into<String>) -> Why {
    Why { setting: setting.into(), value: value.into(), why: text.into(), card }
}

pub fn explain(family_id: &str, precision: &str, bud: &Budget, resolutions: &[u32], batch: u32, rank: u32, total_gb: f32) -> Vec<Why> {
    let fam = family(family_id);
    let card = format!("{} GB", total_gb.round());
    let res = format!("{} px", resolutions.iter().map(u32::to_string).collect::<Vec<_>>().join(" + "));
    let mut rows = Vec::new();
    match family_id {
        "krea2" => {
            rows.push(why("Transformer weights", "8-bit (qfloat8)", true, format!(
                "Krea 2 is 24.5 GB in bf16, more than the whole card. At 8-bit it is {} GB, which leaves room to train at 1024 px with everything on the GPU. qfloat8 is the engine's default 8-bit type for this model.",
                fam.weights("qfloat8"))));
            rows.push(why("Text encoder", "Qwen3-VL-4B, 8-bit, then unloaded", true, "Captions are encoded once before training and cached, so the encoder's memory goes back to the transformer."));
            rows.push(why("Resolutions", res, true, format!(
                "Krea 2 was trained at 256, 512 and 1024, and community runs at 768 are reported to go wrong, so 768 is left out. 1024 px is the largest size where batch 1 fits a {card} card without offloading.")));
            rows.push(why("Timesteps", "linear", false, "The engine author's default for Krea 2; the model applies its own resolution-dependent shift on top."));
        }
        "moss_image" => {
            if precision == "bf16" {
                rows.push(why("Transformer weights", "full bf16, not quantized", true, format!(
                    "The 5B transformer is {} GB in bf16, so a {card} card can train it at full precision. The engine's default is 8-bit because it targets smaller cards; you do not need that trade-off.",
                    fam.weights("bf16"))));
            } else {
                rows.push(why("Transformer weights", "8-bit (convrot8)", true, "This checkpoint is already stored as int8, so it is trained as it is."));
            }
            rows.push(why("Text encoder", "Qwen3.5-9B, int8, then unloaded", true, "Captions are cached before training. The vendor says to keep this encoder frozen, and it is."));
            rows.push(why("Resolutions", res, true, "The vendor recommends 1 to 2 megapixel output, and there is memory to train at 1024 px with batch 2."));
            rows.push(why("Layer 22", "left untouched", false, "MOSS-Image skips block 22 on the negative branch when sampling. Keeping the LoRA off that block preserves the effect."));
            rows.push(why("Timesteps", "shift", false, "The engine author's default for MOSS-Image."));
        }
        _ => {
            rows.push(why("Transformer weights", "int8 (convrot8), as shipped", false, "The 19.5 GB file in your ComfyUI folder is used directly, with no re-quantizing."));
            if bud.offload > 0.0 {
                rows.push(why("Layer offloading", format!("{:.0}% of layers in system RAM", bud.offload * 100.0), true, format!(
                    "19.6 GB of weights plus training memory does not fit in {card}. Keeping {} GB of layers in system RAM is the smallest share that fits; more would only slow training down. The engine marks this feature experimental.",
                    bud.ram_gb)));
            } else {
                rows.push(why("Layer offloading", "off", true, "Everything fits on the card at this resolution, so no layers are streamed from RAM."));
            }
            rows.push(why("Text encoder", "Qwen3-VL-32B, nvfp4, then unloaded", true, "At 14.6 GB it cannot share the card with the transformer. The transformer waits in system RAM while captions are cached."));
            rows.push(why("Resolutions", res, true, "The model is built for 4 megapixels and up. Training lower is faster and lighter but learns less fine detail; 2048 px is the practical ceiling on this card."));
            rows.push(why("Training adapter", "on", false, "The model is step- and guidance-distilled. The adapter stands in for the lost steps during training and is switched off for previews."));
        }
    }
    rows.push(why("Batch size", batch.to_string(), true, format!("This recipe is planned at {} GB of the {} GB that can be used.", bud.need_gb, bud.usable_gb)));
    rows.push(why("Gradient checkpointing", "on", false, "Recomputes activations during the backward pass instead of storing them. Slower per step, far less memory."));
    rows.push(why("Optimizer", "AdamW 8-bit", false, "The engine's default; its optimizer state takes a quarter of the memory of plain AdamW."));
    rows.push(why("Rank", rank.to_string(), fam.rank.is_none(), if fam.rank.is_some() {
        "The engine author's default for this model. Rank sets the LoRA's capacity and file size, not whether the run fits."
    } else {
        "Picked from the training goal (32 for a character, 16 for a style or concept). It sets the LoRA's capacity and file size, not whether the run fits."
    }));
    rows
}

// ------------------------------------------------------------------ form -> plan

#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // dataset is read through the scanned Dataset
pub struct Form {
    pub base: String,
    pub name: String,
    pub trigger: String,
    pub dataset: String,
    pub goal: String,
    pub detail: String,
    pub batch: u32,
    /// Empty means "work it out".
    pub steps: String,
    pub rank: String,
    pub lr: String,
    pub text_encoder: String,
    pub prompts: String,
    pub previews: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Plan {
    pub name: String,
    pub family: String,
    pub base: String,
    pub base_id: String,
    pub goal: String,
    pub detail: String,
    pub precision: String,
    pub resolutions: Vec<u32>,
    pub batch: u32,
    pub rank: u32,
    pub lr: f64,
    pub steps: u32,
    pub save_every: u32,
    pub images: usize,
    pub captioned: usize,
    pub trigger: String,
    pub budget: Budget,
    #[serde(default)]
    pub downloads: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub why: Vec<Why>,
    pub measured_key: String,
    /// The engine config. Not stored in job.json (config.json holds it).
    #[serde(skip)]
    pub config: Value,
}

const DEFAULT_PROMPTS: [(&str, [&str; 4]); 3] = [
    ("character", [
        "{t}, portrait, soft window light, looking at the camera",
        "{t}, full body, walking down a rainy street at night, neon reflections",
        "{t}, sitting in a cafe reading a book, candid photo",
        "{t}, standing on a mountain trail at sunrise, wide shot",
    ]),
    ("style", [
        "a lighthouse on a cliff at dusk, {t}",
        "a cat asleep on a windowsill full of plants, {t}",
        "a crowded night market seen from above, {t}",
        "an old bicycle leaning on a brick wall, {t}",
    ]),
    ("concept", [
        "{t}, on a wooden table, soft daylight",
        "{t}, outdoors in a park, wide shot",
        "{t}, close-up, studio lighting, plain backdrop",
        "{t}, in a busy city street at night",
    ]),
];

pub fn safe_name(s: &str) -> String {
    s.trim().chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

pub fn build_plan(form: &Form, cat: &Catalog, ds: &Dataset, paths: &Paths, cfg: &Settings, total_gb: f32) -> Result<Plan, String> {
    let entry = cat.get(&form.base).ok_or("Pick a base model.")?;
    if let Some(b) = &entry.blocked {
        return Err(format!("{} cannot be trained on: {b}.", entry.label));
    }
    let family_id = entry.family;
    let fam = family(family_id);
    let name = safe_name(&form.name);
    if name.is_empty() {
        return Err("Give the LoRA a name.".into());
    }
    let trigger = form.trigger.trim().to_string();
    let g = goal(&form.goal);
    let d = detail(&form.detail);
    let resolutions = d.resolutions(family_id);
    let top = *resolutions.iter().max().unwrap();
    let batch = form.batch.clamp(1, 4);
    let reserve = cfg.vram_reserve_gb;

    let precision = pick_precision(family_id, entry.precision.as_deref(), total_gb, reserve, top, batch);
    let bud = budget(family_id, precision, top, batch, total_gb, reserve);
    if !bud.fits {
        let extra = if bud.offload > 0.0 { format!(" even with {:.0}% of the model in system RAM", bud.offload * 100.0) } else { String::new() };
        return Err(format!(
            "Batch {batch} at {top} px needs about {} GB{extra}, and this card has {} GB to give. Lower the batch size or the resolution.",
            bud.need_gb, bud.usable_gb
        ));
    }

    let rank: u32 = match form.rank.trim() {
        "" => fam.rank.unwrap_or(g.rank),
        s => s.parse().map_err(|_| "Rank must be a whole number.")?,
    };
    let lr: f64 = match form.lr.trim() {
        "" => g.lr,
        s => s.parse().map_err(|_| "Learning rate must be a number, like 0.0001 or 1e-4.")?,
    };
    let steps: u32 = match form.steps.trim() {
        "" => suggest_steps(g.id, ds.images, batch),
        s => s.parse().map_err(|_| "Steps must be a whole number.")?,
    };
    if steps < 10 {
        return Err("Train for at least 10 steps.".into());
    }
    let save_every = if steps >= 1000 { 250 } else { (steps / 4 / 50 * 50).max(50) };

    let mut downloads = Vec::new();
    let mut warnings = Vec::new();
    let uncaptioned = ds.images - ds.captioned;
    if uncaptioned > 0 && trigger.is_empty() {
        warnings.push(format!(
            "{uncaptioned} of {} images have no caption file and there is no trigger word, so they train with an empty caption. The LoRA then applies whenever it is loaded, with nothing to call it by.",
            ds.images
        ));
    }
    if let Some(dl) = &entry.download {
        downloads.push(format!("{}: {dl}", entry.label));
    }

    let mut model = serde_json::Map::new();
    model.insert("arch".into(), json!(family_id));
    model.insert("low_vram".into(), json!(true));
    let mut model_kwargs = serde_json::Map::new();
    let mut ignore: Vec<&str> = Vec::new();
    let mut sample_guidance = fam.guidance;
    let sample_steps = fam.sample_steps;

    let name_or_path = if let Some(f) = entry.id.strip_prefix("file:") {
        f.to_string()
    } else {
        match entry.id.as_str() {
            "krea2:raw" => "krea/Krea-2-Raw".into(),
            "moss_image:base" => "OpenMOSS-Team/MOSS-Image-Base".into(),
            _ => H3_REPO.into(),
        }
    };
    model.insert("name_or_path".into(), json!(name_or_path));
    model.insert("quantize".into(), json!(precision != "bf16"));
    if precision != "bf16" {
        model.insert("qtype".into(), json!(precision));
    }

    match family_id {
        "krea2" => {
            // Qwen3-VL-4B, quantized; unloaded once the captions are cached
            model.insert("quantize_te".into(), json!(true));
            model.insert("qtype_te".into(), json!("qfloat8"));
            for (repo, file, what) in [
                ("Qwen/Qwen3-VL-4B-Instruct", "config.json", "Qwen3-VL-4B text encoder, 8.3 GB"),
                ("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors", "Qwen-Image VAE, 0.25 GB"),
            ] {
                if settings::hf_cached_file(repo, file).is_none() {
                    downloads.push(what.into());
                }
            }
        }
        "minimax_h3_image" => {
            // both files are used exactly as ComfyUI ships them
            model.insert("quantize_te".into(), json!(true));
            model.insert("qtype_te".into(), json!("nvfp4"));
            model.insert("assistant_lora_path".into(), json!(H3_ADAPTER));
            ignore.push("adaln_proj");
            if bud.offload > 0.0 {
                model.insert("layer_offloading".into(), json!(true));
                model.insert("layer_offloading_transformer_percent".into(), json!((bud.offload as f64 * 100.0).round() / 100.0));
                // the text encoder gets the card to itself while captions are cached
                model.insert("layer_offloading_text_encoder_percent".into(), json!(0.0));
            }
            sample_guidance = 1.0;
        }
        _ => {
            model.insert("quantize_te".into(), json!(true));
            model.insert("qtype_te".into(), json!("convrot8"));
            model_kwargs.insert("uncond_skip_layers".into(), json!([22]));
            // leave the block the negative branch skips to the base model
            ignore.push("layers.22.");
            let te = cat.moss_text_encoders.iter().find(|t| t.id == form.text_encoder).or(cat.moss_text_encoders.first());
            if let Some(te) = te {
                if te.id != "official" {
                    model.insert("te_name_or_path".into(), json!(te.id));
                }
                if let Some(dl) = &te.download {
                    downloads.push(format!("Text encoder: {dl}"));
                }
            }
            match &cat.moss_vae {
                Some(v) if file_name(v) != "flux2-vae.safetensors" => {
                    model.insert("vae_path".into(), json!(v.display().to_string()));
                }
                Some(_) => {}
                None => downloads.push("VAE: 0.3 GB (flux2-vae.safetensors)".into()),
            }
        }
    }
    model.insert("model_kwargs".into(), Value::Object(model_kwargs));

    if ds.small > 0 && top >= 1024 {
        warnings.push(format!(
            "{} of {} checked images are under 768 px on their short side; they will be upscaled for the {top} px bucket.",
            ds.small, ds.checked
        ));
    }
    if bud.offload > 0.0 {
        warnings.push(format!(
            "{:.0}% of the transformer ({} GB) stays in system RAM and is streamed to the card every step. That is what makes this model fit, and it makes each step slower.",
            bud.offload * 100.0,
            bud.ram_gb
        ));
    }
    if ds.images < 8 {
        warnings.push("Fewer than 8 images. Expect an overfit LoRA; 15 to 40 varied images work better.".into());
    }

    let datasets: Vec<Value> = ds
        .folders
        .iter()
        .map(|f| {
            json!({
                "folder_path": f.path.display().to_string(),
                "caption_ext": "txt",
                "default_caption": trigger,
                // captions are encoded once and the text encoder is unloaded, so no per-step dropout
                "caption_dropout_rate": 0,
                "cache_latents_to_disk": true,
                "resolution": resolutions,
                "num_repeats": 1,
            })
        })
        .collect();

    let mut prompts: Vec<String> = form.prompts.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
    if prompts.is_empty() {
        let fallback = match g.id {
            "style" => "in this style",
            "concept" => "the subject",
            _ => "a person",
        };
        let t = if trigger.is_empty() { fallback } else { trigger.as_str() };
        let set = DEFAULT_PROMPTS.iter().find(|(k, _)| *k == g.id).map(|d| d.1).unwrap_or(DEFAULT_PROMPTS[0].1);
        prompts = set.iter().map(|p| p.replace("{t}", t)).collect();
    }
    prompts.truncate(6);

    let mut process = json!({
        "type": "sd_trainer",
        "training_folder": paths.output.display().to_string(),
        "device": "cuda:0",
        "performance_log_every": 250,
        "network": {"type": "lora", "linear": rank, "linear_alpha": rank, "network_kwargs": {"ignore_if_contains": ignore}},
        "save": {"dtype": "bf16", "save_every": save_every, "max_step_saves_to_keep": 6},
        "datasets": datasets,
        "train": {
            "batch_size": batch,
            "steps": steps,
            "gradient_accumulation": 1,
            "train_unet": true,
            "train_text_encoder": false,
            "gradient_checkpointing": true,
            "noise_scheduler": "flowmatch",
            "optimizer": "adamw8bit",
            "optimizer_params": {"weight_decay": 1e-4},
            "lr": lr,
            "timestep_type": fam.timestep_type,
            "content_or_style": "balanced",
            "dtype": "bf16",
            "cache_text_embeddings": true,
            "unload_text_encoder": false,
            "ema_config": {"use_ema": false},
            "skip_first_sample": true,
            "disable_sampling": !form.previews,
            "loss_type": "mse",
        },
        "model": Value::Object(model),
        "sample": {
            "sampler": "flowmatch",
            "sample_every": save_every,
            "width": fam.sample_size,
            "height": fam.sample_size,
            "neg": "",
            "seed": 42,
            "walk_seed": true,
            "guidance_scale": sample_guidance as f64,
            "sample_steps": sample_steps,
            "samples": prompts.iter().map(|p| json!({"prompt": p})).collect::<Vec<_>>(),
        },
    });
    if !trigger.is_empty() {
        process["trigger_word"] = json!(trigger);
    }
    let config = json!({"job": "extension", "config": {"name": name, "process": [process]}, "meta": {"name": "[name]", "version": "1.0"}});

    Ok(Plan {
        name,
        family: family_id.into(),
        base: entry.label.clone(),
        base_id: entry.id.clone(),
        goal: g.id.into(),
        detail: d.id.into(),
        precision: precision.into(),
        resolutions: resolutions.to_vec(),
        batch,
        rank,
        lr,
        steps,
        save_every,
        images: ds.images,
        captioned: ds.captioned,
        why: explain(family_id, precision, &bud, resolutions, batch, rank, total_gb),
        trigger,
        measured_key: measured_key(family_id, precision, top, batch),
        budget: bud,
        downloads,
        warnings,
        config,
    })
}

// ------------------------------------------------------------------ what fits

pub struct FitCell {
    pub need_gb: f32,
    pub fits: bool,
    pub offload: f32,
}

pub struct FitRow {
    pub detail: &'static str,
    pub resolutions: &'static [u32],
    pub cells: Vec<FitCell>,
}

pub struct FitModel {
    pub family: &'static Family,
    pub rows: Vec<FitRow>,
    pub why: Vec<Why>,
}

/// Per model: what fits on this card at each resolution and batch size.
pub fn fit_table(cfg: &Settings, total_gb: f32) -> Vec<FitModel> {
    let reserve = cfg.vram_reserve_gb;
    FAMILIES
        .iter()
        .map(|fam| {
            let rows = DETAILS
                .iter()
                .map(|d| {
                    let res = *d.resolutions(fam.id).iter().max().unwrap();
                    let cells = [1u32, 2, 4]
                        .iter()
                        .map(|&batch| {
                            let p = pick_precision(fam.id, None, total_gb, reserve, res, batch);
                            let b = budget(fam.id, p, res, batch, total_gb, reserve);
                            FitCell { need_gb: b.need_gb, fits: b.fits, offload: b.offload }
                        })
                        .collect();
                    FitRow { detail: d.label, resolutions: d.resolutions(fam.id), cells }
                })
                .collect();
            let res = DETAILS[1].resolutions(fam.id);
            let top = *res.iter().max().unwrap();
            let p = pick_precision(fam.id, None, total_gb, reserve, top, 1);
            let b = budget(fam.id, p, top, 1, total_gb, reserve);
            let why = explain(fam.id, p, &b, res, 1, fam.rank.unwrap_or(GOALS[0].rank), total_gb);
            FitModel { family: fam, rows, why }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat() -> Catalog {
        let b = |id: &str, fam: &'static str| Base {
            id: id.into(),
            family: fam,
            label: id.into(),
            note: String::new(),
            ready: true,
            download: None,
            recommended: true,
            blocked: None,
            precision: None,
            size_gb: None,
        };
        Catalog {
            bases: vec![b("krea2:raw", "krea2"), b("moss_image:base", "moss_image"), b("minimax_h3_image:base", "minimax_h3_image")],
            local: vec![],
            moss_text_encoders: vec![TextEncoder { id: "official".into(), label: "x".into(), download: None }],
            moss_vae: Some(PathBuf::from("flux2-vae.safetensors")),
        }
    }

    fn ds(n: usize, captioned: usize) -> Dataset {
        Dataset { root: "d".into(), images: n, captioned, folders: vec![DatasetFolder { path: "d".into(), images: n, captions: captioned }], ..Default::default() }
    }

    fn form(base: &str) -> Form {
        Form { base: base.into(), name: "t".into(), goal: "character".into(), detail: "balanced".into(), batch: 1, previews: true, ..Default::default() }
    }

    #[test]
    fn plans_match_the_python_version() {
        let paths = Paths::new(std::env::temp_dir());
        let cfg = Settings::default();
        let k = build_plan(&form("krea2:raw"), &cat(), &ds(20, 20), &paths, &cfg, 24.0).unwrap();
        assert_eq!((k.precision.as_str(), k.budget.need_gb, k.steps, k.rank), ("qfloat8", 18.6, 1500, 32));
        let m = build_plan(&Form { batch: 2, ..form("moss_image:base") }, &cat(), &ds(8, 8), &paths, &cfg, 24.0).unwrap();
        assert_eq!((m.precision.as_str(), m.budget.need_gb), ("bf16", 16.3));
        let h = build_plan(&form("minimax_h3_image:base"), &cat(), &ds(8, 8), &paths, &cfg, 24.0).unwrap();
        assert_eq!((h.budget.offload, h.budget.need_gb, h.rank), (0.15, 21.4, 16));
        let hm = build_plan(&Form { detail: "max".into(), ..form("minimax_h3_image:base") }, &cat(), &ds(8, 8), &paths, &cfg, 24.0).unwrap();
        assert_eq!(hm.budget.offload, 0.25);
        assert!(build_plan(&Form { detail: "max".into(), batch: 2, ..form("krea2:raw") }, &cat(), &ds(8, 8), &paths, &cfg, 24.0).is_err());
    }

    #[test]
    fn blank_trigger_with_uncaptioned_images_warns_instead_of_failing() {
        let p = build_plan(&form("krea2:raw"), &cat(), &ds(8, 4), &Paths::new(std::env::temp_dir()), &Settings::default(), 24.0).unwrap();
        assert!(p.warnings.iter().any(|w| w.contains("empty caption")));
        assert!(p.config["config"]["process"][0].get("trigger_word").is_none());
    }

    /// Writes one config per model to DOJO_CONFIG_OUT so the engine's parser can check them.
    #[test]
    fn write_configs_for_the_engine() {
        let Some(out) = std::env::var_os("DOJO_CONFIG_OUT") else { return };
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        let ds = inspect_dataset(&std::env::var("DOJO_TEST_DATASET").unwrap()).unwrap();
        let cfg = Settings::default();
        let real = catalog(&cfg);
        for base in ["krea2:raw", "moss_image:base", "minimax_h3_image:base"] {
            let f = Form { trigger: "zxshape".into(), dataset: ds.root.display().to_string(), ..form(base) };
            let p = build_plan(&f, &real, &ds, &Paths::new(out.clone()), &cfg, 24.0).unwrap();
            let file = out.join(format!("{}.json", p.family));
            std::fs::write(&file, serde_json::to_string_pretty(&p.config).unwrap()).unwrap();
        }
    }
}
