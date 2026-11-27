//! The window: an aurora wallpaper with floating Liquid Glass panels (rust-glass).
//! Pages: New job, Queue, a job's own page, and GPU fit.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rust_glass::prelude::*;

use crate::hardware::{self, Gpu, Monitor};
use crate::jobs::{self, Shared};
use crate::recipes::{self, Catalog, Dataset, Form, Plan};
use crate::settings::{Paths, Settings};
use crate::thumbs::{Thumbs, square_uv};

// ---------------------------------------------------------------- look

pub const ACCENT: Color = Color::hex(0xFF9F45);
pub const GOOD: Color = Color::hex(0x4CC38A);
pub const BAD: Color = Color::hex(0xF46A62);
pub const INFO: Color = Color::hex(0x62A8FF);
pub const WARN: Color = Color::hex(0xF2C14E);
pub const TEXT: Color = Color::rgb(0.94, 0.94, 0.97);
pub const MUTED: Color = Color::rgb(0.66, 0.68, 0.75);
pub const FAINT: Color = Color::rgb(0.48, 0.50, 0.58);
const C_WEIGHTS: Color = Color::hex(0xFF9F45);
const C_ACT: Color = Color::hex(0xB48CFF);
const C_OVER: Color = Color::hex(0x5FD0C4);
const C_OTHER: Color = Color::rgba(0.55, 0.57, 0.64, 0.9);

const GAP: f32 = 14.0;
const SIDEBAR_W: f32 = 222.0;
const HEADER_H: f32 = 58.0;
const STATUS_H: f32 = 40.0;

#[derive(Clone, PartialEq, Debug)]
pub enum Page {
    New,
    Queue,
    Job(String),
    Fit,
}

const NAV: [(Icon, &str); 3] = [(Icon::Plus, "New job"), (Icon::Layers, "Queue"), (Icon::Sliders, "GPU fit")];

enum DsState {
    Empty,
    Loading,
    Ready(Dataset),
    Failed(String),
}

/// A copy of one job, taken under the manager's lock.
#[derive(Clone)]
pub struct JobView {
    pub id: String,
    pub name: String,
    pub status: String,
    pub message: String,
    pub step: u32,
    pub total: u32,
    pub speed: String,
    pub eta: String,
    pub phase: String,
    pub last_loss: Option<f32>,
    pub peak_gb: f32,
    #[allow(dead_code)]
    pub baseline_gb: f32,
    pub spilling: bool,
    pub plan: Plan,
}

impl JobView {
    fn active(&self) -> bool {
        jobs::ACTIVE.contains(&self.status.as_str())
    }
    fn running(&self) -> bool {
        matches!(self.status.as_str(), "running" | "starting")
    }
    fn progress(&self) -> f32 {
        if self.total == 0 { 0.0 } else { (self.step as f32 / self.total as f32).min(1.0) }
    }
    fn info(&self) -> String {
        if self.running() {
            if self.step > 0 { format!("{}{}", self.speed, if self.eta.is_empty() { String::new() } else { format!(", {} left", self.eta) }) } else if self.phase.is_empty() { "Loading the model".into() } else { self.phase.clone() }
        } else {
            self.message.clone()
        }
    }
}

#[derive(Default)]
struct Detail {
    id: String,
    loss: Vec<(u32, f32)>,
    vram: Vec<f32>,
    log: Vec<String>,
    samples: Vec<PathBuf>,
    checkpoints: Vec<String>,
    config: String,
}

fn status_label(s: &str) -> &str {
    match s {
        "queued" => "Queued",
        "waiting" => "Waiting for memory",
        "starting" => "Starting",
        "running" => "Training",
        "done" => "Finished",
        "failed" => "Failed",
        "stopped" => "Stopped",
        "interrupted" => "Interrupted",
        other => other,
    }
}

fn status_color(s: &str) -> Color {
    match s {
        "running" | "starting" => ACCENT,
        "done" => GOOD,
        "failed" => BAD,
        "waiting" => WARN,
        _ => MUTED,
    }
}

pub struct DojoApp {
    pub page: Page,
    pub paths: Paths,
    pub cfg: Settings,
    pub monitor: Monitor,
    pub manager: Shared,
    catalog: Arc<Mutex<(u64, Option<Catalog>)>>,
    thumbs: Thumbs,
    wallpaper: Option<ImageId>,

    // form
    pub family: usize,
    pub base: String,
    pub goal: usize,
    pub detail: usize,
    pub batch: usize,
    pub name: String,
    pub trigger: String,
    pub dataset: String,
    pub steps: String,
    pub rank: String,
    pub lr: String,
    pub te: usize,
    pub prompts: String,
    pub previews: bool,
    pub copy: bool,

    ds: DsState,
    ds_for: String,
    ds_edit: Option<Instant>,
    ds_rx: Option<Receiver<(String, Result<Dataset, String>)>>,
    ds_gen: u64,
    plan_key: String,
    pub plan: Option<Result<Plan, String>>,

    pub views: Vec<JobView>,
    detail_data: Detail,
    detail_at: Option<Instant>,
    status: (String, bool),
    confirm_delete: Option<String>,
    confirm_delete_open: bool,
    confirm_free: bool,
    paused: bool,
}

impl DojoApp {
    pub fn new(paths: Paths, cfg: Settings, start_manager: bool) -> Self {
        let monitor = Monitor::start(cfg.comfy_url.clone());
        let manager = jobs::Manager::new(paths.clone(), cfg.clone());
        let manager: Shared = if start_manager { manager.spawn(monitor.clone()) } else { Arc::new(Mutex::new(manager)) };
        let catalog: Arc<Mutex<(u64, Option<Catalog>)>> = Arc::new(Mutex::new((0, None)));
        {
            let (c, cfg) = (catalog.clone(), cfg.clone());
            std::thread::spawn(move || {
                loop {
                    let cat = recipes::catalog(&cfg);
                    {
                        let mut g = c.lock().unwrap();
                        g.0 += 1;
                        g.1 = Some(cat);
                    }
                    std::thread::sleep(std::time::Duration::from_secs(60));
                }
            });
        }
        let status = if paths.engine_ready() {
            ("Ready. Pick a model and a dataset folder.".to_string(), false)
        } else {
            ("The training engine is not installed yet. Run setup.bat in the Dojo 24 folder.".to_string(), true)
        };
        Self {
            page: Page::New,
            paths,
            cfg,
            monitor,
            manager,
            catalog,
            thumbs: Thumbs::new(),
            wallpaper: None,
            family: 0,
            base: String::new(),
            goal: 0,
            detail: 1,
            batch: 0,
            name: String::new(),
            trigger: String::new(),
            dataset: String::new(),
            steps: String::new(),
            rank: String::new(),
            lr: String::new(),
            te: 0,
            prompts: String::new(),
            previews: true,
            copy: true,
            ds: DsState::Empty,
            ds_for: String::new(),
            ds_edit: None,
            ds_rx: None,
            ds_gen: 0,
            plan_key: String::new(),
            plan: None,
            views: Vec::new(),
            detail_data: Detail::default(),
            detail_at: None,
            status,
            confirm_delete: None,
            confirm_delete_open: false,
            confirm_free: false,
            paused: false,
        }
    }

    pub fn catalog(&self) -> Option<Catalog> {
        self.catalog.lock().unwrap().1.clone()
    }

    #[cfg(test)]
    pub fn wait_for_catalog(&self) {
        let t = Instant::now();
        while self.catalog.lock().unwrap().1.is_none() && t.elapsed().as_secs() < 60 {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    fn family_id(&self) -> &'static str {
        recipes::FAMILIES[self.family].id
    }

    pub fn batch_value(&self) -> u32 {
        [1, 2, 4][self.batch]
    }

    fn gpu(&self) -> Option<Gpu> {
        self.monitor.gpu()
    }

    pub fn set_status(&mut self, text: impl Into<String>, err: bool) {
        self.status = (text.into(), err);
    }

    // ---------------------------------------------------------------- state upkeep

    fn snapshot(&mut self) {
        let m = self.manager.lock().unwrap();
        self.paused = m.paused;
        self.views = m
            .ordered()
            .into_iter()
            .map(|j| JobView {
                id: j.meta.id.clone(),
                name: j.meta.name.clone(),
                status: j.meta.status.clone(),
                message: j.meta.message.clone(),
                step: j.step,
                total: j.total,
                speed: j.speed.clone(),
                eta: j.eta.clone(),
                phase: j.phase.clone(),
                last_loss: j.last_loss(),
                peak_gb: j.meta.peak_gb,
                baseline_gb: j.meta.baseline_gb,
                spilling: j.meta.spilling,
                plan: j.meta.plan.clone(),
            })
            .collect();
        if let Page::Job(id) = &self.page {
            let stale = self.detail_data.id != *id || self.detail_at.is_none_or(|t| t.elapsed().as_secs_f32() > 1.0);
            if stale && let Some(j) = m.get(id) {
                let config = if self.detail_data.id == *id { std::mem::take(&mut self.detail_data.config) } else { std::fs::read_to_string(j.config_path()).unwrap_or_default() };
                self.detail_data = Detail {
                    id: id.clone(),
                    loss: j.loss.clone(),
                    vram: j.vram.clone(),
                    log: j.tail.iter().rev().take(40).rev().cloned().collect(),
                    samples: j.samples(&m.paths),
                    checkpoints: j.checkpoints(&m.paths),
                    config,
                };
                self.detail_at = Some(Instant::now());
            }
        }
    }

    fn poll_dataset(&mut self) {
        if self.dataset != self.ds_for {
            self.ds_for = self.dataset.clone();
            self.ds_edit = Some(Instant::now());
            if self.dataset.trim().is_empty() {
                self.ds = DsState::Empty;
                self.ds_edit = None;
                self.ds_gen += 1;
            }
        }
        if let Some(t) = self.ds_edit
            && t.elapsed().as_secs_f32() > 0.35
        {
            self.ds_edit = None;
            let (tx, rx) = channel();
            let path = self.dataset.clone();
            std::thread::spawn(move || {
                let r = recipes::inspect_dataset(&path);
                let _ = tx.send((path, r));
            });
            self.ds_rx = Some(rx);
            self.ds = DsState::Loading;
        }
        if let Some(rx) = &self.ds_rx
            && let Ok((path, r)) = rx.try_recv()
        {
            self.ds_rx = None;
            if path == self.dataset {
                self.ds = match r {
                    Ok(d) => {
                        if self.name.trim().is_empty()
                            && let Some(n) = d.root.file_name().and_then(|n| n.to_str())
                        {
                            self.name = recipes::safe_name(n);
                        }
                        DsState::Ready(d)
                    }
                    Err(e) => DsState::Failed(e),
                };
                self.ds_gen += 1;
            }
        }
    }

    pub fn dataset_ready(&self) -> Option<&Dataset> {
        match &self.ds {
            DsState::Ready(d) => Some(d),
            _ => None,
        }
    }

    pub fn dataset_loading(&self) -> bool {
        self.ds_edit.is_some() || matches!(self.ds, DsState::Loading)
    }

    pub fn form(&self, cat: &Catalog) -> Form {
        let te = cat.moss_text_encoders.get(self.te).map(|t| t.id.clone()).unwrap_or_default();
        Form {
            base: self.base.clone(),
            name: self.name.clone(),
            trigger: self.trigger.clone(),
            dataset: self.dataset.clone(),
            goal: recipes::GOALS[self.goal].id.into(),
            detail: recipes::DETAILS[self.detail].id.into(),
            batch: self.batch_value(),
            steps: self.steps.clone(),
            rank: self.rank.clone(),
            lr: self.lr.clone(),
            text_encoder: te,
            prompts: self.prompts.clone(),
            previews: self.previews,
        }
    }

    pub fn replan(&mut self) {
        let (cat_gen, cat) = {
            let g = self.catalog.lock().unwrap();
            (g.0, g.1.clone())
        };
        let Some(cat) = cat else {
            self.plan = None;
            return;
        };
        // keep the base on the chosen family
        let fid = self.family_id();
        if cat.get(&self.base).is_none_or(|b| b.family != fid || b.blocked.is_some()) {
            self.base = cat.bases.iter().find(|b| b.family == fid).map(|b| b.id.clone()).unwrap_or_default();
        }
        let total = self.gpu().map_or(24.0, |g| g.total_gb);
        let form = self.form(&cat);
        let key = format!("{form:?}|{cat_gen}|{}|{total:.1}", self.ds_gen);
        if key == self.plan_key {
            return;
        }
        self.plan_key = key;
        self.plan = match (&self.ds, self.dataset.trim().is_empty()) {
            (_, true) => None,
            (DsState::Ready(d), _) => Some(recipes::build_plan(&form, &cat, d, &self.paths, &self.cfg, total)),
            (DsState::Failed(e), _) => Some(Err(e.clone())),
            _ => None,
        };
    }

    pub fn create_job(&mut self) {
        let Some(Ok(plan)) = self.plan.clone() else { return };
        let res = self.manager.lock().unwrap().create(plan, self.copy);
        match res {
            Ok(id) => {
                self.set_status("Job added to the queue.", false);
                self.snapshot();
                self.page = Page::Job(id);
            }
            Err(e) => self.set_status(e, true),
        }
    }

    pub fn job_action(&mut self, id: &str, action: &str) {
        let r = {
            let mut m = self.manager.lock().unwrap();
            match action {
                "stop" => {
                    m.stop(id);
                    Ok(())
                }
                "resume" => m.resume(id),
                "delete" => m.delete(id, false),
                "delete_files" => m.delete(id, true),
                _ => Ok(()),
            }
        };
        match r {
            Ok(()) => {
                if action.starts_with("delete") && self.page == Page::Job(id.to_string()) {
                    self.page = Page::Queue;
                }
            }
            Err(e) => self.set_status(e, true),
        }
        self.detail_at = None;
        self.snapshot();
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.manager.lock().unwrap().paused = paused;
        self.paused = paused;
    }

    fn free_comfy(&mut self, force: bool) {
        let c = self.monitor.comfy();
        if !c.running {
            return self.set_status("ComfyUI is not running, so something else is holding the memory.", true);
        }
        if c.busy && !force {
            self.confirm_free = true;
            return;
        }
        if hardware::comfy_free(&self.cfg.comfy_url) {
            self.set_status("Asked ComfyUI to unload its models. It loads them again on its next prompt.", false);
        } else {
            self.set_status("ComfyUI did not accept the unload request.", true);
        }
    }

    // ---------------------------------------------------------------- frame pieces

    fn wallpaper(&mut self, ctx: &mut Context) {
        let id = *self.wallpaper.get_or_insert_with(|| ctx.add_image(480, 300, aurora(480, 300)));
        let screen = ctx.screen.expand(64.0);
        ctx.background().image(screen, 0.0, id, rect(0.0, 0.0, 1.0, 1.0), Color::WHITE);
    }

    fn sidebar(&mut self, ctx: &mut Context) {
        let r = rect(GAP, GAP, SIDEBAR_W, ctx.screen.height() - 2.0 * GAP);
        let gpu = self.gpu();
        let mut active: Vec<JobView> = self.views.iter().filter(|v| v.active()).cloned().collect();
        active.sort_by_key(|v| !v.running());
        active.truncate(3);
        let mut go: Option<Page> = None;
        Panel::new("sidebar", r).corner(Corner::Radius(26.0)).material(ctx.theme.thick).padding(14.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                let b = ui.allocate(vec2(36.0, 36.0));
                let l = ui.layout_text("24", &TextStyle::new(15.0).weight(Weight::Bold));
                let mut p = ui.painter();
                p.fill(b, Corner::Radius(11.0), ACCENT);
                p.text_centered(b, &l, Color::hex(0x1A1208), 1.0);
                ui.vertical(150.0, |ui| {
                    ui.styled_text("Dojo 24", TextStyle::new(18.0).weight(Weight::Bold), TEXT);
                    ui.styled_text("LoRA training, 24 GB", TextStyle::new(11.5), MUTED);
                });
            });
            ui.space(14.0);
            let sel = match self.page {
                Page::New => 0,
                Page::Queue | Page::Job(_) => 1,
                Page::Fit => 2,
            };
            if let Some(i) = nav(ui, sel, &NAV) {
                // a job's page highlights Queue too, so clicking Queue there still goes back
                let target = [Page::New, Page::Queue, Page::Fit][i].clone();
                if target != self.page {
                    go = Some(target);
                }
            }
            ui.space(12.0);
            for v in &active {
                let w = ui.available_width();
                let r = ui.allocate_response(vec2(w, if v.running() { 52.0 } else { 40.0 }), Sense::Click);
                if let Some(n) = ui.access(r.id, rust_glass::accessibility::Role::Button, r.rect) {
                    n.label(format!("Open job {}", v.name));
                }
                if r.clicked {
                    go = Some(Page::Job(v.id.clone()));
                }
                let name = ui.layout_text(&truncate(&v.name, 22), &TextStyle::new(12.5).weight(Weight::Semibold));
                let sub = ui.layout_text(&truncate(&if v.running() && v.step > 0 { format!("{} / {}", v.step, v.total) } else { status_label(&v.status).to_string() }, 30), &TextStyle::new(11.0));
                let mut p = ui.painter();
                p.fill(r.rect, Corner::Radius(14.0), if r.hovered { Color::white(0.12) } else { Color::white(0.07) });
                p.text(r.rect.min + vec2(12.0, 7.0), &name, TEXT);
                p.text(r.rect.min + vec2(12.0, 24.0), &sub, status_color(&v.status));
                if v.running() {
                    let bar = Rect::from_min_size(r.rect.min + vec2(12.0, 41.0), vec2(w - 24.0, 4.0));
                    p.fill(bar, Corner::Capsule, Color::white(0.12));
                    p.fill(Rect::from_min_size(bar.min, vec2(bar.width() * v.progress(), 4.0)), Corner::Capsule, ACCENT);
                }
            }
            let left = ui.available_height() - 62.0;
            if left > 0.0 {
                ui.space(left);
            }
            match &gpu {
                Some(g) => {
                    ui.horizontal(|ui| {
                        ui.styled_text(&g.name, TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                        ui.horizontal_right(|ui| {
                            if let Some(t) = g.temp_c {
                                ui.styled_text(&format!("{t}°C"), TextStyle::new(12.0), if t >= 80 { BAD } else if t >= 60 { WARN } else { GOOD });
                            }
                            ui.space(4.0);
                        });
                    });
                    let bw = ui.available_width();
                    bar(ui, bw, 6.0, g.used_gb / g.total_gb, ACCENT);
                    ui.styled_text(&format!("{:.1} of {:.0} GB in use", g.used_gb, g.total_gb), TextStyle::new(11.5), MUTED);
                }
                None => {
                    ui.styled_text("No NVIDIA GPU found", TextStyle::new(12.0), BAD);
                }
            }
        });
        if let Some(p) = go {
            self.page = p;
            self.detail_at = None;
        }
    }

    fn header(&mut self, ctx: &mut Context) {
        let x = GAP * 2.0 + SIDEBAR_W;
        let r = rect(x, GAP, ctx.screen.width() - x - GAP, HEADER_H);
        let title = match &self.page {
            Page::New => "New training job".to_string(),
            Page::Queue => "Queue".into(),
            Page::Fit => "What fits on this card".into(),
            Page::Job(id) => self.views.iter().find(|v| &v.id == id).map_or("Job".into(), |v| v.name.clone()),
        };
        let mut act: Option<&'static str> = None;
        let job = match &self.page {
            Page::Job(id) => self.views.iter().find(|v| &v.id == id).cloned(),
            _ => None,
        };
        let plan_ok = matches!(self.plan, Some(Ok(_)));
        let busy = self.views.iter().any(JobView::active);
        let engine = self.paths.engine_ready();
        let paused = self.paused;
        let page = self.page.clone();
        Panel::new("header", r).corner(Corner::Capsule).padding(10.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.allocate(vec2(12.0, 36.0));
                if matches!(page, Page::Job(_)) && ui.add(Button::new("Back").icon(Icon::ChevronLeft).small()).clicked {
                    act = Some("back");
                }
                ui.styled_text(&title, TextStyle::new(19.0).weight(Weight::Bold), TEXT);
                ui.horizontal_right(|ui| {
                    match &page {
                        Page::New => {
                            let label = if busy { "Add to queue" } else { "Create job" };
                            if ui.add_enabled(plan_ok && engine, |ui| ui.add(Button::new(label).icon(Icon::Play).prominent())).clicked {
                                act = Some("create");
                            }
                        }
                        Page::Queue => {
                            if ui.add(Button::new(if paused { "Resume queue" } else { "Pause queue" }).icon(if paused { Icon::Play } else { Icon::Pause })).clicked {
                                act = Some("pause");
                            }
                            if ui.add(Button::new("New job").icon(Icon::Plus)).clicked {
                                act = Some("new");
                            }
                        }
                        Page::Job(_) => {
                            if let Some(j) = &job {
                                if !j.running() && ui.add(Button::new("Delete").icon(Icon::Trash)).clicked {
                                    act = Some("delete");
                                }
                                if matches!(j.status.as_str(), "stopped" | "failed" | "interrupted") && ui.add(Button::new("Resume").icon(Icon::Play).prominent()).clicked {
                                    act = Some("resume");
                                }
                                if j.active() && ui.add(Button::new(if j.running() { "Stop" } else { "Remove from queue" }).icon(Icon::Close).tinted(BAD)).clicked {
                                    act = Some("stop");
                                }
                            }
                        }
                        Page::Fit => {}
                    }
                    ui.space(12.0); // keep clear of the capsule's rounded end
                });
            });
        });
        let job_id = job.map(|j| j.id);
        match act {
            Some("back") => self.page = Page::Queue,
            Some("create") => self.create_job(),
            Some("pause") => self.set_paused(!paused),
            Some("new") => self.page = Page::New,
            Some("delete") => {
                self.confirm_delete = job_id;
                self.confirm_delete_open = true;
            }
            Some(a) => {
                if let Some(id) = job_id {
                    self.job_action(&id, a);
                }
            }
            None => {}
        }
    }

    fn status_bar(&mut self, ctx: &mut Context) {
        let x = GAP * 2.0 + SIDEBAR_W;
        let r = rect(x, ctx.screen.height() - GAP - STATUS_H, ctx.screen.width() - x - GAP, STATUS_H);
        let (text, err) = self.status.clone();
        Panel::new("status", r).corner(Corner::Capsule).padding(6.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.space(10.0);
                let d = ui.allocate(vec2(8.0, 8.0));
                ui.painter().fill(d, Corner::Capsule, if err { BAD } else { GOOD });
                ui.styled_text(&text, TextStyle::new(12.5), if err { BAD } else { TEXT });
            });
        });
    }

    fn content_rect(ctx: &Context) -> Rect {
        let x = GAP * 2.0 + SIDEBAR_W;
        Rect { min: vec2(x, GAP * 2.0 + HEADER_H), max: vec2(ctx.screen.width() - GAP, ctx.screen.height() - GAP * 2.0 - STATUS_H) }
    }

    // ---------------------------------------------------------------- New job

    fn new_page(&mut self, ctx: &mut Context) {
        let area = Self::content_rect(ctx);
        let left_w = (area.width() * 0.47).clamp(420.0, 620.0);
        let left = Rect { min: area.min, max: vec2(area.min.x + left_w, area.max.y) };
        let right = Rect { min: vec2(left.max.x + GAP, area.min.y), max: area.max };
        let short = match (&self.plan, self.gpu()) {
            (Some(Ok(p)), Some(g)) => p.budget.need_gb + 0.5 > g.free_gb,
            _ => false,
        };
        let mem = Rect { min: right.min, max: vec2(right.max.x, right.min.y + if short { 236.0 } else { 196.0 }) };
        let why = Rect { min: vec2(right.min.x, mem.max.y + GAP), max: right.max };
        let cat = self.catalog();
        Panel::new("form", left).corner(Corner::Radius(24.0)).material(ctx.theme.thick).padding(14.0).show(ctx, |ui| {
            ScrollArea::new().id("form_scroll").show(ui, |ui| self.form_ui(ui, cat.as_ref()));
        });
        Panel::new("memory", mem).corner(Corner::Radius(24.0)).padding(16.0).show(ctx, |ui| self.memory_ui(ui));
        Panel::new("why", why).corner(Corner::Radius(24.0)).material(ctx.theme.thick).padding(14.0).show(ctx, |ui| {
            ScrollArea::new().id("why_scroll").show(ui, |ui| self.why_ui(ui));
        });
    }

    fn form_ui(&mut self, ui: &mut Ui, cat: Option<&Catalog>) {
        let w = ui.available_width() - 8.0;
        let Some(cat) = cat else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.secondary("Looking at the models in your ComfyUI folder");
            });
            return;
        };
        section(ui, "Model", |ui| {
            let labels: Vec<&str> = recipes::FAMILIES.iter().map(|f| f.label).collect();
            let r = ui.add(Segmented::new(&mut self.family, &labels).width(w - 32.0));
            ui.ctx.access_label(r.id, "Model family");
            let fam = &recipes::FAMILIES[self.family];
            ui.styled_text(fam.params, TextStyle::new(12.0).wrap(w - 32.0), MUTED);
            let bases: Vec<&recipes::Base> = cat.all().filter(|b| b.family == fam.id).collect();
            let names: Vec<String> = bases
                .iter()
                .map(|b| {
                    let mut s = if b.id.starts_with("file:") { format!("Your checkpoint: {}", b.label) } else { b.label.clone() };
                    if b.blocked.is_some() {
                        s.push_str(" (cannot be loaded)");
                    } else if b.download.is_some() {
                        s.push_str(" (downloads first)");
                    }
                    s
                })
                .collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let mut i = bases.iter().position(|b| b.id == self.base).unwrap_or(0);
            if ComboBox::new(&mut i, &refs).width(w - 32.0).show(ui)
                && let Some(b) = bases.get(i)
            {
                if let Some(reason) = &b.blocked {
                    self.status = (format!("{} cannot be used: {reason}.", b.label), true);
                } else {
                    self.base = b.id.clone();
                }
            }
            if let Some(b) = bases.iter().find(|b| b.id == self.base) {
                ui.styled_text(&b.note, TextStyle::new(12.0).wrap(w - 32.0), MUTED);
                if let Some(dl) = &b.download {
                    ui.styled_text(&format!("First run fetches {dl}."), TextStyle::new(12.0).wrap(w - 32.0), INFO);
                }
            }
            if fam.id == "moss_image" && !cat.moss_text_encoders.is_empty() {
                ui.space(4.0);
                ui.styled_text("Text encoder", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                let tes: Vec<String> = cat.moss_text_encoders.iter().map(|t| if t.download.is_some() { format!("{} (downloads)", t.label) } else { t.label.clone() }).collect();
                let refs: Vec<&str> = tes.iter().map(String::as_str).collect();
                self.te = self.te.min(refs.len() - 1);
                ComboBox::new(&mut self.te, &refs).width(w - 32.0).show(ui);
            }
        });
        ui.space(6.0);
        section(ui, "Dataset", |ui| {
            ui.horizontal(|ui| {
                let r = ui.add(TextEdit::new(&mut self.dataset).hint("Folder of images, or drop one on the window").icon(Icon::Folder).width(w - 140.0));
                ui.ctx.access_label(r.id, "Dataset folder");
                if ui.add(Button::new("Browse").icon(Icon::Folder)).clicked
                    && let Some(p) = rfd::FileDialog::new().pick_folder()
                {
                    self.dataset = p.display().to_string();
                }
            });
            let (text, color) = match &self.ds {
                DsState::Empty => ("Subfolders are included. Put a .txt caption next to each image. Images without one use the trigger word, or an empty caption if that is blank too.".to_string(), MUTED),
                DsState::Loading => ("Looking at the folder".to_string(), MUTED),
                DsState::Failed(e) => (e.clone(), BAD),
                DsState::Ready(d) => {
                    let missing = d.images - d.captioned;
                    let caps = if missing == 0 {
                        "Every image has a caption.".to_string()
                    } else {
                        format!("{missing} have no caption file; those use the trigger word, or an empty caption if you leave it blank.")
                    };
                    (format!("{} images in {} folder{}. {caps}", d.images, d.folders.len(), if d.folders.len() == 1 { "" } else { "s" }), MUTED)
                }
            };
            ui.styled_text(&text, TextStyle::new(12.0).wrap(w - 32.0), color);
            if let DsState::Ready(d) = &self.ds {
                let previews = d.previews.clone();
                let size = ((w - 32.0 - 6.0 * 4.0) / 5.0).min(84.0);
                for row in previews.chunks(5) {
                    ui.horizontal(|ui| {
                        ui.spacing = 6.0;
                        for p in row {
                            let r = ui.allocate(vec2(size, size));
                            let t = self.thumbs.get(p, 192);
                            let mut pt = ui.painter();
                            match t {
                                Some((id, s)) => pt.image(r, Corner::Radius(10.0), id, square_uv(s), Color::WHITE),
                                None => pt.fill(r, Corner::Radius(10.0), Color::white(0.06)),
                            }
                        }
                    });
                }
            }
        });
        ui.space(6.0);
        section(ui, "Job", |ui| {
            ui.horizontal(|ui| {
                ui.vertical((w - 44.0) / 2.0, |ui| {
                    ui.styled_text("LoRA name", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                    let r = ui.add(TextEdit::new(&mut self.name).hint("my_character_v1").width((w - 44.0) / 2.0));
                    ui.ctx.access_label(r.id, "LoRA name");
                });
                ui.vertical((w - 44.0) / 2.0, |ui| {
                    ui.styled_text("Trigger word", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                    let r = ui.add(TextEdit::new(&mut self.trigger).hint("optional").width((w - 44.0) / 2.0));
                    ui.ctx.access_label(r.id, "Trigger word");
                });
            });
            ui.styled_text("What are you teaching it?", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
            let labels: Vec<&str> = recipes::GOALS.iter().map(|g| g.label).collect();
            let r = ui.add(Segmented::new(&mut self.goal, &labels).width(w - 32.0));
            ui.ctx.access_label(r.id, "Training goal");
            ui.styled_text(recipes::GOALS[self.goal].hint, TextStyle::new(12.0), MUTED);
        });
        ui.space(6.0);
        let plan = self.plan.clone();
        section(ui, "Training", |ui| {
            let fid = self.family_id();
            ui.styled_text("Resolution", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
            let labels: Vec<String> = recipes::DETAILS.iter().map(|d| format!("{} {}", d.label, d.resolutions(fid).iter().map(u32::to_string).collect::<Vec<_>>().join("+"))).collect();
            let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            let r = ui.add(Segmented::new(&mut self.detail, &refs).width(w - 32.0));
            ui.ctx.access_label(r.id, "Resolution");
            ui.styled_text("Batch size", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
            let r = ui.add(Segmented::new(&mut self.batch, &["1", "2", "4"]).width(w - 32.0));
            ui.ctx.access_label(r.id, "Batch size");
            let auto = |f: &dyn Fn(&Plan) -> String| plan.as_ref().and_then(|p| p.as_ref().ok()).map_or("auto".to_string(), |p| format!("auto: {}", f(p)));
            let cw = (w - 32.0 - 20.0) / 3.0;
            ui.horizontal(|ui| {
                for (label, value, hint) in [
                    ("Steps", &mut self.steps, auto(&|p| p.steps.to_string())),
                    ("Rank", &mut self.rank, auto(&|p| p.rank.to_string())),
                    ("Learning rate", &mut self.lr, auto(&|p| format!("{:e}", p.lr))),
                ] {
                    ui.vertical(cw, |ui| {
                        ui.styled_text(label, TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                        let r = ui.add(TextEdit::new(value).hint(&hint).width(cw));
                        ui.ctx.access_label(r.id, label);
                    });
                }
            });
        });
        ui.space(6.0);
        section(ui, "Previews and output", |ui| {
            ui.toggle(&mut self.previews, "Render preview images at every checkpoint");
            let dir = self.cfg.lora_dirs.get(self.family_id()).cloned().unwrap_or_else(|| "loras".into());
            ui.toggle(&mut self.copy, &format!("Copy the finished LoRA to ComfyUI {dir}"));
            ui.styled_text("Preview prompts, one per line", TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
            let r = ui.add(TextEdit::new(&mut self.prompts).hint("Leave empty for four standard prompts built from the trigger word.").rows(4).width(w - 32.0));
            ui.ctx.access_label(r.id, "Preview prompts");
        });
        ui.space(10.0);
    }

    fn memory_ui(&mut self, ui: &mut Ui) {
        let Some(g) = self.gpu() else {
            ui.heading("No NVIDIA GPU found");
            ui.secondary("nvidia-smi did not answer, so Dojo 24 cannot plan memory.");
            return;
        };
        let w = ui.available_width();
        let comfy = self.monitor.comfy();
        let mut free_clicked = false;
        match &self.plan {
            Some(Ok(p)) => {
                let b = &p.budget;
                ui.horizontal(|ui| {
                    ui.styled_text(&format!("This job needs about {:.1} of {:.0} GB", b.need_gb, g.total_gb), TextStyle::new(16.0).weight(Weight::Semibold), TEXT);
                    ui.horizontal_right(|ui| {
                        ui.styled_text("estimate", TextStyle::new(11.5), FAINT);
                    });
                });
                let other = g.used_gb;
                let over = (other + b.need_gb - g.total_gb).max(0.0);
                let k = if over > 0.0 { ((g.total_gb - other) / b.need_gb).max(0.0) } else { 1.0 };
                memory_strip(ui, w, g.total_gb, &[(other, C_OTHER), (b.weights_gb * k, C_WEIGHTS), (b.activations_gb * k, C_ACT), (b.overhead_gb * k, C_OVER)]);
                let offl = if b.offload > 0.0 { format!(", {:.0}% in system RAM", b.offload * 100.0) } else { String::new() };
                legend(ui, w, &[
                    (C_WEIGHTS, format!("Model weights, {}{offl}", p.precision), b.weights_gb),
                    (C_ACT, format!("Training at {} px, batch {}", p.resolutions.iter().max().unwrap_or(&0), p.batch), b.activations_gb),
                    (C_OVER, "LoRA, optimizer, CUDA".into(), b.overhead_gb),
                    (C_OTHER, "Other apps now".into(), other),
                ]);
                let short = b.need_gb + 0.5 - g.free_gb;
                if short > 0.0 {
                    ui.horizontal(|ui| {
                        ui.styled_text(&format!("Other apps hold {other:.1} GB, so {short:.1} GB more has to be freed first. A queued job waits and starts by itself."), TextStyle::new(12.0).wrap(w - if comfy.running { 200.0 } else { 0.0 }), WARN);
                        if comfy.running {
                            ui.horizontal_right(|ui| {
                                if ui.add(Button::new("Unload ComfyUI models").small()).clicked {
                                    free_clicked = true;
                                }
                            });
                        }
                    });
                }
            }
            other => {
                ui.styled_text(&format!("{:.1} of {:.0} GB free", g.free_gb, g.total_gb), TextStyle::new(16.0).weight(Weight::Semibold), TEXT);
                memory_strip(ui, w, g.total_gb, &[(g.used_gb, C_OTHER)]);
                legend(ui, w, &[(C_OTHER, "In use by other apps".into(), g.used_gb)]);
                let msg = match other {
                    Some(Err(e)) => (e.clone(), BAD),
                    _ if self.dataset_loading() => ("Looking at the dataset".to_string(), MUTED),
                    _ => ("Pick a dataset folder to see how this job fits on the card.".to_string(), MUTED),
                };
                ui.styled_text(&msg.0, TextStyle::new(12.5).wrap(w), msg.1);
            }
        }
        if free_clicked {
            self.free_comfy(false);
        }
    }

    fn why_ui(&mut self, ui: &mut Ui) {
        let w = ui.available_width() - 8.0;
        let Some(Ok(p)) = self.plan.clone() else {
            ui.styled_text("Why these settings", TextStyle::new(16.0).weight(Weight::Semibold), TEXT);
            ui.secondary("The reasons for every setting appear here once there is a plan.");
            return;
        };
        ui.styled_text(
            &format!("{} on {}: {} images, {} steps, a checkpoint every {} steps.", p.name, p.base, p.images, p.steps, p.save_every),
            TextStyle::new(13.0).weight(Weight::Semibold).wrap(w),
            TEXT,
        );
        for d in &p.downloads {
            note(ui, w, Icon::Download, &format!("First run downloads {d}."), INFO);
        }
        for x in &p.warnings {
            note(ui, w, Icon::Warning, x, WARN);
        }
        ui.space(6.0);
        ui.styled_text("Why these settings on this card", TextStyle::new(15.0).weight(Weight::Semibold), TEXT);
        why_rows(ui, w, &p.why);
        ui.space(6.0);
        ui.collapsing("Engine config", |ui| {
            let text = serde_json::to_string_pretty(&p.config).unwrap_or_default();
            ui.styled_text(&text, TextStyle::new(11.0).mono().wrap(w - 20.0), MUTED);
        });
    }

    // ---------------------------------------------------------------- Queue

    fn queue_page(&mut self, ctx: &mut Context) {
        let area = Self::content_rect(ctx);
        let views = self.views.clone();
        let paused = self.paused;
        let mut act: Option<(String, &'static str)> = None;
        Panel::new("queue", area).corner(Corner::Radius(24.0)).material(ctx.theme.thick).padding(16.0).show(ctx, |ui| {
            let w = ui.available_width() - 8.0;
            let running = views.iter().any(JobView::running);
            ui.horizontal(|ui| {
                let d = ui.allocate(vec2(10.0, 10.0));
                ui.painter().fill(d, Corner::Capsule, if paused { WARN } else if running { GOOD } else { MUTED });
                ui.styled_text(
                    if paused { "Queue paused: nothing new starts until you resume it." } else if running { "Training. Queued jobs start when it finishes." } else { "Idle. A queued job starts as soon as there is memory for it." },
                    TextStyle::new(13.0),
                    TEXT,
                );
            });
            ui.space(8.0);
            ScrollArea::new().id("queue_scroll").show(ui, |ui| {
                if views.is_empty() {
                    ui.space(30.0);
                    ui.align = Align::Center;
                    ui.heading("No jobs yet");
                    ui.secondary("Create one from New job.");
                    return;
                }
                for v in &views {
                    ui.card(|ui| {
                        ui.horizontal(|ui| {
                            ui.styled_text(&v.name, TextStyle::new(15.0).weight(Weight::Semibold), TEXT);
                            pill(ui, status_label(&v.status), status_color(&v.status));
                            ui.styled_text(&format!("{}, {} px, batch {}", v.plan.base, v.plan.resolutions.iter().map(u32::to_string).collect::<Vec<_>>().join("+"), v.plan.batch), TextStyle::new(12.0), MUTED);
                            ui.horizontal_right(|ui| {
                                let (label, action, tint) = if v.active() { ("Stop", "stop", Some(BAD)) } else { ("Delete", "delete", None) };
                                let b = Button::new(label).small();
                                let r = ui.add(if let Some(t) = tint { b.tinted(t) } else { b });
                                ui.ctx.access_label(r.id, format!("{label} {}", v.name));
                                if r.clicked {
                                    act = Some((v.id.clone(), action));
                                }
                                let r = ui.add(Button::new("Open").small());
                                ui.ctx.access_label(r.id, format!("Open {}", v.name));
                                if r.clicked {
                                    act = Some((v.id.clone(), "open"));
                                }
                                ui.space(4.0);
                            });
                        });
                        bar(ui, w - 36.0, 6.0, v.progress(), status_color(&v.status));
                        let info = v.info();
                        ui.styled_text(&format!("Step {} of {}{}", v.step, v.total, if info.is_empty() { String::new() } else { format!(". {info}") }), TextStyle::new(12.0).wrap(w - 40.0), MUTED);
                    });
                    ui.space(4.0);
                }
            });
        });
        match act {
            Some((id, "open")) => {
                self.page = Page::Job(id);
                self.detail_at = None;
            }
            Some((id, "delete")) => {
                self.confirm_delete = Some(id);
                self.confirm_delete_open = true;
            }
            Some((id, a)) => self.job_action(&id, a),
            None => {}
        }
    }

    // ---------------------------------------------------------------- Job page

    fn job_page(&mut self, ctx: &mut Context, id: &str) {
        let area = Self::content_rect(ctx);
        let Some(v) = self.views.iter().find(|v| v.id == id).cloned() else {
            Panel::new("gone", area).corner(Corner::Radius(24.0)).padding(20.0).show(ctx, |ui| {
                ui.heading("This job no longer exists");
            });
            return;
        };
        let side_w = (area.width() * 0.32).clamp(300.0, 420.0);
        let main = Rect { min: area.min, max: vec2(area.max.x - side_w - GAP, area.max.y) };
        let side = Rect { min: vec2(main.max.x + GAP, area.min.y), max: area.max };
        let gpu = self.gpu();
        let detail = std::mem::take(&mut self.detail_data);
        Panel::new("job_main", main).corner(Corner::Radius(24.0)).material(ctx.theme.thick).padding(16.0).show(ctx, |ui| {
            ScrollArea::new().id("job_scroll").show(ui, |ui| {
                let w = ui.available_width() - 8.0;
                ui.horizontal(|ui| {
                    pill(ui, status_label(&v.status), status_color(&v.status));
                    let head = if v.running() && v.step > 0 {
                        format!("{:.0}% done{}", v.progress() * 100.0, if v.eta.is_empty() { String::new() } else { format!(", about {} left", v.eta) })
                    } else if v.running() {
                        if v.phase.is_empty() { "Loading the model".to_string() } else { v.phase.clone() }
                    } else {
                        v.message.clone()
                    };
                    ui.styled_text(&truncate(&head, 110), TextStyle::new(13.0), MUTED);
                });
                ui.space(4.0);
                bar(ui, w, 9.0, v.progress(), ACCENT);
                ui.space(6.0);
                let cw = (w - 4.0 * 10.0) / 5.0;
                ui.horizontal(|ui| {
                    ui.spacing = 10.0;
                    for (label, value) in [
                        ("Step", format!("{} / {}", v.step, v.total)),
                        ("Speed", if v.speed.is_empty() { "–".into() } else { v.speed.clone() }),
                        ("Time left", if v.running() && !v.eta.is_empty() { v.eta.clone() } else { "–".into() }),
                        ("Loss", v.last_loss.map_or("–".into(), |l| format!("{l:.4}"))),
                        ("Peak GPU memory", if v.peak_gb > 0.0 { format!("{:.1} GB", v.peak_gb) } else { "–".into() }),
                    ] {
                        ui.vertical(cw, |ui| {
                            ui.card(|ui| {
                                ui.styled_text(label, TextStyle::new(11.5), MUTED);
                                ui.styled_text(&value, TextStyle::new(18.0).weight(Weight::Bold), TEXT);
                            });
                        });
                    }
                });
                if v.spilling {
                    note(ui, w, Icon::Warning, "GPU memory is pinned at the ceiling. The driver is paging into system RAM, which makes steps many times slower. Stop the job and lower the batch size or resolution.", BAD);
                }
                ui.space(8.0);
                let half = (w - 12.0) / 2.0;
                ui.horizontal(|ui| {
                    ui.spacing = 12.0;
                    let sm = smooth(&detail.loss, (20.0 / detail.loss.len().max(1) as f32).clamp(0.01, 0.1));
                    let (lo, hi) = sm.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
                    chart(ui, half, 150.0, "Loss, smoothed", &sm, ACCENT, lo, hi);
                    let vram: Vec<(f32, f32)> = detail.vram.iter().enumerate().map(|(i, v)| (i as f32, *v)).collect();
                    let total = gpu.as_ref().map_or(24.0, |g| g.total_gb);
                    chart(ui, half, 150.0, &format!("GPU memory, 0 to {total:.0} GB"), &vram, C_ACT, 0.0, total);
                });
                ui.space(10.0);
                ui.styled_text("Samples", TextStyle::new(15.0).weight(Weight::Semibold), TEXT);
                if detail.samples.is_empty() {
                    ui.secondary("Preview images appear here after each checkpoint.");
                } else {
                    let cols = ((w + 8.0) / 150.0).floor().max(2.0) as usize;
                    let size = (w - 8.0 * (cols as f32 - 1.0)) / cols as f32;
                    for row in detail.samples.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.spacing = 8.0;
                            for p in row {
                                let r = ui.allocate_response(vec2(size, size), Sense::Click);
                                let t = self.thumbs.get(p, 384);
                                let mut pt = ui.painter();
                                match t {
                                    Some((img, s)) => pt.image(r.rect, Corner::Radius(12.0), img, square_uv(s), Color::WHITE),
                                    None => pt.fill(r.rect, Corner::Radius(12.0), Color::white(0.06)),
                                }
                                drop(pt);
                                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                                let r = r.on_hover_text(ui, &format!("{name}\nClick to open"));
                                if r.clicked {
                                    let _ = std::process::Command::new("explorer").arg(p).spawn();
                                }
                            }
                        });
                    }
                }
                if !detail.checkpoints.is_empty() {
                    ui.space(6.0);
                    ui.styled_text(&format!("Checkpoints: {}", detail.checkpoints.join(", ")), TextStyle::new(12.0).wrap(w), MUTED);
                }
            });
        });
        let out_dir = self.paths.output.join(&v.name);
        Panel::new("job_side", side).corner(Corner::Radius(24.0)).padding(16.0).show(ctx, |ui| {
            let w = ui.available_width() - 4.0;
            ui.styled_text("Recipe", TextStyle::new(15.0).weight(Weight::Semibold), TEXT);
            let p = &v.plan;
            for (k, val) in [
                ("Model", p.base.clone()),
                ("Goal", p.goal.clone()),
                ("Resolution", format!("{} px", p.resolutions.iter().map(u32::to_string).collect::<Vec<_>>().join(" + "))),
                ("Batch, rank", format!("{}, {}", p.batch, p.rank)),
                ("Learning rate", format!("{:e}", p.lr)),
                ("Weights", format!("{}{}", p.precision, if p.budget.offload > 0.0 { format!(", {:.0}% in system RAM", p.budget.offload * 100.0) } else { String::new() })),
                ("Images", p.images.to_string()),
                ("Planned memory", format!("{:.1} GB", p.budget.need_gb)),
            ] {
                ui.horizontal(|ui| {
                    ui.styled_text(k, TextStyle::new(12.0), MUTED);
                    ui.horizontal_right(|ui| {
                        ui.styled_text(&truncate(&val, 34), TextStyle::new(12.0), TEXT);
                    });
                });
            }
            ui.horizontal(|ui| {
                if ui.add(Button::new("Open output folder").icon(Icon::Folder).small()).clicked && out_dir.is_dir() {
                    let _ = std::process::Command::new("explorer").arg(&out_dir).spawn();
                }
            });
            ui.space(6.0);
            ui.styled_text("Engine log", TextStyle::new(15.0).weight(Weight::Semibold), TEXT);
            let text = if detail.log.is_empty() { "Waiting for the engine.".to_string() } else { detail.log.join("\n") };
            ScrollArea::new().id("log_scroll").height((ui.available_height() - 60.0).max(120.0)).show(ui, |ui| {
                ui.styled_text(&text, TextStyle::new(11.0).mono().wrap(w - 16.0), MUTED);
            });
            ui.collapsing("Engine config", |ui| {
                ui.styled_text(&truncate(&detail.config, 6000), TextStyle::new(10.5).mono().wrap(w - 20.0), MUTED);
            });
        });
        self.detail_data = detail;
    }

    // ---------------------------------------------------------------- GPU fit

    fn fit_page(&mut self, ctx: &mut Context) {
        let area = Self::content_rect(ctx);
        let gpu = self.gpu();
        let total = gpu.as_ref().map_or(24.0, |g| g.total_gb);
        let models = recipes::fit_table(&self.cfg, total);
        let measured = recipes::measured(&self.paths);
        let reserve = self.cfg.vram_reserve_gb;
        Panel::new("fit", area).corner(Corner::Radius(24.0)).material(ctx.theme.thick).padding(16.0).show(ctx, |ui| {
            ScrollArea::new().id("fit_scroll").show(ui, |ui| {
                let w = ui.available_width() - 8.0;
                note(ui, w, Icon::Info, &format!(
                    "Every new job starts from these settings on the {} ({total:.0} GB, with {reserve} GB kept free for Windows). The memory figures are estimates from the model sizes.",
                    gpu.as_ref().map_or("GPU".to_string(), |g| g.name.clone())), INFO);
                for m in &models {
                    ui.space(8.0);
                    section(ui, m.family.label, |ui| {
                        let w = ui.available_width();
                        ui.styled_text(m.family.params, TextStyle::new(12.0).wrap(w), MUTED);
                        ui.space(4.0);
                        fit_grid(ui, w, m);
                        let seen: Vec<String> = measured
                            .iter()
                            .filter(|(k, _)| k.starts_with(&format!("{}|", m.family.id)))
                            .map(|(k, v)| {
                                let parts: Vec<&str> = k.split('|').collect();
                                format!("{} at {} px, batch {}: the driver showed {:.1} GB", parts.get(1).unwrap_or(&""), parts.get(2).unwrap_or(&""), parts.get(3).unwrap_or(&""), v.peak_gb - v.baseline_gb)
                            })
                            .collect();
                        if !seen.is_empty() {
                            ui.styled_text(
                                &format!("Seen in earlier runs here: {}. That is memory PyTorch had reserved, which is more than the job strictly needs, so planning keeps using the estimate.", seen.join("; ")),
                                TextStyle::new(12.0).wrap(w),
                                MUTED,
                            );
                        }
                        ui.space(4.0);
                        why_rows(ui, w, &m.why.iter().filter(|r| r.setting != "Batch size").cloned().collect::<Vec<_>>());
                    });
                }
                ui.space(10.0);
            });
        });
    }
}

impl App for DojoApp {
    fn setup(&mut self, ctx: &mut Context) {
        ctx.theme.dark = true;
        ctx.theme.accent = ACCENT;
        ctx.theme.glass_tint = 0.30;
    }

    fn update(&mut self, ctx: &mut Context, _info: &FrameInfo) {
        self.snapshot();
        self.poll_dataset();
        self.replan();
        let mut keep: Vec<PathBuf> = self.dataset_ready().map(|d| d.previews.clone()).unwrap_or_default();
        keep.extend(self.detail_data.samples.iter().cloned());
        self.thumbs.retain(ctx, &keep);
        let loading = self.thumbs.poll(ctx);
        // background work cannot wake the window, so keep frames coming while it runs
        if loading || self.dataset_loading() || self.views.iter().any(JobView::running) || self.catalog.lock().unwrap().1.is_none() {
            ctx.request_repaint();
        }
        self.wallpaper(ctx);
        match self.page.clone() {
            Page::New => self.new_page(ctx),
            Page::Queue => self.queue_page(ctx),
            Page::Job(id) => self.job_page(ctx, &id),
            Page::Fit => self.fit_page(ctx),
        }
        self.sidebar(ctx);
        self.header(ctx);
        self.status_bar(ctx);

        let name = self.confirm_delete.as_ref().and_then(|id| self.views.iter().find(|v| &v.id == id)).map_or(String::new(), |v| v.name.clone());
        let msg = format!("{name} leaves the list. Delete its checkpoints and previews too? A LoRA already copied to ComfyUI stays.");
        match alert(ctx, Id::new("confirm_delete"), &mut self.confirm_delete_open, "Delete this job?", &msg, &["Cancel", "Keep files", "Delete files"]) {
            Some(1) => {
                if let Some(id) = self.confirm_delete.take() {
                    self.job_action(&id, "delete");
                }
            }
            Some(2) => {
                if let Some(id) = self.confirm_delete.take() {
                    self.job_action(&id, "delete_files");
                }
            }
            _ => {}
        }
        if alert(ctx, Id::new("confirm_free"), &mut self.confirm_free, "ComfyUI is generating", "Unloading its models now breaks the job it is running.", &["Cancel", "Unload anyway"]) == Some(1) {
            self.free_comfy(true);
        }
    }

    fn file_dropped(&mut self, _ctx: &mut Context, path: &std::path::Path) {
        let p = if path.is_dir() { path.to_path_buf() } else { path.parent().map_or_else(|| path.to_path_buf(), |d| d.to_path_buf()) };
        self.dataset = p.display().to_string();
        self.page = Page::New;
    }
}

// ---------------------------------------------------------------- pieces

/// A dark aurora with ember, violet and teal glows. Generated once, drawn stretched.
fn aurora(w: u32, h: u32) -> Vec<u8> {
    let blobs = [(0.15f32, 0.20f32, 0.40f32, [0.95f32, 0.52, 0.18]), (0.86, 0.18, 0.36, [0.48, 0.30, 0.92]), (0.65, 0.92, 0.42, [0.14, 0.62, 0.66]), (0.06, 0.95, 0.30, [0.85, 0.28, 0.40])];
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut c = [0.045 + 0.02 * v, 0.035 + 0.012 * u, 0.06 + 0.03 * (1.0 - v)];
            for (bx, by, r, col) in blobs {
                let d = ((u - bx) * 1.6).powi(2) + (v - by).powi(2);
                let k = (-d / (r * r)).exp() * 0.5;
                for i in 0..3 {
                    c[i] += col[i] * k;
                }
            }
            for v in c {
                out.push((v.clamp(0.0, 1.0).powf(1.0 / 1.15) * 255.0) as u8);
            }
            out.push(255);
        }
    }
    out
}

pub fn truncate(s: &str, n: usize) -> String {
    let n = n.max(4);
    if s.chars().count() > n { format!("{}…", s.chars().take(n - 1).collect::<String>()) } else { s.to_string() }
}

/// A progress bar across the full width (rust-glass's own stops at 460 points).
fn bar(ui: &mut Ui, w: f32, h: f32, value: f32, color: Color) {
    let r = ui.allocate(vec2(w, h));
    let mut p = ui.painter();
    p.fill(r, Corner::Capsule, Color::white(0.10));
    let v = value.clamp(0.0, 1.0);
    if v > 0.0005 {
        p.fill(Rect::from_min_size(r.min, vec2((r.width() * v).max(h), h)), Corner::Capsule, color);
    }
}

/// The sidebar list: rust-glass's nav_list look (sliding capsule), but it reports every
/// click, including one on the item that is already highlighted.
fn nav(ui: &mut Ui, selected: usize, items: &[(Icon, &str)]) -> Option<usize> {
    let w = ui.available_width();
    let h = 38.0;
    let id = ui.make_id("dojo_nav");
    let old = ui.spacing;
    ui.spacing = 2.0;
    let top = ui.next_pos().y;
    let y = ui.ctx.animate(id, selected as f32, Tuning::new(0.34, 0.78));
    let hl = Rect::from_min_size(vec2(ui.cursor.x, top + y * (h + 2.0)), vec2(w, h));
    let accent = ui.ctx.theme.accent.lerp(Color::WHITE, 0.35 * ui.surface.darkness());
    let strong = ui.surface.strong_fill_color();
    ui.painter().fill(hl, Corner::Capsule, strong);
    let mut clicked = None;
    for (i, (icon, label)) in items.iter().enumerate() {
        let r = ui.allocate_response(vec2(w, h), Sense::Click);
        ui.ctx.focusable(r.id);
        if r.clicked {
            clicked = Some(i);
        }
        if let Some(n) = ui.access(r.id, rust_glass::accessibility::Role::Tab, r.rect) {
            n.label(*label).selected(selected == i);
        }
        if r.hovered {
            ui.ctx.cursor = rust_glass::context::Cursor::Pointer;
        }
        let hov = ui.ctx.animate_bool(r.id.with("h"), r.hovered && i != selected, Tuning::SNAP);
        let on = ui.ctx.animate_bool(r.id.with("on"), i == selected, Tuning::SNAP);
        let st = TextStyle::new(ui.ctx.theme.body_size).weight(if i == selected { Weight::Semibold } else { Weight::Medium });
        let l = ui.layout_text(label, &st);
        let (f, t) = (ui.surface.fill_color(), ui.surface.text_color());
        let mut p = ui.painter();
        p.fill(r.rect, Corner::Capsule, f.fade(hov * 0.7));
        p.icon(vec2(r.rect.min.x + 22.0, r.rect.center().y), 18.0, *icon, t.lerp(accent, on));
        p.text(vec2(r.rect.min.x + 42.0, r.rect.center().y - l.size.y * 0.5), &l, t);
        ui.focus_ring(r.id, r.rect, Corner::Capsule);
    }
    ui.spacing = old;
    clicked
}

fn pill(ui: &mut Ui, text: &str, color: Color) {
    let l = ui.layout_text(text, &TextStyle::new(11.5).weight(Weight::Semibold));
    let r = ui.allocate(l.size + vec2(16.0, 6.0));
    let mut p = ui.painter();
    p.fill(r, Corner::Capsule, color.alpha(0.18));
    p.text_centered(r, &l, color, 1.0);
}

fn note(ui: &mut Ui, w: f32, icon: Icon, text: &str, color: Color) {
    let st = TextStyle::new(12.5).wrap(w - 52.0);
    let l = ui.layout_text(text, &st);
    let r = ui.allocate(vec2(w, l.size.y + 16.0));
    let mut p = ui.painter();
    p.fill(r, Corner::Radius(12.0), color.alpha(0.12));
    p.stroke(r, Corner::Radius(12.0), 1.0, color.alpha(0.35));
    p.icon(vec2(r.min.x + 18.0, r.min.y + 8.0 + 9.0), 16.0, icon, color);
    p.text(vec2(r.min.x + 36.0, r.min.y + 8.0), &l, TEXT);
}

/// The card's memory as 1 GB cells, filled left to right by `segments` (GB, colour).
fn memory_strip(ui: &mut Ui, w: f32, total: f32, segments: &[(f32, Color)]) {
    let r = ui.allocate(vec2(w, 34.0));
    let mut p = ui.painter();
    p.fill(r, Corner::Radius(8.0), Color::white(0.07));
    let mut x = r.min.x;
    for (gb, c) in segments {
        let sw = (r.width() * gb / total).max(0.0).min(r.max.x - x);
        if sw > 0.0 {
            p.fill(Rect { min: vec2(x, r.min.y), max: vec2(x + sw, r.max.y) }, Corner::Radius(3.0), *c);
            x += sw;
        }
    }
    let cells = total.round().max(1.0) as usize;
    for i in 1..cells {
        let cx = r.min.x + r.width() * i as f32 / cells as f32;
        p.line(vec2(cx, r.min.y), vec2(cx, r.max.y), 2.0, Color::black(0.45));
    }
    let scale: Vec<_> = (0..=6).map(|i| (i, ui_layout(i * 4))).collect();
    drop(p);
    let row = ui.allocate(vec2(w, 14.0));
    let labels: Vec<_> = scale.iter().map(|(i, s)| (*i, ui.layout_text(s, &TextStyle::new(10.0)))).collect();
    let mut p = ui.painter();
    for (i, l) in labels {
        let fx = row.min.x + row.width() * (i as f32 * 4.0 / total).min(1.0) - if i == 6 { l.size.x } else if i == 0 { 0.0 } else { l.size.x / 2.0 };
        p.text(vec2(fx, row.min.y), &l, FAINT);
    }
}

fn ui_layout(gb: i32) -> String {
    if gb == 24 { "24 GB".into() } else { gb.to_string() }
}

fn legend(ui: &mut Ui, w: f32, items: &[(Color, String, f32)]) {
    let _ = w;
    for chunk in items.chunks(2) {
        ui.horizontal(|ui| {
            for (c, label, gb) in chunk {
                let d = ui.allocate(vec2(10.0, 10.0));
                ui.painter().fill(d, Corner::Radius(2.0), *c);
                ui.styled_text(&format!("{label}  {gb:.1} GB"), TextStyle::new(12.0), MUTED);
                ui.space(14.0);
            }
        });
    }
}

fn why_rows(ui: &mut Ui, w: f32, rows: &[recipes::Why]) {
    for r in rows {
        ui.card(|ui| {
            ui.horizontal(|ui| {
                ui.styled_text(&r.setting, TextStyle::new(13.0).weight(Weight::Semibold), TEXT);
                ui.styled_text(&r.value, TextStyle::new(13.0), ACCENT);
                ui.horizontal_right(|ui| pill(ui, if r.card { "this card" } else { "engine default" }, if r.card { INFO } else { MUTED }));
            });
            ui.styled_text(&r.why, TextStyle::new(12.0).wrap(w - 36.0), MUTED);
        });
    }
}

fn fit_grid(ui: &mut Ui, w: f32, m: &recipes::FitModel) {
    let first = (w * 0.34).min(220.0);
    let cw = (w - first) / 3.0;
    let head = ["Resolution", "Batch 1", "Batch 2", "Batch 4"];
    ui.horizontal(|ui| {
        ui.spacing = 0.0;
        for (i, h) in head.iter().enumerate() {
            ui.vertical(if i == 0 { first } else { cw }, |ui| {
                ui.styled_text(h, TextStyle::new(11.5).weight(Weight::Semibold), FAINT);
            });
        }
    });
    for row in &m.rows {
        ui.horizontal(|ui| {
            ui.spacing = 0.0;
            ui.vertical(first, |ui| {
                ui.styled_text(row.detail, TextStyle::new(12.5).weight(Weight::Semibold), TEXT);
                ui.styled_text(&format!("{} px", row.resolutions.iter().map(u32::to_string).collect::<Vec<_>>().join(" + ")), TextStyle::new(11.0), MUTED);
            });
            for c in &row.cells {
                ui.vertical(cw, |ui| {
                    let (text, color) = if !c.fits { ("does not fit".to_string(), FAINT) } else if c.offload > 0.3 { (format!("{:.1} GB", c.need_gb), WARN) } else { (format!("{:.1} GB", c.need_gb), GOOD) };
                    ui.styled_text(&text, TextStyle::new(13.0).weight(Weight::Semibold), color);
                    if c.fits && c.offload > 0.0 {
                        ui.styled_text(&format!("{:.0}% in RAM", c.offload * 100.0), TextStyle::new(11.0), MUTED);
                    }
                });
            }
        });
    }
    ui.styled_text("Green fits on the card. Amber fits only by streaming more than 30% of the model from system RAM, which is slow.", TextStyle::new(11.5).wrap(w), FAINT);
}

fn smooth(points: &[(u32, f32)], k: f32) -> Vec<(f32, f32)> {
    let mut a: Option<f32> = None;
    points
        .iter()
        .map(|(x, y)| {
            let v = a.map_or(*y, |prev| prev + (y - prev) * k);
            a = Some(v);
            (*x as f32, v)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn chart(ui: &mut Ui, w: f32, h: f32, title: &str, points: &[(f32, f32)], color: Color, y0: f32, y1: f32) {
    ui.vertical(w, |ui| {
        ui.styled_text(title, TextStyle::new(12.0), MUTED);
        let r = ui.allocate(vec2(w, h));
        let empty = ui.layout_text("No data yet", &TextStyle::new(12.0));
        let mut p = ui.painter();
        p.fill(r, Corner::Radius(12.0), Color::black(0.28));
        if points.len() < 2 {
            p.text_centered(r, &empty, FAINT, 1.0);
            return;
        }
        let step = (points.len() / 300).max(1);
        let pts: Vec<(f32, f32)> = points.iter().step_by(step).copied().collect();
        let (x0, x1) = (pts[0].0, pts[pts.len() - 1].0);
        let span = if (y1 - y0).abs() < 1e-6 { 1.0 } else { y1 - y0 };
        let map = |pt: (f32, f32)| vec2(r.min.x + 8.0 + (pt.0 - x0) / (x1 - x0).max(1e-6) * (r.width() - 16.0), r.max.y - 8.0 - (pt.1 - y0) / span * (r.height() - 16.0));
        for wpair in pts.windows(2) {
            p.line(map(wpair[0]), map(wpair[1]), 2.0, color);
        }
    });
}
