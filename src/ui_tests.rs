//! Drives the real window off-screen with rust-glass's `Headless` runner (GPU, no window).
//! Widgets are found by their accessibility label.
//!
//!   $env:DOJO_TEST_DATASET = "<folder with a few images>"
//!   cargo test --release ui_ -- --ignored --test-threads 1 --nocapture
//! Screenshots land in `$DOJO_SNAP_DIR` (default: temp dir). Jobs are written to a throwaway
//! folder and the queue never starts, so nothing trains.

use crate::app::{DojoApp, Page};
use crate::settings::{self, Paths};
use rust_glass::accessibility::Role;
use rust_glass::headless::Headless;
use rust_glass::input::Event;
use rust_glass::prelude::*;
use std::path::PathBuf;

struct Driver {
    h: Headless,
    app: DojoApp,
    _tmp: PathBuf,
}

impl Driver {
    fn new(w: u32, hgt: u32) -> Self {
        let real = Paths::new(settings::root());
        let tmp = std::env::temp_dir().join(format!("dojo-ui-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let mut paths = Paths::new(tmp.clone());
        paths.engine_dir = real.engine_dir.clone();
        paths.engine_py = real.engine_py.clone();
        let cfg = settings::load(&real);
        let mut h = Headless::new(w, hgt, 1.0).expect("headless GPU");
        let mut app = DojoApp::new(paths, cfg, false);
        app.setup(&mut h.ctx);
        h.ctx.set_accessibility(true);
        app.wait_for_catalog();
        let mut d = Self { h, app, _tmp: tmp };
        d.settle();
        d
    }

    fn settle(&mut self) {
        self.h.settle(&mut self.app, 3.0);
        self.h.step(&mut self.app, 1.0 / 60.0, &[]);
    }

    fn step(&mut self, events: &[Event]) {
        self.h.step(&mut self.app, 1.0 / 60.0, events);
    }

    fn find(&self, label: &str) -> Vec2 {
        let nodes: Vec<_> = self.h.ctx.access_nodes().iter().filter(|n| !matches!(n.role, Role::Label | Role::Heading | Role::Group | Role::ScrollView | Role::List)).collect();
        let exact: Vec<_> = nodes.iter().filter(|n| n.label == label).collect();
        let hits = if exact.is_empty() { nodes.iter().filter(|n| n.label.starts_with(label)).collect() } else { exact };
        match hits.as_slice() {
            [one] => one.rect.center(),
            [] => panic!("no widget labelled {label:?}; have: {:?}", nodes.iter().map(|n| &n.label).collect::<Vec<_>>()),
            many => panic!("{} widgets labelled {label:?}", many.len()),
        }
    }

    fn click(&mut self, label: &str) {
        let p = self.find(label);
        self.step(&[Event::PointerMoved(p)]);
        self.step(&[Event::PointerDown]);
        self.step(&[Event::PointerUp]);
        self.settle();
    }

    /// Steps until `done` holds (the dataset scan and thumbnails run on threads).
    fn until(&mut self, what: &str, mut done: impl FnMut(&DojoApp) -> bool) {
        let t = std::time::Instant::now();
        while !done(&self.app) {
            assert!(t.elapsed().as_secs() < 30, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(30));
            self.step(&[]);
        }
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(15));
            self.step(&[]);
        }
        self.settle();
    }

    fn snap(&mut self, name: &str) {
        let dir = std::env::var_os("DOJO_SNAP_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ppp = self.h.ctx.pixels_per_point;
        let (w, h) = ((self.h.ctx.screen.width() * ppp).round() as u32, (self.h.ctx.screen.height() * ppp).round() as u32);
        image::RgbaImage::from_raw(w, h, self.h.pixels()).expect("pixel buffer").save(dir.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
#[ignore]
fn ui_plan_every_model_then_queue_a_job() {
    let ds = std::env::var("DOJO_TEST_DATASET").expect("set DOJO_TEST_DATASET");
    let mut d = Driver::new(1440, 920);
    d.snap("0_empty");
    d.app.dataset = ds;
    d.app.trigger = "zxshape".into();
    for fam in 0..3 {
        d.app.family = fam;
        d.until("a plan", |a| a.plan.as_ref().is_some_and(|p| p.is_ok()));
        let p = d.app.plan.clone().unwrap().unwrap();
        println!("{}: {} {:?} need {} GB offload {}", p.family, p.precision, p.resolutions, p.budget.need_gb, p.budget.offload);
        d.snap(&format!("1_new_{}", p.family));
    }
    // blank trigger with uncaptioned images is allowed
    d.app.family = 0;
    d.app.trigger.clear();
    d.until("a plan without trigger", |a| a.plan.as_ref().is_some_and(|p| p.as_ref().is_ok_and(|p| p.family == "krea2" && p.trigger.is_empty())));

    d.click("Create job");
    assert!(matches!(d.app.page, Page::Job(_)), "Create job should open the job page");
    d.snap("2_job");
    d.click("Queue");
    assert_eq!(d.app.page, Page::Queue);
    assert_eq!(d.app.views.len(), 1);
    assert_eq!(d.app.views[0].status, "queued");
    d.snap("3_queue");
    d.click("GPU fit");
    assert_eq!(d.app.page, Page::Fit);
    d.snap("4_fit");
    d.click("Queue");
    d.click("Stop dataset_shapes");
    assert_eq!(d.app.views[0].status, "stopped");
    d.click("Delete dataset_shapes");
    d.snap("5_confirm_delete");
    d.click("Delete files");
    assert!(d.app.views.is_empty(), "job should be gone");
}

/// Opens the jobs already in this Dojo 24 folder (read-only: the queue never runs here).
#[test]
#[ignore]
fn ui_existing_jobs() {
    if std::env::var_os("DOJO_REAL_JOBS").is_none() {
        return;
    }
    let real = Paths::new(settings::root());
    let cfg = settings::load(&real);
    let mut h = Headless::new(1440, 920, 1.0).expect("headless GPU");
    let mut app = DojoApp::new(real, cfg, false);
    app.setup(&mut h.ctx);
    h.ctx.set_accessibility(true);
    let mut d = Driver { h, app, _tmp: PathBuf::new() };
    d.settle();
    let done = d.app.views.iter().find(|v| v.status == "done").map(|v| v.id.clone()).expect("a finished job");
    d.app.page = Page::Queue;
    d.settle();
    d.snap("6_real_queue");
    d.app.page = Page::Job(done);
    d.until("samples", |_| false || true);
    for _ in 0..120 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        d.step(&[]);
    }
    d.settle();
    d.snap("7_real_job");
}
