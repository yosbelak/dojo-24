//! Dojo 24: LoRA training for Krea 2, MOSS-Image and MiniMax-H3 Image, planned for a 24 GB GPU.
//! Training runs in ostris' ai-toolkit (engine\); this window plans, queues and follows the jobs.
//! The interface is drawn with rust-glass (Liquid Glass style, wgpu).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod hardware;
mod jobs;
mod recipes;
mod settings;
mod sniff;
mod thumbs;
#[cfg(test)]
mod ui_tests;
#[cfg(test)]
mod readme_shots;

fn main() -> Result<(), String> {
    let paths = settings::Paths::new(settings::root());
    let cfg = settings::load(&paths);
    let mut app = app::DojoApp::new(paths, cfg, true);
    if let Some(folder) = std::env::args_os().nth(1) {
        app.dataset = std::path::PathBuf::from(folder).display().to_string();
    }
    let options = rust_glass::Options { title: "Dojo 24".into(), size: (1440.0, 920.0), min_size: (1100.0, 680.0), ..Default::default() };
    rust_glass::run(options, app)
}
