//! Thumbnails decoded on a worker thread and uploaded to the GPU on the UI thread.

use rust_glass::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

enum Slot {
    Pending,
    Ready(ImageId, (u32, u32)),
    Failed,
}

pub struct Thumbs {
    slots: HashMap<PathBuf, Slot>,
    req: Sender<(PathBuf, u32)>,
    done: Receiver<(PathBuf, Option<(u32, u32, Vec<u8>)>)>,
}

impl Thumbs {
    pub fn new() -> Self {
        let (req, rx) = channel::<(PathBuf, u32)>();
        let (tx, done) = channel();
        std::thread::spawn(move || {
            while let Ok((p, max)) = rx.recv() {
                let img = image::open(&p).ok().map(|i| {
                    let t = i.thumbnail(max, max).to_rgba8();
                    (t.width(), t.height(), t.into_raw())
                });
                if tx.send((p, img)).is_err() {
                    break;
                }
            }
        });
        Self { slots: HashMap::new(), req, done }
    }

    /// Uploads whatever the worker finished. Call once per frame.
    pub fn poll(&mut self, ctx: &mut Context) -> bool {
        let mut any = false;
        while let Ok((p, img)) = self.done.try_recv() {
            any = true;
            let slot = match img {
                Some((w, h, px)) => Slot::Ready(ctx.add_image(w, h, px), (w, h)),
                None => Slot::Failed,
            };
            self.slots.insert(p, slot);
        }
        any || self.slots.values().any(|s| matches!(s, Slot::Pending))
    }

    /// The image and its pixel size, starting a decode the first time a path is asked for.
    pub fn get(&mut self, path: &Path, max: u32) -> Option<(ImageId, (u32, u32))> {
        match self.slots.get(path) {
            Some(Slot::Ready(id, s)) => Some((*id, *s)),
            Some(_) => None,
            None => {
                self.slots.insert(path.to_path_buf(), Slot::Pending);
                let _ = self.req.send((path.to_path_buf(), max));
                None
            }
        }
    }

    /// Frees every image whose path is not in `keep`.
    pub fn retain(&mut self, ctx: &mut Context, keep: &[PathBuf]) {
        let drop: Vec<PathBuf> = self.slots.keys().filter(|p| !keep.contains(p)).cloned().collect();
        for p in drop {
            if let Some(Slot::Ready(id, _)) = self.slots.remove(&p) {
                ctx.remove_image(id);
            }
        }
    }
}

/// UV rect that crops an image of `size` to a square, centred.
pub fn square_uv(size: (u32, u32)) -> Rect {
    let (w, h) = (size.0 as f32, size.1 as f32);
    if w > h {
        let f = h / w;
        rect((1.0 - f) / 2.0, 0.0, f, 1.0)
    } else {
        let f = w / h.max(1.0);
        rect(0.0, (1.0 - f) / 2.0, 1.0, f)
    }
}
