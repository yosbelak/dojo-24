//! Reads a .safetensors header (no tensor data) and works out what the file is:
//! which model family, how it is stored, and whether the engine can open it as a base.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

const MAX_HEADER: u64 = 256 << 20;

#[derive(Clone, Debug, Default)]
pub struct Info {
    pub family: &'static str,
    pub precision: String,
    /// "header" (quant config in metadata), "markers" (`.comfy_quant` tensors) or "".
    pub quant_layout: &'static str,
    /// Keys start with `model.diffusion_model.` / `diffusion_model.`.
    pub prefixed: bool,
    pub size_gb: f32,
    pub error: Option<String>,
}

fn read_header(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|_| "not a safetensors file".to_string())?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > MAX_HEADER {
        return Err("implausible safetensors header".into());
    }
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    match serde_json::from_slice(&buf).map_err(|e| e.to_string())? {
        serde_json::Value::Object(m) => Ok(m),
        _ => Err("header is not an object".into()),
    }
}

fn strip(key: &str) -> &str {
    key.strip_prefix("model.diffusion_model.").or_else(|| key.strip_prefix("diffusion_model.")).unwrap_or(key)
}

pub fn sniff(path: &Path) -> Info {
    let size_gb = std::fs::metadata(path).map(|m| m.len() as f32 / (1u64 << 30) as f32).unwrap_or(0.0);
    let mut info = Info { family: "unknown", precision: "unknown".into(), size_gb, ..Default::default() };
    let mut header = match read_header(path) {
        Ok(h) => h,
        Err(e) => {
            info.error = Some(e);
            return info;
        }
    };
    let meta = header.remove("__metadata__");
    info.prefixed = header.keys().any(|k| k.starts_with("model.diffusion_model.") || k.starts_with("diffusion_model."));
    let keys: Vec<&str> = header.keys().map(|k| strip(k)).collect();

    // Krea 2: SingleStreamDiT with blocks.N.* and a text fusion stack.
    // MOSS-Image: layers.N.* with fused qkv projections, 3072 wide.
    let has_blocks = keys.iter().any(|k| k.starts_with("blocks."));
    let has_layers = keys.iter().any(|k| k.starts_with("layers."));
    let has_txtfusion = keys.iter().any(|k| k.starts_with("txtfusion."));
    let has_qkv = keys.iter().any(|k| k.contains(".qkv."));
    let lora = header.keys().any(|k| k.contains(".lora_A.") || k.contains(".lora_down.") || k.starts_with("lora_"));
    info.family = if has_blocks && has_txtfusion {
        "krea2"
    } else if has_layers && has_qkv && !has_blocks {
        "moss_image"
    } else if lora {
        "lora"
    } else {
        "unknown"
    };

    // How it is stored: the dtype holding the most bytes.
    let mut bytes: HashMap<String, u64> = HashMap::new();
    for (k, v) in &header {
        if k.ends_with(".comfy_quant") {
            info.quant_layout = "markers";
        }
        if let (Some(dt), Some(off)) = (v.get("dtype").and_then(|d| d.as_str()), v.get("data_offsets").and_then(|o| o.as_array())) {
            let a = off.first().and_then(|x| x.as_u64()).unwrap_or(0);
            let b = off.get(1).and_then(|x| x.as_u64()).unwrap_or(0);
            *bytes.entry(dt.to_string()).or_default() += b.saturating_sub(a);
        }
    }
    if meta.as_ref().and_then(|m| m.get("_quantization_metadata")).is_some() {
        info.quant_layout = "header";
    }
    let dominant = bytes.iter().max_by_key(|(_, b)| **b).map(|(d, _)| d.as_str()).unwrap_or("?");
    info.precision = match dominant {
        "I8" | "U8" => "int8",
        "F8_E4M3" => "fp8",
        "BF16" => "bf16",
        "F16" => "fp16",
        "F32" => "fp32",
        other => return Info { precision: other.to_lowercase(), ..info },
    }
    .into();
    info
}

/// Why the engine cannot train on this checkpoint, or None when it can.
pub fn blocked_reason(info: &Info) -> Option<String> {
    if let Some(e) = &info.error {
        return Some(format!("unreadable: {e}"));
    }
    if info.prefixed {
        return Some("its keys carry a ComfyUI checkpoint prefix".into());
    }
    if info.family == "krea2" && info.quant_layout == "header" {
        return Some("it is quantized in a layout the Krea 2 loader does not read".into());
    }
    if !matches!(info.precision.as_str(), "int8" | "bf16" | "fp16" | "fp32") {
        return Some(format!("{} weights", info.precision));
    }
    None
}
