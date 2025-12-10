use crate::slot::SGLRuneSlot;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Save to disk atomically: write tmp then rename.
pub fn write_rune_atomic<P: AsRef<Path>>(path: P, slots: &[SGLRuneSlot]) -> Result<()> {
    let p = path.as_ref();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Failed to create dir {}", parent.display()))?;
    }

    let tmp = p.with_extension("tmp");
    let json = serde_json::to_string_pretty(&slots).context("Serialize slots to JSON")?;
    fs::write(&tmp, json.as_bytes()).with_context(|| format!("Write tmp file {}", tmp.display()))?;
    fs::rename(&tmp, p).with_context(|| format!("Rename tmp to {}", p.display()))?;
    Ok(())
}

/// Read file gracefully; returns empty vec when missing or empty
pub fn read_rune<P: AsRef<Path>>(path: P) -> Result<Vec<SGLRuneSlot>> {
    let p = path.as_ref();
    if !p.exists() { return Ok(vec![]); }
    let raw = fs::read_to_string(p).with_context(|| format!("Read file {}", p.display()))?;
    if raw.trim().is_empty() { return Ok(vec![]); }
    let slots: Vec<SGLRuneSlot> = serde_json::from_str(&raw).context("Deserialize slots JSON")?;
    Ok(slots)
}
