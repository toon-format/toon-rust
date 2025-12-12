//! JSON helpers — RUNE is the priority, JSON is the bridge.
//!
//! These helpers keep JSON as the interchange format while pushing to / from
//! canonical RUNE strings using `rune-format`'s encode/decode defaults.

use serde_json::Value;
use yoshi::{Result, yoshi};

/// Convert a JSON string into a RUNE string (encoded with default options).
pub fn json_to_rune_string(json: &str) -> Result<String> {
    let value: Value = serde_json::from_str(json)
        .map_err(|e| yoshi!("serde_json parse error: {}", e))?;
    rune_format::encode_object(value, &rune_format::types::EncodeOptions::default())
        .map_err(|e| yoshi!("rune-format encode error: {}", e))
}

/// Convert a RUNE string into a pretty-printed JSON string.
pub fn rune_string_to_json(rune: &str) -> Result<String> {
    let value: Value = rune_format::decode_default(rune)
        .map_err(|e| yoshi!("rune-format decode error: {}", e))?;
    serde_json::to_string_pretty(&value)
        .map_err(|e| yoshi!("serde_json stringify error: {}", e))
}

/// Parse JSON string into serde_json::Value (utility).
pub fn json_to_value(json: &str) -> Result<Value> {
    serde_json::from_str(json).map_err(|e| yoshi!("serde_json parse error: {}", e))
}

/// Encode serde_json::Value into RUNE string (utility).
pub fn value_to_rune(value: Value) -> Result<String> {
    rune_format::encode_object(value, &rune_format::types::EncodeOptions::default())
        .map_err(|e| yoshi!("rune-format encode error: {}", e))
}
