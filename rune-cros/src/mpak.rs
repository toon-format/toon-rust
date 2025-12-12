/* src/mpak/mpak.rs */
//!▫~•◦-------------------------------‣
//! # MessagePack format conversion module for cross-format data processing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-cros to achieve efficient conversion between MessagePack and JSON/RUNE formats.
//!
//! ### Key Capabilities
//! - **MessagePack to JSON Conversion**: Converts MessagePack binary data to JSON string representation.
//! - **JSON to MessagePack Conversion**: Converts JSON strings to MessagePack binary format.
//! - **MessagePack to RUNE Conversion**: Converts MessagePack data directly to RUNE format.
//! - **RUNE to MessagePack Conversion**: Converts RUNE format data to MessagePack binary.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `lib` and other format converters.
//! Functions adhere to the `Result<T>` return type and integrate with the `yoshi` error handling system.
//!
//! ### Example
//! ```rust
//! use crate::rune_cros::{mpak_to_json_string, json_string_to_mpak};
//!
//! let mpak_data: &[u8] = &[/* MessagePack binary data */];
//! let json_result = mpak_to_json_string(mpak_data).unwrap();
//! let mpak_result = json_string_to_mpak(&json_result).unwrap();
//!
//! // The results can now be used for further processing.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "mpak")]
use yoshi::{Result, yoshi};
#[cfg(not(feature = "mpak"))]
use yoshi::{Result, buck};

#[cfg(feature = "mpak")]
/// Convert MessagePack bytes to JSON string
pub fn mpak_to_json_string(bytes: &[u8]) -> Result<String> {
    let json_val: serde_json::Value = rmp_serde::from_slice(bytes)
        .map_err(|e| yoshi!("rmp-serde decode error: {}", e))?;
    let s = serde_json::to_string_pretty(&json_val)
        .map_err(|e| yoshi!("serde_json stringify error: {}", e))?;
    Ok(s)
}

#[cfg(not(feature = "mpak"))]
pub fn mpak_to_json_string(_bytes: &[u8]) -> Result<String> { buck!("mpak feature not enabled") }

#[cfg(feature = "mpak")]
/// Convert JSON string into MessagePack bytes
pub fn json_string_to_mpak(json: &str) -> Result<Vec<u8>> {
    let j: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| yoshi!("serde_json parse error: {}", e))?;
    let mut buf = Vec::new();
    rmp_serde::encode::write(&mut buf, &j)
        .map_err(|e| yoshi!("rmp-serde encode error: {}", e))?;
    Ok(buf)
}

#[cfg(not(feature = "mpak"))]
pub fn json_string_to_mpak(_json: &str) -> Result<Vec<u8>> { buck!("mpak feature not enabled") }

#[cfg(feature = "mpak")]
/// Convert MessagePack bytes to Rune string
pub fn mpak_to_rune_string(bytes: &[u8]) -> Result<String> {
    let json_val: serde_json::Value = rmp_serde::from_slice(bytes)
        .map_err(|e| yoshi!("rmp-serde decode error: {}", e))?;
    let rune = rune_format::encode_object(json_val, &rune_format::types::EncodeOptions::default())
        .map_err(|e| yoshi!("rune-format encode error: {}", e))?;
    Ok(rune)
}

#[cfg(feature = "mpak")]
/// Convert Rune string into MessagePack bytes
pub fn rune_string_to_mpak(rune: &str) -> Result<Vec<u8>> {
    let v: serde_json::Value = rune_format::decode_default(rune)
        .map_err(|e| yoshi!("rune-format decode error: {}", e))?;
    let json = serde_json::to_string(&v)
        .map_err(|e| yoshi!("serde_json stringify error: {}", e))?;
    let mut buf = Vec::new();
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| yoshi!("serde_json parse error: {}", e))?;
    rmp_serde::encode::write(&mut buf, &value)
        .map_err(|e| yoshi!("rmp-serde encode error: {}", e))?;
    Ok(buf)
}
