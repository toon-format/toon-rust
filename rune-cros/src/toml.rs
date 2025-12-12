/* src/toml/toml.rs */
//!▫~•◦-------------------------------‣
//! # TOML format conversion module for configuration data processing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-cros to achieve efficient conversion between TOML and JSON/RUNE formats.
//!
//! ### Key Capabilities
//! - **TOML to JSON Conversion**: Converts TOML configuration data to JSON string representation.
//! - **JSON to TOML Conversion**: Converts JSON strings to TOML format.
//! - **TOML to RUNE Conversion**: Converts TOML data directly to RUNE format.
//! - **RUNE to TOML Conversion**: Converts RUNE format data to TOML configuration format.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `lib` and other format converters.
//! Functions adhere to the `Result<String>` return type and integrate with the `yoshi` error handling system.
//!
//! ### Example
//! ```rust
//! use crate::rune_cros::{toml_to_json_string, json_to_toml_string};
//!
//! let toml_data = r#"key = "value""#;
//! let json_result = toml_to_json_string(toml_data).unwrap();
//! let toml_result = json_to_toml_string(&json_result).unwrap();
//!
//! // The results can now be used for further processing.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![allow(unreachable_code)]

use yoshi::{Result, buck};

#[cfg(feature = "toml")]
pub fn toml_to_json_string(toml_str: &str) -> Result<String> {
    match toml::from_str::<toml::Value>(toml_str) {
        Ok(v) => match serde_json::to_value(v) {
            Ok(json_val) => match serde_json::to_string_pretty(&json_val) {
                Ok(s) => Ok(s),
                Err(e) => Err(buck!("serde_json stringify error: {}", e)),
            },
            Err(e) => Err(buck!("serde_json conversion error: {}", e)),
        },
        Err(e) => Err(buck!("toml parse error: {}", e)),
    }
}

#[cfg(not(feature = "toml"))]
pub fn toml_to_json_string(_toml: &str) -> Result<String> { buck!("toml feature not enabled") }

#[cfg(feature = "toml")]
pub fn json_to_toml_string(json: &str) -> Result<String> {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(j) => match toml::to_string_pretty(&j) {
            Ok(t) => Ok(t),
            Err(e) => Err(buck!("toml serialize error: {}", e)),
        },
        Err(e) => Err(buck!("serde_json parse error: {}", e)),
    }
}

#[cfg(not(feature = "toml"))]
pub fn json_to_toml_string(_json: &str) -> Result<String> { buck!("toml feature not enabled") }

#[cfg(feature = "toml")]
pub fn toml_to_rune_string(toml: &str) -> Result<String> {
    match toml::from_str::<toml::Value>(toml) {
        Ok(v) => match serde_json::to_value(v) {
            Ok(json_val) => match rune_format::encode_object(json_val, &rune_format::types::EncodeOptions::default()) {
                Ok(rune) => Ok(rune),
                Err(e) => Err(buck!("rune-format encode error: {}", e)),
            },
            Err(e) => Err(buck!("serde_json conversion error: {}", e)),
        },
        Err(e) => Err(buck!("toml parse error: {}", e)),
    }
}

#[cfg(feature = "toml")]
pub fn rune_string_to_toml(rune: &str) -> Result<String> {
    match rune_format::decode_default::<serde_json::Value>(rune) {
        Ok(v) => match toml::to_string_pretty(&v) {
            Ok(t) => Ok(t),
            Err(e) => Err(buck!("toml serialize error: {}", e)),
        },
        Err(e) => Err(buck!("rune-format decode error: {}", e)),
    }
}
