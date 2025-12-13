/* src/yaml/yaml.rs */
//!▫~•◦-------------------------------‣
//! # YAML format conversion module for human-readable data serialization.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-cros to achieve efficient conversion between YAML and JSON/RUNE formats.
//!
//! ### Key Capabilities
//! - **YAML to JSON Conversion**: Converts YAML configuration data to JSON string representation.
//! - **JSON to YAML Conversion**: Converts JSON strings to YAML format.
//! - **YAML to RUNE Conversion**: Converts YAML data directly to RUNE format.
//! - **RUNE to YAML Conversion**: Converts RUNE format data to YAML configuration format.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `lib` and other format converters.
//! Functions adhere to the `Result<String>` return type and integrate with the `yoshi` error handling system.
//!
//! ### Example
//! ```rust
//! use crate::rune_cros::{yaml_to_json_string, json_string_to_yaml};
//!
//! let yaml_data = r#"key: value"#;
//! let json_result = yaml_to_json_string(yaml_data).unwrap();
//! let yaml_result = json_string_to_yaml(&json_result).unwrap();
//!
//! // The results can now be used for further processing.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![allow(unreachable_code)]

use yoshi::{Result, buck};

#[cfg(feature = "yaml")]
pub fn yaml_to_json_string(yaml: &str) -> Result<String> {
    match serde_yaml_ng::from_str::<serde_json::Value>(yaml) {
        Ok(json_val) => match serde_json::to_string_pretty(&json_val) {
            Ok(s) => Ok(s),
            Err(e) => Err(buck!("serde_json stringify error: {}", e)),
        },
        Err(e) => Err(buck!("yaml parse error: {}", e)),
    }
}

#[cfg(not(feature = "yaml"))]
pub fn yaml_to_json_string(_yaml: &str) -> Result<String> { buck!("yaml feature not enabled") }

#[cfg(feature = "yaml")]
pub fn json_string_to_yaml(json: &str) -> Result<String> {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(j) => match serde_yaml_ng::to_string(&j) {
            Ok(s) => Ok(s),
            Err(e) => Err(buck!("yaml serialize error: {}", e)),
        },
        Err(e) => Err(buck!("serde_json parse error: {}", e)),
    }
}

#[cfg(not(feature = "yaml"))]
pub fn json_string_to_yaml(_json: &str) -> Result<String> { buck!("yaml feature not enabled") }

#[cfg(feature = "yaml")]
pub fn yaml_to_rune_string(yaml: &str) -> Result<String> {
    match serde_yaml_ng::from_str::<serde_json::Value>(yaml) {
        Ok(json_val) => match rune_format::encode_object(json_val, &rune_format::types::EncodeOptions::default()) {
            Ok(rune) => Ok(rune),
            Err(e) => Err(buck!("rune-format encode error: {}", e)),
        },
        Err(e) => Err(buck!("yaml parse error: {}", e)),
    }
}

#[cfg(feature = "yaml")]
pub fn rune_string_to_yaml(rune: &str) -> Result<String> {
    match rune_format::decode_default::<serde_json::Value>(rune) {
        Ok(v) => match serde_yaml_ng::to_string(&v) {
            Ok(y) => Ok(y),
            Err(e) => Err(buck!("yaml serialize error: {}", e)),
        },
        Err(e) => Err(buck!("rune-format decode error: {}", e)),
    }
}
