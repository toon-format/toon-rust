/* src/lib/lib.rs */
//!▫~•◦-------------------------------‣
//! # Cross-format conversion library for RUNE data processing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-cros to achieve seamless conversion between RUNE, JSON, and other data formats.
//!
//! ### Key Capabilities
//! - **Multi-format Support**: Provides conversion utilities for Arrow, MessagePack, NumPy, Parquet, TOML, and YAML formats.
//! - **Modular Architecture**: Organizes functionality into separate modules for each format type.
//! - **Feature-gated Dependencies**: Uses conditional compilation to manage optional dependencies and reduce binary size.
//!
//! ### Architectural Notes
//! This module serves as the main entry point for the rune-cros crate, re-exporting functionality from all format-specific modules.
//! It integrates with the `yoshi` error handling system and provides a unified API for cross-format conversions.
//!
//! ### Example
//! ```rust
//! use crate::rune_cros::{toml_to_json_string, json_to_yaml_string};
//!
//! let toml_data = r#"key = "value""#;
//! let json_result = toml_to_json_string(toml_data).unwrap();
//! let yaml_result = json_to_yaml_string(&json_result).unwrap();
//!
//! // The results can now be used for further processing.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

//! Cross-format helpers for RUNE <-> JSON and other formats
//!
//! The crate provides small wrappers to convert between RUNE/JSON/and other
//! formats (YAML, TOML, MessagePack, Arrow, Parquet, NumPy). Big libraries are feature-gated.

pub mod arow;
pub mod mpak;
pub mod npy;
pub mod prqt;
pub mod toml;
pub mod json;
pub mod yaml;

pub use arow::*;
pub use mpak::*;
pub use npy::*;
pub use prqt::*;
pub use toml::*;
pub use json::*;
pub use yaml::*;

pub use yoshi::{Result, buck};
