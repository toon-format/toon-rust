/* rune-xero/src/types/mod.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Types Module
//!▫~•◦-----------------------------‣
//!
//! Central hub for RUNE type definitions, options, and errors.
//! Re-exports core primitives used across the Encoder and Decoder.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Fixed typo: delimeter -> delimiter
pub mod delimiter;
pub mod errors;
pub mod folding;
pub mod options;
pub mod value;

pub use delimiter::Delimiter;
pub use errors::{ErrorContext, RuneError, RuneResult};
pub use folding::{KeyFoldingMode, PathExpansionMode, is_identifier_segment};
pub use options::{DecodeOptions, EncodeOptions, Indent};
pub use value::{JsonValue, Number, Value};
