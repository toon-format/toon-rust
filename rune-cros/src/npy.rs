/* src/npy/npy.rs */
//!▫~•◦-------------------------------‣
//! # NumPy format conversion module for scientific data processing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-cros to achieve efficient conversion between NumPy and JSON formats.
//!
//! ### Key Capabilities
//! - **NumPy to JSON Conversion**: Converts NumPy .npy binary data to JSON array representation.
//! - **Float64 Support**: Specializes in handling 64-bit floating point numerical data.
//! - **Feature-gated Implementation**: Uses conditional compilation to enable/disable NumPy support based on feature flags.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `lib` and other format converters.
//! Functions adhere to the `Result<String>` return type and integrate with the `yoshi` error handling system.
//!
//! ### Example
//! ```rust
//! use crate::rune_cros::{npy_to_json};
//!
//! let npy_data: &[u8] = &[/* NumPy binary data */];
//! let result = npy_to_json(npy_data).unwrap();
//!
//! // The 'result' contains JSON representation of the NumPy array.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use yoshi::{Result, buck};
#[cfg(feature = "npy")]
use yoshi::yoshi;
/// Basic Numpy `.npy` read/write helper (Float64) — feature-gated
///
/// NOTE: This minimal implementation supports `f64` typed numpy arrays only
/// (single-dimension or multi-dimensional) and converts them into nested JSON
/// arrays. It uses `ndarray` and `ndarray-npy` when the `npy` feature is enabled.
#[cfg(feature = "npy")]
pub fn npy_to_json(bytes: &[u8]) -> Result<String> {
    use npy::NpyData;

    // Parse NPY payload as f64s and flatten to a Vec<f64>.
    let data: NpyData<f64> = NpyData::from_bytes(bytes)
        .map_err(|e| yoshi!("npy decode error: {}", e))?;
    let values: Vec<f64> = data.to_vec();

    // Represent as JSON array (flat). Multi-dimensional arrays are flattened here;
    // we can extend to shape-aware nesting later if needed.
    serde_json::to_string_pretty(&values)
        .map_err(|e| yoshi!("serde_json stringify error: {}", e))
}

#[cfg(not(feature = "npy"))]
pub fn npy_to_json(_bytes: &[u8]) -> Result<String> { buck!("npy feature is disabled; compile with '--features npy'") }
