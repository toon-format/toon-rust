/* e8/gf8/src/bitcodec/mod.rs */
//! Bit encoding and decoding for `Gf8` vectors.
//!
//! This module provides both lossy and lossless quantization approaches for `Gf8` vectors.
//! The choice between them depends on whether you need perfect reconstruction (lossless)
//! or maximum compression with reasonable accuracy (lossy).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;

pub mod lossless;
pub mod lossy;

// Re-export the main types and functions for convenience
pub use lossless::{
    Gf8LosslessCode, gf8_from_lossless_code, gf8_to_lossless_code, gf8_to_lossless_code_closest,
};
pub use lossy::{Gf8BitSig, bits_from_u8_le, bits_to_u8_le, gf8_from_code, gf8_to_code};

/// Convenience function that attempts lossless encoding first, falls back to lossy.
///
/// This function tries to encode the `Gf8` using the lossless approach first (checking
/// against the 240 canonical roots). If that fails (because the vector is not an
/// exact E8 lattice point), it falls back to the lossy sign-based encoding.
///
/// # Returns
/// - `Ok(Gf8LosslessCode)` if the vector is an exact E8 lattice point
/// - `Err(Gf8BitSig )` if the vector needed lossy encoding
pub fn gf8_to_best_code(gf: &Gf8) -> Result<Gf8LosslessCode, Gf8BitSig> {
    // Try lossless first (exact match on 240 roots)
    if let Some(lossless_code) = lossless::gf8_to_lossless_code(gf) {
        return Ok(lossless_code);
    }

    // Fall back to lossy (sign bits)
    Err(lossy::gf8_to_code(gf))
}

/// Reconstructs a `Gf8` from either a lossless or lossy code.
///
/// This function handles both types of codes and returns the appropriate `Gf8`.
pub fn gf8_from_best_code(
    lossless_code: Option<Gf8LosslessCode>,
    lossy_code: Option<Gf8BitSig>,
) -> Option<Gf8> {
    match (lossless_code, lossy_code) {
        (Some(code), _) => Some(lossless::gf8_from_lossless_code(code)),
        (None, Some(code)) => Some(lossy::gf8_from_code(code)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gf8;

    #[test]
    fn test_best_code_selection() {
        // Test with an exact E8 lattice point (from D8 subset)
        let lattice_point = Gf8::from_bits_even_parity([1, 0, 1, 1, 0, 0, 1, 0]);
        let result = gf8_to_best_code(&lattice_point);

        // Should use lossless encoding
        assert!(
            result.is_ok(),
            "E8 lattice points should use lossless encoding"
        );

        // Test with an arbitrary vector (not on lattice)
        let arbitrary = Gf8::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let result = gf8_to_best_code(&arbitrary);

        // Should fall back to lossy encoding
        assert!(
            result.is_err(),
            "Arbitrary vectors should use lossy encoding"
        );
    }
}
