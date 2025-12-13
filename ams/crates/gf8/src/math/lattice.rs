/* e8/gf8/src/math/lattice.rs */
//! E₈-inspired lattice utilities for `Gf8`.
//!
//! # e8 Primitives – Lattice Module
//!▫~•◦-------------------------------------‣
//!
//! This module provides simple, E₈-inspired lattice helpers for `Gf8`. The
//! intent is not to fully model the 240-root E₈ lattice, but to offer practical
//! gf8s that align with the sign-parity structure already used by
//! `Gf8::from_bits_even_parity` and the `Gf8BitSig ` bitcodec.
//!
//! ### Key Capabilities
//! - **Shell Quantization:** Project arbitrary 8D vectors onto an E₈-like ±1
//!   shell with even parity.
//! - **Chordal & Angular Distances:** Compute distances between `Gf8` values
//!   on the unit sphere.
//!
//! These utilities are suitable for building E8B-like compression schemes,
//! approximate lookup tables, or shell-based neighborhood computations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;

/// Quantize an arbitrary 8D vector onto an E₈-like ±1 shell with even parity.
///
/// This function:
/// - Interprets the sign of each coordinate (`< 0.0` -> 1, `>= 0.0` -> 0) as a bit.
/// - Enforces an even number of 1s by flipping the last bit if necessary.
/// - Uses `Gf8::from_bits_even_parity` to construct a normalized `Gf8`.
///
/// This is aligned with the parity logic used by the `Gf8BitSig ` bitcodec.
pub fn quantize_to_e8_shell(coords: &[f32; 8]) -> Gf8 {
    let mut bits = [0u8; 8];
    let mut set_bits = 0u32;

    for (i, &c) in coords.iter().enumerate() {
        if c < 0.0 {
            bits[i] = 1;
            set_bits += 1;
        } else {
            bits[i] = 0;
        }
    }

    if set_bits % 2 == 1 {
        bits[7] ^= 1;
    }

    Gf8::from_bits_even_parity(bits)
}

/// Quantize a slice of 8D vectors onto the E₈-like shell.
///
/// `src` and `dst` must have the same length. Each source vector is quantized
/// independently using [`quantize_to_e8_shell`].
pub fn quantize_slice_to_e8_shell(src: &[[f32; 8]], dst: &mut [Gf8]) {
    assert_eq!(src.len(), dst.len(), "src and dst length mismatch");

    for (coords, gf) in src.iter().zip(dst.iter_mut()) {
        *gf = quantize_to_e8_shell(coords);
    }
}

/// Compute the squared chordal distance between two `Gf8` directions in ℝ⁸.
///
/// For unit-normalized vectors `a` and `b`, this is:
///
/// ```text
/// ||a - b||² = 2 - 2 * dot(a, b)
/// ```
///
/// This is often a convenient substitute for geodesic distance in E₈-based
/// search and clustering.
#[inline]
pub fn gf8_chordal_distance2(a: &Gf8, b: &Gf8) -> f32 {
    let a_coords = a.coords();
    let b_coords = b.coords();

    let mut dot = 0.0f32;
    for (&av, &bv) in a_coords.iter().zip(b_coords.iter()) {
        dot += av * bv;
    }

    let dot_clamped = dot.clamp(-1.0, 1.0);
    2.0 - 2.0 * dot_clamped
}

/// Compute the chordal distance between two `Gf8` directions in ℝ⁸.
#[inline]
pub fn gf8_chordal_distance(a: &Gf8, b: &Gf8) -> f32 {
    gf8_chordal_distance2(a, b).sqrt()
}

/// Compute the geodesic distance (angle in radians) between two `Gf8`
/// directions on the unit 8-sphere.
///
/// This is equivalent to `acos(dot(a, b))`, clamped for numeric stability.
#[inline]
pub fn gf8_geodesic_distance(a: &Gf8, b: &Gf8) -> f32 {
    let a_coords = a.coords();
    let b_coords = b.coords();

    let mut dot = 0.0f32;
    for (&av, &bv) in a_coords.iter().zip(b_coords.iter()) {
        dot += av * bv;
    }

    let cos = dot.clamp(-1.0, 1.0);
    cos.acos()
}
