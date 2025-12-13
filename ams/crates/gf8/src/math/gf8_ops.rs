/* e8/gf8/src/math/gf8_ops.rs */
//! Geometric operations on `Gf8` directions.
//!
//! # e8 Primitives – Gf8 Ops
//!▫~•◦-------------------------------------‣
//!
//! This module provides higher-level operations on `Gf8` directions that are
//! useful for building semantic flows, attention mechanisms, and interpolation
//! schemes on the unit 8-sphere.
//!
//! ### Key Capabilities
//! - **Cosine Similarity & Angle:** Compute similarity and angular distance
//!   between two `Gf8` directions.
//! - **Lerp & Slerp:** Linear and spherical interpolation between `Gf8`
//!   directions, renormalized to the unit sphere.
//! - **Slice Utilities:** Helpers for applying interpolation across slices of
//!   `Gf8` values.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;

/// Compute the cosine similarity between two `Gf8` directions.
///
/// `Gf8` values are expected to be unit-normalized; this function clamps the
/// dot product into `[-1.0, 1.0]` for numerical stability.
///
/// If either vector is effectively zero-length (should not happen for valid
/// `Gf8`), this returns `0.0`.
#[inline]
pub fn gf8_cosine_similarity(a: &Gf8, b: &Gf8) -> f32 {
    let a_coords = a.coords();
    let b_coords = b.coords();

    let mut dot = 0.0f32;
    let mut a_n2 = 0.0f32;
    let mut b_n2 = 0.0f32;

    for (&av, &bv) in a_coords.iter().zip(b_coords.iter()) {
        dot += av * bv;
        a_n2 += av * av;
        b_n2 += bv * bv;
    }

    if a_n2 == 0.0 || b_n2 == 0.0 {
        return 0.0;
    }

    let denom = (a_n2 * b_n2).sqrt();
    let cos = dot / denom;
    cos.clamp(-1.0, 1.0)
}

/// Compute the angle (in radians) between two `Gf8` directions.
///
/// This is defined as `acos(cosine_similarity(a, b))`. The result lies in
/// `[0, π]`. If either vector is degenerate, this returns `0.0`.
#[inline]
pub fn gf8_angle(a: &Gf8, b: &Gf8) -> f32 {
    let cos = gf8_cosine_similarity(a, b);
    cos.acos()
}

/// Linearly interpolate between two `Gf8` directions and renormalize.
///
/// This performs:
///
/// ```text
/// v(t) = (1 - t) * a + t * b
/// ```
///
/// followed by renormalization onto the unit sphere. It is a simple, fast
/// approximation of spherical interpolation.
///
/// - `t = 0.0` → `a`
/// - `t = 1.0` → `b`
pub fn gf8_lerp(a: &Gf8, b: &Gf8, t: f32) -> Gf8 {
    let a_coords = a.coords();
    let b_coords = b.coords();

    let mut v = [0.0f32; 8];
    let one_minus_t = 1.0 - t;

    for (i, (a_c, b_c)) in a_coords.iter().zip(b_coords.iter()).enumerate() {
        v[i] = one_minus_t * (*a_c) + t * (*b_c);
    }

    Gf8::from_coords(v)
}

/// Spherical linear interpolation (SLERP) between two `Gf8` directions.
///
/// This computes the geodesic interpolation on the unit 8-sphere. For small
/// angles or nearly identical vectors, it falls back to `gf8_lerp` for
/// numerical stability.
///
/// - `t = 0.0` → `a`
/// - `t = 1.0` → `b`
pub fn gf8_slerp(a: &Gf8, b: &Gf8, t: f32) -> Gf8 {
    let a_coords = a.coords();
    let b_coords = b.coords();

    let mut dot = 0.0f32;
    for (&ac, &bc) in a_coords.iter().zip(b_coords.iter()) {
        dot += ac * bc;
    }

    let mut cos_theta = dot.clamp(-1.0, 1.0);
    // If vectors point in opposite directions, SLERP is ambiguous; flip `b`
    // to choose the shorter path.
    let mut b_adjusted = [0.0f32; 8];
    if cos_theta < 0.0 {
        cos_theta = -cos_theta;
        for (i, &val) in b_coords.iter().enumerate() {
            b_adjusted[i] = -val;
        }
    } else {
        b_adjusted.copy_from_slice(b_coords);
    }

    let theta = cos_theta.acos();

    // If angle is small, fall back to LERP.
    if theta.abs() < 1e-4 {
        return gf8_lerp(a, &Gf8::from_coords(b_adjusted), t);
    }

    let sin_theta = theta.sin();
    let w1 = ((1.0 - t) * theta).sin() / sin_theta;
    let w2 = (t * theta).sin() / sin_theta;

    let mut v = [0.0f32; 8];
    for (i, &ac) in a_coords.iter().enumerate() {
        v[i] = w1 * ac + w2 * b_adjusted[i];
    }

    Gf8::from_coords(v)
}

/// Apply linear interpolation between pairs of `Gf8` values across slices,
/// writing the result into `dst`.
///
/// All slices must have the same length.
pub fn gf8_lerp_slice(dst: &mut [Gf8], a: &[Gf8], b: &[Gf8], t: f32) {
    assert_eq!(dst.len(), a.len(), "dst and a length mismatch");
    assert_eq!(dst.len(), b.len(), "dst and b length mismatch");

    for ((dst_v, a_v), b_v) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *dst_v = gf8_lerp(a_v, b_v, t);
    }
}
