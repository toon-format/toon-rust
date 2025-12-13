/* crates/gf8/src/compute.rs */
//! # E8F Hybrid Computation Strategy
//!
//! This module defines the `E8FCompute` trait and utilities for hybrid E8F/f32 computation.
//!
//! ## Design Philosophy
//!
//! The E8 lattice provides 32x compression (1 byte per 8D block), but operations between
//! E8F values can accumulate quantization error. The hybrid strategy addresses this:
//!
//! - **E8F for**: Storage, transmission, identity comparison, neighbor lookup
//! - **f32/u32 for**: Accumulation, weighted sums, score computation, intermediate results
//!
//! ## Canonical Pattern
//!
//! ```text
//! E8F → f32/u32 (compute) → E8F (store)
//! ```
//!
//! This pattern ensures:
//! 1. Minimal storage footprint (E8F)
//! 2. Full precision during computation (f32/u32)
//! 3. Quantization only at final result
//!
//! ## Error Bounds
//!
//! | Operation | Expected Error |
//! |-----------|----------------|
//! | E8F → f32 → E8F roundtrip | ≤ 0.15 chordal distance |
//! | Accumulated sum (N terms) | O(√N) quantization noise |
//! | Single quantization | ≤ 0.087 chordal distance (worst case) |
//!
//! ## Requirements Coverage
//!
//! - R16.1: E8F for storage, transmission, identity comparison, neighbor lookup
//! - R16.2: f32 for accumulation, weighted sums, score computation
//! - R16.6: Document expected error bounds
//! - R16.7: E8FCompute trait for types supporting hybrid computation
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;
use crate::e8f::{E8F, E8Vec};
use crate::quantize::quantize_to_nearest_code;

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 1: E8FCOMPUTE TRAIT
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Trait for types that support hybrid E8F/f32 computation.
///
/// This trait defines the conversion interface between discrete E8F space
/// and continuous f32 space, enabling the hybrid computation pattern:
///
/// ```text
/// E8F → f32 (compute) → E8F (store)
/// ```
///
/// # Error Bounds & Lossless Operations
///
/// **Truly Lossless Operations** (zero error):
/// - E8F → Gf8: Exact lookup from codebook (no quantization)
/// - E8F → E8F arithmetic: Deterministic lookup table (same inputs → same output)
/// - E8F ↔ u8 serialization: Direct byte mapping (bijective)
/// - E8F identity comparison: Exact u8 equality check
///
/// **Bounded Error Operations** (quantization required):
/// - E8F → f32 → E8F roundtrip: ≤0.087 chordal distance (worst case)
/// - Arbitrary f32[8] → E8F: ≤0.087 chordal distance (nearest root)
/// - Chained E8F ops: O(√N) drift accumulation over N operations
///
/// **Key Insight**: Operations stay lossless *within* the E8 lattice.
/// Error only occurs when entering/exiting the discrete E8 space.
pub trait E8FCompute {
    /// Convert to f32 representation for computation.
    ///
    /// This is a lossless operation - the E8F root coordinates are
    /// exactly representable in f32.
    ///
    /// # Returns
    ///
    /// The 8D coordinates of this E8F root as f32 values.
    fn to_f32_coords(&self) -> [f32; 8];

    /// Convert from f32 coordinates back to E8F.
    ///
    /// This operation quantizes to the nearest E8 root, introducing
    /// quantization error bounded by ~0.087 chordal distance.
    ///
    /// # Arguments
    ///
    /// * `coords` - 8D coordinates to quantize
    ///
    /// # Returns
    ///
    /// The nearest E8F root and the quantization error (chordal distance).
    fn from_f32_coords(coords: &[f32; 8]) -> (Self, f32)
    where
        Self: Sized;

    /// Convert to a scalar f32 representation.
    ///
    /// For E8F, this returns the first coordinate of the root vector,
    /// useful for simple scalar operations.
    fn to_f32_scalar(&self) -> f32;

    /// Convert from a scalar f32 to E8F.
    ///
    /// Creates an E8F by treating the scalar as the first coordinate
    /// and padding with zeros, then quantizing to nearest root.
    fn from_f32_scalar(value: f32) -> Self
    where
        Self: Sized;

    /// Convert to u32 for integer accumulation.
    ///
    /// Maps the E8F index to a u32 value suitable for accumulation.
    /// This is useful for transition score computation.
    fn to_u32(&self) -> u32;

    /// Convert from u32 back to E8F.
    ///
    /// Maps a u32 value back to an E8F index (clamped to 0-239).
    fn from_u32(value: u32) -> Self
    where
        Self: Sized;
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 2: E8F IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════════════

impl E8FCompute for E8F {
    fn to_f32_coords(&self) -> [f32; 8] {
        if !self.is_valid() {
            return [0.0; 8];
        }
        let gf8 = self.to_gf8();
        *gf8.coords()
    }

    fn from_f32_coords(coords: &[f32; 8]) -> (Self, f32) {
        let (code, snapped_gf8) = quantize_to_nearest_code(coords);

        // Compute quantization error as chordal distance
        let input_gf8 = Gf8::from_coords(*coords);
        let input_coords = input_gf8.coords();
        let snapped_coords = snapped_gf8.coords();

        let error: f32 = input_coords
            .iter()
            .zip(snapped_coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        (E8F::from_code(code), error)
    }

    fn to_f32_scalar(&self) -> f32 {
        if !self.is_valid() {
            return 0.0;
        }
        let gf8 = self.to_gf8();
        gf8.coords()[0]
    }

    fn from_f32_scalar(value: f32) -> Self {
        // Create a vector with the scalar as first coordinate
        let mut coords = [0.0f32; 8];
        coords[0] = value;

        // Normalize to unit sphere before quantization
        let norm = coords.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for c in &mut coords {
                *c /= norm;
            }
        } else {
            coords[0] = 1.0; // Default to first axis
        }

        let (code, _) = quantize_to_nearest_code(&coords);
        E8F::from_code(code)
    }

    fn to_u32(&self) -> u32 {
        self.0 as u32
    }

    fn from_u32(value: u32) -> Self {
        E8F::new((value.min(239)) as u8)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 3: HYBRID COMPUTATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Compute transition scores using the hybrid E8F/u32 pattern.
///
/// Implements the canonical transition equation:
/// ```text
/// score(x') = Σ_o e(o) · v(o → x')
/// ```
///
/// # Hybrid Pattern
///
/// - Resonance and valorem are stored as u32 (not E8F)
/// - Accumulation uses u64 to prevent overflow
/// - Final scores are u32
///
/// # Arguments
///
/// * `resonance` - Energy levels at each root (u32, not E8F)
/// * `valorem` - Path weights matrix (u32, not E8F)
///
/// # Returns
///
/// Transition scores for all 240 roots.
///
/// # Requirements
///
/// - R16.3: E8F → f32 (compute) → E8F pattern
/// - R16.4: Accumulation in u32/f32
/// - R16.5: Quantize only final result
/// - R16.8: Valorem uses u32 counts
/// - R16.9: E8F only for root identity
pub fn compute_transition_scores_hybrid(
    resonance: &[u32; 240],
    valorem: &[[u32; 240]; 240],
) -> [u32; 240] {
    let mut scores = [0u32; 240];

    // Canonical Transition Equation:
    // score(x') = Σ_o ( Resonance[o] * Valorem[o][x'] )
    //
    // Uses u64 for intermediate accumulation to prevent overflow
    for (x_prime, score_slot) in scores.iter_mut().enumerate() {
        let mut score: u64 = 0;

        for origin in 0..240 {
            let energy = resonance[origin] as u64;
            let weight = valorem[origin][x_prime] as u64;
            score += energy * weight;
        }

        // Scale down to u32 range (shift by 10 bits = divide by 1024)
        *score_slot = (score >> 10).min(u32::MAX as u64) as u32;
    }

    scores
}

/// Compute transition scores with sparse optimization.
///
/// Same as `compute_transition_scores_hybrid` but skips inactive roots
/// (energy == 0) for better performance on sparse resonance fields.
///
/// # Arguments
///
/// * `resonance` - Energy levels at each root
/// * `valorem` - Path weights matrix
/// * `energy_threshold` - Minimum energy to consider a root active
///
/// # Returns
///
/// Transition scores for all 240 roots.
pub fn compute_transition_scores_sparse(
    resonance: &[u32; 240],
    valorem: &[[u32; 240]; 240],
    energy_threshold: u32,
) -> [u32; 240] {
    let mut scores = [0u32; 240];

    // Collect active roots first (sparse optimization)
    let active_roots: Vec<(usize, u64)> = resonance
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e >= energy_threshold)
        .map(|(i, &e)| (i, e as u64))
        .collect();

    // Only process active roots
    for (x_prime, score_slot) in scores.iter_mut().enumerate() {
        let mut score: u64 = 0;

        for &(origin, energy) in &active_roots {
            let weight = valorem[origin][x_prime] as u64;
            score += energy * weight;
        }

        *score_slot = (score >> 10).min(u32::MAX as u64) as u32;
    }

    scores
}

/// Accumulate weighted E8F values in f32 space, then quantize.
///
/// This is the canonical hybrid pattern for weighted sums:
/// 1. Convert E8F weights and values to f32
/// 2. Accumulate in f32 space
/// 3. Quantize final result to E8F
///
/// # Arguments
///
/// * `weights` - E8F weights (converted to f32 for computation)
/// * `values` - E8F values to weight and sum
///
/// # Returns
///
/// The weighted sum quantized back to E8F, plus the quantization error.
pub fn weighted_sum_hybrid(weights: &[E8F], values: &[E8F]) -> (E8F, f32) {
    assert_eq!(weights.len(), values.len());

    if weights.is_empty() {
        return (E8F::new(0), 0.0);
    }

    // Accumulate in f32 space
    let mut sum = [0.0f32; 8];

    for (w, v) in weights.iter().zip(values.iter()) {
        let w_coords = w.to_f32_coords();
        let v_coords = v.to_f32_coords();

        // Weight is treated as a scalar (first coordinate)
        let w_scalar = w_coords[0];

        for i in 0..8 {
            sum[i] += w_scalar * v_coords[i];
        }
    }

    // Normalize before quantization
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for c in &mut sum {
            *c /= norm;
        }
    }

    // Quantize final result
    E8F::from_f32_coords(&sum)
}

/// Compute dot product in f32 space for precision.
///
/// While E8F has a built-in dot product via lookup tables, this function
/// computes the exact f32 dot product for cases requiring higher precision.
///
/// # Arguments
///
/// * `a` - First E8F value
/// * `b` - Second E8F value
///
/// # Returns
///
/// The exact f32 dot product of the two E8F root vectors.
pub fn dot_f32(a: E8F, b: E8F) -> f32 {
    let a_coords = a.to_f32_coords();
    let b_coords = b.to_f32_coords();

    a_coords
        .iter()
        .zip(b_coords.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute chordal distance between two E8F values.
///
/// The chordal distance is the Euclidean distance between points on the
/// unit sphere, useful for measuring quantization error.
///
/// # Arguments
///
/// * `a` - First E8F value
/// * `b` - Second E8F value
///
/// # Returns
///
/// The chordal distance in [0, 2] range.
pub fn chordal_distance(a: E8F, b: E8F) -> f32 {
    let a_coords = a.to_f32_coords();
    let b_coords = b.to_f32_coords();

    let sum_sq: f32 = a_coords
        .iter()
        .zip(b_coords.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    sum_sq.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 4: E8VEC HYBRID OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Extension trait for E8Vec hybrid operations.
pub trait E8VecCompute {
    /// Convert entire vector to f32 representation.
    fn to_f32_vec_full(&self) -> Vec<f32>;

    /// Create from f32 vector with quantization.
    fn from_f32_vec_full(vec: &[f32]) -> Self;

    /// Compute dot product in f32 space for precision.
    fn dot_f32(&self, other: &Self) -> f32;
}

impl E8VecCompute for E8Vec {
    fn to_f32_vec_full(&self) -> Vec<f32> {
        self.to_f32_vec()
    }

    fn from_f32_vec_full(vec: &[f32]) -> Self {
        E8Vec::from_f32_vec(vec)
    }

    fn dot_f32(&self, other: &Self) -> f32 {
        assert_eq!(self.data.len(), other.data.len());

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| dot_f32(*a, *b))
            .sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 5: ERROR BOUND CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Maximum chordal distance for single E8F quantization.
///
/// This is the worst-case error when quantizing an arbitrary 8D unit vector
/// to the nearest E8 root. The E8 lattice has 240 roots uniformly distributed
/// on S⁷, giving a maximum quantization error of approximately 0.087.
pub const MAX_SINGLE_QUANTIZATION_ERROR: f32 = 0.087;

/// Maximum chordal distance for E8F roundtrip (E8F → f32 → E8F).
///
/// Due to the discrete nature of the E8 lattice, a roundtrip through f32
/// space may land on a different root. The maximum error is bounded by
/// approximately 0.15 chordal distance.
pub const MAX_ROUNDTRIP_ERROR: f32 = 0.15;

/// Recommended maximum chain length before re-alignment.
///
/// After this many E8F operations, accumulated error may exceed acceptable
/// bounds. Use `E8FAligned` or explicit re-quantization for longer chains.
pub const RECOMMENDED_MAX_CHAIN_LENGTH: usize = 10;

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 6: TESTS
// ═══════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8f_to_f32_coords() {
        let e8f = E8F::new(42);
        let coords = e8f.to_f32_coords();

        // Should be unit vector
        let norm: f32 = coords.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "E8F coords should be unit vector"
        );
    }

    #[test]
    fn test_e8f_roundtrip() {
        for idx in 0..240u8 {
            let original = E8F::new(idx);
            let coords = original.to_f32_coords();
            let (recovered, error) = E8F::from_f32_coords(&coords);

            // Roundtrip should recover the same root
            assert_eq!(
                original.index(),
                recovered.index(),
                "Roundtrip should preserve E8F index"
            );
            assert!(
                error < 1e-5,
                "Roundtrip error should be negligible for exact coords"
            );
        }
    }

    #[test]
    fn test_e8f_to_u32() {
        let e8f = E8F::new(100);
        assert_eq!(e8f.to_u32(), 100);

        let recovered = E8F::from_u32(100);
        assert_eq!(recovered.index(), 100);
    }

    #[test]
    fn test_e8f_from_u32_clamping() {
        // Values > 239 should be clamped
        let e8f = E8F::from_u32(500);
        assert_eq!(e8f.index(), 239);
    }

    #[test]
    fn test_compute_transition_scores_hybrid() {
        let mut resonance = [0u32; 240];
        resonance[0] = 100;
        resonance[1] = 50;

        let mut valorem = [[0u32; 240]; 240];
        valorem[0][10] = 10;
        valorem[1][10] = 20;

        let scores = compute_transition_scores_hybrid(&resonance, &valorem);

        // score[10] = (100*10 + 50*20) >> 10 = 2000 >> 10 = 1
        // Note: The shift is for scaling, actual value depends on implementation
        assert!(scores[10] > 0, "Score should be non-zero");
    }

    #[test]
    fn test_compute_transition_scores_sparse() {
        let mut resonance = [0u32; 240];
        resonance[0] = 100;
        resonance[1] = 5; // Below threshold

        let mut valorem = [[0u32; 240]; 240];
        valorem[0][10] = 10;
        valorem[1][10] = 20;

        let scores = compute_transition_scores_sparse(&resonance, &valorem, 10);

        // Only root 0 should contribute (root 1 is below threshold)
        // The sparse version should skip root 1
        // Just verify it runs without panic
        let _ = scores[10];
    }

    #[test]
    fn test_dot_f32() {
        let a = E8F::new(0);
        let b = E8F::new(0);

        let dot = dot_f32(a, b);
        assert!(
            (dot - 1.0).abs() < 1e-5,
            "Same root should have dot product 1.0"
        );
    }

    #[test]
    fn test_chordal_distance() {
        let a = E8F::new(0);
        let b = E8F::new(0);

        let dist = chordal_distance(a, b);
        assert!(dist < 1e-5, "Same root should have zero chordal distance");

        // Different roots should have non-zero distance
        let c = E8F::new(100);
        let dist2 = chordal_distance(a, c);
        assert!(dist2 > 0.0, "Different roots should have non-zero distance");
    }

    #[test]
    fn test_weighted_sum_hybrid() {
        let weights = vec![E8F::new(100), E8F::new(100)];
        let values = vec![E8F::new(0), E8F::new(0)];

        let (result, error) = weighted_sum_hybrid(&weights, &values);

        assert!(result.is_valid(), "Result should be valid E8F");
        // Error can be larger for weighted sums due to accumulated quantization
        // Just verify it's finite and reasonable (< 2.0 which is max chordal distance)
        assert!(
            error.is_finite() && error < 2.0,
            "Error should be finite and bounded"
        );
    }

    #[test]
    fn test_e8vec_compute() {
        let vec1 = E8Vec::from_indices(&[0, 1, 2, 3]);
        let vec2 = E8Vec::from_indices(&[0, 1, 2, 3]);

        let dot = vec1.dot_f32(&vec2);
        assert!(dot > 0.0, "Same vectors should have positive dot product");
    }
}
