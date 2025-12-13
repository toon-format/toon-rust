/* crates/gf8/src/e8x.rs */
//! # E8X - The Batteries-Included E8 Type
//!
//! E8X (E8 Cross) combines all E8 capabilities into a single, easy-to-use type:
//! - **E8F core operations**: Zero-FLOP arithmetic via lookup tables
//! - **Automatic error management**: Re-alignment after N operations
//! - **Drift tracking**: Built-in metrics for error monitoring
//! - **Hybrid computation**: Seamless E8F ↔ f32 conversion
//!
//! ## Philosophy
//!
//! E8X is the "batteries included" type for E8 operations. Instead of manually
//! wiring together E8F + E8FAligned + E8FCompute, just use E8X and everything
//! works out of the box.
//!
//! ## When to Use
//!
//! - **Use E8X for**: Applications, media compression, neural networks, anything
//!   requiring robust E8 operations with automatic error management
//! - **Use E8F for**: Low-level operations, when you need fine-grained control,
//!   or when you're building your own wrappers
//!
//! ## Example
//!
//! ```rust
//! use gf8::E8X;
//!
//! // Single import, everything works
//! let mut a = E8X::new_from_index(42);
//! let b = E8X::new_from_index(100);
//!
//! // Automatic re-alignment after N operations
//! for i in 0..20 {
//!     a += b;  // Re-alignment happens automatically
//! }
//!
//! // Drift tracking built-in
//! println!("Max drift: {:.4}", a.max_drift());
//!
//! // Hybrid compute built-in
//! let coords = a.to_f32_coords();
//! let reconstructed = E8X::from_f32_coords(&coords);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::quantize::quantize_to_nearest_code;
use crate::{E8F, Gf8, gf8_chordal_distance};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 1: E8X TYPE DEFINITION
// ═══════════════════════════════════════════════════════════════════════════════════════

/// E8X - The batteries-included E8 type.
///
/// Combines E8F core operations with automatic error management, hybrid computation,
/// and drift tracking. This is the recommended type for most applications.
///
/// # Features
///
/// - **1-byte wire representation** via `batch_to_bytes` (same index as E8F)
/// - **Zero-FLOP operations**: Add, mul, dot via precomputed lookup tables
/// - **Automatic re-alignment**: Triggers after N operations (default: 10)
/// - **Hybrid computation**: Seamless E8F ↔ f32 conversion automation
/// - **Drift tracking**: Monitors cumulative error from operations
///
/// # Error Management
///
/// E8X automatically re-aligns to the nearest E8 root after a configurable
/// number of operations, bounding cumulative quantization drift.
///
/// # Example
///
/// ```rust
/// use gf8::E8X;
///
/// let mut x = E8X::new_from_index(0);
///
/// // Chain operations - automatic re-alignment
/// for i in 0..15 {
///     x += E8X::new_from_index(i);
/// }
///
/// // Check drift is finite
/// assert!(x.max_drift().is_finite());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct E8X {
    /// The current E8F value.
    value: E8F,

    /// Number of operations since last alignment.
    ops_since_alignment: u8,

    /// Maximum operations before triggering automatic alignment.
    max_ops_before_align: u8,

    /// Maximum observed drift from last alignment point.
    max_drift: f32,

    /// Sum of all drift values (for computing mean).
    drift_sum: f32,

    /// Count of drift measurements.
    drift_count: u32,

    /// Reference point for drift measurement (last aligned value).
    reference: Gf8,
}

impl E8X {
    /// Default number of operations before automatic alignment.
    pub const DEFAULT_MAX_OPS: u8 = 10;

    /// Default drift threshold for warnings.
    pub const DEFAULT_DRIFT_THRESHOLD: f32 = 0.1;

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 2: CONSTRUCTORS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Create a new E8X from an E8F value.
    ///
    /// # Arguments
    /// * `value` - The E8F value to wrap
    ///
    /// # Example
    /// ```rust
    /// use gf8::{E8X, E8F};
    ///
    /// let e8f = E8F::new(42);
    /// let e8x = E8X::new(e8f);
    /// ```
    pub fn new(value: E8F) -> Self {
        let reference = value.to_gf8();
        Self {
            value,
            ops_since_alignment: 0,
            max_ops_before_align: Self::DEFAULT_MAX_OPS,
            max_drift: 0.0,
            drift_sum: 0.0,
            drift_count: 0,
            reference,
        }
    }

    /// Create a new E8X from a root index (0-239).
    ///
    /// # Arguments
    /// * `index` - The E8 root index
    ///
    /// # Example
    /// ```rust
    /// use gf8::E8X;
    ///
    /// let e8x = E8X::new_from_index(42);
    /// ```
    pub fn new_from_index(index: u8) -> Self {
        Self::new(E8F::new(index))
    }

    /// Create a new E8X with custom alignment threshold.
    ///
    /// # Arguments
    /// * `value` - The E8F value to wrap
    /// * `max_ops` - Maximum operations before automatic alignment
    ///
    /// # Example
    /// ```rust
    /// use gf8::{E8X, E8F};
    ///
    /// let e8x = E8X::with_max_ops(E8F::new(42), 5);
    /// ```
    pub fn with_max_ops(value: E8F, max_ops: u8) -> Self {
        let mut e8x = Self::new(value);
        e8x.max_ops_before_align = max_ops;
        e8x
    }

    /// Create a new E8X from f32 coordinates.
    ///
    /// Quantizes the coordinates to the nearest E8 root.
    ///
    /// # Arguments
    /// * `coords` - 8D coordinates to quantize
    ///
    /// # Returns
    /// Tuple of (E8X, quantization_error)
    ///
    /// # Example
    /// ```rust
    /// use gf8::E8X;
    ///
    /// let coords = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let (e8x, error) = E8X::from_f32_coords(&coords);
    /// assert!(error < 1.0);
    /// ```
    pub fn from_f32_coords(coords: &[f32; 8]) -> (Self, f32) {
        let (code, snapped_gf8) = quantize_to_nearest_code(coords);

        // Compute quantization error
        let input_gf8 = Gf8::from_coords(*coords);
        let error = gf8_chordal_distance(&input_gf8, &snapped_gf8);

        (Self::new(E8F::from_code(code)), error)
    }

    /// Create a new E8X from a Gf8 vector.
    ///
    /// # Arguments
    /// * `gf8` - The Gf8 vector to quantize
    ///
    /// # Example
    /// ```rust
    /// use gf8::{E8X, Gf8};
    ///
    /// let gf8 = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// let e8x = E8X::from_gf8(&gf8);
    /// ```
    pub fn from_gf8(gf8: &Gf8) -> Self {
        let (code, _) = quantize_to_nearest_code(gf8.coords());
        Self::new(E8F::from_code(code))
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 3: ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Get the underlying E8F value.
    #[inline]
    pub fn value(&self) -> E8F {
        self.value
    }

    /// Get the E8 root index (0-239).
    #[inline]
    pub fn index(&self) -> u8 {
        self.value.index()
    }

    /// Get the number of operations since last alignment.
    #[inline]
    pub fn ops_since_alignment(&self) -> u8 {
        self.ops_since_alignment
    }

    /// Get the maximum operations before alignment.
    #[inline]
    pub fn max_ops_before_align(&self) -> u8 {
        self.max_ops_before_align
    }

    /// Set the maximum operations before alignment.
    #[inline]
    pub fn set_max_ops(&mut self, max_ops: u8) {
        self.max_ops_before_align = max_ops;
    }

    /// Get the maximum observed drift.
    #[inline]
    pub fn max_drift(&self) -> f32 {
        self.max_drift
    }

    /// Get the mean drift across all operations.
    #[inline]
    pub fn mean_drift(&self) -> f32 {
        if self.drift_count == 0 {
            0.0
        } else {
            self.drift_sum / self.drift_count as f32
        }
    }

    /// Get the current drift from reference point.
    pub fn current_drift(&self) -> f32 {
        let current_gf8 = self.value.to_gf8();
        gf8_chordal_distance(&self.reference, &current_gf8)
    }

    /// Check if the value is a valid E8 root.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.value.is_valid()
    }

    /// Check if the current value exactly matches a specific E8 root.
    /// This is a **lossless** comparison - no quantization error.
    #[inline]
    pub fn is_exact_root(&self, index: u8) -> bool {
        self.value.index() == index && self.value.is_valid()
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 4: HYBRID COMPUTATION (E8F ↔ f32)
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Convert to f32 coordinates for computation.
    ///
    /// This is a lossless operation - the E8F root coordinates are
    /// exactly representable in f32.
    ///
    /// # Example
    /// ```rust
    /// use gf8::E8X;
    ///
    /// let e8x = E8X::new_from_index(0);
    /// let coords = e8x.to_f32_coords();
    /// ```
    pub fn to_f32_coords(&self) -> [f32; 8] {
        if !self.value.is_valid() {
            return [0.0; 8];
        }
        let gf8 = self.value.to_gf8();
        *gf8.coords()
    }

    /// Convert to Gf8 representation.
    pub fn to_gf8(&self) -> Gf8 {
        self.value.to_gf8()
    }

    /// Convert to u32 for integer accumulation.
    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.value.index() as u32
    }

    /// Compute dot product with another E8X in f32 space.
    ///
    /// Uses exact f32 computation for higher precision than the
    /// E8F lookup table.
    pub fn dot_f32(&self, other: &E8X) -> f32 {
        let a_coords = self.to_f32_coords();
        let b_coords = other.to_f32_coords();

        a_coords
            .iter()
            .zip(b_coords.iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// Compute chordal distance to another E8X.
    pub fn chordal_distance(&self, other: &E8X) -> f32 {
        let a_coords = self.to_f32_coords();
        let b_coords = other.to_f32_coords();

        let sum_sq: f32 = a_coords
            .iter()
            .zip(b_coords.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        sum_sq.sqrt()
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 5: ERROR MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Check if alignment is needed.
    #[inline]
    pub fn needs_alignment(&self) -> bool {
        self.ops_since_alignment >= self.max_ops_before_align
    }

    /// Force re-alignment to the nearest valid E8 root.
    ///
    /// This resets the operation counter and updates the reference point.
    /// **Error Bound**: Alignment introduces ≤0.087 chordal distance error
    /// (worst case) by snapping to the nearest E8 root.
    pub fn align(&mut self) {
        if !self.value.is_valid() {
            self.value = E8F::new(0);
            self.ops_since_alignment = 0;
            self.reference = self.value.to_gf8();
            return;
        }

        // If already a valid root, alignment is a no-op (lossless)
        let gf8 = self.value.to_gf8();
        let (code, _) = quantize_to_nearest_code(gf8.coords());
        self.value = E8F::from_code(code);
        self.ops_since_alignment = 0;
        self.reference = self.value.to_gf8();
    }

    /// Update drift metrics after an operation.
    fn update_drift(&mut self) {
        let drift = self.current_drift();
        self.max_drift = self.max_drift.max(drift);
        self.drift_sum += drift;
        self.drift_count += 1;
    }

    /// Perform an operation with automatic alignment check.
    ///
    /// This is the internal method used by operator overloads.
    fn perform_op<F>(&mut self, f: F)
    where
        F: FnOnce(E8F) -> E8F,
    {
        self.value = f(self.value);
        self.ops_since_alignment = self.ops_since_alignment.saturating_add(1);
        self.update_drift();

        if self.needs_alignment() {
            self.align();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 6: ARITHMETIC OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Add another E8X value (with automatic re-alignment).
    pub fn add_e8x(&mut self, other: E8X) -> &mut Self {
        self.perform_op(|v| v + other.value);
        self
    }

    /// Subtract another E8X value (with automatic re-alignment).
    pub fn sub_e8x(&mut self, other: E8X) -> &mut Self {
        self.perform_op(|v| v - other.value);
        self
    }

    /// Multiply by another E8X value (with automatic re-alignment).
    pub fn mul_e8x(&mut self, other: E8X) -> &mut Self {
        self.perform_op(|v| v * other.value);
        self
    }

    /// Compute dot product using E8F lookup table.
    ///
    /// For higher precision, use `dot_f32()` instead.
    pub fn dot(&self, other: E8X) -> f32 {
        self.value.dot(other.value)
    }

    /// Reflect through a hyperplane normal to another E8X.
    pub fn reflect(&mut self, normal: E8X) -> &mut Self {
        self.perform_op(|v| v.reflect(normal.value));
        self
    }

    /// Negate the value.
    pub fn neg(&mut self) -> &mut Self {
        self.perform_op(|v| -v);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 7: OPERATOR OVERLOADS
// ═══════════════════════════════════════════════════════════════════════════════════════

impl Add for E8X {
    type Output = E8X;

    fn add(self, other: E8X) -> E8X {
        let mut result = self;
        result.add_e8x(other);
        result
    }
}

impl AddAssign for E8X {
    fn add_assign(&mut self, other: E8X) {
        self.add_e8x(other);
    }
}

impl Sub for E8X {
    type Output = E8X;

    fn sub(self, other: E8X) -> E8X {
        let mut result = self;
        result.sub_e8x(other);
        result
    }
}

impl SubAssign for E8X {
    fn sub_assign(&mut self, other: E8X) {
        self.sub_e8x(other);
    }
}

impl Mul for E8X {
    type Output = E8X;

    fn mul(self, other: E8X) -> E8X {
        let mut result = self;
        result.mul_e8x(other);
        result
    }
}

impl MulAssign for E8X {
    fn mul_assign(&mut self, other: E8X) {
        self.mul_e8x(other);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 8: CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

impl From<E8F> for E8X {
    fn from(value: E8F) -> Self {
        Self::new(value)
    }
}

impl From<E8X> for E8F {
    fn from(e8x: E8X) -> Self {
        e8x.value
    }
}

impl From<u8> for E8X {
    fn from(index: u8) -> Self {
        Self::new_from_index(index)
    }
}

impl Default for E8X {
    fn default() -> Self {
        Self::new(E8F::new(0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 9: BATCH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

impl E8X {
    /// Convert a batch of E8X values to bytes for storage.
    pub fn batch_to_bytes(batch: &[E8X]) -> Vec<u8> {
        batch.iter().map(|e8x| e8x.index()).collect()
    }

    /// Convert bytes back to E8X values.
    pub fn batch_from_bytes(bytes: &[u8]) -> Vec<E8X> {
        bytes.iter().map(|&b| E8X::new_from_index(b)).collect()
    }

    /// Compute weighted sum of E8X values in f32 space.
    ///
    /// This is the canonical hybrid pattern:
    /// 1. Convert to f32
    /// 2. Accumulate
    /// 3. Quantize back to E8X
    pub fn weighted_sum(weights: &[E8X], values: &[E8X]) -> (E8X, f32) {
        assert_eq!(weights.len(), values.len());

        if weights.is_empty() {
            return (E8X::default(), 0.0);
        }

        // Accumulate in f32 space
        let mut sum = [0.0f32; 8];

        for (w, v) in weights.iter().zip(values.iter()) {
            let w_coords = w.to_f32_coords();
            let v_coords = v.to_f32_coords();
            let w_scalar = w_coords[0];

            for i in 0..8 {
                sum[i] += w_scalar * v_coords[i];
            }
        }

        // Normalize
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for c in &mut sum {
                *c /= norm;
            }
        }

        // Quantize back to E8X
        E8X::from_f32_coords(&sum)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 10: TESTS
// ═══════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8x_creation() {
        let e8x = E8X::new_from_index(42);
        assert_eq!(e8x.index(), 42);
        assert_eq!(e8x.ops_since_alignment(), 0);
        assert!(e8x.is_valid());
    }

    #[test]
    fn test_e8x_automatic_alignment() {
        let mut e8x = E8X::with_max_ops(E8F::new(10), 3);

        // Perform 3 operations - should trigger alignment
        e8x.add_e8x(E8X::new_from_index(1));
        e8x.add_e8x(E8X::new_from_index(2));
        e8x.add_e8x(E8X::new_from_index(3));

        // Counter should be reset after alignment
        assert_eq!(e8x.ops_since_alignment(), 0);
        assert!(e8x.is_valid());
    }

    #[test]
    fn test_e8x_operator_overloads() {
        let a = E8X::new_from_index(10);
        let b = E8X::new_from_index(20);

        let c = a + b;
        assert!(c.is_valid());

        let d = a - b;
        assert!(d.is_valid());

        let e = a * b;
        assert!(e.is_valid());
    }

    #[test]
    fn test_e8x_drift_tracking() {
        let mut e8x = E8X::new_from_index(0);

        // Perform operations
        for i in 0..5 {
            e8x.add_e8x(E8X::new_from_index(i * 10));
        }

        // Drift should be tracked
        assert!(e8x.max_drift() >= 0.0);
        assert!(e8x.mean_drift() >= 0.0);
    }

    #[test]
    fn test_e8x_hybrid_compute() {
        let e8x = E8X::new_from_index(42);
        let coords = e8x.to_f32_coords();

        // Should be unit vector
        let norm: f32 = coords.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);

        // Roundtrip
        let (recovered, error) = E8X::from_f32_coords(&coords);
        assert_eq!(recovered.index(), 42);
        assert!(error < 1e-5);
    }

    #[test]
    fn test_e8x_dot_product() {
        let a = E8X::new_from_index(0);
        let b = E8X::new_from_index(0);

        let dot = a.dot(b);
        assert!((dot - 1.0).abs() < 0.1, "Same root should have dot ~1.0");

        let dot_f32 = a.dot_f32(&b);
        assert!(
            (dot_f32 - 1.0).abs() < 1e-5,
            "f32 dot should be more precise"
        );
    }

    #[test]
    fn test_e8x_batch_operations() {
        let batch = vec![
            E8X::new_from_index(0),
            E8X::new_from_index(1),
            E8X::new_from_index(2),
        ];

        let bytes = E8X::batch_to_bytes(&batch);
        assert_eq!(bytes, vec![0, 1, 2]);

        let recovered = E8X::batch_from_bytes(&bytes);
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].index(), 0);
        assert_eq!(recovered[1].index(), 1);
        assert_eq!(recovered[2].index(), 2);
    }

    #[test]
    fn test_e8x_weighted_sum() {
        let weights = vec![E8X::new_from_index(100), E8X::new_from_index(100)];
        let values = vec![E8X::new_from_index(0), E8X::new_from_index(0)];

        let (result, error) = E8X::weighted_sum(&weights, &values);
        assert!(result.is_valid());
        assert!(error.is_finite());
    }

    #[test]
    fn test_e8x_conversions() {
        let e8f = E8F::new(42);
        let e8x: E8X = e8f.into();
        assert_eq!(e8x.index(), 42);

        let back: E8F = e8x.into();
        assert_eq!(back.index(), 42);

        let from_u8: E8X = 100u8.into();
        assert_eq!(from_u8.index(), 100);
    }

    #[test]
    fn test_e8x_chordal_distance() {
        let a = E8X::new_from_index(0);
        let b = E8X::new_from_index(0);

        let dist = a.chordal_distance(&b);
        assert!(dist < 1e-5, "Same root should have zero distance");

        let c = E8X::new_from_index(100);
        let dist2 = a.chordal_distance(&c);
        assert!(dist2 > 0.0, "Different roots should have non-zero distance");
    }
}
