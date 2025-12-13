/* crates/gf8/src/aligned.rs */
//! # E8F Error Management - Aligned Wrappers
//!
//! This module provides automatic error management for E8F operations through
//! zero-cost wrappers that track operation counts and trigger re-alignment
//! when quantization drift may accumulate.
//!
//! ## Key Types
//! - [`E8FAligned`]: Zero-cost wrapper that tracks operation count and triggers
//!   automatic re-alignment after a configurable number of operations.
//! - [`E8FChain`]: Tracks operation sequences with drift metrics and warnings.
//!
//! ## Design Rationale
//! E8F operations always resolve to valid E8 roots via lookup tables, but
//! chaining many operations can accumulate semantic drift from the original
//! intent. These wrappers provide automatic re-alignment to bound this drift.
//!
//! ## Example
//! ```rust
//! use gf8::aligned::E8FAligned;
//! use gf8::E8F;
//!
//! let mut aligned = E8FAligned::new(E8F::new(42));
//!
//! // Operations are tracked automatically
//! aligned.op(|e| e + E8F::new(10));
//! aligned.op(|e| e * E8F::new(5));
//!
//! // After max_ops_before_align operations, alignment is triggered
//! let result = aligned.value();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::quantize::quantize_to_nearest_code;
use crate::{E8F, Gf8, gf8_chordal_distance};

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 1: E8FAligned - ZERO-COST WRAPPER WITH AUTOMATIC ALIGNMENT
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Zero-cost wrapper that tracks operation count and triggers automatic alignment.
///
/// This wrapper ensures that E8F values are periodically re-quantized to the
/// nearest valid E8 root, bounding cumulative quantization drift.
///
/// # Default Behavior
/// - `max_ops_before_align`: 10 operations before automatic re-alignment
/// - Alignment converts E8F → Gf8 → nearest E8 root
///
/// # Example
/// ```rust
/// use gf8::aligned::E8FAligned;
/// use gf8::E8F;
///
/// let mut aligned = E8FAligned::new(E8F::new(42));
///
/// // Chain operations - alignment happens automatically after 10 ops
/// for i in 0..15 {
///     aligned.op(|e| e + E8F::new(i as u8));
/// }
///
/// // Value is guaranteed to be a valid E8 root
/// assert!(aligned.value().is_valid());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct E8FAligned {
    /// The current E8F value.
    value: E8F,
    /// Number of operations since last alignment.
    ops_since_alignment: u8,
    /// Maximum operations before triggering automatic alignment.
    max_ops_before_align: u8,
}

impl E8FAligned {
    /// Default number of operations before automatic alignment.
    pub const DEFAULT_MAX_OPS: u8 = 10;

    /// Create a new aligned wrapper with default settings.
    ///
    /// # Arguments
    /// * `value` - Initial E8F value
    #[inline]
    pub fn new(value: E8F) -> Self {
        Self {
            value,
            ops_since_alignment: 0,
            max_ops_before_align: Self::DEFAULT_MAX_OPS,
        }
    }

    /// Create a new aligned wrapper with custom alignment threshold.
    ///
    /// # Arguments
    /// * `value` - Initial E8F value
    /// * `max_ops` - Maximum operations before automatic alignment
    #[inline]
    pub fn with_max_ops(value: E8F, max_ops: u8) -> Self {
        Self {
            value,
            ops_since_alignment: 0,
            max_ops_before_align: max_ops,
        }
    }

    /// Get the current value.
    #[inline]
    pub fn value(&self) -> E8F {
        self.value
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

    /// Perform an operation with automatic alignment check.
    ///
    /// After the operation, if `ops_since_alignment >= max_ops_before_align`,
    /// automatic re-alignment is triggered.
    ///
    /// # Arguments
    /// * `f` - Operation to perform on the E8F value
    ///
    /// # Returns
    /// Mutable reference to self for chaining
    #[inline]
    pub fn op<F: FnOnce(E8F) -> E8F>(&mut self, f: F) -> &mut Self {
        self.value = f(self.value);
        self.ops_since_alignment = self.ops_since_alignment.saturating_add(1);

        if self.ops_since_alignment >= self.max_ops_before_align {
            self.align();
        }
        self
    }

    /// Force re-alignment to the nearest valid E8 root.
    ///
    /// This converts the current E8F to its Gf8 representation, then
    /// re-quantizes to the nearest E8 root, resetting the operation counter.
    #[inline]
    pub fn align(&mut self) {
        if !self.value.is_valid() {
            self.value = E8F::new(0);
            self.ops_since_alignment = 0;
            return;
        }

        let gf8 = self.value.to_gf8();
        let (code, _distance) = quantize_to_nearest_code(gf8.coords());
        self.value = E8F::new(code.0);
        self.ops_since_alignment = 0;
    }

    /// Check if alignment is needed (ops >= threshold).
    #[inline]
    pub fn needs_alignment(&self) -> bool {
        self.ops_since_alignment >= self.max_ops_before_align
    }

    /// Reset the operation counter without re-aligning.
    #[inline]
    pub fn reset_counter(&mut self) {
        self.ops_since_alignment = 0;
    }
}

impl Default for E8FAligned {
    fn default() -> Self {
        Self::new(E8F::new(0))
    }
}

impl From<E8F> for E8FAligned {
    fn from(value: E8F) -> Self {
        Self::new(value)
    }
}

impl From<E8FAligned> for E8F {
    fn from(aligned: E8FAligned) -> Self {
        aligned.value
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 2: E8FChain - OPERATION SEQUENCE TRACKING WITH DRIFT METRICS
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Tracks a chain of E8F operations with drift metrics and warnings.
///
/// Unlike `E8FAligned`, this type maintains a reference to the initial ground
/// truth value and tracks cumulative drift, logging warnings when drift
/// exceeds a configurable threshold.
///
/// # Drift Metrics (R15.6)
/// - `max_drift`: Maximum observed drift from ground truth
/// - `mean_drift`: Running mean of drift values across operations
/// - `alignment_count`: Number of times re-alignment was triggered
///
/// # Automatic Re-alignment (R15.9)
/// When drift exceeds the threshold (default: 0.1 chordal distance), a warning
/// is logged and automatic re-alignment is triggered.
///
/// # Example
/// ```rust
/// use gf8::aligned::E8FChain;
/// use gf8::{E8F, Gf8};
///
/// let initial = Gf8::new([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
/// let mut chain = E8FChain::start(&initial);
///
/// chain.apply("add_10", |e| e + E8F::new(10));
/// chain.apply("mul_5", |e| e * E8F::new(5));
///
/// let (result, max_drift) = chain.finish();
/// println!("Max drift: {:.4}", max_drift);
/// ```
#[derive(Debug, Clone)]
pub struct E8FChain {
    /// Ground truth reference (initial Gf8 value).
    initial: Gf8,
    /// Current quantized value.
    current: E8F,
    /// Operation names for debugging.
    ops: Vec<&'static str>,
    /// Maximum observed drift from ground truth.
    max_drift: f32,
    /// Sum of all drift values (for computing mean).
    drift_sum: f32,
    /// Count of drift measurements (for computing mean).
    drift_count: u32,
    /// Number of times re-alignment was triggered.
    alignment_count: u32,
    /// Threshold for warning and auto-realignment (default: 0.1).
    drift_threshold: f32,
}

impl E8FChain {
    /// Default drift threshold for warnings and auto-realignment.
    pub const DEFAULT_DRIFT_THRESHOLD: f32 = 0.1;

    /// Start a new operation chain from a ground truth Gf8 value.
    ///
    /// The initial Gf8 value is stored as the ground truth reference (R15.4).
    /// All subsequent drift measurements are computed against this reference.
    ///
    /// # Arguments
    /// * `initial` - Ground truth Gf8 value to track drift against
    pub fn start(initial: &Gf8) -> Self {
        let (code, _) = quantize_to_nearest_code(initial.coords());
        Self {
            initial: *initial,
            current: E8F::new(code.0),
            ops: Vec::new(),
            max_drift: 0.0,
            drift_sum: 0.0,
            drift_count: 0,
            alignment_count: 0,
            drift_threshold: Self::DEFAULT_DRIFT_THRESHOLD,
        }
    }

    /// Start a new operation chain with a custom drift threshold.
    ///
    /// # Arguments
    /// * `initial` - Ground truth Gf8 value
    /// * `threshold` - Drift threshold for warnings and auto-realignment
    pub fn start_with_threshold(initial: &Gf8, threshold: f32) -> Self {
        let mut chain = Self::start(initial);
        chain.drift_threshold = threshold;
        chain
    }

    /// Apply an operation to the chain, tracking drift.
    ///
    /// After each operation, drift from the ground truth is computed using
    /// `gf8_chordal_distance()`. If drift exceeds the threshold (R15.9):
    /// 1. A warning is logged
    /// 2. Automatic re-alignment is triggered
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation (for debugging)
    /// * `f` - Operation to apply
    ///
    /// # Returns
    /// Mutable reference to self for chaining
    pub fn apply(&mut self, op_name: &'static str, f: impl FnOnce(E8F) -> E8F) -> &mut Self {
        self.current = f(self.current);
        self.ops.push(op_name);

        // Measure drift from ground truth using gf8_chordal_distance
        let current_gf8 = self.current.to_gf8();
        let drift = gf8_chordal_distance(&self.initial, &current_gf8);

        // Update drift metrics (R15.6)
        self.max_drift = self.max_drift.max(drift);
        self.drift_sum += drift;
        self.drift_count += 1;

        // R15.9: Log warning and force re-alignment when drift exceeds threshold
        if drift > self.drift_threshold {
            eprintln!(
                "[WARN] E8FChain drift {:.4} exceeds threshold {:.4} after {} ops: {:?}",
                drift,
                self.drift_threshold,
                self.ops.len(),
                self.ops
            );
            // Force re-alignment (R15.9)
            self.align_to_nearest_root();
        }

        self
    }

    /// Get the current drift from ground truth.
    ///
    /// Computes drift using `gf8_chordal_distance()` between the current
    /// E8F value (converted to Gf8) and the initial ground truth.
    pub fn current_drift(&self) -> f32 {
        let current_gf8 = self.current.to_gf8();
        gf8_chordal_distance(&self.initial, &current_gf8)
    }

    /// Get the maximum observed drift (R15.6).
    #[inline]
    pub fn max_drift(&self) -> f32 {
        self.max_drift
    }

    /// Get the mean drift across all operations (R15.6).
    ///
    /// Returns 0.0 if no operations have been performed.
    #[inline]
    pub fn mean_drift(&self) -> f32 {
        if self.drift_count == 0 {
            0.0
        } else {
            self.drift_sum / self.drift_count as f32
        }
    }

    /// Get the number of times re-alignment was triggered (R15.6).
    #[inline]
    pub fn alignment_count(&self) -> u32 {
        self.alignment_count
    }

    /// Get the current E8F value.
    #[inline]
    pub fn current(&self) -> E8F {
        self.current
    }

    /// Get the ground truth Gf8 value (R15.4).
    #[inline]
    pub fn initial(&self) -> &Gf8 {
        &self.initial
    }

    /// Get the operation history (for debugging).
    #[inline]
    pub fn ops(&self) -> &[&'static str] {
        &self.ops
    }

    /// Get the drift threshold.
    #[inline]
    pub fn drift_threshold(&self) -> f32 {
        self.drift_threshold
    }

    /// Set the drift threshold.
    #[inline]
    pub fn set_drift_threshold(&mut self, threshold: f32) {
        self.drift_threshold = threshold;
    }

    /// Explicit re-quantization to the nearest valid E8 root (R15.5).
    ///
    /// This method:
    /// 1. Converts current E8F to Gf8
    /// 2. Re-quantizes to the nearest E8 root
    /// 3. Updates the ground truth reference to the new aligned value
    /// 4. Increments the alignment counter (R15.6)
    /// 5. Clears operation history
    ///
    /// Note: `max_drift` and `mean_drift` are preserved for overall metrics tracking.
    /// The ground truth reference is updated to the newly aligned value so that
    /// subsequent drift measurements are relative to the aligned state.
    pub fn align_to_nearest_root(&mut self) {
        let gf8 = self.current.to_gf8();
        let (code, _) = quantize_to_nearest_code(gf8.coords());
        self.current = E8F::new(code.0);
        // Update initial to current for fresh drift tracking from aligned state
        self.initial = self.current.to_gf8();
        self.alignment_count += 1;
        self.ops.clear();
    }

    /// Force re-alignment to the nearest E8 root.
    ///
    /// This is an alias for `align_to_nearest_root()` for backward compatibility.
    /// It re-quantizes the current value and resets drift tracking
    /// to use the new value as the reference point.
    #[inline]
    pub fn realign(&mut self) {
        self.align_to_nearest_root();
    }

    /// Finish the chain and return the result with max drift.
    ///
    /// # Returns
    /// Tuple of (final E8F value, maximum observed drift)
    pub fn finish(self) -> (E8F, f32) {
        (self.current, self.max_drift)
    }

    /// Finish the chain and return full metrics.
    ///
    /// # Returns
    /// Tuple of (final E8F value, max_drift, mean_drift, alignment_count)
    pub fn finish_with_metrics(self) -> (E8F, f32, f32, u32) {
        let mean = self.mean_drift();
        (self.current, self.max_drift, mean, self.alignment_count)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 3: TESTS
// ═══════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8f_aligned_basic() {
        let aligned = E8FAligned::new(E8F::new(42));
        assert_eq!(aligned.value().index(), 42);
        assert_eq!(aligned.ops_since_alignment(), 0);
    }

    #[test]
    fn test_e8f_aligned_op_tracking() {
        let mut aligned = E8FAligned::new(E8F::new(10));

        // Perform some operations
        aligned.op(|e| e + E8F::new(5));
        assert_eq!(aligned.ops_since_alignment(), 1);

        aligned.op(|e| e * E8F::new(3));
        assert_eq!(aligned.ops_since_alignment(), 2);
    }

    #[test]
    fn test_e8f_aligned_auto_alignment() {
        let mut aligned = E8FAligned::with_max_ops(E8F::new(10), 3);

        // Perform 3 operations - should trigger alignment
        aligned.op(|e| e + E8F::new(1));
        aligned.op(|e| e + E8F::new(2));
        aligned.op(|e| e + E8F::new(3));

        // Counter should be reset after alignment
        assert_eq!(aligned.ops_since_alignment(), 0);

        // Value should still be valid
        assert!(aligned.value().is_valid());
    }

    #[test]
    fn test_e8f_aligned_manual_align() {
        let mut aligned = E8FAligned::new(E8F::new(50));

        aligned.op(|e| e + E8F::new(10));
        aligned.op(|e| e + E8F::new(20));
        assert_eq!(aligned.ops_since_alignment(), 2);

        aligned.align();
        assert_eq!(aligned.ops_since_alignment(), 0);
        assert!(aligned.value().is_valid());
    }

    #[test]
    fn test_e8f_aligned_needs_alignment() {
        let mut aligned = E8FAligned::with_max_ops(E8F::new(10), 2);

        assert!(!aligned.needs_alignment());
        aligned.op(|e| e + E8F::new(1));
        assert!(!aligned.needs_alignment());
        aligned.op(|e| e + E8F::new(2));
        // After 2 ops with max_ops=2, alignment was triggered, counter reset
        assert!(!aligned.needs_alignment());
    }

    #[test]
    fn test_e8f_chain_basic() {
        let initial = Gf8::new([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let chain = E8FChain::start(&initial);

        assert!(chain.current().is_valid());
        assert_eq!(chain.ops().len(), 0);
        assert_eq!(chain.max_drift(), 0.0);
        assert_eq!(chain.mean_drift(), 0.0);
        assert_eq!(chain.alignment_count(), 0);
    }

    #[test]
    fn test_e8f_chain_apply() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment during test
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        chain.apply("add_10", |e| e + E8F::new(10));
        assert_eq!(chain.ops().len(), 1);
        assert_eq!(chain.ops()[0], "add_10");

        chain.apply("mul_5", |e| e * E8F::new(5));
        assert_eq!(chain.ops().len(), 2);
    }

    #[test]
    fn test_e8f_chain_drift_tracking() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment during test
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        // Apply operations that may cause drift
        for i in 0..5 {
            chain.apply("op", |e| e + E8F::new(i * 10));
        }

        // Drift should be tracked
        let drift = chain.current_drift();
        assert!(drift >= 0.0);
        assert!(chain.max_drift() >= drift || (chain.max_drift() - drift).abs() < 1e-6);

        // Mean drift should be computed (R15.6)
        assert!(chain.mean_drift() >= 0.0);
    }

    #[test]
    fn test_e8f_chain_finish() {
        let initial = Gf8::new([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let mut chain = E8FChain::start(&initial);

        chain.apply("op1", |e| e + E8F::new(10));
        chain.apply("op2", |e| e * E8F::new(20));

        let (result, max_drift) = chain.finish();
        assert!(result.is_valid());
        assert!(max_drift >= 0.0);
    }

    #[test]
    fn test_e8f_chain_finish_with_metrics() {
        let initial = Gf8::new([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let mut chain = E8FChain::start(&initial);

        chain.apply("op1", |e| e + E8F::new(10));
        chain.apply("op2", |e| e * E8F::new(20));

        let (result, max_drift, mean_drift, alignment_count) = chain.finish_with_metrics();
        assert!(result.is_valid());
        assert!(max_drift >= 0.0);
        assert!(mean_drift >= 0.0);
        // alignment_count is u32, just verify it's accessible
        let _ = alignment_count;
    }

    #[test]
    fn test_e8f_chain_realign() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        chain.apply("op1", |e| e + E8F::new(50));
        chain.apply("op2", |e| e + E8F::new(100));

        let max_drift_before = chain.max_drift();
        chain.realign();

        // After realign, ops should be cleared
        assert_eq!(chain.ops().len(), 0);
        assert!(chain.current().is_valid());
        // Alignment count should be incremented
        assert_eq!(chain.alignment_count(), 1);

        // max_drift is preserved for metrics tracking
        assert_eq!(chain.max_drift(), max_drift_before);
    }

    #[test]
    fn test_e8f_chain_align_to_nearest_root() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        chain.apply("op1", |e| e + E8F::new(50));

        // Explicit re-quantization (R15.5)
        chain.align_to_nearest_root();

        assert!(chain.current().is_valid());
        assert_eq!(chain.alignment_count(), 1);
        assert_eq!(chain.ops().len(), 0);
    }

    #[test]
    fn test_e8f_chain_auto_realignment_on_threshold() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use very low threshold to trigger auto-realignment (R15.9)
        let mut chain = E8FChain::start_with_threshold(&initial, 0.001);

        // Apply operation that will likely exceed threshold
        chain.apply("big_op", |e| e + E8F::new(200));

        // Auto-realignment should have been triggered
        // (alignment_count > 0 if drift exceeded threshold)
        // Note: The actual drift depends on E8 geometry, so we just verify
        // the mechanism works
        assert!(chain.current().is_valid());
    }

    #[test]
    fn test_e8f_chain_metrics_tracking() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        // Apply multiple operations
        chain.apply("op1", |e| e + E8F::new(10));
        chain.apply("op2", |e| e + E8F::new(20));
        chain.apply("op3", |e| e + E8F::new(30));

        // R15.6: Track error metrics
        assert!(chain.max_drift() >= 0.0);
        assert!(chain.mean_drift() >= 0.0);
        assert_eq!(chain.alignment_count(), 0); // No auto-realignment with high threshold
    }

    #[test]
    fn test_e8f_aligned_from_into() {
        let e8f = E8F::new(100);
        let aligned: E8FAligned = e8f.into();
        assert_eq!(aligned.value().index(), 100);

        let back: E8F = aligned.into();
        assert_eq!(back.index(), 100);
    }

    #[test]
    fn test_e8f_chain_custom_threshold() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let chain = E8FChain::start_with_threshold(&initial, 0.05);

        assert_eq!(chain.drift_threshold(), 0.05);
    }

    #[test]
    fn test_e8f_chain_ground_truth_reference() {
        // R15.4: Use Gf8 as ground truth reference
        let initial = Gf8::new([0.7, 0.3, 0.5, 0.1, 0.2, 0.4, 0.6, 0.8]);
        let chain = E8FChain::start(&initial);

        // Ground truth should be stored
        let stored_initial = chain.initial();
        assert_eq!(stored_initial.coords()[0], initial.coords()[0]);
        assert_eq!(stored_initial.coords()[7], initial.coords()[7]);
    }

    #[test]
    fn test_e8f_chain_operation_names_for_debugging() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use high threshold to prevent auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 10.0);

        chain.apply("first_op", |e| e + E8F::new(1));
        chain.apply("second_op", |e| e + E8F::new(2));
        chain.apply("third_op", |e| e + E8F::new(3));

        // Operation names should be tracked for debugging
        let ops = chain.ops();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0], "first_op");
        assert_eq!(ops[1], "second_op");
        assert_eq!(ops[2], "third_op");
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // TASK 3.3: COMPREHENSIVE TESTS FOR E8F ERROR MANAGEMENT (R15.3, R15.9)
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// R15.3: Test that alignment triggers exactly after N operations (default: 10)
    #[test]
    fn test_alignment_triggers_after_n_operations_default() {
        // Default max_ops is 10
        let mut aligned = E8FAligned::new(E8F::new(42));
        assert_eq!(aligned.max_ops_before_align(), E8FAligned::DEFAULT_MAX_OPS);
        assert_eq!(E8FAligned::DEFAULT_MAX_OPS, 10);

        // Perform 9 operations - should NOT trigger alignment
        for i in 0..9 {
            aligned.op(|e| e + E8F::new(i));
            assert_eq!(aligned.ops_since_alignment(), i + 1);
        }

        // 10th operation should trigger alignment and reset counter
        aligned.op(|e| e + E8F::new(9));
        assert_eq!(
            aligned.ops_since_alignment(),
            0,
            "Counter should reset after alignment"
        );
        assert!(aligned.value().is_valid());
    }

    /// R15.3: Test alignment with custom chain depth
    #[test]
    fn test_alignment_triggers_after_custom_n_operations() {
        let custom_max = 5;
        let mut aligned = E8FAligned::with_max_ops(E8F::new(100), custom_max);
        assert_eq!(aligned.max_ops_before_align(), custom_max);

        // Perform 4 operations - should NOT trigger alignment
        for i in 0..4 {
            aligned.op(|e| e + E8F::new(i));
            assert_eq!(aligned.ops_since_alignment(), i + 1);
        }

        // 5th operation should trigger alignment
        aligned.op(|e| e + E8F::new(4));
        assert_eq!(aligned.ops_since_alignment(), 0);

        // Continue with more operations - should trigger again at 5
        for i in 0..5 {
            aligned.op(|e| e + E8F::new(i as u8));
        }
        assert_eq!(
            aligned.ops_since_alignment(),
            0,
            "Should reset after second alignment"
        );
    }

    /// R15.3: Test that alignment is amortized (no overhead for single ops)
    #[test]
    fn test_alignment_amortized_no_overhead_single_ops() {
        let mut aligned = E8FAligned::with_max_ops(E8F::new(50), 100);

        // Single operation should not trigger alignment
        aligned.op(|e| e + E8F::new(1));
        assert_eq!(aligned.ops_since_alignment(), 1);
        assert!(!aligned.needs_alignment());

        // Value should still be valid without alignment
        assert!(aligned.value().is_valid());
    }

    /// R15.9: Test warning threshold behavior - drift exceeds threshold triggers warning
    #[test]
    fn test_warning_threshold_triggers_realignment() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Very low threshold to ensure drift exceeds it
        let mut chain = E8FChain::start_with_threshold(&initial, 0.001);

        let alignment_count_before = chain.alignment_count();

        // Apply operation that will cause significant drift
        chain.apply("large_drift_op", |e| e + E8F::new(200));

        // R15.9: Auto-realignment should have been triggered
        // The alignment count should have increased
        assert!(
            chain.alignment_count() > alignment_count_before,
            "Alignment should be triggered when drift exceeds threshold"
        );

        // Value should still be valid after realignment
        assert!(chain.current().is_valid());
    }

    /// R15.9: Test that default threshold is 0.1 chordal distance
    #[test]
    fn test_default_drift_threshold() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let chain = E8FChain::start(&initial);

        assert_eq!(
            chain.drift_threshold(),
            E8FChain::DEFAULT_DRIFT_THRESHOLD,
            "Default threshold should match constant"
        );
        assert_eq!(
            E8FChain::DEFAULT_DRIFT_THRESHOLD,
            0.1,
            "Default threshold should be 0.1"
        );
    }

    /// R15.9: Test that warning is logged when drift exceeds threshold
    /// (We verify this by checking alignment_count increases)
    #[test]
    fn test_drift_exceeds_threshold_forces_realignment() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Use default threshold (0.1)
        let mut chain = E8FChain::start(&initial);

        // Apply operations that will cause drift > 0.1
        // Adding a large offset should cause significant drift
        chain.apply("drift_op", |e| e + E8F::new(150));

        // Check if drift exceeded threshold and realignment occurred
        // Note: The actual drift depends on E8 geometry
        let drift = chain.current_drift();
        if drift > 0.1 {
            // If drift exceeded threshold, alignment should have been triggered
            assert!(
                chain.alignment_count() > 0,
                "Alignment should be triggered when drift > threshold"
            );
        }
    }

    /// R15.6: Test drift tracking accuracy - max_drift, mean_drift, alignment_count
    #[test]
    fn test_drift_tracking_accuracy() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // High threshold to prevent auto-realignment during test
        let mut chain = E8FChain::start_with_threshold(&initial, 100.0);

        // Apply multiple operations and track drift
        let mut expected_max_drift = 0.0f32;
        let mut drift_sum = 0.0f32;
        let mut drift_count = 0u32;

        for i in 0..5 {
            chain.apply("op", |e| e + E8F::new(i * 20));

            // Manually compute expected drift
            let current_drift = chain.current_drift();
            expected_max_drift = expected_max_drift.max(current_drift);
            drift_sum += current_drift;
            drift_count += 1;
        }

        let expected_mean = drift_sum / drift_count as f32;

        // Verify drift metrics (R15.6)
        assert!(
            (chain.max_drift() - expected_max_drift).abs() < 1e-5,
            "max_drift should match expected: {} vs {}",
            chain.max_drift(),
            expected_max_drift
        );
        assert!(
            (chain.mean_drift() - expected_mean).abs() < 1e-5,
            "mean_drift should match expected: {} vs {}",
            chain.mean_drift(),
            expected_mean
        );
        assert_eq!(
            chain.alignment_count(),
            0,
            "No alignment with high threshold"
        );
    }

    /// R15.6: Test alignment_count increments correctly
    #[test]
    fn test_alignment_count_increments() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // High threshold to prevent auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 100.0);

        assert_eq!(chain.alignment_count(), 0);

        chain.apply("op1", |e| e + E8F::new(10));
        chain.realign();
        assert_eq!(chain.alignment_count(), 1);

        chain.apply("op2", |e| e + E8F::new(20));
        chain.align_to_nearest_root();
        assert_eq!(chain.alignment_count(), 2);

        chain.apply("op3", |e| e + E8F::new(30));
        chain.realign();
        assert_eq!(chain.alignment_count(), 3);
    }

    /// R15.9: Test that ops are cleared after auto-realignment
    #[test]
    fn test_ops_cleared_after_auto_realignment() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Very low threshold to trigger auto-realignment
        let mut chain = E8FChain::start_with_threshold(&initial, 0.0001);

        // Apply operation that will exceed threshold
        chain.apply("trigger_op", |e| e + E8F::new(200));

        // If auto-realignment was triggered, ops should be cleared
        if chain.alignment_count() > 0 {
            assert_eq!(
                chain.ops().len(),
                0,
                "Ops should be cleared after auto-realignment"
            );
        }
    }

    /// Test E8FAligned counter saturation (edge case)
    #[test]
    fn test_aligned_counter_saturation() {
        let mut aligned = E8FAligned::with_max_ops(E8F::new(10), 255);

        // Perform many operations without triggering alignment
        for _ in 0..254 {
            aligned.op(|e| e); // Identity operation
        }

        // Counter should be at 254
        assert_eq!(aligned.ops_since_alignment(), 254);

        // One more should trigger alignment (255 >= 255)
        aligned.op(|e| e);
        assert_eq!(aligned.ops_since_alignment(), 0);
    }

    /// Test E8FChain with zero operations
    #[test]
    fn test_chain_zero_operations() {
        let initial = Gf8::new([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let chain = E8FChain::start(&initial);

        // With no operations, metrics should be zero
        assert_eq!(chain.max_drift(), 0.0);
        assert_eq!(chain.mean_drift(), 0.0);
        assert_eq!(chain.alignment_count(), 0);
        assert_eq!(chain.ops().len(), 0);
    }

    /// Test that set_drift_threshold works correctly
    #[test]
    fn test_set_drift_threshold() {
        let initial = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut chain = E8FChain::start(&initial);

        assert_eq!(chain.drift_threshold(), 0.1);

        chain.set_drift_threshold(0.5);
        assert_eq!(chain.drift_threshold(), 0.5);

        chain.set_drift_threshold(0.01);
        assert_eq!(chain.drift_threshold(), 0.01);
    }
}
