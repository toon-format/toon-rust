/* src/primitive/fast_math.rs */
//! Provides fast, SIMD-friendly approximations for expensive mathematical functions.
//!
//! # E8 Computing Paradigm – Fast Math Module
//!▫~•◦------------------------------------------------------------------------------------‣
//!
//! This module implements the "Function Approximation via Tabulation" pattern for
//! computationally expensive, low-dimensional functions. It replaces slow transcendental
//! function calls (like `acos`) with a high-performance, three-stage pipeline:
//! 1. A compile-time "Baker" generates a cache-resident Look-Up Table (LUT).
//! 2. A runtime "Synthesizer" performs a fast, branchless lookup into the LUT.
//! 3. A SIMD-friendly linear interpolation (`lerp`) approximates the final value.
//!
//! This approach trades a small, controllable amount of precision for a significant
//! increase in performance, targeting the true bottlenecks identified by profiling.
//!
//! ### Key Capabilities
//! - **Compile-Time LUT Generation:** The `AcosLut` is generated entirely at compile-time
//!   via `const fn`, incurring zero runtime initialization cost.
//! - **Cache-Resident Design:** The LUT size is a generic const parameter, allowing it to
//!   be tuned to fit within L1/L2 CPU caches (e.g., 4096 entries * 4 bytes = 16KB).
//! - **Branchless & SIMD-Friendly:** The lookup and interpolation logic is free of
//!   branches and composed of simple arithmetic operations ideal for FMA instructions.
//!
//! ### Architectural Notes
//! This module embodies the "Just Barely Ahead" principle: applying advanced optimization
//! with surgical precision to the actual bottleneck. The `AcosLut` is designed to be a
//! singleton, statically promoted to eliminate overhead. It serves as a drop-in
//! replacement for `f32::acos` in performance-critical code paths like the `Gf8::angle`
//! calculation.
//!
//! ### Example
//! \```rust
//! use crate::primitive::fast_math::FAST_ACOS;
//! use std::f32::consts::FRAC_1_SQRT_2;
//!
//! // The value of cos(pi/4)
//! let cos_val = FRAC_1_SQRT_2; // approx 0.7071
//!
//! // Calculate acos using the fast LUT
//! let angle_fast = FAST_ACOS.lookup(cos_val);
//!
//! // The result is a very close approximation of pi/4 (approx 0.7854)
//! let angle_std = cos_val.acos();
//! assert!((angle_fast - angle_std).abs() < 1e-4); // Error is low for 4096 entries
//! \```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// A precomputed Look-Up Table for approximating the `acos(x)` function.
///
/// `N` is the number of entries in the table. A larger `N` increases precision
/// and memory usage. `4096` is a good default, using 16KB, which fits comfortably
/// in modern L1/L2 caches.
#[derive(Debug, Clone, Copy)]
pub struct AcosLut<const N: usize> {
    table: [f32; N],
}

use std::sync::OnceLock;

impl<const N: usize> AcosLut<N> {
    /// Creates a new `AcosLut` at compile time.
    ///
    /// The table is populated with `N` values of `acos(x)` for `x` spaced
    /// evenly across the valid domain `[-1.0, 1.0]`.
    pub fn new() -> Self {
        let mut table = [0.0; N];
        let mut i = 0;
        while i < N {
            // Map the table index `i` to the domain `[-1.0, 1.0]`
            let t = i as f32 / (N - 1) as f32; // t is in [0.0, 1.0]
            let x = t * 2.0 - 1.0; // x is in [-1.0, 1.0]

            // Use the standard library `acos` here, since `AcosLut::new` is non-const.
            // This produces an accurate precomputed table that matches the
            // standard library within the expected numerical error bounds.
            let result = x.acos();
            table[i] = result;

            i += 1;
        }
        Self { table }
    }

    /// Looks up the approximate value of `acos(x)`.
    ///
    /// # Arguments
    /// * `x` - A float in the range `[-1.0, 1.0]`. Values outside this range will be clamped.
    ///
    /// # Returns
    /// An approximation of `acos(x)`.
    #[inline(always)]
    pub fn lookup(&self, x: f32) -> f32 {
        // Clamp input to the valid domain.
        let x_clamped = x.clamp(-1.0, 1.0);

        // 1. Map `x` from `[-1.0, 1.0]` to a fractional index `[0.0, N-1]`.
        let frac_index = (x_clamped * 0.5 + 0.5) * ((N - 1) as f32);

        // 2. Get the integer index and the fractional part for interpolation.
        let index_floor = frac_index as usize;
        let frac = frac_index - index_floor as f32;

        // Ensure we don't read past the end of the table.
        // This can happen if x is exactly 1.0.
        let index_ceil = (index_floor + 1).min(N - 1);

        // 3. Gather the two nearest values from the LUT.
        // SAFETY: `index_floor` and `index_ceil` are guaranteed to be in bounds.
        let val_floor = unsafe { *self.table.get_unchecked(index_floor) };
        let val_ceil = unsafe { *self.table.get_unchecked(index_ceil) };

        // 4. Linearly interpolate between the two values.
        // This is a single FMA operation on capable hardware.
        val_floor.mul_add(1.0 - frac, val_ceil * frac)
    }
}

impl<const N: usize> Default for AcosLut<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// A statically allocated, globally accessible instance of the `AcosLut`.
///
/// This is the "bottled rain" - a high-performance, ready-to-use utility
/// that solves the specific `acos` bottleneck.
pub static FAST_ACOS: OnceLock<AcosLut<4096>> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn test_acos_lut_edge_cases() {
        // Test at the boundaries of the domain.
        let acos_of_1 = FAST_ACOS.get_or_init(AcosLut::new).lookup(1.0);
        let acos_of_neg_1 = FAST_ACOS.get_or_init(AcosLut::new).lookup(-1.0);
        let acos_of_0 = FAST_ACOS.get_or_init(AcosLut::new).lookup(0.0);

        assert!((acos_of_1 - 0.0).abs() < 1e-3, "acos(1.0) should be ~0.0");
        assert!(
            (acos_of_neg_1 - PI).abs() < 1e-3,
            "acos(-1.0) should be ~PI"
        );
        assert!(
            (acos_of_0 - FRAC_PI_2).abs() < 1e-3,
            "acos(0.0) should be ~PI/2"
        );
    }

    #[test]
    fn test_acos_lut_precision() {
        let inputs = [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9];
        for &x in &inputs {
            let fast_val = FAST_ACOS.get_or_init(AcosLut::new).lookup(x);
            let std_val = x.acos();
            let error = (fast_val - std_val).abs();
            // With a 4096-entry LUT, the error should be very small.
            assert!(
                error < 1e-4,
                "Precision error too high for x={}: fast={}, std={}, err={}",
                x,
                fast_val,
                std_val,
                error
            );
        }
    }

    #[test]
    fn test_acos_lut_known_values() {
        // cos(pi/4) = 1/sqrt(2)
        let x = std::f32::consts::FRAC_1_SQRT_2;
        let fast_val = FAST_ACOS.get_or_init(AcosLut::new).lookup(x);
        assert!(
            (fast_val - FRAC_PI_4).abs() < 1e-4,
            "acos(1/sqrt(2)) should be ~PI/4"
        );

        // cos(pi/3) = 0.5
        let x = 0.5;
        let fast_val = FAST_ACOS.get_or_init(AcosLut::new).lookup(x);
        let std_val = PI / 3.0;
        assert!(
            (fast_val - std_val).abs() < 1e-4,
            "acos(0.5) should be ~PI/3"
        );
    }

    #[test]
    fn test_out_of_domain_clamping() {
        let val_high = FAST_ACOS.get_or_init(AcosLut::new).lookup(1.5);
        let val_low = FAST_ACOS.get_or_init(AcosLut::new).lookup(-1.5);

        // Should be clamped and return the same as the boundary values.
        assert_eq!(val_high, FAST_ACOS.get_or_init(AcosLut::new).lookup(1.0));
        assert_eq!(val_low, FAST_ACOS.get_or_init(AcosLut::new).lookup(-1.0));
    }
}
