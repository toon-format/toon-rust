/* e8/gf8/src/simd.rs */
//! SIMD-accelerated operations for `Gf8` using x86_64 AVX intrinsics.
//!
//! # e8 Primitives – Gf8 SIMD Module
//!▫~•◦---------------------------------‣
//!
//! This module provides hardware-accelerated versions of core `Gf8` arithmetic
//! operations. It leverages the perfect alignment between `Gf8`'s 8 `f32` components
//! and the 256-bit SIMD registers found in modern x86_64 CPUs.
//!
//! ### Key Capabilities
//! - **Runtime Feature Detection:** Safely checks for AVX support at runtime before executing `unsafe` intrinsic code.
//! - **Scalar Fallback:** Automatically falls back to standard scalar operations on non-x86 platforms or CPUs without AVX.
//! - **Accelerated Operations:** Provides SIMD versions for dot product, norm, addition, and subtraction.
//!
//! ### Architectural Notes
//! This module is a prime example of how `Gf8`'s fixed dimensionality enables direct
//! hardware mapping. The public functions are safe wrappers that abstract away the
//! `unsafe` nature of CPU intrinsics and the complexity of runtime dispatch.
//!
//! ### Example
//! ```rust
//! use gf8::{Gf8, gf8_add_simd, gf8_dot_simd};
//!
//! let a = Gf8::from_scalar(1.0);
//! let b = Gf8::from_scalar(-0.5);
//!
//! // These functions will use AVX if available, or scalar math otherwise.
//! let sum_vec = gf8_add_simd(&a, &b);
//! let dot_product = gf8_dot_simd(&a, &b);
//!
//! println!("SIMD-accelerated dot product: {}", dot_product);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{Gf8, intrinsics_for_f32_width};

// Gate architecture-specific modules.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Prints a summary of available SIMD capabilities for debugging.
pub fn print_simd_capabilities() {
    println!("--- SIMD Capabilities ---");
    #[cfg(target_arch = "x86_64")]
    {
        println!("Architecture: x86_64");
        println!("AVX enabled: {}", is_x86_feature_detected!("avx"));
        println!("AVX2 enabled: {}", is_x86_feature_detected!("avx2"));
        println!("FMA enabled: {}", is_x86_feature_detected!("fma"));
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("Architecture: aarch64");
        println!("NEON enabled: {}", is_aarch64_feature_detected!("neon"));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("Architecture: Not x86_64 or aarch64. Scalar fallback only.");
    }
    println!("-------------------------");
}

/// Returns a list of available 256-bit f32 intrinsic names for analysis.
pub fn get_available_f32_256_intrinsics() -> Vec<&'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        return intrinsics_for_f32_width(256)
            .filter(|i| {
                let tech = i.technology;
                (tech.contains("AVX2") && is_x86_feature_detected!("avx2"))
                    || (tech.contains("AVX") && is_x86_feature_detected!("avx"))
                    || (tech.contains("FMA") && is_x86_feature_detected!("fma"))
            })
            .map(|i| i.name)
            .collect();
    }
    // Return an empty vector for non-x86 architectures.
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

/// Performs SIMD-accelerated addition of two `Gf8` values, with scalar fallback.
#[inline]
pub fn gf8_add_simd(a: &Gf8, b: &Gf8) -> Gf8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // Safety: We've confirmed AVX is available at runtime.
            unsafe {
                let va = _mm256_loadu_ps(a.coords().as_ptr());
                let vb = _mm256_loadu_ps(b.coords().as_ptr());
                let sum = _mm256_add_ps(va, vb);

                let mut result_coords = [0.0f32; 8];
                _mm256_storeu_ps(result_coords.as_mut_ptr(), sum);
                // Return a new, normalized Gf8 to preserve the invariant.
                return Gf8::new(result_coords);
            }
        }
    }
    // Fallback to the scalar implementation.
    *a + *b
}

/// Performs SIMD-accelerated subtraction of two `Gf8` values, with scalar fallback.
#[inline]
pub fn gf8_sub_simd(a: &Gf8, b: &Gf8) -> Gf8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // Safety: We've confirmed AVX is available at runtime.
            unsafe {
                let va = _mm256_loadu_ps(a.coords().as_ptr());
                let vb = _mm256_loadu_ps(b.coords().as_ptr());
                let diff = _mm256_sub_ps(va, vb);

                let mut result_coords = [0.0f32; 8];
                _mm256_storeu_ps(result_coords.as_mut_ptr(), diff);
                // Return a new, normalized Gf8.
                return Gf8::new(result_coords);
            }
        }
    }
    // Fallback to the scalar implementation.
    *a - *b
}

/// Computes the dot product of two `Gf8` values using SIMD, with scalar fallback.
#[inline]
pub fn gf8_dot_simd(a: &Gf8, b: &Gf8) -> f32 {
    dot_product(*a.coords(), *b.coords())
}

/// Computes the squared L2 norm of a `Gf8` using SIMD, with scalar fallback.
#[inline]
pub fn gf8_norm2_simd(a: &Gf8) -> f32 {
    dot_product(*a.coords(), *a.coords())
}

/// Performs SIMD-accelerated in-place addition over slices: `dst[i] += src[i]`.
pub fn gf8_add_inplace_slice_simd(dst: &mut [Gf8], src: &[Gf8]) -> Result<(), &'static str> {
    if dst.len() != src.len() {
        return Err("Slice lengths must match for in-place addition.");
    }
    // This function can be further optimized with manual unrolling, but for now,
    // we delegate to the robust `gf8_add_simd` for correctness.
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = gf8_add_simd(d, s);
    }
    Ok(())
}

/// SIMD-accelerated matrix-vector multiplication, with scalar fallback.
pub fn gf8_matvec_simd(mat: &[[f32; 8]; 8], vec: &Gf8) -> Gf8 {
    let mut result_coords = [0.0f32; 8];
    for (i, row) in mat.iter().enumerate() {
        result_coords[i] = dot_product(*row, *vec.coords());
    }
    Gf8::new(result_coords)
}

// --- Private Implementation Details ---

/// The primary, runtime-dispatching dot product implementation.
///
/// This function is the single source of truth for dot products. It checks for CPU
/// features at runtime and calls the most optimal available kernel.
#[inline]
pub fn dot_product(a: [f32; 8], b: [f32; 8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            // FMA is fastest on modern CPUs that support it (implies AVX/AVX2).
            return unsafe { dot_product_fma(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { dot_product_avx(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon(a, b) };
        }
    }

    // Scalar fallback for all other cases.
    dot_product_scalar(a, b)
}

/// Scalar dot product implementation (fallback).
#[inline]
fn dot_product_scalar(a: [f32; 8], b: [f32; 8]) -> f32 {
    let mut sum = 0.0;
    for i in 0..8 {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON implementation for dot product on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_neon(a: [f32; 8], b: [f32; 8]) -> f32 {
    let a1 = vld1q_f32(a.as_ptr());
    let a2 = vld1q_f32(a.as_ptr().add(4));
    let b1 = vld1q_f32(b.as_ptr());
    let b2 = vld1q_f32(b.as_ptr().add(4));
    let acc1 = vmulq_f32(a1, b1);
    let acc2 = vmulq_f32(a2, b2);
    let sum = vaddq_f32(acc1, acc2);
    vaddvq_f32(sum)
}

/// AVX implementation for dot product on x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn dot_product_avx(a: [f32; 8], b: [f32; 8]) -> f32 {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        // Use the `_mm256_dp_ps` intrinsic for a combined dot product.
        // The 0xf1 mask means: multiply lanes 0-3, sum them, and place in lane 0;
        // multiply lanes 4-7, sum them, and place in lane 4.
        let prod = _mm256_dp_ps(va, vb, 0xf1);
        let lo = _mm256_castps256_ps128(prod); // Low 128 bits
        let hi = _mm256_extractf128_ps(prod, 1); // High 128 bits
        let sum = _mm_add_ss(lo, hi); // Add the two sums
        _mm_cvtss_f32(sum)
    }
}

/// AVX+FMA implementation for dot product on x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn dot_product_fma(a: [f32; 8], b: [f32; 8]) -> f32 {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        // This is identical to the AVX version but allows the compiler to use FMA.
        // The `_mm256_dp_ps` is often the most efficient way to do this.
        let prod = _mm256_dp_ps(va, vb, 0xf1);
        let lo = _mm256_castps256_ps128(prod);
        let hi = _mm256_extractf128_ps(prod, 1);
        let sum = _mm_add_ss(lo, hi);
        _mm_cvtss_f32(sum)
    }
}
