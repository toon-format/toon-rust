/* src/rune/hydron/gf8.rs */
//! A foundational 8-dimensional geometric float gf8, inspired by E₈ lattice properties.
//!
//! # e8 Primitives – Gf8 Module
//!▫~•◦-----------------------------‣
//!
//! This module provides the `Gf8` type, a core numeric gf8 for the e8 ecosystem.
//! It is designed to replace standard floating-point numbers in contexts where geometric
//! stability, intrinsic normalization, and binary-addressable states are paramount.
//!
//! ### Key Capabilities
//! - **Geometric Representation:** `Gf8` represents a value as a normalized 8D vector on the unit sphere (S⁷).
//! - **Binary Encoding:** Provides a constructor from 8 bits that maps to a unique, stable direction in 8D space, enforcing an E₈-like even parity constraint.
//! - **Geometric Arithmetic:** All arithmetic operations (add, sub) are geometric, preserving the unit-norm constraint by re-projecting results onto the sphere.
//! - **Tensor-like API:** Implements `Deref` and a `Gf8Tensor` trait, allowing it to be used seamlessly as a small, fixed-size tensor.
//!
//! ### Architectural Notes
//! `Gf8` is the cornerstone of the e8 compute and data model. Its fixed dimensionality is a perfect
//! match for 256-bit SIMD registers (e.g., AVX), enabling highly efficient hardware acceleration.
//! It serves as the basis for E8B codes, E8DB keys, and the E8 LLM's numerical representation.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::hydron::gf8::{Gf8, Gf8Tensor};
//!
//! // Create a Gf8 from a binary pattern (0b10101010)
//! let bits = [0, 1, 0, 1, 0, 1, 0, 1];
//! let a = Gf8::from_bits_even_parity(bits);
//!
//! // Create another Gf8 from a different pattern
//! let b = Gf8::from_scalar(-0.5);
//!
//! // Compute the dot product (cosine similarity)
//! let similarity = a.dot(b.coords());
//!
//! // `Gf8` can be treated like a slice
//! println!("Gf8 'a' has {} dimensions.", a.as_slice().len());
//! println!("Similarity between a and b: {}", similarity);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::rune::hydron::{
    fisher::FisherLayer,
    hyperbolic::HyperbolicLayer,
    lorentzian::{LorentzianLayer, SpacetimePoint},
    quaternion::QuaternionOps,
    spherical::SphericalLayer,
    symplectic::SymplecticLayer,
};

/// A tiny tensor-like trait for GF8.
///
/// This provides an explicit contract for types that can be viewed as a slice of floats,
/// intended for use in generic, tensor-aware code.
pub trait Gf8Tensor {
    /// Returns the underlying data as an immutable slice.
    fn as_slice(&self) -> &[f32];
    /// Returns the underlying data as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [f32];
}

/// A GF8 (GeoFloat8), an 8-dimensional geometric float gf8.
///
/// It is internally represented by an array of 8 `f32`s, which is always
/// normalized to have a unit L2 norm (i.e., it lies on the surface of an
/// 8D hypersphere). This property provides intrinsic stability and makes it suitable
/// for representing directions, rotations, and normalized semantic states.
///
/// The only exception to the unit-norm rule is the zero vector, which has a norm of 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gf8 {
    coords: [f32; 8],
}

impl Gf8 {
    /// The zero vector, representing a neutral or null state.
    pub const ZERO: Self = Self { coords: [0.0; 8] };

    /// Constructs a `Gf8` from raw 8D coordinates, normalizing them to unit length.
    ///
    /// If the input vector has a magnitude of zero, the zero `Gf8` is returned.
    #[inline]
    pub fn new(coords: [f32; 8]) -> Self {
        let mut v = Self { coords };
        v.renormalize();
        v
    }

    /// Constructs a `Gf8` from raw 8D coordinates.
    ///
    /// This is an alias for [`Gf8::new`], provided for clarity when working in
    /// math-heavy code where "from_coords" more clearly expresses intent than "new".
    #[inline]
    pub fn from_coords(coords: [f32; 8]) -> Self {
        Self::new(coords)
    }

    /// Constructs a `Gf8` from 8 bits, mapping them to an E₈-like ±1 pattern.
    ///
    /// The mapping is `0 -> +1.0` and `1 -> -1.0`. To satisfy an E₈-like constraint,
    /// the number of `-1.0` entries is forced to be even by flipping the sign of
    /// the last coordinate if necessary. The resulting vector is then normalized
    /// to unit length.
    pub fn from_bits_even_parity(bits: [u8; 8]) -> Self {
        let mut coords = [0.0f32; 8];
        let mut neg_count = 0usize;

        for (i, &b) in bits.iter().enumerate() {
            if b == 0 {
                coords[i] = 1.0;
            } else {
                coords[i] = -1.0;
                neg_count += 1;
            }
        }

        if neg_count % 2 == 1 {
            // Flip the sign of the last coordinate to enforce even parity.
            coords[7] = -coords[7];
        }

        // Normalize the resulting vector to place it on the unit sphere.
        // A pure ±1 vector has a norm of sqrt(8).
        Self::new(coords)
    }

    /// Constructs a `Gf8` by embedding a scalar along the first axis.
    ///
    /// The resulting `Gf8` will be `[signum(x), 0.0, ..., 0.0]`. This provides a simple
    /// way to represent scalar magnitudes directionally.
    pub fn from_scalar(x: f32) -> Self {
        let mut coords = [0.0; 8];
        coords[0] = x;
        Self::new(coords)
    }

    /// Retrieves the raw coordinate data as a slice.
    #[inline]
    pub fn coords(&self) -> &[f32; 8] {
        &self.coords
    }

    /// Approximates a scalar value by projecting the `Gf8` onto the first axis.
    ///
    /// Since `Gf8` is a unit vector, this value will be in the range `[-1.0, 1.0]`.
    #[inline]
    pub fn to_scalar(&self) -> f32 {
        self.coords[0]
    }

    /// Computes the dot product with another 8D vector.
    ///
    /// For two unit vectors, this is equivalent to their cosine similarity.
    /// This method is backed by a runtime-dispatching SIMD implementation
    /// for maximum performance.
    #[inline(always)]
    pub fn dot(&self, other: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            self::simd::dot_product(self.coords, *other)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0f32;
            for i in 0..8 {
                sum += self.coords[i] * other[i];
            }
            sum
        }
    }

    /// Computes the squared L2 norm. For a valid `Gf8`, this is always `1.0` (or `0.0` for zero).
    #[inline]
    pub fn norm2(&self) -> f32 {
        #[cfg(feature = "simd")]
        {
            self::simd::norm2_scalar(&self.coords)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.coords.iter().map(|x| x * x).sum::<f32>()
        }
    }

    /// Computes the L2 norm. For a valid `Gf8`, this is always `1.0` (or `0.0` for zero).
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm2().sqrt()
    }

    /// Re-normalizes the `Gf8` in-place to ensure it remains a unit vector.
    /// This is useful after performing arithmetic operations that may alter the magnitude.
    pub fn renormalize(&mut self) {
        let n2 = self.norm2();
        if n2 > 0.0 {
            let inv_norm = 1.0 / n2.sqrt();
            for x in &mut self.coords {
                *x *= inv_norm;
            }
        }
    }

    // --- Geometry Operations ---

    /// Spherical geometry: computes geodesic distance to another Gf8 (unit sphere).
    pub fn spherical_distance_to(&self, other: &Self) -> f32 {
        SphericalLayer::distance(self.coords(), other.coords())
    }

    /// Spherical geometry: spherical linear interpolation between two Gf8 values.
    pub fn spherical_slerp(&self, other: &Self, t: f32) -> Self {
        Self::new(SphericalLayer::slerp(self.coords(), other.coords(), t))
    }

    /// Spherical geometry: computes the antipodal (opposite) point on the sphere.
    pub fn spherical_antipodal(&self) -> Self {
        Self {
            coords: SphericalLayer::antipodal(self.coords()),
        }
    }

    /// Hyperbolic geometry: computes distance in the Poincaré ball model.
    /// Since Gf8 coords are on the unit sphere, we map them to the interior ball first.
    pub fn hyperbolic_distance_to(&self, other: &Self) -> f32 {
        // Map sphere coords to ball interior by scaling down
        let self_ball = self.coords.map(|x| x * 0.95);
        let other_ball = other.coords.map(|x| x * 0.95);
        HyperbolicLayer::distance(&self_ball, &other_ball)
    }

    /// Hyperbolic geometry: Möbius addition in Poincaré ball.
    pub fn hyperbolic_mobius_add(&self, other: &Self) -> Self {
        let self_ball = self.coords.map(|x| x * 0.95);
        let other_ball = other.coords.map(|x| x * 0.95);
        let result_ball = HyperbolicLayer::mobius_add(&self_ball, &other_ball);
        // Map back to sphere? Or return as ball coords?
        Self::new(result_ball)
    }

    /// Fisher information geometry: distance with Fisher information matrix.
    pub fn fisher_distance_to(&self, other: &Self, fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        FisherLayer::fisher_distance(self.coords(), other.coords(), fisher_matrix)
    }

    /// Fisher information: uncertainty from Fisher matrix.
    pub fn fisher_uncertainty(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        FisherLayer::uncertainty(fisher_matrix)
    }

    /// Quaternion operations: convert Gf8 to quaternion.
    pub fn to_quaternion(&self) -> [f32; 4] {
        QuaternionOps::from_e8_spinor(self.coords())
    }

    /// Lorentzian geometry: check if other point is in past light cone.
    pub fn lorentzian_in_past_light_cone(&self, other: &Self) -> bool {
        let p1_coords: [f64; 8] = self.coords.map(|x| x as f64);
        let p2_coords: [f64; 8] = other.coords.map(|x| x as f64);
        let p1 = SpacetimePoint::new(p1_coords);
        let p2 = SpacetimePoint::new(p2_coords);

        // Create a temporary LorentzianLayer to check causality
        let layer = LorentzianLayer::new();
        layer.in_past_light_cone(&p1, &p2)
    }

    /// Symplectic geometry: compute Hamiltonian treating Gf8 as position, momentum as zero.
    pub fn symplectic_hamiltonian(&self) -> f32 {
        let zero_momentum = [0.0f32; 8];
        SymplecticLayer::new().hamiltonian(self.coords(), &zero_momentum)
    }
}

impl Gf8 {
    // --- Associated Geometry Functions ---

    /// Spherical geometry: compute Fréchet mean of multiple Gf8 values.
    pub fn spherical_mean(points: &[Self]) -> Self {
        let coords: Vec<[f32; 8]> = points.iter().map(|p| *p.coords()).collect();
        let mean_coords = SphericalLayer::mean(&coords);
        Self::new(mean_coords)
    }

    /// Fisher information: compute Fisher matrix from resonance data.
    pub fn fisher_matrix_from_resonance(resonance: &[u32; 240]) -> [[f32; 8]; 8] {
        FisherLayer::fisher_matrix(resonance)
    }

    /// Quaternion: SLERP between two quaternions derived from Gf8 values.
    pub fn quaternion_slerp_from_gf8(a: &Self, b: &Self, t: f32) -> Self {
        let qa = a.to_quaternion();
        let qb = b.to_quaternion();
        let slerped = QuaternionOps::slerp(&qa, &qb, t);
        // Map quaternion back to Gf8 somehow - this is a bit hacky
        // Use the first 4 quaternion components as coords, pad with 0s, normalize
        let mut coords = [0.0f32; 8];
        coords[0..4].copy_from_slice(&slerped);
        Self::new(coords)
    }
}

impl Gf8Tensor for Gf8 {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        &self.coords
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.coords
    }
}

impl Default for Gf8 {
    /// The default `Gf8` is the zero vector.
    fn default() -> Self {
        Self::ZERO
    }
}

impl Deref for Gf8 {
    type Target = [f32; 8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.coords
    }
}

impl DerefMut for Gf8 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coords
    }
}

/// Geometric addition: performs element-wise vector addition and then
/// re-normalizes the result, projecting it back onto the unit sphere.
impl Add for Gf8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0f32; 8];
        for (i, (&a, &b)) in self.coords.iter().zip(rhs.coords.iter()).enumerate() {
            coords[i] = a + b;
        }
        Self::new(coords)
    }
}

impl AddAssign for Gf8 {
    fn add_assign(&mut self, rhs: Self) {
        for (i, &v) in rhs.coords.iter().enumerate() {
            self.coords[i] += v;
        }
        self.renormalize();
    }
}

/// Geometric subtraction: performs element-wise vector subtraction and then
/// re-normalizes the result, projecting it back onto the unit sphere.
impl Sub for Gf8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0f32; 8];
        for (i, (&a, &b)) in self.coords.iter().zip(rhs.coords.iter()).enumerate() {
            coords[i] = a - b;
        }
        Self::new(coords)
    }
}

impl SubAssign for Gf8 {
    fn sub_assign(&mut self, rhs: Self) {
        for (i, &v) in rhs.coords.iter().enumerate() {
            self.coords[i] -= v;
        }
        self.renormalize();
    }
}

/// Scalar multiplication. The result is re-normalized, so this operation primarily
/// affects the vector's direction (flipping it if the scalar is negative).
impl Mul<f32> for Gf8 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut coords = self.coords;
        for x in &mut coords {
            *x *= rhs;
        }
        Self::new(coords)
    }
}

impl MulAssign<f32> for Gf8 {
    fn mul_assign(&mut self, rhs: f32) {
        for x in &mut self.coords {
            *x *= rhs;
        }
        self.renormalize();
    }
}

/// Negation: flips the direction of the vector. The norm remains unchanged.
impl Neg for Gf8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut coords = self.coords;
        for x in &mut coords {
            *x = -*x;
        }
        Self { coords }
    }
}

/// Embedded SIMD module
#[cfg(feature = "simd")]
mod simd {
    use crate::rune::hydron::intrinsics_for_f32_width;

    // Gate architecture-specific modules.
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    /// Prints a summary of available SIMD capabilities for debugging.
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    /// Performs SIMD-accelerated addition of two raw f32 arrays.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_add_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr());
                    let vb = _mm256_loadu_ps(b.as_ptr());
                    let sum = _mm256_add_ps(va, vb);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), sum);
                    return result;
                }
            }
        }
        // Fallback
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = a[i] + b[i];
        }
        result
    }

    /// Performs SIMD-accelerated subtraction of two raw f32 arrays.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_sub_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr());
                    let vb = _mm256_loadu_ps(b.as_ptr());
                    let diff = _mm256_sub_ps(va, vb);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), diff);
                    return result;
                }
            }
        }
        // Fallback
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = a[i] - b[i];
        }
        result
    }

    /// Computes the dot product of two raw f32 arrays using SIMD.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_dot_simd(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        dot_product(*a, *b)
    }

    /// Computes the squared L2 norm of a raw f32 array using SIMD.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_norm2_simd(a: &[f32; 8]) -> f32 {
        dot_product(*a, *a)
    }

    /// Performs SIMD-accelerated in-place addition for raw f32 arrays: `dst[i] += src[i]`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_add_inplace_slice_simd(dst: &mut [f32; 8], src: &[f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let vdst = _mm256_loadu_ps(dst.as_ptr());
                    let vsrc = _mm256_loadu_ps(src.as_ptr());
                    let sum = _mm256_add_ps(vdst, vsrc);
                    _mm256_storeu_ps(dst.as_mut_ptr(), sum);
                }
                return;
            }
        }
        // Fallback to scalar
        for i in 0..8 {
            dst[i] += src[i];
        }
    }

    /// SIMD-accelerated matrix-vector multiplication for raw arrays.
    #[cfg(feature = "simd")]
    pub fn gf8_matvec_simd(mat: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                return matvec_simd_avx(mat, vec);
            }
        }
        // Fallback to using dot product
        let mut result = [0.0f32; 8];
        for (i, row) in mat.iter().enumerate() {
            result[i] = dot_product(*row, *vec);
        }
        result
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

    /// Scalar norm2 implementation for small arrays where SIMD overhead is not worth it.
    #[inline]
    pub fn norm2_scalar(a: &[f32; 8]) -> f32 {
        a.iter().map(|x| x * x).sum()
    }

    /// AVX implementation for matrix-vector multiplication.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline]
    #[allow(dead_code)]
    unsafe fn matvec_simd_avx(mat: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
        let mut result = [0.0f32; 8];
        let vvec = unsafe { _mm256_loadu_ps(vec.as_ptr()) };

        // Process each row
        for i in 0..8 {
            let row = &mat[i];
            let vrow = unsafe { _mm256_loadu_ps(row.as_ptr()) };
            let prod = _mm256_dp_ps(vrow, vvec, 0xf1); // Dot product with mask 0xf1
            let lo = _mm256_castps256_ps128(prod);
            let hi = _mm256_extractf128_ps(prod, 1);
            let sum = _mm_add_ss(lo, hi);
            result[i] = _mm_cvtss_f32(sum);
        }

        result
    }
}

// Re-export SIMD functions when feature is enabled
#[cfg(feature = "simd")]
pub use simd::{
    get_available_f32_256_intrinsics, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd,
    gf8_matvec_simd, gf8_norm2_simd, gf8_sub_simd, print_simd_capabilities,
};
