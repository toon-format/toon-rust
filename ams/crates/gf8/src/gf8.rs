/* e8/gf8/src/gf8.rs */
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
//! use gf8::{Gf8, Gf8Tensor};
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

use crate::simd;
use std::{
    convert::TryFrom,
    fmt,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Neg, Sub, SubAssign},
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
#[derive(Clone, Copy, Debug)]
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
    /// The resulting `Gf8` will be `[x, 0.0, ..., 0.0]` and then renormalized.
    /// This provides a simple way to represent scalar magnitudes directionally.
    pub fn from_scalar(x: f32) -> Self {
        let mut coords = [0.0; 8];
        coords[0] = x;
        Self::new(coords)
    }

    /// Constructs a `Gf8` from a shared slice reference.
    ///
    /// This is a zero-allocation, bounds-checked constructor: it copies exactly
    /// 8 elements from the slice into the internal storage and then normalizes.
    ///
    /// Returns `None` if the slice length is not exactly 8.
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() != 8 {
            return None;
        }
        let mut coords = [0.0f32; 8];
        coords.copy_from_slice(slice);
        Some(Self::new(coords))
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
        simd::dot_product(self.coords, *other)
    }

    /// Computes the squared L2 norm. For a valid `Gf8`, this is always `1.0` (or `0.0` for zero).
    #[inline]
    pub fn norm2(&self) -> f32 {
        self.coords.iter().map(|&x| x * x).sum()
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

/// Zero-copy style references into the internal coordinates.
///
/// These mirror the "drop-in" ergonomics of fraction/decimal crates by allowing
/// `Gf8` to participate naturally in APIs that expect slices.
impl AsRef<[f32]> for Gf8 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.coords
    }
}

impl AsMut<[f32]> for Gf8 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.coords
    }
}

/// Conversions between scalars, arrays, and `Gf8`.
///
/// These provide the same kind of "just use it like a float" feel that the
/// `fraction` crate exposes for lossless numerics, but specialized for your
/// 8D geometric representation.
impl From<[f32; 8]> for Gf8 {
    #[inline]
    fn from(coords: [f32; 8]) -> Self {
        Self::new(coords)
    }
}

impl From<f32> for Gf8 {
    #[inline]
    fn from(x: f32) -> Self {
        Self::from_scalar(x)
    }
}

impl From<Gf8> for [f32; 8] {
    #[inline]
    fn from(v: Gf8) -> [f32; 8] {
        v.coords
    }
}

impl From<Gf8> for f32 {
    #[inline]
    fn from(v: Gf8) -> f32 {
        v.to_scalar()
    }
}

impl TryFrom<&[f32]> for Gf8 {
    type Error = &'static str;

    #[inline]
    fn try_from(slice: &[f32]) -> Result<Self, Self::Error> {
        Gf8::from_slice(slice).ok_or("Gf8::try_from(&[f32]): expected slice of length 8")
    }
}

impl PartialEq for Gf8 {
    fn eq(&self, other: &Self) -> bool {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for Gf8 {}

impl PartialOrd for Gf8 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Gf8 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for (a, b) in self.coords.iter().zip(other.coords.iter()) {
            let cmp = a.to_bits().cmp(&b.to_bits());
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl std::hash::Hash for Gf8 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for c in &self.coords {
            state.write_u32(c.to_bits());
        }
    }
}

/// Human-readable formatting for debugging and logging.
///
/// Example:
/// ```rust
/// use gf8::Gf8;
/// let v = Gf8::from_scalar(0.5);
/// println!("{}", v); // Gf8[0.50, 0.00, ..., 0.00]
/// ```
impl fmt::Display for Gf8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gf8[{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            self.coords[0],
            self.coords[1],
            self.coords[2],
            self.coords[3],
            self.coords[4],
            self.coords[5],
            self.coords[6],
            self.coords[7],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn order_is_lexicographic_on_bits() {
        let a = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Gf8::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(b < a, "lex order should compare first differing coord via bits");
    }

    #[test]
    fn hash_and_eq_are_stable() {
        let v1 = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v2 = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let mut set = HashSet::new();
        assert!(set.insert(v1));
        assert!(!set.insert(v2), "bit-identical coords should hash equal");
    }

    #[test]
    fn collections_accept_gf8_as_key() {
        let v1 = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v2 = Gf8::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut tree = BTreeSet::new();
        tree.insert(v1);
        tree.insert(v2);
        assert_eq!(tree.len(), 2);
    }
}
