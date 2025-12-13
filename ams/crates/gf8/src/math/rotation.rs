/* e8/gf8/src/math/rotation.rs */
//! 8×8 orthogonal operators for rotating `Gf8` directions.
//!
//! # e8 Primitives – Rotation Module
//!▫~•◦-------------------------------------‣
//!
//! This module defines `Gf8Rotation`, a small wrapper around an 8×8 matrix
//! intended to act as an orthogonal operator on `Gf8` directions. It is useful
//! for implementing:
//!
//! - Learned or fixed E₈-like rotations,
//! - Reflections (sign-flip diagonals),
//! - Householder-style transforms for semantic flows.
//!
//! The matrix is stored explicitly as `[[f32; 8]; 8]` for simplicity. All
//! construction functions aim to produce orthogonal (or approximately orthogonal)
//! operators suitable for use with `Gf8`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;
#[cfg(feature = "simd")]
use crate::simd;

/// An 8×8 operator for transforming `Gf8` directions.
///
/// The matrix is conceptually applied as:
///
/// ```text
/// y = M * x
/// ```
///
/// where `x` and `y` are 8D vectors. Most constructors aim to produce orthogonal
/// matrices (e.g., identity, diagonal sign flips, Householder reflections), so
/// that the result remains on or near the unit sphere once re-normalized into a
/// `Gf8`.
#[derive(Debug, Clone, Copy)]
pub struct Gf8Rotation {
    mat: [[f32; 8]; 8],
}

impl Gf8Rotation {
    /// Construct the identity rotation (no change).
    pub fn identity() -> Self {
        let mut mat = [[0.0f32; 8]; 8];
        for (i, row) in mat.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self { mat }
    }

    /// Construct a diagonal sign-flip operator from a diagonal vector.
    ///
    /// Each entry should be +1.0 or -1.0 for a perfect reflection. Values
    /// outside of ±1.0 are allowed but will no longer be exactly orthogonal.
    pub fn from_diagonal(diag: [f32; 8]) -> Self {
        let mut mat = [[0.0f32; 8]; 8];
        for (i, &v) in diag.iter().enumerate() {
            mat[i][i] = v;
        }
        Self { mat }
    }

    /// Construct a Householder reflection around the hyperplane with normal `n`.
    ///
    /// This builds:
    ///
    /// ```text
    /// H = I - 2 * (n nᵀ) / (nᵀ n)
    /// ```
    ///
    /// If `n` is all zeros, this returns the identity.
    pub fn from_householder(n: [f32; 8]) -> Self {
        // Compute squared norm of n.
        let mut n2 = 0.0f32;
        for &v in n.iter() {
            n2 += v * v;
        }

        if n2 == 0.0 {
            return Self::identity();
        }

        let inv_n2 = 1.0 / n2;

        let mut mat = [[0.0f32; 8]; 8];
        for (i, row) in mat.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let delta = if i == j { 1.0 } else { 0.0 };
                *v = delta - 2.0 * n[i] * n[j] * inv_n2;
            }
        }

        Self { mat }
    }

    /// Apply this rotation to a `Gf8` direction and re-normalize to the unit
    /// sphere.
    ///
    /// The result is returned as a new `Gf8` value.
    #[cfg(feature = "simd")]
    pub fn apply(&self, v: &Gf8) -> Gf8 {
        // Compile-time enabled SIMD path. This will use the most optimized
        // implementation available for the target architecture.
        simd::gf8_matvec_simd(&self.mat, v)
    }

    #[cfg(not(feature = "simd"))]
    pub fn apply(&self, v: &Gf8) -> Gf8 {
        // Scalar fallback: explicit matrix-vector multiply with no runtime
        // SIMD dispatch. This path is selected when the `simd` feature is
        // disabled at compile-time.
        let x = v.coords();
        let mut y = [0.0f32; 8];

        for (i, row) in self.mat.iter().enumerate() {
            y[i] = row.iter().zip(x.iter()).map(|(&m, &v)| m * v).sum();
        }

        Gf8::from_coords(y)
    }

    /// Checks whether this matrix is orthogonal within a relative tolerance.
    ///
    /// This verifies that Mᵀ * M ≈ I (i.e., rows are mutually orthonormal).
    /// Useful for unit tests and debug assertions in high-assurance paths.
    pub fn is_orthogonal(&self, eps: f32) -> bool {
        // Compute Mᵀ * M
        let mut mm = [[0.0f32; 8]; 8];
        for (i, row) in mm.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                // Dot product of column i and column j of M.
                let acc: f32 = self.mat.iter().map(|r| r[i] * r[j]).sum();
                *v = acc;
            }
        }

        for (i, row) in mm.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let ideal = if i == j { 1.0f32 } else { 0.0f32 };
                if (v - ideal).abs() > eps {
                    return false;
                }
            }
        }

        true
    }

    /// Compose two rotations: `self ∘ other`.
    ///
    /// The resulting operator applies `other` first, then `self`.
    pub fn compose(&self, other: &Gf8Rotation) -> Gf8Rotation {
        let mut mat = [[0.0f32; 8]; 8];

        for (i, row) in mat.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, _) in self.mat[i].iter().enumerate() {
                    acc += self.mat[i][k] * other.mat[k][j];
                }
                *v = acc;
            }
        }

        Gf8Rotation { mat }
    }

    /// Access the underlying matrix.
    #[inline]
    pub fn as_matrix(&self) -> &[[f32; 8]; 8] {
        &self.mat
    }

    /// Mutable access to the underlying matrix.
    ///
    /// This allows advanced callers to construct custom operators directly.
    #[inline]
    pub fn as_matrix_mut(&mut self) -> &mut [[f32; 8]; 8] {
        &mut self.mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32; 8], b: &[f32; 8], eps: f32) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(&av, &bv)| (av - bv).abs() <= eps)
    }

    #[test]
    fn identity_apply_is_noop() {
        let r = Gf8Rotation::identity();
        let v = Gf8::from_scalar(1.0);
        let out = r.apply(&v);
        assert!(approx_eq(out.coords(), v.coords(), 1e-6));
    }

    #[test]
    fn compose_with_identity_is_idempotent() {
        let r = Gf8Rotation::from_diagonal([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let comp = r.compose(&Gf8Rotation::identity());
        for (row_comp, row_r) in comp.as_matrix().iter().zip(r.as_matrix().iter()) {
            assert!(approx_eq(row_comp, row_r, 1e-6));
        }
    }

    #[test]
    fn householder_reflects_first_axis() {
        // Householder that reflects across the hyperplane orthogonal to the x-axis.
        let n = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let h = Gf8Rotation::from_householder(n);
        let v = Gf8::from_scalar(1.0);
        let out = h.apply(&v);
        // The reflection should flip the first coordinate.
        assert!(out.coords()[0] < -0.9999);
        // Other coordinates remain near zero.
        for &val in out.coords()[1..].iter() {
            assert!(val.abs() < 1e-6);
        }
        // Verify that the operator is orthogonal.
        assert!(h.is_orthogonal(1e-5));
    }
}
