/* src/hyperbolic.rs */
//! Hyperbolic H8 Geometry Layer - Poincaré Ball Model
//!
//! Implements hyperbolic geometry in 8 dimensions using the Poincaré ball model.
//! Points lie inside the unit ball with hyperbolic distance metric.
//!
//! Key operations:
//! - Projection to Poincaré ball interior
//! - Hyperbolic distance using arcosh formula
//! - Möbius addition for hyperbolic translation
//! - Geodesic interpolation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Hyperbolic H8 layer using Poincaré ball model
pub struct HyperbolicLayer;

impl HyperbolicLayer {
    /// Project an E8 coordinate into the Poincaré ball interior
    /// Ensures ||x|| < 1 by normalization
    pub fn project(coords: &[f32; 8]) -> [f32; 8] {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_norm2_simd;
            let norm_sq = gf8_norm2_simd(coords);

            if norm_sq < 1e-8 {
                return [0.0; 8]; // Origin
            }

            let norm = norm_sq.sqrt();

            // Scale to fit in ball with margin
            let scale = if norm >= 0.95 { 0.95 / norm } else { 1.0 };

            coords.map(|x| x * scale)
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_sq: f32 = coords.iter().map(|x| x * x).sum();

            if norm_sq < 1e-8 {
                return [0.0; 8]; // Origin
            }

            let norm = norm_sq.sqrt();

            // Scale to fit in ball with margin
            let scale = if norm >= 0.95 { 0.95 / norm } else { 1.0 };

            coords.map(|x| x * scale)
        }
    }

    /// Compute hyperbolic distance in Poincaré ball
    /// d_H(x, y) = arcosh(1 + 2||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
    pub fn distance(x: &[f32; 8], y: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            use super::gf8::{gf8_norm2_simd, gf8_sub_simd};
            let diff = gf8_sub_simd(x, y);
            let diff_sq = gf8_norm2_simd(&diff);
            let x_norm_sq = gf8_norm2_simd(x);
            let y_norm_sq = gf8_norm2_simd(y);

            // Hyperbolic distance formula
            let numerator = 2.0 * diff_sq;
            let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

            if denominator < 1e-8 {
                return f32::INFINITY; // Points on boundary
            }

            let ratio = 1.0 + numerator / denominator;
            ratio.max(1.0).acosh()
        }
        #[cfg(not(feature = "simd"))]
        {
            // Optimized scalar computation - avoid intermediate powi(2)
            let mut diff_sq = 0.0;
            let mut x_norm_sq = 0.0;
            let mut y_norm_sq = 0.0;

            for i in 0..8 {
                let diff = x[i] - y[i];
                diff_sq += diff * diff;
                x_norm_sq += x[i] * x[i];
                y_norm_sq += y[i] * y[i];
            }

            // Hyperbolic distance formula
            let numerator = 2.0 * diff_sq;
            let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

            if denominator < 1e-8 {
                return f32::INFINITY; // Points on boundary
            }

            let ratio = 1.0 + numerator / denominator;
            ratio.max(1.0).acosh()
        }
    }

    /// Möbius addition: a ⊕ b (hyperbolic translation)
    /// a ⊕ b = ((1 + 2⟨a,b⟩ + ||b||²)a + (1 - ||a||²)b) / (1 + 2⟨a,b⟩ + ||a||²||b||²)
    pub fn mobius_add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(feature = "simd")]
        let (a_norm_sq, b_norm_sq, dot_ab) = {
            use super::gf8::{gf8_dot_simd, gf8_norm2_simd};
            (gf8_norm2_simd(a), gf8_norm2_simd(b), gf8_dot_simd(a, b))
        };
        #[cfg(not(feature = "simd"))]
        let (a_norm_sq, b_norm_sq, dot_ab) = {
            let a_norm_sq: f32 = a.iter().map(|x| x * x).sum();
            let b_norm_sq: f32 = b.iter().map(|x| x * x).sum();
            let dot_ab: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
            (a_norm_sq, b_norm_sq, dot_ab)
        };

        let numerator_a_coeff = 1.0 + 2.0 * dot_ab + b_norm_sq;
        let numerator_b_coeff = 1.0 - a_norm_sq;
        let denominator = 1.0 + 2.0 * dot_ab + a_norm_sq * b_norm_sq;

        if denominator.abs() < 1e-8 {
            return [0.0; 8]; // Degenerate case
        }

        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = (numerator_a_coeff * a[i] + numerator_b_coeff * b[i]) / denominator;
        }

        result
    }

    /// Compute hyperbolic distance from origin
    pub fn norm(x: &[f32; 8]) -> f32 {
        let origin = [0.0f32; 8];
        Self::distance(&origin, x)
    }

    /// Interpolate along hyperbolic geodesic between two points
    /// Uses parameter t ∈ [0, 1]
    pub fn interpolate(x: &[f32; 8], y: &[f32; 8], t: f32) -> [f32; 8] {
        let t_clamped = t.clamp(0.0, 1.0);

        // Hyperbolic geodesic: use Möbius addition with scaling
        // First translate to make x the origin, interpolate, then translate back
        let neg_x = x.map(|xi| -xi);

        // Translate y to origin frame: y' = (-x) ⊕ y
        let y_translated = Self::mobius_add(&neg_x, y);

        // Scale y' by t
        let y_scaled = y_translated.map(|yi| yi * t_clamped);

        // Translate back: x ⊕ (t * y')
        Self::mobius_add(x, &y_scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_project() {
        // Test projection keeps points inside ball
        let coords = [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let projected = HyperbolicLayer::project(&coords);

        let norm_sq: f32 = projected.iter().map(|x| x * x).sum();
        assert!(norm_sq < 1.0, "Projected point should be inside unit ball");
    }

    #[test]
    fn test_hyperbolic_distance() {
        // Distance from origin
        let x = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let origin = [0.0; 8];

        let dist = HyperbolicLayer::distance(&origin, &x);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_hyperbolic_distance_symmetry() {
        let x = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let dist_xy = HyperbolicLayer::distance(&x, &y);
        let dist_yx = HyperbolicLayer::distance(&y, &x);

        assert!(
            (dist_xy - dist_yx).abs() < 1e-6,
            "Distance should be symmetric"
        );
    }

    #[test]
    fn test_mobius_add_identity() {
        let x = [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let origin = [0.0; 8];

        // x ⊕ 0 = x
        let result = HyperbolicLayer::mobius_add(&x, &origin);
        for i in 0..8 {
            assert!((result[i] - x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hyperbolic_interpolate() {
        let x = [0.0; 8];
        let y = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // At t=0, should be x
        let interp_0 = HyperbolicLayer::interpolate(&x, &y, 0.0);
        for i in 0..8 {
            assert!((interp_0[i] - x[i]).abs() < 1e-6);
        }

        // At t=1, should be y
        let interp_1 = HyperbolicLayer::interpolate(&x, &y, 1.0);
        for i in 0..8 {
            assert!((interp_1[i] - y[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_hyperbolic_norm() {
        let x = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let norm = HyperbolicLayer::norm(&x);

        assert!(norm > 0.0);
        assert!(norm.is_finite());

        // Norm at origin should be 0
        let origin = [0.0; 8];
        let norm_origin = HyperbolicLayer::norm(&origin);
        assert!(norm_origin.abs() < 1e-6);
    }
}
