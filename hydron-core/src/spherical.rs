//! Spherical S7 Geometry Layer - Unit 7-Sphere
//!
//! Implements spherical geometry on the unit 7-sphere in 8-dimensional space.
//! All points satisfy the constraint ||x|| = 1.
//!
//! Key operations:
//! - Projection to unit sphere
//! - Geodesic distance using arccos
//! - SLERP (Spherical Linear Interpolation)
//! - Antipodal points and mean computation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Spherical S7 layer - unit 7-sphere in 8D
pub struct SphericalLayer;

impl SphericalLayer {
    /// Project coordinates onto the unit 7-sphere (||x|| = 1)
    pub fn project(coords: &[f32; 8]) -> [f32; 8] {
        let norm_sq: f32 = coords.iter().map(|x| x * x).sum();

        if norm_sq < 1e-8 {
            // If at origin, project to arbitrary point on sphere
            let mut result = [0.0f32; 8];
            result[0] = 1.0;
            return result;
        }

        let norm = norm_sq.sqrt();
        coords.map(|x| x / norm)
    }

    /// Compute geodesic distance on S7 using arccos
    /// d(x, y) = arccos(x · y) for unit vectors
    pub fn distance(x: &[f32; 8], y: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_dot_simd;
            let dot = gf8_dot_simd(x, y).clamp(-1.0, 1.0);
            dot.acos()
        }
        #[cfg(not(feature = "simd"))]
        {
            let dot: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
            let dot_clamped = dot.clamp(-1.0, 1.0);
            dot_clamped.acos()
        }
    }

    /// Spherical linear interpolation (SLERP)
    /// Smoothly interpolate along great circle between x and y
    /// t ∈ [0, 1]
    pub fn slerp(x: &[f32; 8], y: &[f32; 8], t: f32) -> [f32; 8] {
        let t_clamped = t.clamp(0.0, 1.0);

        // Compute angle between vectors
        #[cfg(feature = "simd")]
        let dot = {
            use super::gf8::gf8_dot_simd;
            gf8_dot_simd(x, y)
        };
        #[cfg(not(feature = "simd"))]
        let dot: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        let dot_clamped = dot.clamp(-1.0, 1.0);

        // If vectors are very close, use linear interpolation
        if dot_clamped > 0.9995 {
            let mut result = [0.0f32; 8];
            for i in 0..8 {
                result[i] = x[i] + t_clamped * (y[i] - x[i]);
            }
            return Self::project(&result); // Renormalize
        }

        let theta = dot_clamped.acos();
        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-8 {
            return *x; // Degenerate case
        }

        // SLERP formula: (sin((1-t)*θ) * x + sin(t*θ) * y) / sin(θ)
        let scale1 = ((1.0 - t_clamped) * theta).sin() / sin_theta;
        let scale2 = (t_clamped * theta).sin() / sin_theta;

        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = scale1 * x[i] + scale2 * y[i];
        }

        result
    }

    /// Compute normalized entropy from probability distribution
    /// Returns value in [0, 1] where 0 = uniform distribution, 1 = concentrated
    /// Optimized to avoid vector allocation and compute in single pass
    pub fn normalized_entropy(distribution: &[f32]) -> f32 {
        let sum: f32 = distribution.iter().sum();

        if sum < 1e-8 {
            return 0.0; // No information
        }

        let n = distribution.len() as f32;
        if n < 1e-8 {
            return 0.0;
        }

        // Compute entropy in single pass without intermediate vector allocation
        let mut entropy = 0.0;
        for &x in distribution {
            if x > 1e-8 {
                let p = x / sum;
                entropy -= p * p.ln();
            }
        }

        // Max entropy for n elements is ln(n)
        let max_entropy = n.ln();

        // Return normalized entropy (0 = uniform, 1 = concentrated)
        1.0 - (entropy / max_entropy)
    }

    /// Get antipodal point (opposite point on sphere)
    pub fn antipodal(x: &[f32; 8]) -> [f32; 8] {
        x.map(|xi| -xi)
    }

    /// Compute Fréchet mean (average on sphere) for multiple points
    /// Uses iterative optimization
    pub fn mean(points: &[[f32; 8]]) -> [f32; 8] {
        if points.is_empty() {
            return Self::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        if points.len() == 1 {
            return points[0];
        }

        // Start with Euclidean mean, then project
        let mut mean = [0.0f32; 8];
        for point in points {
            for i in 0..8 {
                mean[i] += point[i];
            }
        }

        Self::project(&mean)
    }

    /// Compute geodesic from point in direction (tangent vector)
    /// Returns point at distance t along geodesic
    pub fn geodesic(point: &[f32; 8], direction: &[f32; 8], t: f32) -> [f32; 8] {
        // Project direction to tangent space (remove component parallel to point)
        let dot: f32 = point
            .iter()
            .zip(direction.iter())
            .map(|(pi, di)| pi * di)
            .sum();

        let mut tangent = [0.0f32; 8];
        for i in 0..8 {
            tangent[i] = direction[i] - dot * point[i];
        }

        // Normalize tangent
        let tangent_norm_sq: f32 = tangent.iter().map(|x| x * x).sum();
        if tangent_norm_sq < 1e-8 {
            return *point; // No movement
        }

        let tangent_norm = tangent_norm_sq.sqrt();
        for t in &mut tangent {
            *t /= tangent_norm;
        }

        // Geodesic: point * cos(t) + tangent * sin(t)
        let mut result = [0.0f32; 8];
        let cos_t = t.cos();
        let sin_t = t.sin();
        for i in 0..8 {
            result[i] = point[i] * cos_t + tangent[i] * sin_t;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_project() {
        let coords = [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let projected = SphericalLayer::project(&coords);

        // Check that result is on unit sphere
        let norm_sq: f32 = projected.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "Point should be on unit sphere"
        );
    }

    #[test]
    fn test_spherical_distance() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let dist = SphericalLayer::distance(&x, &y);

        // Should be π/2 (90 degrees)
        use std::f32::consts::FRAC_PI_2;
        assert!((dist - FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_spherical_slerp() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // At t=0, should be x
        let interp_0 = SphericalLayer::slerp(&x, &y, 0.0);
        for i in 0..8 {
            assert!((interp_0[i] - x[i]).abs() < 1e-6);
        }

        // At t=1, should be y
        let interp_1 = SphericalLayer::slerp(&x, &y, 1.0);
        for i in 0..8 {
            assert!((interp_1[i] - y[i]).abs() < 1e-5);
        }

        // Midpoint should be on unit sphere
        let mid = SphericalLayer::slerp(&x, &y, 0.5);
        let norm_sq: f32 = mid.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spherical_normalized_entropy() {
        // Uniform distribution has high entropy (returns ~0)
        let uniform = vec![1.0; 10];
        let entropy_uniform = SphericalLayer::normalized_entropy(&uniform);
        assert!(entropy_uniform < 0.1);

        // Peaked distribution has low entropy (returns ~1)
        let peaked = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let entropy_peaked = SphericalLayer::normalized_entropy(&peaked);
        assert!(entropy_peaked > 0.8);
    }

    #[test]
    fn test_spherical_antipodal() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let antipodal = SphericalLayer::antipodal(&x);

        // Distance should be π (180 degrees)
        use std::f32::consts::PI;
        let dist = SphericalLayer::distance(&x, &antipodal);
        assert!((dist - PI).abs() < 1e-5);
    }

    #[test]
    fn test_spherical_mean() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let mean = SphericalLayer::mean(&[x, y]);

        // Mean should be on unit sphere
        let norm_sq: f32 = mean.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }
}
