/* src/quaternion.rs */
//! Quaternion Operations for Phase Transitions
//!
//! Implements quaternion algebra for rotations and phase transitions in E8.
//! Quaternions are used for smooth interpolation (SLERP) and representing
//! rotational symmetries in the E8 lattice.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Quaternion operations
pub struct QuaternionOps;

impl QuaternionOps {
    /// Normalize a quaternion to unit length
    pub fn normalize(q: &[f32; 4]) -> [f32; 4] {
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if norm < 1e-8 {
            [1.0, 0.0, 0.0, 0.0] // Identity quaternion
        } else {
            [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
        }
    }

    /// Multiply two quaternions (non-commutative)
    /// q1 * q2 = [w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2]
    pub fn multiply(q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
        let w1 = q1[0];
        let v1 = [q1[1], q1[2], q1[3]];
        let w2 = q2[0];
        let v2 = [q2[1], q2[2], q2[3]];

        // Scalar part: w1*w2 - v1·v2
        let w = w1 * w2 - (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);

        // Vector part: w1*v2 + w2*v1 + v1×v2
        let i = w1 * v2[0] + w2 * v1[0] + (v1[1] * v2[2] - v1[2] * v2[1]);
        let j = w1 * v2[1] + w2 * v1[1] + (v1[2] * v2[0] - v1[0] * v2[2]);
        let k = w1 * v2[2] + w2 * v1[2] + (v1[0] * v2[1] - v1[1] * v2[0]);

        [w, i, j, k]
    }

    /// Compute quaternion conjugate: q* = [w, -v]
    pub fn conjugate(q: &[f32; 4]) -> [f32; 4] {
        [q[0], -q[1], -q[2], -q[3]]
    }

    /// Compute quaternion inverse: q⁻¹ = q* / |q|²
    pub fn inverse(q: &[f32; 4]) -> [f32; 4] {
        let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if norm_sq < 1e-8 {
            [0.0, 0.0, 0.0, 0.0] // Undefined
        } else {
            let conj = Self::conjugate(q);
            [
                conj[0] / norm_sq,
                conj[1] / norm_sq,
                conj[2] / norm_sq,
                conj[3] / norm_sq,
            ]
        }
    }

    /// Spherical linear interpolation (SLERP)
    /// Smoothly interpolate between q1 and q2 by parameter t ∈ [0, 1]
    pub fn slerp(q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4] {
        let q1_norm = Self::normalize(q1);
        let mut q2_norm = Self::normalize(q2);

        // Compute dot product
        let mut dot = q1_norm[0] * q2_norm[0]
            + q1_norm[1] * q2_norm[1]
            + q1_norm[2] * q2_norm[2]
            + q1_norm[3] * q2_norm[3];

        // If dot is negative, negate one quaternion to take shorter path
        if dot < 0.0 {
            q2_norm = [-q2_norm[0], -q2_norm[1], -q2_norm[2], -q2_norm[3]];
            dot = -dot;
        }

        // Clamp dot to [-1, 1] for numerical stability
        dot = dot.clamp(-1.0, 1.0);

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            let result = [
                q1_norm[0] + t * (q2_norm[0] - q1_norm[0]),
                q1_norm[1] + t * (q2_norm[1] - q1_norm[1]),
                q1_norm[2] + t * (q2_norm[2] - q1_norm[2]),
                q1_norm[3] + t * (q2_norm[3] - q1_norm[3]),
            ];
            return Self::normalize(&result);
        }

        // Compute angle between quaternions
        let theta = dot.acos();
        let sin_theta = theta.sin();

        // SLERP formula: (sin((1-t)*θ) * q1 + sin(t*θ) * q2) / sin(θ)
        let scale1 = ((1.0 - t) * theta).sin() / sin_theta;
        let scale2 = (t * theta).sin() / sin_theta;

        [
            scale1 * q1_norm[0] + scale2 * q2_norm[0],
            scale1 * q1_norm[1] + scale2 * q2_norm[1],
            scale1 * q1_norm[2] + scale2 * q2_norm[2],
            scale1 * q1_norm[3] + scale2 * q2_norm[3],
        ]
    }

    /// Create quaternion from axis-angle representation
    /// q = [cos(θ/2), sin(θ/2) * axis]
    pub fn from_axis_angle(axis: &[f32; 3], angle: f32) -> [f32; 4] {
        let half_angle = angle * 0.5;
        let sin_half = half_angle.sin();

        // Normalize axis
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if axis_norm < 1e-8 {
            return [1.0, 0.0, 0.0, 0.0]; // Identity
        }

        let ax = axis[0] / axis_norm;
        let ay = axis[1] / axis_norm;
        let az = axis[2] / axis_norm;

        [
            half_angle.cos(),
            sin_half * ax,
            sin_half * ay,
            sin_half * az,
        ]
    }

    /// Rotate a 3D vector by a quaternion
    /// v' = q * v * q⁻¹ (treating v as quaternion [0, v])
    pub fn rotate_vector(q: &[f32; 4], v: &[f32; 3]) -> [f32; 3] {
        let q_norm = Self::normalize(q);
        let v_quat = [0.0, v[0], v[1], v[2]];

        // Compute q * v
        let qv = Self::multiply(&q_norm, &v_quat);

        // Compute (q * v) * q⁻¹
        let q_inv = Self::conjugate(&q_norm); // For unit quaternions, conjugate = inverse
        let result = Self::multiply(&qv, &q_inv);

        [result[1], result[2], result[3]]
    }

    /// Quaternion dot product
    pub fn dot(q1: &[f32; 4], q2: &[f32; 4]) -> f32 {
        q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    }

    /// Convert E8 spinor root (indices 112-239) to quaternion
    /// Uses simplified mapping from 8D E8 coordinates to 4D quaternion
    pub fn from_e8_spinor(e8_coords: &[f32; 8]) -> [f32; 4] {
        // Simple projection: map 8D to 4D by averaging pairs
        let w = (e8_coords[0] + e8_coords[1]) * 0.5;
        let i = (e8_coords[2] + e8_coords[3]) * 0.5;
        let j = (e8_coords[4] + e8_coords[5]) * 0.5;
        let k = (e8_coords[6] + e8_coords[7]) * 0.5;

        Self::normalize(&[w, i, j, k])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_normalize() {
        let q = [1.0, 1.0, 1.0, 1.0];
        let q_norm = QuaternionOps::normalize(&q);
        let norm = (q_norm[0] * q_norm[0]
            + q_norm[1] * q_norm[1]
            + q_norm[2] * q_norm[2]
            + q_norm[3] * q_norm[3])
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_multiply() {
        let q1 = [1.0, 0.0, 0.0, 0.0]; // Identity
        let q2 = [0.0, 1.0, 0.0, 0.0]; // i
        let result = QuaternionOps::multiply(&q1, &q2);
        assert_eq!(result, [0.0, 1.0, 0.0, 0.0]);

        // i * i = -1
        let i = [0.0, 1.0, 0.0, 0.0];
        let result = QuaternionOps::multiply(&i, &i);
        assert!((result[0] + 1.0).abs() < 1e-6); // Scalar part should be -1
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = [1.0, 2.0, 3.0, 4.0];
        let conj = QuaternionOps::conjugate(&q);
        assert_eq!(conj, [1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_quaternion_slerp() {
        let q1 = [1.0, 0.0, 0.0, 0.0]; // Identity
        let q2 = [0.0, 1.0, 0.0, 0.0]; // 180° rotation around x-axis

        let mid = QuaternionOps::slerp(&q1, &q2, 0.5);
        // Should be halfway rotation
        let expected = QuaternionOps::normalize(&[0.5, 0.5, 0.0, 0.0]);
        for i in 0..4 {
            assert!((mid[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        use std::f32::consts::PI;

        // 90° rotation around z-axis
        let q = QuaternionOps::from_axis_angle(&[0.0, 0.0, 1.0], PI / 2.0);

        // Rotate vector [1, 0, 0]
        let v = [1.0, 0.0, 0.0];
        let rotated = QuaternionOps::rotate_vector(&q, &v);

        // Should rotate to [0, 1, 0]
        assert!((rotated[0] - 0.0).abs() < 1e-5);
        assert!((rotated[1] - 1.0).abs() < 1e-5);
        assert!((rotated[2] - 0.0).abs() < 1e-5);
    }
}
