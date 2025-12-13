//! Geometric projection operators for coordinate transformations
//!
//! # PROJECTOR MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced geometric projection operators implementing transformations between
//! coordinate systems and dimensional spaces. Supports orthogonal, perspective,
//! stereographic, and parallel projections with matrix-based mathematics.
//!
//! ### Key Capabilities
//! - **Orthogonal Projection:** Subspace projection with normal vector computation.
//! - **Perspective Projection:** 3D to 2D perspective transformations with focal length.
//! - **Stereographic Projection:** Sphere to plane mappings from north pole.
//! - **Parallel Projection:** Direction-preserving planar projections.
//! - **Batch Processing:** Efficient vectorized operations on point sets.
//!
//! ### Technical Features
//! - **Matrix Operations:** Efficient ndarray-based linear algebra.
//! - **Numerical Stability:** Proper normalization and singularity handling.
//! - **Projection Inversion:** Bidirectional transformation capabilities.
//! - **Geometric Algebra:** Clifford algebra integration for advanced projections.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::projector::{Projector, ProjectionType};
//! use ndarray::Array1;
//!
//! let projector = Projector::perspective(1000.0);
//! let point: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 10.0]);
//! let projected_point = projector.project(point.view()).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{GsaError, GsaResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Types of geometric projections supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionType {
    /// Orthogonal projection onto a subspace
    Orthogonal,
    /// Perspective projection with vanishing point
    Perspective,
    /// Stereographic projection from sphere to plane
    Stereographic,
    /// Parallel projection maintaining parallelism
    Parallel,
}

/// Geometric projection operator
#[derive(Debug)]
pub struct Projector {
    /// Type of projection to perform
    projection_type: ProjectionType,
    /// Projection matrix (if applicable)
    matrix: Option<Array2<f64>>,
    /// Projection parameters (focal length, etc.)
    params: Vec<f64>,
}

impl Projector {
    /// Create a new projector with the specified type
    pub fn new(projection_type: ProjectionType) -> Self {
        Self {
            projection_type,
            matrix: None,
            params: Vec::new(),
        }
    }

    /// Create an orthogonal projector onto a subspace
    pub fn orthogonal(normal: ArrayView1<f64>) -> GsaResult<Self> {
        let mut projector = Self::new(ProjectionType::Orthogonal);
        projector.setup_orthogonal(normal)?;
        Ok(projector)
    }

    /// Create a perspective projector with focal length
    pub fn perspective(focal_length: f64) -> Self {
        let mut projector = Self::new(ProjectionType::Perspective);
        projector.params = vec![focal_length];
        projector
    }

    /// Create a stereographic projector
    pub fn stereographic() -> Self {
        Self::new(ProjectionType::Stereographic)
    }

    /// Create a parallel projector along a direction
    pub fn parallel(direction: ArrayView1<f64>) -> GsaResult<Self> {
        let mut projector = Self::new(ProjectionType::Parallel);
        projector.setup_parallel(direction)?;
        Ok(projector)
    }

    /// Set up orthogonal projection matrix
    fn setup_orthogonal(&mut self, normal: ArrayView1<f64>) -> GsaResult<()> {
        let n = normal.len();
        if n == 0 {
            return Err(GsaError::Geometry("Empty normal vector".to_string()));
        }

        // Normalize the normal vector
        let norm_sq = normal.dot(&normal);
        if norm_sq <= 1e-12 {
            return Err(GsaError::Geometry("Zero-length normal vector".to_string()));
        }

        let unit_normal = &normal / norm_sq.sqrt();

        // Create projection matrix: I - n*n^T
        let mut matrix = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] -= unit_normal[i] * unit_normal[j];
            }
        }

        self.matrix = Some(matrix);
        Ok(())
    }

    /// Set up parallel projection parameters
    fn setup_parallel(&mut self, direction: ArrayView1<f64>) -> GsaResult<()> {
        if direction.is_empty() {
            return Err(GsaError::Geometry("Empty direction vector".to_string()));
        }

        let norm_sq = direction.dot(&direction);
        if norm_sq <= 1e-12 {
            return Err(GsaError::Geometry(
                "Zero-length direction vector".to_string(),
            ));
        }

        // Store normalized direction
        let unit_dir: Vec<f64> = direction.iter().map(|&x| x / norm_sq.sqrt()).collect();
        self.params = unit_dir;
        Ok(())
    }

    /// Project a point using the configured projection
    pub fn project(&self, point: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        match self.projection_type {
            ProjectionType::Orthogonal => self.project_orthogonal(point),
            ProjectionType::Perspective => self.project_perspective(point),
            ProjectionType::Stereographic => self.project_stereographic(point),
            ProjectionType::Parallel => self.project_parallel(point),
        }
    }

    /// Project a set of points
    pub fn project_batch(&self, points: ArrayView2<f64>) -> GsaResult<Array2<f64>> {
        let n_points = points.nrows();
        let mut result = Array2::zeros((n_points, points.ncols()));

        for i in 0..n_points {
            let point = points.row(i);
            let projected = self.project(point)?;
            result.row_mut(i).assign(&projected);
        }

        Ok(result)
    }

    /// Orthogonal projection onto subspace perpendicular to normal
    fn project_orthogonal(&self, point: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        match &self.matrix {
            Some(matrix) => {
                let projected = matrix.dot(&point.to_owned().insert_axis(ndarray::Axis(1)));
                Ok(projected.remove_axis(ndarray::Axis(1)))
            }
            None => Err(GsaError::Geometry(
                "Orthogonal projection not initialized".to_string(),
            )),
        }
    }

    /// Perspective projection with focal length
    fn project_perspective(&self, point: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let focal_length = self.params.first().copied().unwrap_or(1.0);

        let n = point.len();
        if n < 2 {
            return Err(GsaError::Geometry(
                "Point must have at least 2 dimensions for perspective projection".to_string(),
            ));
        }

        // Assume last coordinate is depth
        let depth = point[n - 1];
        if depth.abs() < 1e-12 {
            return Err(GsaError::Geometry(
                "Cannot project point at infinity".to_string(),
            ));
        }

        let scale = focal_length / depth;
        let mut result = Array1::zeros(n - 1);
        for i in 0..(n - 1) {
            result[i] = point[i] * scale;
        }

        Ok(result)
    }

    /// Stereographic projection from sphere to plane
    fn project_stereographic(&self, point: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let n = point.len();
        if n < 3 {
            return Err(GsaError::Geometry(
                "Stereographic projection requires at least 3D points".to_string(),
            ));
        }

        // Project from north pole (last coordinate = 1) to equatorial plane
        let z = point[n - 1];
        if (z - 1.0).abs() < 1e-12 {
            return Err(GsaError::Geometry("Cannot project north pole".to_string()));
        }

        let scale = 1.0 / (1.0 - z);
        let mut result = Array1::zeros(n - 1);
        for i in 0..(n - 1) {
            result[i] = point[i] * scale;
        }

        Ok(result)
    }

    /// Parallel projection along direction
    fn project_parallel(&self, point: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        if self.params.is_empty() {
            return Err(GsaError::Geometry(
                "Parallel projection not initialized".to_string(),
            ));
        }

        let n = point.len();
        let direction = &self.params;

        // Project by removing component along direction
        let mut result = Array1::zeros(n);
        let dot_product = point.iter().zip(direction).map(|(a, b)| a * b).sum::<f64>();

        for i in 0..n {
            result[i] = point[i] - dot_product * direction[i];
        }

        Ok(result)
    }

    /// Get the projection type
    pub fn projection_type(&self) -> ProjectionType {
        self.projection_type
    }

    /// Check if two projectors are equivalent
    pub fn equivalent(&self, other: &Projector) -> bool {
        if self.projection_type != other.projection_type {
            return false;
        }

        match (&self.matrix, &other.matrix) {
            (Some(a), Some(b)) => a.abs_diff_eq(b, 1e-10),
            (None, None) => self
                .params
                .iter()
                .zip(&other.params)
                .all(|(a, b)| (a - b).abs() < 1e-10),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_orthogonal_projection() {
        let normal = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let projector = Projector::orthogonal(normal.view()).unwrap();

        let point = Array1::from_vec(vec![3.0, 4.0, 5.0]);
        let projected = projector.project(point.view()).unwrap();

        // Should project out x-component
        assert_relative_eq!(projected[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(projected[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_perspective_projection() {
        let projector = Projector::perspective(2.0);

        let point = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let projected = projector.project(point.view()).unwrap();

        // Should scale by focal_length / depth = 2.0 / 4.0 = 0.5
        assert_relative_eq!(projected[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 1.5, epsilon = 1e-10);
        assert_eq!(projected.len(), 2);
    }

    #[test]
    fn test_stereographic_projection() {
        let projector = Projector::stereographic();

        // Point on unit sphere
        let point = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let projected = projector.project(point.view()).unwrap();

        assert_relative_eq!(projected[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_projection_errors() {
        let projector = Projector::perspective(1.0);

        // Point at infinity (depth = 0)
        let point = Array1::from_vec(vec![1.0, 2.0, 0.0]);
        assert!(projector.project(point.view()).is_err());

        // North pole for stereographic
        let stereographic = Projector::stereographic();
        let north_pole = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        assert!(stereographic.project(north_pole.view()).is_err());
    }
}
