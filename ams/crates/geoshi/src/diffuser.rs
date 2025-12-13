//! Diffusion processes for geometric cognition and smoothing
//!
//! # DIFFUSION MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced diffusion algorithms for geometric cognition, implementing heat
//! equations, anisotropic smoothing, and information propagation across geometric
//! structures. Enables gradient-preserving noise reduction and topological smoothing.
//!
//! ### Key Capabilities
//! - **Heat Equation Diffusion:** Classic Laplacian smoothing over geometric domains.
//! - **Anisotropic Diffusion:** Edge-preserving smoothing with conductance functions.
//! - **Weighted Diffusion:** Custom kernel-based information propagation.
//! - **Boundary Conditions:** Zero, Neumann, and periodic boundary handling.
//! - **Graph-Based Diffusion:** Topology-aware information spreading on arbitrary graphs.
//!
//! ### Technical Features
//! - **Convergence Control:** Configurable iteration limits and convergence thresholds.
//! - **Multi-Stencil Support:** 4-point, 9-point, and custom convolution kernels.
//! - **Performance Optimized:** Efficient ndarray operations with zero-copy views.
//! - **Perona-Malik Implementation:** Classic anisotropic diffusion algorithm.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::diffuser::{Diffuser, DiffusionConfig, DiffusionMethod};
//! use ndarray::Array2;
//!
//! let config = DiffusionConfig {
//!     method: DiffusionMethod::Heat,
//!     iterations: 100,
//!     time_step: 0.1,
//!     ..Default::default()
//! };
//!
//! let diffuser = Diffuser::new(config);
//! let values: Array2<f64> = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let result = diffuser.diffuse_grid(values.view()).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use ndarray::{Array2, ArrayView2};

/// Diffusion method for value propagation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffusionMethod {
    /// Simple averaging diffusion
    Average,
    /// Heat equation diffusion
    Heat,
    /// Weighted diffusion with custom weights
    Weighted,
}

/// Diffusion configuration parameters
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub method: DiffusionMethod,
    pub iterations: usize,
    pub time_step: f64,
    pub convergence_threshold: f64,
    pub boundary_conditions: BoundaryCondition,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            method: DiffusionMethod::Heat,
            iterations: 100,
            time_step: 0.1,
            convergence_threshold: 1e-6,
            boundary_conditions: BoundaryCondition::Zero,
        }
    }
}

/// Boundary condition types for diffusion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Zero boundary values
    Zero,
    /// Neumann boundary (zero gradient)
    Neumann,
    /// Periodic boundaries
    Periodic,
}

/// Diffusion solver for geometric structures
pub struct Diffuser {
    config: DiffusionConfig,
}

impl Diffuser {
    /// Create a new diffuser with given configuration
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Diffuse values across a 2D grid using finite differences
    pub fn diffuse_grid(&self, initial_values: ArrayView2<f64>) -> GsaResult<Array2<f64>> {
        let mut current = initial_values.to_owned();
        let mut next = current.clone();

        for _iteration in 0..self.config.iterations {
            let max_change = match self.config.method {
                DiffusionMethod::Heat => self.heat_equation_step(&current, &mut next),
                DiffusionMethod::Average => self.average_diffusion_step(&current, &mut next),
                DiffusionMethod::Weighted => self.weighted_diffusion_step(&current, &mut next),
            }?;

            if max_change < self.config.convergence_threshold {
                break;
            }

            std::mem::swap(&mut current, &mut next);
        }

        Ok(current)
    }

    /// Apply heat equation diffusion step
    fn heat_equation_step(&self, current: &Array2<f64>, next: &mut Array2<f64>) -> GsaResult<f64> {
        let (rows, cols) = current.dim();
        let mut max_change = 0.0f64;

        for i in 0..rows {
            for j in 0..cols {
                let neighbors = self.get_neighbors(current, i, j);
                let center = current[[i, j]];
                let laplacian = self.compute_laplacian(center, &neighbors);

                let new_value = current[[i, j]] + self.config.time_step * laplacian;
                let change = (new_value - current[[i, j]]).abs();
                max_change = max_change.max(change);

                next[[i, j]] = new_value;
            }
        }

        Ok(max_change)
    }

    /// Apply simple averaging diffusion
    fn average_diffusion_step(
        &self,
        current: &Array2<f64>,
        next: &mut Array2<f64>,
    ) -> GsaResult<f64> {
        let (rows, cols) = current.dim();
        let mut max_change = 0.0f64;

        for i in 0..rows {
            for j in 0..cols {
                let neighbors = self.get_neighbors(current, i, j);
                // Include the current cell in the average for stability
                let sum: f64 = neighbors.iter().sum::<f64>() + current[[i, j]];
                let avg = sum / (neighbors.len() as f64 + 1.0);

                let change = (avg - current[[i, j]]).abs();
                max_change = max_change.max(change);

                next[[i, j]] = avg;
            }
        }

        Ok(max_change)
    }

    /// Apply weighted diffusion with custom kernel
    fn weighted_diffusion_step(
        &self,
        current: &Array2<f64>,
        next: &mut Array2<f64>,
    ) -> GsaResult<f64> {
        let kernel = self.gaussian_kernel(3, 1.0);
        let (rows, cols) = current.dim();
        let mut max_change = 0.0f64;

        for i in 0..rows {
            for j in 0..cols {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for ki in 0..kernel.nrows() {
                    for kj in 0..kernel.ncols() {
                        let ni = i as isize + ki as isize - 1;
                        let nj = j as isize + kj as isize - 1;

                        if ni >= 0 && ni < rows as isize && nj >= 0 && nj < cols as isize {
                            let weight = kernel[[ki, kj]];
                            weighted_sum += weight * current[[ni as usize, nj as usize]];
                            weight_sum += weight;
                        }
                    }
                }

                let new_value = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    current[[i, j]]
                };
                let change = (new_value - current[[i, j]]).abs();
                max_change = max_change.max(change);

                next[[i, j]] = new_value;
            }
        }

        Ok(max_change)
    }

    /// Get neighboring values for a grid position
    fn get_neighbors(&self, grid: &Array2<f64>, i: usize, j: usize) -> Vec<f64> {
        let (rows, cols) = grid.dim();
        let mut neighbors = Vec::new();

        let directions = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];

        for (di, dj) in directions {
            let ni = i as isize + di;
            let nj = j as isize + dj;

            match self.config.boundary_conditions {
                BoundaryCondition::Zero => {
                    if ni >= 0 && ni < rows as isize && nj >= 0 && nj < cols as isize {
                        neighbors.push(grid[[ni as usize, nj as usize]]);
                    } else {
                        neighbors.push(0.0);
                    }
                }
                BoundaryCondition::Neumann => {
                    let clamped_i = ni.max(0).min(rows as isize - 1) as usize;
                    let clamped_j = nj.max(0).min(cols as isize - 1) as usize;
                    neighbors.push(grid[[clamped_i, clamped_j]]);
                }
                BoundaryCondition::Periodic => {
                    let wrapped_i = ((ni % rows as isize) + rows as isize) % rows as isize;
                    let wrapped_j = ((nj % cols as isize) + cols as isize) % cols as isize;
                    neighbors.push(grid[[wrapped_i as usize, wrapped_j as usize]]);
                }
            }
        }

        neighbors
    }

    /// Compute Laplacian using finite differences
    fn compute_laplacian(&self, center: f64, neighbors: &[f64]) -> f64 {
        // 4-point stencil: ∇²u ≈ (u_north + u_south + u_east + u_west - 4*u_center) / h²
        // With h=1, this simplifies to sum of neighbors - 4 * center
        // neighbors are ordered: [north, south, east, west]
        let sum_neighbors = neighbors.iter().sum::<f64>();
        sum_neighbors - (neighbors.len() as f64) * center
    }

    /// Generate Gaussian kernel for weighted diffusion
    fn gaussian_kernel(&self, size: usize, sigma: f64) -> Array2<f64> {
        let mut kernel = Array2::zeros((size, size));
        let center = (size / 2) as f64;

        for i in 0..size {
            for j in 0..size {
                let x = i as f64 - center;
                let y = j as f64 - center;
                let distance_sq = x * x + y * y;
                kernel[[i, j]] = (-distance_sq / (2.0 * sigma * sigma)).exp();
            }
        }

        // Normalize
        let sum: f64 = kernel.iter().sum();
        if sum > 0.0 {
            kernel /= sum;
        }

        kernel
    }
}

/// Anisotropic diffusion for edge-preserving smoothing
pub struct AnisotropicDiffuser {
    config: DiffusionConfig,
    conductance_function: fn(f64) -> f64,
}

impl AnisotropicDiffuser {
    /// Create new anisotropic diffuser with Perona-Malik function
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            conductance_function: Self::perona_malik,
        }
    }

    /// Perona-Malik conductance function
    fn perona_malik(gradient: f64) -> f64 {
        (-(gradient * gradient) / (2.0 * 0.01 * 0.01)).exp()
    }

    /// Apply anisotropic diffusion
    pub fn diffuse(&self, initial_values: ArrayView2<f64>) -> GsaResult<Array2<f64>> {
        let mut current = initial_values.to_owned();
        let mut next = current.clone();

        for _iteration in 0..self.config.iterations {
            self.anisotropic_step(&current, &mut next)?;
            std::mem::swap(&mut current, &mut next);
        }

        Ok(current)
    }

    /// Single anisotropic diffusion step
    fn anisotropic_step(&self, current: &Array2<f64>, next: &mut Array2<f64>) -> GsaResult<()> {
        let (rows, cols) = current.dim();

        for i in 0..rows {
            for j in 0..cols {
                let north = if i > 0 {
                    current[[i - 1, j]]
                } else {
                    current[[i, j]]
                };
                let south = if i < rows - 1 {
                    current[[i + 1, j]]
                } else {
                    current[[i, j]]
                };
                let east = if j < cols - 1 {
                    current[[i, j + 1]]
                } else {
                    current[[i, j]]
                };
                let west = if j > 0 {
                    current[[i, j - 1]]
                } else {
                    current[[i, j]]
                };

                let center = current[[i, j]];

                let cn = (self.conductance_function)((center - north).abs());
                let cs = (self.conductance_function)((center - south).abs());
                let ce = (self.conductance_function)((center - east).abs());
                let cw = (self.conductance_function)((center - west).abs());

                next[[i, j]] = center
                    + self.config.time_step
                        * (cn * (north - center)
                            + cs * (south - center)
                            + ce * (east - center)
                            + cw * (west - center));
            }
        }

        Ok(())
    }
}

/// Graph-based diffusion for arbitrary topologies
pub struct GraphDiffuser {
    adjacency: Vec<Vec<usize>>,
    weights: Vec<Vec<f64>>,
}

impl GraphDiffuser {
    /// Create new graph diffuser from adjacency list
    pub fn new(adjacency: Vec<Vec<usize>>, weights: Vec<Vec<f64>>) -> Self {
        Self { adjacency, weights }
    }

    /// Diffuse values across graph nodes
    pub fn diffuse(&self, initial_values: &[f64], iterations: usize) -> GsaResult<Vec<f64>> {
        if initial_values.len() != self.adjacency.len() {
            return Err(crate::GsaError::Geometry(
                "Value array size doesn't match graph size".to_string(),
            ));
        }

        let mut current = initial_values.to_vec();
        let mut next = current.clone();

        for _ in 0..iterations {
            self.graph_diffusion_step(&current, &mut next)?;
            std::mem::swap(&mut current, &mut next);
        }

        Ok(current)
    }

    /// Single graph diffusion step
    fn graph_diffusion_step(&self, current: &[f64], next: &mut [f64]) -> GsaResult<()> {
        for i in 0..self.adjacency.len() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for &neighbor in &self.adjacency[i] {
                let weight = self.weights[i][neighbor];
                weighted_sum += weight * current[neighbor];
                total_weight += weight;
            }

            // include some of the node's own value to stabilize updates and avoid
            // oscillating values on bipartite-like graphs. This mirrors a simple
            // relaxation step (Jacobi with self-weight) where self contribution
            // is weighted by 1.0.
            weighted_sum += current[i];
            total_weight += 1.0;

            next[i] = weighted_sum / total_weight;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_heat_equation_diffusion() {
        let diffuser = Diffuser::new(DiffusionConfig::default());
        let mut initial = Array2::zeros((5, 5));

        // Set center point to 1.0
        initial[[2, 2]] = 1.0;

        let result = diffuser.diffuse_grid(initial.view()).unwrap();

        // Center should still be highest
        assert!(result[[2, 2]] > result[[0, 0]]);
        assert!(result[[2, 2]] > result[[1, 1]]);

        // Values should spread out
        let total: f64 = result.iter().sum();
        // For zero boundary conditions mass is not strictly conserved (it may leak out),
        // but it should still be non-negative and not artificially inflated.
        assert!(total >= 0.0);
        assert!(total <= 1.0);
    }

    #[test]
    fn test_average_diffusion() {
        let config = DiffusionConfig {
            method: DiffusionMethod::Average,
            iterations: 1,
            ..Default::default()
        };
        let diffuser = Diffuser::new(config);

        let mut initial = Array2::zeros((3, 3));
        initial[[1, 1]] = 2.0;

        let result = diffuser.diffuse_grid(initial.view()).unwrap();

        // Center should become average of neighbors plus self (4 neighbors + self = 5)
        // Sum = 2.0. Count = 5. Avg = 0.4
        assert!((result[[1, 1]] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_conditions() {
        let config = DiffusionConfig {
            boundary_conditions: BoundaryCondition::Zero,
            method: DiffusionMethod::Average,
            iterations: 1,
            ..Default::default()
        };
        let diffuser = Diffuser::new(config);

        let mut initial = Array2::zeros((3, 3));
        initial[[0, 0]] = 1.0;

        let result = diffuser.diffuse_grid(initial.view()).unwrap();

        // Corner should become average including zero boundaries and self (4 neighbors + self = 5)
        // Sum = 1.0. Avg = 0.2.
        assert!((result[[0, 0]] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_anisotropic_diffusion() {
        let config = DiffusionConfig {
            iterations: 1,
            ..Default::default()
        };
        let diffuser = AnisotropicDiffuser::new(config);

        let mut initial = Array2::zeros((5, 5));
        initial[[2, 2]] = 1.0;

        let result = diffuser.diffuse(initial.view()).unwrap();

        // Should preserve edges while smoothing
        assert!(result[[2, 2]] > result[[1, 1]]);
    }

    #[test]
    fn test_graph_diffusion() {
        // Simple line graph: 0 -- 1 -- 2
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let weights = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];

        let diffuser = GraphDiffuser::new(adjacency, weights);
        let initial = vec![1.0, 0.0, 0.0];
        // Run 3 iterations to allow propagation to reach node 2
        let result = diffuser.diffuse(&initial, 3).unwrap();

        // Value should diffuse from node 0 to neighbors
        assert!(result[0] < initial[0]); // Should decrease
        assert!(result[1] > initial[1]); // Should increase
        assert!(result[2] > initial[2]); // Should increase slightly
    }
}
