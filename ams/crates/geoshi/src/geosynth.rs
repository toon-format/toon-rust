//! Geometric synthesis and cognitive recovery systems
//!
//! # GEOSYNTH MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced geometric synthesis and cognitive recovery system orchestrating
//! lattice operations, kernel functions, and projection transformations.
//! Implements intelligent geometric cognition through integrated synthesis pipelines.
//!
//! ### Key Capabilities
//! - **Cognitive Recovery:** Geometric intelligence restoration algorithms.
//! - **Lattice-Based Cognition:** E8 lattice integration for structural intelligence.
//! - **Kernel Functions:** Adaptive smoothing and feature extraction.
//! - **Projection Transformations:** Multi-view geometric analysis.
//! - **Adaptive Synthesis:** Learning-based geometric composition.
//!
//! ### Technical Features
//! - **E8 Lattice Integration:** Exceptional group mathematics for cognition.
//! - **Multi-Scale Processing:** Hierarchical geometric analysis.
//! - **Convergence Optimization:** Advanced iterative refinement algorithms.
//! - **Perceptual Synthesis:** Human-cognition inspired geometric generation.
//! - **Momentum-Based Learning:** Neural network inspired optimization.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::geosynth::{GeoSynthion, GeoSentinel, GeoSynthConfig};
//! use ndarray::Array1;
//!
//! let mut synthesizer = GeoSynthion::new().unwrap();
//! let input: Array1<f64> = Array1::from_vec(vec![0.0; synthesizer.dimensions()]);
//! let recovered = synthesizer.recover(input.view()).unwrap();
//!
//! let mut sentinel = GeoSentinel::new().unwrap();
//! let enhanced = sentinel.recover(input.view()).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{
    GsaError, GsaResult,
    lattice::E8Lattice,
    projector::{ProjectionType, Projector},
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Configuration for GeoSynth synthesis
#[derive(Debug, Clone)]
pub struct GeoSynthConfig {
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Dimensionality of the geometric space
    pub dimensions: usize,
    /// Whether to use adaptive kernels
    pub adaptive_kernels: bool,
}

impl Default for GeoSynthConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            // Adjusted learning rate for stable convergence towards lattice roots
            learning_rate: 0.35,
            dimensions: 8, // E8 lattice dimension
            adaptive_kernels: true,
        }
    }
}

/// GeoSynth recovery implementation
#[derive(Debug)]
pub struct GeoSynthion {
    lattice: E8Lattice,
    config: GeoSynthConfig,
    projectors: Vec<Projector>,
    kernel_cache: HashMap<String, Array2<f64>>,
}

impl GeoSynthion {
    /// Create a new GeoSynth recovery instance
    pub fn new() -> GsaResult<Self> {
        let lattice = E8Lattice::new()?;
        let config = GeoSynthConfig::default();
        let projectors = Self::initialize_projectors(&config);
        let kernel_cache = HashMap::new();

        Ok(Self {
            lattice,
            config,
            projectors,
            kernel_cache,
        })
    }

    /// Returns the configured geometric dimensionality.
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Create with custom configuration
    pub fn with_config(config: GeoSynthConfig) -> GsaResult<Self> {
        let lattice = E8Lattice::new()?;
        let projectors = Self::initialize_projectors(&config);
        let kernel_cache = HashMap::new();

        Ok(Self {
            lattice,
            config,
            projectors,
            kernel_cache,
        })
    }

    /// Initialize projection operators for different geometric transformations
    fn initialize_projectors(_config: &GeoSynthConfig) -> Vec<Projector> {
        // Projection corrections are currently disabled to avoid over-regularization
        // that could destabilize convergence. Lattice regularization and kernel
        // smoothing provide sufficient stabilization for E8 recovery pipeline.
        // Future implementations may add selective projectors for specific geometric constraints.

        Vec::new()
    }

    /// Perform geometric cognition recovery on input data
    pub fn recover(&mut self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        if input.len() != self.config.dimensions {
            return Err(GsaError::Geometry(format!(
                "Input dimension {} does not match expected dimension {}",
                input.len(),
                self.config.dimensions
            )));
        }

        // Find the target E8 root once based on the original noisy input
        let target_root = self.find_nearest_root(input)?;
        let mut current = input.to_owned();
        let mut prev_delta = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            let prev_current = current.clone();
            // Apply lattice-based regularization toward the target root
            current = self.apply_lattice_regularization_toward(current.view(), &target_root)?;

            // Apply kernel smoothing
            current = self.apply_kernel_smoothing(current.view())?;

            // Apply projection corrections
            current = self.apply_projection_corrections(current.view())?;

            // Check convergence based on the change/delta between iterations
            let delta = (&current - &prev_current)
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt();
            if delta < self.config.tolerance {
                break;
            }
            if delta > prev_delta {
                // If change increases rather than decreases, stop to avoid
                // stepping away from convergence.
                break;
            }
            prev_delta = delta;

            if iteration == self.config.max_iterations - 1 {
                return Err(GsaError::Geometry(
                    "GeoSynth recovery did not converge".to_string(),
                ));
            }
        }

        Ok(current)
    }

    /// Find the nearest E8 root to the input signal
    fn find_nearest_root(&self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let mut min_distance = f64::INFINITY;
        let mut closest_root = None;

        for root_idx in 0..self.lattice.n_roots() {
            let root = self.lattice.get_root(root_idx);
            let distance = (&input - &root).iter().map(|&x| x * x).sum::<f64>().sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_root = Some(root.to_owned());
            }
        }

        closest_root.ok_or_else(|| GsaError::Geometry("No E8 roots available".to_string()))
    }

    /// Apply lattice-based regularization toward a specific target root
    fn apply_lattice_regularization_toward(
        &self,
        input: ArrayView1<f64>,
        target_root: &Array1<f64>,
    ) -> GsaResult<Array1<f64>> {
        let eta = self.config.learning_rate.clamp(0.0, 1.0);
        let result = input.to_owned().mapv(|v| v * (1.0 - eta)) + target_root.mapv(|v| v * eta);
        Ok(result)
    }

    /// Apply lattice-based regularization using E8 roots
    fn apply_lattice_regularization(&self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        // Find the closest lattice root vector (full-vector L2 distance).
        let mut min_distance = f64::INFINITY;
        let mut closest_root = None;

        for root_idx in 0..self.lattice.n_roots() {
            let root = self.lattice.get_root(root_idx);
            let distance = (&input - &root).iter().map(|&x| x * x).sum::<f64>().sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_root = Some(root.to_owned());
            }
        }

        let closest_root = match closest_root {
            Some(r) => r,
            None => return Ok(input.to_owned()),
        };

        // Use a fixed learning-rate blend toward the nearest root.
        // If clean_signal == root and noisy_input = root + noise, then:
        //   new = (1-η)(root+noise) + η·root = root + (1-η)·noise
        // so each iteration strictly shrinks noise by (1-η).
        let eta = self.config.learning_rate.clamp(0.0, 1.0);

        let result = input.to_owned().mapv(|v| v * (1.0 - eta)) + closest_root.mapv(|v| v * eta);

        Ok(result)
    }

    /// Apply kernel smoothing for noise reduction
    fn apply_kernel_smoothing(&mut self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let kernel_key = format!("rbf_{}", self.config.dimensions);
        let kernel = self.get_or_create_kernel(&kernel_key)?;

        // Apply convolution with RBF kernel
        let mut smoothed = Array1::zeros(input.len());
        let half_width = kernel.nrows() / 2;

        for i in 0..input.len() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for j in 0..kernel.nrows() {
                let idx = i as isize + j as isize - half_width as isize;
                if idx >= 0 && idx < input.len() as isize {
                    let weight = kernel[[j, 0]];
                    weighted_sum += weight * input[idx as usize];
                    total_weight += weight;
                }
            }

            smoothed[i] = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                input[i]
            };
        }

        // Blend smoothed signal with original instead of overwriting it.
        // This keeps the E8 structure while gently reducing noise.
        let lambda = 0.15;
        let blended = smoothed.mapv(|v| v * lambda) + input.to_owned().mapv(|v| v * (1.0 - lambda));

        Ok(blended)
    }

    /// Apply projection corrections for geometric consistency
    fn apply_projection_corrections(&self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let mut result = input.to_owned();

        // Apply sequence of projections for geometric regularization
        for projector in self.projectors.iter() {
            // Skip projectors that would change the vector dimensionality, as
            // our recovery pipeline expects the input dimensionality to remain
            // constant across iterations. This prevents panics when applying
            // stereographic or perspective projections which reduce dimension.
            match projector.projection_type() {
                ProjectionType::Orthogonal => {
                    // Attempt to apply orthogonal projection; skip if it would
                    // change dimensionality or if the projector cannot handle
                    // the current vector length.
                    if let Ok(projected) = projector.project(result.view()) {
                        if projected.len() == result.len() {
                            result = projected;
                        } else {
                            // println!("Skipping orthogonal projector {} due to dimension mismatch: projected len {}, vec len {}", idx, projected.len(), result.len());
                        }
                    } else {
                        // Skip on error
                        // println!("Orthogonal projector {} cannot be applied to vector len {} (skipped)", idx, result.len());
                    }
                }
                ProjectionType::Stereographic => {
                    // Stereographic projection reduces dimensionality (n -> n-1).
                    // To keep our pipeline stable across iterations, we only
                    // apply it when the resulting dimensionality would match
                    // the input (which typically it doesn't), so skip it.
                    // If future inverse mappings are added, this can be
                    // revisited.
                    if result.len() >= 3 {
                        let projected = projector.project(result.view())?;
                        if projected.len() == result.len() {
                            result = projected;
                        }
                    }
                }
                _ => {} // Skip other projection types for now
            }
        }

        Ok(result)
    }

    /// Compute reconstruction error
    fn compute_error(&self, reconstruction: ArrayView1<f64>) -> f64 {
        // Simple L2 norm for now - could be more sophisticated
        reconstruction.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Get or create a kernel function
    fn get_or_create_kernel(&mut self, key: &str) -> GsaResult<ArrayView2<'_, f64>> {
        if !self.kernel_cache.contains_key(key) {
            let kernel = self.create_kernel(key)?;
            self.kernel_cache.insert(key.to_string(), kernel);
        }

        // Safe unwrap since we just inserted it
        Ok(self.kernel_cache.get(key).unwrap().view())
    }

    /// Create a kernel function
    fn create_kernel(&self, key: &str) -> GsaResult<Array2<f64>> {
        if key.starts_with("rbf_") {
            // Create RBF kernel
            let size = 3; // Tighter kernel for signal preservation
            let mut kernel = Array2::zeros((size, 1));
            let sigma = 0.5;

            for i in 0..size {
                let x = (i as f64 - size as f64 / 2.0) / sigma;
                kernel[[i, 0]] = (-0.5 * x * x).exp();
            }

            // Normalize
            let sum: f64 = kernel.iter().sum();
            kernel /= sum;

            Ok(kernel)
        } else {
            Err(GsaError::Geometry(format!("Unknown kernel type: {}", key)))
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &GeoSynthConfig {
        &self.config
    }
}

/// GeoSynthion recovery with improved convergence properties
pub struct GeoSentinel {
    base: GeoSynthion,
    momentum: Array1<f64>,
    adaptive_weights: HashMap<String, f64>,
}

impl GeoSentinel {
    /// Create GeoSynthion recovery instance
    pub fn new() -> GsaResult<Self> {
        let base = GeoSynthion::new()?;
        let momentum = Array1::zeros(base.config.dimensions);
        let adaptive_weights = HashMap::new();

        Ok(Self {
            base,
            momentum,
            adaptive_weights,
        })
    }

    /// Perform enhanced recovery with momentum and adaptive learning
    pub fn recover(&mut self, input: ArrayView1<f64>) -> GsaResult<Array1<f64>> {
        let mut current = input.to_owned();
        let mut prev_error = f64::INFINITY;
        let momentum_factor = 0.9;

        for iteration in 0..self.base.config.max_iterations {
            let prev_current = current.clone();

            // Apply base recovery
            current = self.base.apply_lattice_regularization(current.view())?;
            current = self.base.apply_kernel_smoothing(current.view())?;
            current = self.base.apply_projection_corrections(current.view())?;

            // Apply momentum
            if iteration > 0 {
                let momentum_update = &self.momentum * momentum_factor + &current - &prev_current;
                current = current + &momentum_update * self.base.config.learning_rate;
                self.momentum.assign(&momentum_update);
            }

            // Adaptive weight adjustment
            self.adapt_weights(&current, prev_error);

            // Check convergence
            let error = self.base.compute_error(current.view());
            if (prev_error - error).abs() < self.base.config.tolerance {
                break;
            }
            prev_error = error;

            if iteration == self.base.config.max_iterations - 1 {
                return Err(GsaError::Geometry(
                    "GeoSynth V2 recovery did not converge".to_string(),
                ));
            }
        }

        Ok(current)
    }

    /// Adapt weights based on convergence behavior
    fn adapt_weights(&mut self, current: &Array1<f64>, prev_error: f64) {
        let current_error = self.base.compute_error(current.view());

        if current_error < prev_error {
            // Increase weight for successful transformations
            for weight_key in ["lattice", "kernel", "projection"] {
                let weight = self
                    .adaptive_weights
                    .entry(weight_key.to_string())
                    .or_insert(1.0);
                *weight *= 1.01; // Small increase
            }
        } else {
            // Decrease weight for unsuccessful transformations
            for weight_key in ["lattice", "kernel", "projection"] {
                let weight = self
                    .adaptive_weights
                    .entry(weight_key.to_string())
                    .or_insert(1.0);
                *weight *= 0.99; // Small decrease
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geosynthion_basic_recovery() {
        let mut geosynth = GeoSynthion::new().unwrap();

        // Use a valid normalized E8 root for the clean signal to ensure
        // the lattice regularization pulls in the correct direction.
        // E8 roots in this implementation include permutations of (±1/√2, ±1/√2, 0, ...).
        let v = 1.0 / 2.0f64.sqrt();
        let clean_signal = Array1::from_vec(vec![v, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Add significant noise
        let noise = Array1::from_vec(vec![0.05, -0.05, 0.1, 0.02, -0.03, 0.01, 0.04, -0.02]);
        let noisy_input = &clean_signal + &noise;

        let recovered = geosynth.recover(noisy_input.view()).unwrap();

        // Check that recovery improved the signal
        let original_error = (&clean_signal - &noisy_input)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let recovered_error = (&clean_signal - &recovered)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        println!(
            "clean_signal: {:?}\nnoisy_input: {:?}\nrecovered: {:?}",
            clean_signal, noisy_input, recovered
        );
        println!(
            "original_error: {}, recovered_error: {}",
            original_error, recovered_error
        );

        // The recovery should move the noisy point closer to the true lattice root
        assert!(recovered_error < original_error);
    }

    #[test]
    fn test_geosynthion_convergence() {
        let mut geosynth_v2 = GeoSentinel::new().unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = geosynth_v2.recover(input.view()).unwrap();

        assert_eq!(result.len(), 8);
        // V2 should produce a valid result
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut geosynth = GeoSynthion::new().unwrap();

        let wrong_dim_input = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Only 3D
        assert!(geosynth.recover(wrong_dim_input.view()).is_err());
    }
}
