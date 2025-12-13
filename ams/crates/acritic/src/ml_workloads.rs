/* arcmoon-suite/src/ml_workloads.rs */
//! Machine Learning Workload Generators
//!
//! This module provides standardized, statistically relevant ML workload generators
//! for benchmarking machine learning performance. It generates synthetic datasets
//! with controllable statistical properties (distributions, noise, correlations)
//! to simulate realistic training and inference scenarios.
//!
//! ### Capabilities
//! - **Synthetic Data Generation:** Creates verifiable datasets for Classification, Regression, and Time-Series.
//! - **Statistical Control:** Fine-grained control over noise levels, complexity (non-linearity), and uncertainty.
//! - **Memory Optimization:** Uses pre-allocation and efficient iterators to minimize generation overhead.
//!
//! ### Architectural Notes
//! This module is designed to feed the `benchmarks` module. It prioritizes correctness of
//! statistical properties over raw generation speed, ensuring benchmarks run against
//! non-trivial data distributions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Standard ML workload types for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    Classification,
    Regression,
    TimeSeriesForecasting,
    ReinforcementLearning,
    GenerativeModeling,
    UncertaintyQuantification,
}

/// ML workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    pub workload_type: WorkloadType,
    pub input_size: usize,
    pub output_size: usize,
    pub sample_count: usize,
    /// Complexity factor (0.0 to 1.0).
    /// Higher values introduce more non-linearity or dimensionality.
    pub complexity: f64,
    /// Noise variance (0.0 to 1.0).
    pub noise_level: f64,
    /// Aleatoric uncertainty factor (0.0 to 1.0).
    pub uncertainty_level: f64,
    pub seed: u64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            workload_type: WorkloadType::Classification,
            input_size: 100,
            output_size: 10,
            sample_count: 1000,
            complexity: 0.5,
            noise_level: 0.1,
            uncertainty_level: 0.05,
            seed: 42,
        }
    }
}

/// Generated workload data for different ML task types
#[derive(Debug, Clone)]
pub enum WorkloadData {
    Classification {
        inputs: Vec<Vec<f32>>,
        labels: Vec<usize>,
    },
    Regression {
        inputs: Vec<Vec<f32>>,
        targets: Vec<f32>,
    },
    TimeSeriesForecasting {
        sequences: Vec<Vec<f32>>,
        forecasts: Vec<Vec<f32>>,
        horizon: usize,
    },
    ReinforcementLearning {
        states: Vec<Vec<f32>>,
        actions: Vec<f32>,
        rewards: Vec<f32>,
        episode_length: usize,
    },
    GenerativeModeling {
        samples: Vec<Vec<f32>>,
        latent_codes: Vec<Vec<f32>>,
        generation_steps: usize,
    },
    UncertaintyQuantification {
        inputs: Vec<Vec<f32>>,
        targets: Vec<f32>,
        uncertainties: Vec<f32>,
    },
}

/// ML workload generator for standardized benchmarking
pub struct WorkloadGenerator {
    rng: StdRng,
    config: WorkloadConfig,
}

impl WorkloadGenerator {
    pub fn new(config: WorkloadConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self { rng, config }
    }

    /// Generate workload data based on configuration
    pub fn generate_workload(&mut self) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        match self.config.workload_type {
            WorkloadType::Classification => self.generate_classification_workload(),
            WorkloadType::Regression => self.generate_regression_workload(),
            WorkloadType::TimeSeriesForecasting => self.generate_timeseries_workload(),
            WorkloadType::ReinforcementLearning => self.generate_rl_workload(),
            WorkloadType::GenerativeModeling => self.generate_generative_workload(),
            WorkloadType::UncertaintyQuantification => self.generate_uncertainty_workload(),
        }
    }

    /// Helper: Generate a standard normal random variable (Box-Muller transform)
    /// Used to create realistic Gaussian noise without extra dependencies.
    #[inline]
    fn sample_normal(&mut self) -> f64 {
        let u1: f64 = self.rng.random();
        let u2: f64 = self.rng.random();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        r * theta.cos()
    }

    /// Generate reinforcement learning workload (States, Actions, Rewards)
    fn generate_rl_workload(&mut self) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let episode_length = (self.config.sample_count as f64).sqrt() as usize;
        let num_episodes = self.config.sample_count / episode_length.max(1);

        let mut states = Vec::with_capacity(self.config.sample_count);
        let mut actions = Vec::with_capacity(self.config.sample_count);
        let mut rewards = Vec::with_capacity(self.config.sample_count);

        // Simulate episodes with Markov property (next state depends on prev state)
        for _ in 0..num_episodes {
            // Initialize episode state
            let mut current_state_vec: Vec<f32> = (0..self.config.input_size)
                .map(|_| self.rng.random_range(-1.0..1.0))
                .collect();

            for _ in 0..episode_length {
                states.push(current_state_vec.clone());

                // Simple linear policy simulation
                let action_idx =
                    (current_state_vec[0].abs() * self.config.output_size as f32) as usize;
                let action = (action_idx % self.config.output_size) as f32;
                actions.push(action);

                // Reward simulation: Function of state and action
                let base_reward =
                    current_state_vec.iter().sum::<f32>() / (self.config.input_size as f32);
                let noise = self.sample_normal() * self.config.noise_level;
                rewards.push((base_reward + noise as f32).clamp(-1.0, 1.0));

                // State transition: S_next = S_curr + Noise + Drift
                for val in &mut current_state_vec {
                    *val += (self.rng.random_range(-0.1..0.1) * self.config.complexity) as f32;
                    *val = val.clamp(-2.0, 2.0);
                }
                // Simple linear policy simulation
                let action_idx =
                    (current_state_vec[0].abs() * self.config.output_size as f32) as usize;
                let action = (action_idx % self.config.output_size) as f32;
                actions.push(action);

                // Reward simulation: Function of state and action
                let base_reward =
                    current_state_vec.iter().sum::<f32>() / (self.config.input_size as f32);
                let noise = self.sample_normal() * self.config.noise_level;
                rewards.push((base_reward + noise as f32).clamp(-1.0, 1.0));

                // State transition: S_next = S_curr + Noise + Drift
                for val in &mut current_state_vec {
                    *val += (self.rng.random_range(-0.1..0.1) * self.config.complexity) as f32;
                    *val = val.clamp(-2.0, 2.0);
                }
            }
        }

        // Fill remaining if exact sample count needed
        while states.len() < self.config.sample_count {
            let state: Vec<f32> = (0..self.config.input_size).map(|_| 0.0).collect();
            states.push(state);
            actions.push(0.0);
            rewards.push(0.0);
        }

        Ok(WorkloadData::ReinforcementLearning {
            states,
            actions,
            rewards,
            episode_length,
        })
    }

    /// Generate generative modeling workload (Latent Codes -> Samples)
    /// Simulates a decoder network output using a Mixture of Gaussians.
    fn generate_generative_workload(&mut self) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let latent_dim = (self.config.input_size as f64).log2().max(2.0) as usize;

        let mut samples = Vec::with_capacity(self.config.sample_count);
        let mut latent_codes = Vec::with_capacity(self.config.sample_count);

        // Pre-generate random projection matrix for "decoding" simulation
        let mut projection_matrix = Vec::with_capacity(latent_dim * self.config.input_size);
        for _ in 0..(latent_dim * self.config.input_size) {
            projection_matrix.push(self.rng.random_range(-1.0..1.0));
        }

        for _ in 0..self.config.sample_count {
            // Generate latent code (Standard Normal)
            let mut latent = Vec::with_capacity(latent_dim);
            for _ in 0..latent_dim {
                latent.push(self.sample_normal() as f32);
            }
            latent_codes.push(latent.clone());

            // "Decode" to sample space via linear projection + non-linearity
            let mut sample = Vec::with_capacity(self.config.input_size);
            for i in 0..self.config.input_size {
                let mut val = 0.0;
                for j in 0..latent_dim {
                    val += latent[j] * projection_matrix[i * latent_dim + j] as f32;
                }
                // Add non-linearity
                val = val.tanh();
                // Add noise
                val += (self.sample_normal() * self.config.noise_level) as f32;
                sample.push(val);
            }
            samples.push(sample);
        }

        Ok(WorkloadData::GenerativeModeling {
            samples,
            latent_codes,
            generation_steps: 50,
        })
    }

    /// Generate complex distribution sample (Simulated Multi-modal)
    #[inline]
    fn generate_complex_distribution_sample(&mut self) -> f64 {
        let mode = self.rng.random_range(0..3);
        let (mu, sigma) = match mode {
            0 => (-2.0, 0.5),
            1 => (0.0, 1.0),
            _ => (2.0, 0.5),
        };

        let raw = self.sample_normal();
        let value = mu + raw * sigma;

        // Apply non-linear distortion based on complexity
        if self.config.complexity > 0.5 {
            value + (value * 2.0).sin() * 0.2
        } else {
            value
        }
    }

    /// Generate classification workload (Cluster-based)
    fn generate_classification_workload(
        &mut self,
    ) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let mut inputs = Vec::with_capacity(self.config.sample_count);
        let mut labels = Vec::with_capacity(self.config.sample_count);

        // Generate random centroids for each class
        let num_classes = self.config.output_size;
        let mut centroids = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            let centroid: Vec<f32> = (0..self.config.input_size)
                .map(|_| self.rng.random_range(-3.0..3.0))
                .collect();
            centroids.push(centroid);
        }

        for _ in 0..self.config.sample_count {
            let label = self.rng.random_range(0..num_classes);
            let centroid = &centroids[label];

            let mut input = Vec::with_capacity(self.config.input_size);
            for &center_val in centroid {
                // Add noise to create cloud around centroid
                let noise = self.sample_normal() as f32 * (0.5 + self.config.noise_level as f32);
                input.push(center_val + noise);
            }

            inputs.push(input);
            labels.push(label);
        }

        Ok(WorkloadData::Classification { inputs, labels })
    }

    /// Generate regression workload (Linear + Non-linear components)
    fn generate_regression_workload(&mut self) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let mut inputs = Vec::with_capacity(self.config.sample_count);
        let mut targets = Vec::with_capacity(self.config.sample_count);

        // Define true weights for the regression problem
        let weights: Vec<f32> = (0..self.config.input_size)
            .map(|_| self.rng.random_range(-1.0..1.0))
            .collect();
        let bias = self.rng.random_range(-0.5..0.5);

        for _ in 0..self.config.sample_count {
            let mut input = Vec::with_capacity(self.config.input_size);
            let mut linear_term = 0.0;

            for &weight in weights.iter().take(self.config.input_size) {
                let val = self.rng.random_range(-1.0..1.0);
                input.push(val as f32);
                linear_term += val as f32 * weight;
            }
            inputs.push(input);

            // y = w*x + b + noise + non_linearity
            let noise = (self.sample_normal() * self.config.noise_level) as f32;
            let non_linear = if self.config.complexity > 0.0 {
                (linear_term * self.config.complexity as f32).sin()
            } else {
                0.0
            };

            targets.push(linear_term + bias + noise + non_linear);
        }

        Ok(WorkloadData::Regression { inputs, targets })
    }

    /// Generate time series forecasting workload (Trend + Seasonality + Noise)
    fn generate_timeseries_workload(&mut self) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let sequence_length = self.config.input_size;
        let horizon = (sequence_length as f64 * 0.2) as usize;
        let mut sequences = Vec::with_capacity(self.config.sample_count);
        let mut forecasts = Vec::with_capacity(self.config.sample_count);

        // Base frequencies for seasonality
        let freq1 = self.rng.random_range(5.0..15.0);
        let freq2 = self.rng.random_range(20.0..50.0);

        for i in 0..self.config.sample_count {
            let mut sequence = Vec::with_capacity(sequence_length);
            let mut forecast = Vec::with_capacity(horizon);

            // Each sample has a slightly different phase/trend offset
            let phase_shift = self.rng.random_range(0.0..2.0 * PI);
            let trend_slope = self.rng.random_range(-0.05..0.05);
            let start_time = (i * 2) as f64;

            for t_step in 0..(sequence_length + horizon) {
                let t = start_time + t_step as f64;

                // Components
                let trend = trend_slope * t;
                let season1 = (t / freq1 + phase_shift).sin();
                let season2 = (t / freq2).cos() * 0.5 * self.config.complexity;
                let noise = self.sample_normal() * self.config.noise_level;

                // Use the complex distribution sample generator for added realism
                let complex_component = if self.config.complexity > 0.7 {
                    self.generate_complex_distribution_sample() * 0.1
                } else {
                    0.0
                };

                let value = (trend + season1 + season2 + noise + complex_component) as f32;

                if t_step < sequence_length {
                    sequence.push(value);
                } else {
                    forecast.push(value);
                }
            }

            sequences.push(sequence);
            forecasts.push(forecast);
        }

        Ok(WorkloadData::TimeSeriesForecasting {
            sequences,
            forecasts,
            horizon,
        })
    }

    /// Generate uncertainty quantification workload (Heteroscedastic noise)
    fn generate_uncertainty_workload(
        &mut self,
    ) -> Result<WorkloadData, Box<dyn std::error::Error>> {
        let mut inputs = Vec::with_capacity(self.config.sample_count);
        let mut targets = Vec::with_capacity(self.config.sample_count);
        let mut uncertainties = Vec::with_capacity(self.config.sample_count);

        for _ in 0..self.config.sample_count {
            let mut input = Vec::with_capacity(self.config.input_size);
            let mut signal = 0.0;
            let mut local_uncertainty_factor = 0.0;

            for i in 0..self.config.input_size {
                let val: f32 = self.rng.random_range(-2.0..2.0);
                input.push(val);

                // Signal function
                signal += val * (i as f32).sin();

                // Uncertainty depends on input magnitude (Heteroscedastic)
                local_uncertainty_factor += val.abs();
            }

            // Normalize uncertainty factor roughly
            local_uncertainty_factor = (local_uncertainty_factor / self.config.input_size as f32)
                * self.config.uncertainty_level as f32;

            inputs.push(input);
            uncertainties.push(local_uncertainty_factor);

            let noise = (self.sample_normal() as f32) * local_uncertainty_factor;
            targets.push(signal + noise);
        }

        Ok(WorkloadData::UncertaintyQuantification {
            inputs,
            targets,
            uncertainties,
        })
    }
}

/// Predefined workload configurations for common benchmarks
pub struct StandardWorkloads;

impl StandardWorkloads {
    /// Small classification benchmark suitable for unit testing
    pub fn small_classification() -> WorkloadConfig {
        WorkloadConfig {
            workload_type: WorkloadType::Classification,
            input_size: 50,
            output_size: 5,
            sample_count: 500,
            complexity: 0.3,
            noise_level: 0.05,
            uncertainty_level: 0.02,
            seed: 42,
        }
    }

    /// Large regression benchmark for throughput testing
    pub fn large_regression() -> WorkloadConfig {
        WorkloadConfig {
            workload_type: WorkloadType::Regression,
            input_size: 200,
            output_size: 50,
            sample_count: 2000,
            complexity: 0.7,
            noise_level: 0.1,
            uncertainty_level: 0.05,
            seed: 123,
        }
    }

    /// Time series forecasting benchmark with seasonal patterns
    pub fn timeseries_forecast() -> WorkloadConfig {
        WorkloadConfig {
            workload_type: WorkloadType::TimeSeriesForecasting,
            input_size: 100, // 100 time steps
            output_size: 20, // 20 step forecast
            sample_count: 1000,
            complexity: 0.6,
            noise_level: 0.08,
            uncertainty_level: 0.04,
            seed: 456,
        }
    }

    /// High uncertainty quantification benchmark for robust ML testing
    pub fn high_uncertainty() -> WorkloadConfig {
        WorkloadConfig {
            workload_type: WorkloadType::UncertaintyQuantification,
            input_size: 80,
            output_size: 20,
            sample_count: 1500,
            complexity: 0.8,
            noise_level: 0.15,
            uncertainty_level: 0.1,
            seed: 789,
        }
    }
}
