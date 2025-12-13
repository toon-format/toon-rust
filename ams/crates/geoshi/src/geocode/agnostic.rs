/* src/geocode/agnostic.rs */
//! Language-agnostic code geometry primitives and metrics.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ndarray::Array1;

/// Function identifier for tracking in the geometric space
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnId {
    pub module: String,
    pub name: String,
}

/// Metrics for code geometric embedding in 8D E8 space.
/// All values normalized to [-1, 1] range using tanh scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct HydronMetrics {
    /// Cyclomatic complexity (control flow branches/loops)
    pub cyclomatic_complexity: f64,
    /// Density of unsafe blocks (unsafe statements / total statements)
    pub unsafe_density: f64,
    /// Mutable borrow usage (count &mut params and patterns)
    pub mutable_borrows: f64,
    /// Generic complexity (where clauses + type params count)
    pub generics_complexity: f64,
    /// Distinct lifetime parameter count
    pub lifetimes: f64,
    /// Function parameter count
    pub args: f64,
    /// Call chain depth (dependency adjacency)
    pub dependency_depth: f64,
    /// Approximate line count (statement-based)
    pub line_count: f64,
}

/// Builder for HydronMetrics to avoid too many constructor arguments
#[derive(Debug, Default)]
pub struct HydronMetricsBuilder {
    cyclomatic_complexity: usize,
    unsafe_density: f64,
    mutable_borrows: usize,
    generics_complexity: usize,
    lifetimes: usize,
    args: usize,
    dependency_depth: usize,
    line_count: usize,
}

impl HydronMetricsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cyclomatic_complexity(mut self, value: usize) -> Self {
        self.cyclomatic_complexity = value;
        self
    }

    pub fn unsafe_density(mut self, value: f64) -> Self {
        self.unsafe_density = value;
        self
    }

    pub fn mutable_borrows(mut self, value: usize) -> Self {
        self.mutable_borrows = value;
        self
    }

    pub fn generics_complexity(mut self, value: usize) -> Self {
        self.generics_complexity = value;
        self
    }

    pub fn lifetimes(mut self, value: usize) -> Self {
        self.lifetimes = value;
        self
    }

    pub fn args(mut self, value: usize) -> Self {
        self.args = value;
        self
    }

    pub fn dependency_depth(mut self, value: usize) -> Self {
        self.dependency_depth = value;
        self
    }

    pub fn line_count(mut self, value: usize) -> Self {
        self.line_count = value;
        self
    }

    pub fn build(self) -> HydronMetrics {
        HydronMetrics {
            cyclomatic_complexity: HydronMetrics::normalize(self.cyclomatic_complexity as f64),
            unsafe_density: HydronMetrics::normalize(self.unsafe_density),
            mutable_borrows: HydronMetrics::normalize(self.mutable_borrows as f64),
            generics_complexity: HydronMetrics::normalize(self.generics_complexity as f64),
            lifetimes: HydronMetrics::normalize(self.lifetimes as f64),
            args: HydronMetrics::normalize(self.args as f64),
            dependency_depth: HydronMetrics::normalize(self.dependency_depth as f64),
            line_count: HydronMetrics::normalize(self.line_count as f64),
        }
    }
}

impl HydronMetrics {
    // NOTE: The explicit `new(...)` constructor was removed in favor of
    // the `HydronMetricsBuilder` which offers clearer call sites and
    // flexibility for future fields. Use the builder to construct metrics:
    //
    // HydronMetricsBuilder::new()
    //     .cyclomatic_complexity(5)
    //     .unsafe_density(0.1)
    //     ...
    //     .build();

    /// Normalize value to [-1, 1] using bounded tanh scaling.
    /// Values naturally cluster around 0, with extreme values approaching ±1.
    pub fn normalize(value: f64) -> f64 {
        (value / 10.0).tanh()
    }

    /// Convert to 8D E8 vector representation.
    pub fn to_e8_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.cyclomatic_complexity,
            self.unsafe_density,
            self.mutable_borrows,
            self.generics_complexity,
            self.lifetimes,
            self.args,
            self.dependency_depth,
            self.line_count,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hydron_metrics_creation() {
        let metrics = HydronMetricsBuilder::new()
            .cyclomatic_complexity(5)
            .unsafe_density(0.1)
            .mutable_borrows(2)
            .generics_complexity(1)
            .lifetimes(3)
            .args(4)
            .dependency_depth(0)
            .line_count(10)
            .build();

        // Check normalization is within [-1, 1]
        assert!(metrics.cyclomatic_complexity >= -1.0 && metrics.cyclomatic_complexity <= 1.0);
        assert!(metrics.unsafe_density >= -1.0 && metrics.unsafe_density <= 1.0);
        assert!(metrics.mutable_borrows >= -1.0 && metrics.mutable_borrows <= 1.0);
        assert!(metrics.generics_complexity >= -1.0 && metrics.generics_complexity <= 1.0);
        assert!(metrics.lifetimes >= -1.0 && metrics.lifetimes <= 1.0);
        assert!(metrics.args >= -1.0 && metrics.args <= 1.0);
        assert!(metrics.dependency_depth >= -1.0 && metrics.dependency_depth <= 1.0);
        assert!(metrics.line_count >= -1.0 && metrics.line_count <= 1.0);
    }

    #[test]
    fn test_to_e8_vector() {
        let metrics = HydronMetricsBuilder::new()
            .cyclomatic_complexity(1)
            .unsafe_density(0.0)
            .mutable_borrows(0)
            .generics_complexity(0)
            .lifetimes(0)
            .args(1)
            .dependency_depth(0)
            .line_count(1)
            .build();
        let vec = metrics.to_e8_vector();

        assert_eq!(vec.len(), 8);
        assert!(vec.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_normalization_bounds() {
        // Small values should be around 0
        assert_relative_eq!(HydronMetrics::normalize(0.0), 0.0, epsilon = 0.01);
        assert_relative_eq!(HydronMetrics::normalize(1.0), 0.099, epsilon = 0.01);

        // Larger values approach ±1 but don't exceed
        assert!(HydronMetrics::normalize(50.0) <= 1.0);
        assert!(HydronMetrics::normalize(50.0) > 0.0);
    }
}
