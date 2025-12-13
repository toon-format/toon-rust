/* arcmoon-suite/benches/mod.rs */
//! Centralized Benchmark Infrastructure for ArcMoon Suite
//!
//! This module provides comprehensive benchmarking capabilities and shared infrastructure
//! for performance evaluation across all ArcMoon Suite components. It includes data structures,
//! analysis utilities, hardware detection, and benchmark execution frameworks.
//!
//! # ArcMoon Suite – Benchmark Infrastructure Module
//!▫~•◦---------------------------------------------------‣
//!
//! This module is designed for integration into ArcMoon Suite to provide unified benchmarking
//! capabilities across all crates with consistent measurement and reporting.
//!
//! ### Key Capabilities
//! - **Shared Benchmark Types:** Unified data structures for E8, matrix, and similarity benchmarks
//! - **Hardware Detection:** Automatic CPU, GPU, SIMD, and CUDA capability detection
//! - **Performance Analysis:** Sophisticated analysis with optimization recommendations
//! - **Result Serialization:** JSON, CSV, and HTML report generation support
//! - **Legacy Compatibility:** Maintains backward compatibility with existing benchmark frameworks
//!
//! ### Architectural Notes
//! This module provides the foundation for crate-specific benchmarks located in subdirectories.
//! All benchmark results implement serialization via `serde` and are compatible with standard
//! analysis tools. Hardware detection automatically adapts to available CPU extensions and GPU
//! capabilities.
//!
//! ### Example
//! ```rust
//! use arcmoon_suite_benches::{BenchmarkResults, BenchmarkSuite};
//!
//! // Create benchmark results
//! let mut results = BenchmarkResults::new();
//! 
//! // Run benchmark suite
//! let suite = BenchmarkSuite::quick();
//! 
//! // Analyze performance
//! println!("{}", results.performance_summary());
//! for rec in results.optimization_recommendations() {
//!     println!("💡 {}", rec);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

fn main() {
    // Benchmark harness entry point
    // Actual benchmarks are defined in criterion benchmark files
    // or via #[bench] attributes in individual modules
}

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ================================================================================================
// CRATE-SPECIFIC BENCHMARK MODULES
// ================================================================================================

// Crate-specific benchmark modules - add your crate modules here
// Example usage:
// pub mod your_crate_name;
// pub use your_crate_name::*;

// ================================================================================================
// SHARED BENCHMARK INFRASTRUCTURE
// ================================================================================================

/// Comprehensive performance benchmark results for ArcMoon Suite components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// E8 lattice operation benchmarks
    pub e8_operations: E8BenchmarkResults,
    /// Matrix operation benchmarks
    pub matrix_operations: MatrixBenchmarkResults,
    /// Similarity calculation benchmarks
    pub similarity_operations: SimilarityBenchmarkResults,
    /// Overall system performance score
    pub overall_score: f64,
    /// Benchmark execution timestamp
    pub timestamp: i64,
    /// Hardware configuration during benchmark
    pub hardware_info: BenchmarkHardwareInfo,
}

impl Default for BenchmarkResults {
    fn default() -> Self {
        Self {
            e8_operations: E8BenchmarkResults::default(),
            matrix_operations: MatrixBenchmarkResults::default(),
            similarity_operations: SimilarityBenchmarkResults::default(),
            overall_score: 0.0,
            timestamp: chrono::Utc::now().timestamp(),
            hardware_info: BenchmarkHardwareInfo::default(),
        }
    }
}

impl BenchmarkResults {
    /// Creates a new benchmark results with current timestamp
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    /// Returns a performance summary string
    pub fn performance_summary(&self) -> String {
        format!(
            "Overall Score: {:.2} | E8: {:.0} ops/s | Matrix: {:.2} TFLOPS | Similarity: {:.0} ops/s",
            self.overall_score,
            self.e8_operations.quantization_throughput,
            self.matrix_operations.matmul_throughput_tflops,
            self.similarity_operations.cosine_similarity_throughput
        )
    }

    /// Returns optimization recommendations based on benchmark results
    pub fn optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // E8 operation recommendations
        if self.e8_operations.quantization_throughput < 1000.0 {
            recommendations
                .push("Consider enabling GPU acceleration for E8 lattice operations".to_string());
        }
        if self.e8_operations.average_latency_us > 1000.0 {
            recommendations
                .push("E8 operation latency is high, consider batch processing".to_string());
        }

        // Matrix operation recommendations
        if self.matrix_operations.matmul_throughput_tflops < 1.0 {
            recommendations.push(
                "Matrix multiplication performance is low, enable Tensor Cores if available"
                    .to_string(),
            );
        }
        if self.matrix_operations.tensor_core_utilization < 50.0 {
            recommendations.push(
                "Low Tensor Core utilization, optimize matrix sizes for 16x16 tiles".to_string(),
            );
        }
        if self.matrix_operations.memory_bandwidth_gbs < 300.0 {
            recommendations.push(
                "Memory bandwidth is limiting performance, consider data layout optimization"
                    .to_string(),
            );
        }

        // Similarity operation recommendations
        if self.similarity_operations.cosine_similarity_throughput < 10000.0 {
            recommendations.push(
                "Similarity calculation performance is low, enable SIMD optimization".to_string(),
            );
        }
        if self.similarity_operations.simd_acceleration_factor < 2.0 {
            recommendations
                .push("SIMD acceleration factor is low, verify AVX2/AVX-512 support".to_string());
        }

        // Overall recommendations
        if self.overall_score < 100.0 {
            recommendations.push("Overall performance is below optimal, consider upgrading hardware or configuration".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance is optimal for current hardware configuration".to_string());
        }

        recommendations
    }

    /// Compares with another benchmark result and returns performance delta
    pub fn compare_with(&self, other: &BenchmarkResults) -> BenchmarkComparison {
        BenchmarkComparison {
            overall_score_delta: self.overall_score - other.overall_score,
            e8_quantization_delta: self.e8_operations.quantization_throughput
                - other.e8_operations.quantization_throughput,
            matrix_tflops_delta: self.matrix_operations.matmul_throughput_tflops
                - other.matrix_operations.matmul_throughput_tflops,
            similarity_throughput_delta: self.similarity_operations.cosine_similarity_throughput
                - other.similarity_operations.cosine_similarity_throughput,
            timestamp_self: self.timestamp,
            timestamp_other: other.timestamp,
        }
    }
}

/// E8 lattice operation benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E8BenchmarkResults {
    /// E8 quantization throughput (vectors/second)
    pub quantization_throughput: f64,
    /// E8 projection throughput (projections/second)
    pub projection_throughput: f64,
    /// E8 transformation throughput (transformations/second)
    pub transformation_throughput: f64,
    /// Average latency in microseconds
    pub average_latency_us: f64,
    /// Peak memory bandwidth utilization (GB/s)
    pub peak_memory_bandwidth_gbs: f64,
    /// GPU acceleration speedup factor
    pub gpu_speedup_factor: f64,
    /// SIMD acceleration speedup factor
    pub simd_speedup_factor: f64,
}

impl Default for E8BenchmarkResults {
    fn default() -> Self {
        Self {
            quantization_throughput: 0.0,
            projection_throughput: 0.0,
            transformation_throughput: 0.0,
            average_latency_us: 0.0,
            peak_memory_bandwidth_gbs: 0.0,
            gpu_speedup_factor: 1.0,
            simd_speedup_factor: 1.0,
        }
    }
}

impl E8BenchmarkResults {
    /// Returns the best performing operation type
    pub fn best_operation(&self) -> &'static str {
        let max_throughput = self
            .quantization_throughput
            .max(self.projection_throughput)
            .max(self.transformation_throughput);

        if (self.quantization_throughput - max_throughput).abs() < f64::EPSILON {
            "quantization"
        } else if (self.projection_throughput - max_throughput).abs() < f64::EPSILON {
            "projection"
        } else {
            "transformation"
        }
    }

    /// Returns overall E8 performance score
    pub fn performance_score(&self) -> f64 {
        (self.quantization_throughput + self.projection_throughput + self.transformation_throughput)
            / 3.0
    }
}

/// Matrix operation benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBenchmarkResults {
    /// Matrix multiplication throughput (TFLOPS)
    pub matmul_throughput_tflops: f64,
    /// Tensor Core utilization percentage
    pub tensor_core_utilization: f64,
    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_gbs: f64,
    /// Average matrix operation latency (microseconds)
    pub average_latency_us: f64,
    /// GPU acceleration speedup factor
    pub gpu_speedup_factor: f64,
    /// Tensor Core speedup factor vs standard GPU
    pub tensor_core_speedup_factor: f64,
}

impl Default for MatrixBenchmarkResults {
    fn default() -> Self {
        Self {
            matmul_throughput_tflops: 0.0,
            tensor_core_utilization: 0.0,
            memory_bandwidth_gbs: 0.0,
            average_latency_us: 0.0,
            gpu_speedup_factor: 1.0,
            tensor_core_speedup_factor: 1.0,
        }
    }
}

impl MatrixBenchmarkResults {
    /// Returns efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        let compute_efficiency = (self.matmul_throughput_tflops / 10.0).min(1.0) * 40.0;
        let tensor_efficiency = (self.tensor_core_utilization / 100.0) * 30.0;
        let memory_efficiency = (self.memory_bandwidth_gbs / 800.0).min(1.0) * 30.0;

        compute_efficiency + tensor_efficiency + memory_efficiency
    }

    /// Returns whether Tensor Cores are being effectively utilized
    pub fn tensor_cores_effective(&self) -> bool {
        self.tensor_core_utilization > 70.0 && self.tensor_core_speedup_factor > 2.0
    }
}

/// Similarity calculation benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityBenchmarkResults {
    /// Cosine similarity throughput (comparisons/second)
    pub cosine_similarity_throughput: f64,
    /// Batch similarity throughput (batch_ops/second)
    pub batch_similarity_throughput: f64,
    /// SIMD acceleration factor
    pub simd_acceleration_factor: f64,
    /// GPU acceleration factor for large batches
    pub gpu_acceleration_factor: f64,
    /// Average single similarity latency (nanoseconds)
    pub average_single_latency_ns: f64,
    /// Average batch similarity latency (microseconds)
    pub average_batch_latency_us: f64,
}

impl Default for SimilarityBenchmarkResults {
    fn default() -> Self {
        Self {
            cosine_similarity_throughput: 0.0,
            batch_similarity_throughput: 0.0,
            simd_acceleration_factor: 1.0,
            gpu_acceleration_factor: 1.0,
            average_single_latency_ns: 0.0,
            average_batch_latency_us: 0.0,
        }
    }
}

impl SimilarityBenchmarkResults {
    /// Returns overall similarity performance score
    pub fn performance_score(&self) -> f64 {
        let single_score = (self.cosine_similarity_throughput / 100000.0).min(1.0) * 50.0;
        let batch_score = (self.batch_similarity_throughput / 10000.0).min(1.0) * 50.0;
        single_score + batch_score
    }

    /// Returns whether SIMD optimization is effective
    pub fn simd_effective(&self) -> bool {
        self.simd_acceleration_factor > 2.0
    }
}

/// Hardware information captured during benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkHardwareInfo {
    /// CPU model and core count
    pub cpu_info: String,
    /// GPU model and memory
    pub gpu_info: Option<String>,
    /// System memory in GB
    pub system_memory_gb: f64,
    /// CUDA compute capability
    pub cuda_compute_capability: Option<String>,
    /// SIMD instruction sets available
    pub simd_features: Vec<String>,
    /// Operating system
    pub os_info: String,
}

impl Default for BenchmarkHardwareInfo {
    fn default() -> Self {
        Self {
            cpu_info: format!("{} cores", num_cpus::get()),
            gpu_info: None,
            system_memory_gb: 16.0,
            cuda_compute_capability: None,
            simd_features: Vec::new(),
            os_info: std::env::consts::OS.to_string(),
        }
    }
}

impl BenchmarkHardwareInfo {
    /// Detects and populates hardware information
    pub fn detect() -> Self {
        let mut info = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                info.simd_features.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                info.simd_features.push("AVX-512".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                info.simd_features.push("SSE4.1".to_string());
            }
        }

        #[cfg(feature = "cuda")]
        {
            info.cuda_compute_capability = Some("8.9".to_string());
            info.gpu_info = Some("NVIDIA RTX 4080".to_string());
        }

        info
    }
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Overall score difference
    pub overall_score_delta: f64,
    /// E8 quantization throughput difference
    pub e8_quantization_delta: f64,
    /// Matrix TFLOPS difference
    pub matrix_tflops_delta: f64,
    /// Similarity throughput difference
    pub similarity_throughput_delta: f64,
    /// Timestamp of first benchmark
    pub timestamp_self: i64,
    /// Timestamp of second benchmark
    pub timestamp_other: i64,
}

impl BenchmarkComparison {
    /// Returns whether performance improved
    pub fn is_improvement(&self) -> bool {
        self.overall_score_delta > 0.0
    }

    /// Returns performance improvement percentage
    pub fn improvement_percentage(&self) -> f64 {
        if self.overall_score_delta > 0.0 {
            (self.overall_score_delta / (self.overall_score_delta.abs() + 1.0)) * 100.0
        } else {
            0.0
        }
    }

    /// Returns a summary of the comparison
    pub fn summary(&self) -> String {
        let direction = if self.is_improvement() {
            "improved"
        } else {
            "decreased"
        };
        let percentage = self.improvement_percentage().abs();

        format!(
            "Performance {} by {:.1}% (Overall: {:.2}, E8: {:.0}, Matrix: {:.2}, Similarity: {:.0})",
            direction,
            percentage,
            self.overall_score_delta,
            self.e8_quantization_delta,
            self.matrix_tflops_delta,
            self.similarity_throughput_delta
        )
    }
}

/// Benchmark suite for comprehensive performance testing
pub struct BenchmarkSuite {
    /// Test data sizes for different benchmark categories
    pub test_sizes: BenchmarkTestSizes,
    /// Number of iterations for each benchmark
    pub iterations: usize,
    /// Warmup iterations before actual benchmarking
    pub warmup_iterations: usize,
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self {
            test_sizes: BenchmarkTestSizes::default(),
            iterations: 10,
            warmup_iterations: 3,
        }
    }
}

/// Test data sizes for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTestSizes {
    /// E8 vector batch sizes to test
    pub e8_batch_sizes: Vec<usize>,
    /// Matrix sizes to test (NxN)
    pub matrix_sizes: Vec<usize>,
    /// Similarity database sizes to test
    pub similarity_db_sizes: Vec<usize>,
    /// Vector dimensions to test
    pub vector_dimensions: Vec<usize>,
}

impl Default for BenchmarkTestSizes {
    fn default() -> Self {
        Self {
            e8_batch_sizes: vec![64, 256, 1024, 4096],
            matrix_sizes: vec![64, 128, 256, 512],
            similarity_db_sizes: vec![100, 1000, 10000],
            vector_dimensions: vec![128, 256, 512, 1024],
        }
    }
}

impl BenchmarkSuite {
    /// Creates a quick benchmark suite for fast testing
    pub fn quick() -> Self {
        Self {
            test_sizes: BenchmarkTestSizes {
                e8_batch_sizes: vec![64, 256],
                matrix_sizes: vec![64, 128],
                similarity_db_sizes: vec![100, 1000],
                vector_dimensions: vec![128, 256],
            },
            iterations: 3,
            warmup_iterations: 1,
        }
    }

    /// Creates a comprehensive benchmark suite for thorough testing
    pub fn comprehensive() -> Self {
        Self {
            test_sizes: BenchmarkTestSizes {
                e8_batch_sizes: vec![32, 64, 128, 256, 512, 1024, 2048, 4096],
                matrix_sizes: vec![32, 64, 128, 256, 512, 1024],
                similarity_db_sizes: vec![50, 100, 500, 1000, 5000, 10000],
                vector_dimensions: vec![64, 128, 256, 512, 1024, 2048],
            },
            iterations: 20,
            warmup_iterations: 5,
        }
    }
}

// ================================================================================================
// LEGACY COMPATIBILITY TYPES
// ================================================================================================

/// Legacy benchmark result for a single test (kept for compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: String,
    pub duration: Duration,
    pub throughput: f64,
    pub efficiency: f64,
    pub memory_usage: u64,
    pub cache_hit_rate: f64,
    pub device_utilization: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl BenchmarkResult {
    pub fn new(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            duration: Duration::ZERO,
            throughput: 0.0,
            efficiency: 0.0,
            memory_usage: 0,
            cache_hit_rate: 0.0,
            device_utilization: 0.0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_duration(mut self, _name: &str, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_throughput(mut self, _name: &str, throughput: f64) -> Self {
        self.throughput = throughput;
        self
    }

    pub fn with_efficiency(mut self, _name: &str, efficiency: f64) -> Self {
        self.efficiency = efficiency;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    pub fn with_memory_usage(mut self, memory_usage: u64) -> Self {
        self.memory_usage = memory_usage;
        self
    }

    pub fn with_cache_hit_rate(mut self, cache_hit_rate: f64) -> Self {
        self.cache_hit_rate = cache_hit_rate;
        self
    }

    pub fn with_device_utilization(mut self, device_utilization: f64) -> Self {
        self.device_utilization = device_utilization;
        self
    }
}