/* arcmoon-suite/src/lib.rs */
//! ArcMoon Suite Performance Suite Library
//!
//! This library provides comprehensive benchmarking capabilities for the ArcMoon Suit
//! ecosystem, including performance analysis, report generation, and visualization.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod benchmarks;
pub mod ml_workloads;
pub mod performance_analysis;
pub mod report_generator;
pub mod test_harness;

// Re-export main types for convenience
pub use benchmarks::{QuantumBenchmark, QuaternionBenchmark};
pub use ml_workloads::{StandardWorkloads, WorkloadConfig, WorkloadGenerator};
pub use performance_analysis::{BenchmarkResult, BenchmarkResults, PerformanceAnalyzer};
pub use report_generator::ReportGenerator;
pub use test_harness::{ExecutionType, TestHarness, TestHarnessConfig};
