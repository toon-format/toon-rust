/* arcmoon-suite/tests/mod.rs */
//! # ArcMoon's Centralized Testing Suite Infrastructure
//!
//! This module provides comprehensive testing capabilities and shared infrastructure
//! for test execution leveraging all of ArcMoon automation and reporting components.
//! It includes test result structures, analysis utilities, hardware detection,
//! and test execution frameworks.
//!
//! ## Architecture
//!
//! - **Shared Infrastructure**: Common types and utilities used by all crate tests
//! - **Hardware Detection**: Automatic detection of CPU, GPU, and SIMD capabilities  
//! - **Test Analysis**: Sophisticated analysis and failure pattern recognition
//! - **Result Comparison**: Tools for comparing test results across runs
//! - **Centralized Organization**: All crate-specific tests organized by crate name
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Crate-specific test modules - add your crate modules here
// Example usage:
// pub mod your_crate_name;
// pub use your_crate_name::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ================================================================================================
// SHARED TEST INFRASTRUCTURE
// ================================================================================================

/// Comprehensive test execution results for ArcMoon  components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Unit test results
    pub unit_tests: UnitTestResults,
    /// Integration test results
    pub integration_tests: IntegrationTestResults,
    /// Performance test results
    pub performance_tests: PerformanceTestResults,
    /// Overall test success rate
    pub overall_success_rate: f64,
    /// Test execution timestamp
    pub timestamp: i64,
    /// Hardware configuration during testing
    pub hardware_info: TestHardwareInfo,
}

impl Default for TestResults {
    fn default() -> Self {
        Self {
            unit_tests: UnitTestResults::default(),
            integration_tests: IntegrationTestResults::default(),
            performance_tests: PerformanceTestResults::default(),
            overall_success_rate: 0.0,
            timestamp: chrono::Utc::now().timestamp(),
            hardware_info: TestHardwareInfo::default(),
        }
    }
}

impl TestResults {
    /// Creates a new test results with current timestamp
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    /// Returns a test summary string
    pub fn test_summary(&self) -> String {
        format!(
            "Success Rate: {:.1}% | Unit: {}/{} | Integration: {}/{} | Performance: {}/{}",
            self.overall_success_rate * 100.0,
            self.unit_tests.passed,
            self.unit_tests.total,
            self.integration_tests.passed,
            self.integration_tests.total,
            self.performance_tests.passed,
            self.performance_tests.total
        )
    }

    /// Returns test failure analysis and recommendations
    pub fn failure_analysis(&self) -> Vec<String> {
        let mut analysis = Vec::new();

        // Unit test analysis
        if self.unit_tests.success_rate() < 0.95 {
            analysis.push(format!(
                "Unit test success rate is {:.1}%, investigate failing tests",
                self.unit_tests.success_rate() * 100.0
            ));
        }

        // Integration test analysis
        if self.integration_tests.success_rate() < 0.90 {
            analysis.push(format!(
                "Integration test success rate is {:.1}%, check component interactions",
                self.integration_tests.success_rate() * 100.0
            ));
        }

        // Performance test analysis
        if self.performance_tests.success_rate() < 0.85 {
            analysis.push(format!(
                "Performance test success rate is {:.1}%, review performance requirements",
                self.performance_tests.success_rate() * 100.0
            ));
        }

        // Execution time analysis
        if self.unit_tests.average_execution_time_ms > 1000.0 {
            analysis.push("Unit tests are taking too long, consider optimization".to_string());
        }

        if self.integration_tests.average_execution_time_ms > 10000.0 {
            analysis.push(
                "Integration tests are taking too long, consider parallelization".to_string(),
            );
        }

        // Memory usage analysis
        if self.unit_tests.peak_memory_usage_mb > 1000.0 {
            analysis.push("Unit tests are using excessive memory".to_string());
        }

        if analysis.is_empty() {
            analysis.push("All tests are performing within expected parameters".to_string());
        }

        analysis
    }

    /// Compares with another test result and returns analysis
    pub fn compare_with(&self, other: &TestResults) -> TestComparison {
        TestComparison {
            success_rate_delta: self.overall_success_rate - other.overall_success_rate,
            unit_test_delta: self.unit_tests.passed as i32 - other.unit_tests.passed as i32,
            integration_test_delta: self.integration_tests.passed as i32
                - other.integration_tests.passed as i32,
            performance_test_delta: self.performance_tests.passed as i32
                - other.performance_tests.passed as i32,
            execution_time_delta: self.total_execution_time() - other.total_execution_time(),
            timestamp_self: self.timestamp,
            timestamp_other: other.timestamp,
        }
    }

    /// Returns total execution time across all test categories
    pub fn total_execution_time(&self) -> Duration {
        Duration::from_millis(
            (self.unit_tests.total_execution_time_ms
                + self.integration_tests.total_execution_time_ms
                + self.performance_tests.total_execution_time_ms) as u64,
        )
    }
}

/// Unit test execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitTestResults {
    /// Number of tests passed
    pub passed: usize,
    /// Number of tests failed
    pub failed: usize,
    /// Total number of tests
    pub total: usize,
    /// Average execution time per test (milliseconds)
    pub average_execution_time_ms: f64,
    /// Total execution time (milliseconds)
    pub total_execution_time_ms: f64,
    /// Peak memory usage during testing (MB)
    pub peak_memory_usage_mb: f64,
    /// Test coverage percentage
    pub coverage_percentage: f64,
}

impl Default for UnitTestResults {
    fn default() -> Self {
        Self {
            passed: 0,
            failed: 0,
            total: 0,
            average_execution_time_ms: 0.0,
            total_execution_time_ms: 0.0,
            peak_memory_usage_mb: 0.0,
            coverage_percentage: 0.0,
        }
    }
}

impl UnitTestResults {
    /// Returns success rate (0.0-1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    /// Returns whether unit tests are healthy
    pub fn is_healthy(&self) -> bool {
        self.success_rate() >= 0.95 && self.average_execution_time_ms < 1000.0
    }
}

/// Integration test execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTestResults {
    /// Number of tests passed
    pub passed: usize,
    /// Number of tests failed
    pub failed: usize,
    /// Total number of tests
    pub total: usize,
    /// Average execution time per test (milliseconds)
    pub average_execution_time_ms: f64,
    /// Total execution time (milliseconds)
    pub total_execution_time_ms: f64,
    /// Peak memory usage during testing (MB)
    pub peak_memory_usage_mb: f64,
    /// Number of components tested
    pub components_tested: usize,
    /// Cross-component interaction success rate
    pub interaction_success_rate: f64,
}

impl Default for IntegrationTestResults {
    fn default() -> Self {
        Self {
            passed: 0,
            failed: 0,
            total: 0,
            average_execution_time_ms: 0.0,
            total_execution_time_ms: 0.0,
            peak_memory_usage_mb: 0.0,
            components_tested: 0,
            interaction_success_rate: 0.0,
        }
    }
}

impl IntegrationTestResults {
    /// Returns success rate (0.0-1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    /// Returns whether integration tests are healthy
    pub fn is_healthy(&self) -> bool {
        self.success_rate() >= 0.90 && self.interaction_success_rate >= 0.85
    }
}

/// Performance test execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestResults {
    /// Number of tests passed
    pub passed: usize,
    /// Number of tests failed
    pub failed: usize,
    /// Total number of tests
    pub total: usize,
    /// Average execution time per test (milliseconds)
    pub average_execution_time_ms: f64,
    /// Total execution time (milliseconds)
    pub total_execution_time_ms: f64,
    /// Peak memory usage during testing (MB)
    pub peak_memory_usage_mb: f64,
    /// Performance regression count
    pub performance_regressions: usize,
    /// Average performance improvement percentage
    pub average_performance_improvement: f64,
}

impl Default for PerformanceTestResults {
    fn default() -> Self {
        Self {
            passed: 0,
            failed: 0,
            total: 0,
            average_execution_time_ms: 0.0,
            total_execution_time_ms: 0.0,
            peak_memory_usage_mb: 0.0,
            performance_regressions: 0,
            average_performance_improvement: 0.0,
        }
    }
}

impl PerformanceTestResults {
    /// Returns success rate (0.0-1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    /// Returns whether performance tests are healthy
    pub fn is_healthy(&self) -> bool {
        self.success_rate() >= 0.85 && self.performance_regressions == 0
    }
}

/// Hardware information captured during testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestHardwareInfo {
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
    /// Rust version used for testing
    pub rust_version: String,
}

impl Default for TestHardwareInfo {
    fn default() -> Self {
        Self {
            cpu_info: format!("{} cores", num_cpus::get()),
            gpu_info: None,
            system_memory_gb: 16.0, // Default assumption
            cuda_compute_capability: None,
            simd_features: Vec::new(),
            os_info: std::env::consts::OS.to_string(),
            rust_version: "1.89.0".to_string(), // Default assumption
        }
    }
}

impl TestHardwareInfo {
    /// Detects and populates hardware information
    pub fn detect() -> Self {
        let mut info = Self::default();

        // Detect SIMD features
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

        // Detect CUDA capability (placeholder)
        #[cfg(feature = "cuda")]
        {
            info.cuda_compute_capability = Some("8.9".to_string()); // Ada Lovelace
            info.gpu_info = Some("NVIDIA RTX 4080".to_string());
        }

        info
    }
}

/// Test comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestComparison {
    /// Overall success rate difference
    pub success_rate_delta: f64,
    /// Unit test pass count difference
    pub unit_test_delta: i32,
    /// Integration test pass count difference
    pub integration_test_delta: i32,
    /// Performance test pass count difference
    pub performance_test_delta: i32,
    /// Total execution time difference
    pub execution_time_delta: Duration,
    /// Timestamp of first test run
    pub timestamp_self: i64,
    /// Timestamp of second test run
    pub timestamp_other: i64,
}

impl TestComparison {
    /// Returns whether test results improved
    pub fn is_improvement(&self) -> bool {
        self.success_rate_delta > 0.0
    }

    /// Returns improvement percentage
    pub fn improvement_percentage(&self) -> f64 {
        if self.success_rate_delta > 0.0 {
            self.success_rate_delta * 100.0
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
            "Test results {} by {:.1}% (Unit: {:+}, Integration: {:+}, Performance: {:+})",
            direction,
            percentage,
            self.unit_test_delta,
            self.integration_test_delta,
            self.performance_test_delta
        )
    }
}

/// Test suite for comprehensive testing
pub struct TestSuite {
    /// Test configuration parameters
    pub test_config: TestConfig,
    /// Number of test iterations for flaky test detection
    pub iterations: usize,
    /// Timeout for individual tests (seconds)
    pub test_timeout_seconds: u64,
}

impl Default for TestSuite {
    fn default() -> Self {
        Self {
            test_config: TestConfig::default(),
            iterations: 1,
            test_timeout_seconds: 300, // 5 minutes
        }
    }
}

/// Test configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Whether to run unit tests
    pub run_unit_tests: bool,
    /// Whether to run integration tests
    pub run_integration_tests: bool,
    /// Whether to run performance tests
    pub run_performance_tests: bool,
    /// Whether to capture test output
    pub capture_output: bool,
    /// Whether to run tests in parallel
    pub parallel_execution: bool,
    /// Maximum number of parallel test threads
    pub max_parallel_threads: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            run_unit_tests: true,
            run_integration_tests: true,
            run_performance_tests: true,
            capture_output: true,
            parallel_execution: true,
            max_parallel_threads: num_cpus::get(),
        }
    }
}

impl TestSuite {
    /// Creates a quick test suite for fast validation
    pub fn quick() -> Self {
        Self {
            test_config: TestConfig {
                run_unit_tests: true,
                run_integration_tests: false,
                run_performance_tests: false,
                capture_output: true,
                parallel_execution: true,
                max_parallel_threads: num_cpus::get(),
            },
            iterations: 1,
            test_timeout_seconds: 60, // 1 minute
        }
    }

    /// Creates a comprehensive test suite for thorough validation
    pub fn comprehensive() -> Self {
        Self {
            test_config: TestConfig {
                run_unit_tests: true,
                run_integration_tests: true,
                run_performance_tests: true,
                capture_output: true,
                parallel_execution: true,
                max_parallel_threads: num_cpus::get(),
            },
            iterations: 3,             // Run tests multiple times to detect flaky tests
            test_timeout_seconds: 600, // 10 minutes
        }
    }
}

// ================================================================================================
// LEGACY COMPATIBILITY TYPES (for backward compatibility)
// ================================================================================================

/// Legacy test result for a single test (kept for compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub category: String,
    pub passed: bool,
    pub duration: Duration,
    pub error_message: Option<String>,
    pub memory_usage: u64, // Peak memory usage in bytes
    pub cpu_usage: f64,    // CPU usage percentage
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TestResult {
    pub fn new(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            passed: false,
            duration: Duration::ZERO,
            error_message: None,
            memory_usage: 0,
            cpu_usage: 0.0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_success(mut self) -> Self {
        self.passed = true;
        self
    }

    pub fn with_failure(mut self, error_message: &str) -> Self {
        self.passed = false;
        self.error_message = Some(error_message.to_string());
        self
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
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

    pub fn with_cpu_usage(mut self, cpu_usage: f64) -> Self {
        self.cpu_usage = cpu_usage;
        self
    }
}

/// Legacy collection of test results (kept for compatibility)
#[derive(Debug, Default)]
pub struct LegacyTestResults {
    pub results: Vec<TestResult>,
}

impl LegacyTestResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    pub fn get_category_results(&self, category: &str) -> Vec<&TestResult> {
        self.results
            .iter()
            .filter(|r| r.category == category)
            .collect()
    }

    pub fn success_rate(&self, category: &str) -> f64 {
        let results = self.get_category_results(category);
        if results.is_empty() {
            return 0.0;
        }
        let passed = results.iter().filter(|r| r.passed).count();
        passed as f64 / results.len() as f64
    }

    pub fn average_duration(&self, category: &str) -> Duration {
        let results = self.get_category_results(category);
        if results.is_empty() {
            return Duration::ZERO;
        }
        let total_ms: u64 = results.iter().map(|r| r.duration.as_millis() as u64).sum();
        Duration::from_millis(total_ms / results.len() as u64)
    }
}
