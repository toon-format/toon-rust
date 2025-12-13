/* arcmoon-suite/src/performance_analysis.rs */
//! Performance Analysis and Statistics Evaluation
//!
//! This module provides comprehensive performance analysis capabilities for all
//! benchmark results, including statistical analysis, trend detection, bottleneck identification,
//! and performance regression tracking.
//!
//! ### Capabilities
//! - **Statistical Aggregation:** Computes means, variances, and standard deviations for throughput/efficiency.
//! - **Trend Analysis:** Uses linear regression to detect performance trajectories over time.
//! - **Bottleneck Detection:** Heuristic analysis of memory, cache, and compute metrics to identify constraints.
//! - **Regression Detection:** Compares current run against historical data to flag performance drops.
//!
//! ### Architectural Notes
//! This module acts as the analytical engine of the suite. It transforms raw `BenchmarkResult` data
//! into actionable intelligence (`PerformanceAnalysis`) for the reporting layer.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::test_harness::SerializableDuration;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Core Data Structures
// ---------------------------------------------------------------------------

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: String,
    pub duration: SerializableDuration,
    pub duration_ns: u64,
    pub throughput: f64,         // Operations per second
    pub efficiency: f64,         // Efficiency score (0.0-1.0)
    pub efficiency_score: f64,   // Alias for efficiency
    pub memory_usage: u64,       // Peak memory usage in bytes
    pub memory_usage_bytes: u64, // Alias for memory_usage
    pub cache_hit_rate: f64,     // Cache hit rate (0.0-1.0)
    pub device_utilization: f64, // Device utilization (0.0-1.0)
    pub metadata: HashMap<String, serde_json::Value>,
}

impl BenchmarkResult {
    pub fn new(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            duration: SerializableDuration::from(Duration::ZERO),
            duration_ns: 0,
            throughput: 0.0,
            efficiency: 0.0,
            efficiency_score: 0.0,
            memory_usage: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
            device_utilization: 0.0,
            metadata: HashMap::new(),
        }
    }
}

/// Collection of benchmark results
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    pub fn get_category_results(&self, category: &str) -> Vec<&BenchmarkResult> {
        self.results
            .iter()
            .filter(|r| r.category == category)
            .collect()
    }

    pub fn average_throughput(&self, category: &str) -> f64 {
        let results = self.get_category_results(category);
        if results.is_empty() {
            return 0.0;
        }
        results.iter().map(|r| r.throughput).sum::<f64>() / results.len() as f64
    }

    pub fn average_efficiency(&self, category: &str) -> f64 {
        let results = self.get_category_results(category);
        if results.is_empty() {
            return 0.0;
        }
        results.iter().map(|r| r.efficiency_score).sum::<f64>() / results.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Analysis Structures
// ---------------------------------------------------------------------------

/// Performance summary statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_benchmarks: usize,
    pub total_duration: Duration,
    pub average_throughput: f64,
    pub average_efficiency: f64,
    pub best_performing_category: String,
    pub worst_performing_category: String,
    pub overall_score: f64,
    pub total_execution_time: Duration,
    pub memory_efficiency: f64,
    pub device_utilization: f64,
}

/// Category-specific analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct CategoryAnalysis {
    pub category: String,
    pub benchmark_count: usize,
    pub average_throughput: f64,
    pub throughput_std_dev: f64,
    pub average_efficiency: f64,
    pub efficiency_std_dev: f64,
    pub best_benchmark: String,
    pub worst_benchmark: String,
    pub performance_consistency: f64, // 0.0 to 1.0
    pub efficiency_score: f64,
    pub memory_usage_avg: u64,
    pub bottlenecks: Vec<String>,
}

/// Efficiency ranking entry
#[derive(Debug, Serialize, Deserialize)]
pub struct EfficiencyRanking {
    pub rank: usize,
    pub benchmark_name: String,
    pub category: String,
    pub efficiency_score: f64,
    pub throughput: f64,
    pub relative_performance: f64, // Compared to category average
}

/// Performance trend analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub category: String,
    pub trend_type: TrendType,
    pub trend_strength: f64, // -1.0 to 1.0
    pub confidence: f64,     // 0.0 to 1.0
    pub description: String,
    pub metric: String,
    pub trend_direction: TrendDirection,
    pub slope: f64,
    pub r_squared: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TrendType {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
    Volatile,
}

/// Bottleneck analysis results
#[derive(Debug, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottlenecks: Vec<Bottleneck>,
    pub resource_utilization: ResourceUtilization,
    pub scaling_analysis: ScalingAnalysis,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub resource: String,
    pub severity: f64, // 0.0 to 1.0
    pub affected_benchmarks: Vec<String>,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BottleneckType {
    Memory,
    Compute,
    IO,
    Synchronization,
    Cache,
    Network,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub cpu_efficiency: f64,
    pub memory_utilization: f64,
    pub memory_efficiency: f64,
    pub cache_efficiency: f64,
    pub gpu_utilization: f64,
    pub gpu_efficiency: f64,
    pub io_utilization: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub linear_scaling_score: f64,
    pub parallel_efficiency: f64,
    pub optimal_batch_sizes: HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub affected_area: String,
    pub description: String,
    pub benchmark_name: String,
    pub category: String,
    pub metric: String,
    pub previous_value: f64,
    pub current_value: f64,
    pub regression_percentage: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RegressionType {
    Throughput,
    Latency,
    Memory,
    Accuracy,
    Stability,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,    // < 5% regression
    Moderate, // 5-15% regression
    Major,    // 15-30% regression
    Critical, // > 30% regression
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub title: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: ImplementationEffort,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Top-level analysis container
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub summary: PerformanceSummary,
    pub category_analysis: HashMap<String, CategoryAnalysis>,
    pub efficiency_rankings: Vec<EfficiencyRanking>,
    pub performance_trends: Vec<PerformanceTrend>,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub recommendations: Vec<PerformanceRecommendation>,
}

// ---------------------------------------------------------------------------
// Performance Analyzer Logic
// ---------------------------------------------------------------------------

pub struct PerformanceAnalyzer {
    results: HashMap<String, BenchmarkResults>,
    historical_data: Vec<PerformanceAnalysis>,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            historical_data: Vec::new(),
        }
    }

    /// Add benchmark results for a category
    pub fn add_results(&mut self, category: &str, results: BenchmarkResults) {
        self.results.insert(category.to_string(), results);
    }

    /// Add a single benchmark result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        let category = result.category.clone();
        self.results.entry(category).or_default().add_result(result);
    }

    /// Perform comprehensive performance analysis
    pub fn analyze(&self) -> PerformanceAnalysis {
        info!("🔍 Performing comprehensive performance analysis");

        let summary = self.compute_summary();
        let category_analysis = self.analyze_categories();
        let efficiency_rankings = self.compute_efficiency_rankings();
        let performance_trends = self.analyze_trends();
        let bottleneck_analysis = self.analyze_bottlenecks();
        let recommendations =
            self.generate_recommendations(&category_analysis, &bottleneck_analysis);

        PerformanceAnalysis {
            summary,
            category_analysis,
            efficiency_rankings,
            performance_trends,
            bottleneck_analysis,
            recommendations,
        }
    }

    /// Compute overall performance summary
    fn compute_summary(&self) -> PerformanceSummary {
        let mut total_benchmarks = 0;
        let mut total_duration = Duration::ZERO;
        let mut throughput_sum = 0.0;
        let mut efficiency_sum = 0.0;
        let mut mem_util_sum = 0.0;
        let mut device_util_sum = 0.0;
        let mut category_scores = HashMap::new();

        for (category, results) in &self.results {
            let category_throughput = results.average_throughput(category);
            let category_efficiency = results.average_efficiency(category);

            // Score is a blend of log-throughput and efficiency
            let category_score =
                (category_throughput.ln().max(0.0) + category_efficiency * 10.0) / 2.0;

            category_scores.insert(category.clone(), category_score);

            for result in &results.results {
                total_benchmarks += 1;
                total_duration += std::time::Duration::from(result.duration);
                throughput_sum += result.throughput;
                efficiency_sum += result.efficiency;

                // Memory efficiency heuristic: 1.0 - (usage / baseline_max)
                // Simplified here to direct normalization if usage > 0
                let mem_score = if result.memory_usage > 0 {
                    1.0 / (1.0 + (result.memory_usage as f64 / 1_000_000_000.0).ln().max(0.0))
                } else {
                    1.0
                };
                mem_util_sum += mem_score;
                device_util_sum += result.device_utilization;
            }
        }

        let count_f64 = total_benchmarks as f64;
        let average_throughput = if total_benchmarks > 0 {
            throughput_sum / count_f64
        } else {
            0.0
        };
        let average_efficiency = if total_benchmarks > 0 {
            efficiency_sum / count_f64
        } else {
            0.0
        };
        let avg_mem_efficiency = if total_benchmarks > 0 {
            mem_util_sum / count_f64
        } else {
            0.0
        };
        let avg_device_util = if total_benchmarks > 0 {
            device_util_sum / count_f64
        } else {
            0.0
        };

        let best_category = category_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        let worst_category = category_scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        let overall_score = (average_throughput.ln().max(0.0) + average_efficiency * 10.0) / 2.0;

        PerformanceSummary {
            total_benchmarks,
            total_duration,
            average_throughput,
            average_efficiency,
            best_performing_category: best_category,
            worst_performing_category: worst_category,
            overall_score,
            total_execution_time: total_duration,
            memory_efficiency: avg_mem_efficiency,
            device_utilization: avg_device_util,
        }
    }

    /// Analyze performance by category
    fn analyze_categories(&self) -> HashMap<String, CategoryAnalysis> {
        let mut category_analysis = HashMap::new();

        for (category, results) in &self.results {
            let category_results = results.get_category_results(category);

            if category_results.is_empty() {
                continue;
            }

            let throughputs: Vec<f64> = category_results.iter().map(|r| r.throughput).collect();
            let efficiencies: Vec<f64> = category_results.iter().map(|r| r.efficiency).collect();

            let count = throughputs.len() as f64;
            let avg_throughput = throughputs.iter().sum::<f64>() / count;
            let avg_efficiency = efficiencies.iter().sum::<f64>() / count;

            let throughput_variance = throughputs
                .iter()
                .map(|t| (t - avg_throughput).powi(2))
                .sum::<f64>()
                / count;
            let throughput_std_dev = throughput_variance.sqrt();

            let efficiency_variance = efficiencies
                .iter()
                .map(|e| (e - avg_efficiency).powi(2))
                .sum::<f64>()
                / count;
            let efficiency_std_dev = efficiency_variance.sqrt();

            let best_benchmark = category_results
                .iter()
                .max_by(|a, b| {
                    a.efficiency
                        .partial_cmp(&b.efficiency)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|r| r.name.clone())
                .unwrap_or_default();

            let worst_benchmark = category_results
                .iter()
                .min_by(|a, b| {
                    a.efficiency
                        .partial_cmp(&b.efficiency)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|r| r.name.clone())
                .unwrap_or_default();

            // Performance consistency: lower std dev = higher consistency
            let performance_consistency =
                1.0 / (1.0 + throughput_std_dev / avg_throughput.max(1e-9));

            let memory_usage_avg = category_results.iter().map(|r| r.memory_usage).sum::<u64>()
                / category_results.len() as u64;

            let analysis = CategoryAnalysis {
                category: category.clone(),
                benchmark_count: category_results.len(),
                average_throughput: avg_throughput,
                throughput_std_dev,
                average_efficiency: avg_efficiency,
                efficiency_std_dev,
                best_benchmark,
                worst_benchmark,
                performance_consistency,
                efficiency_score: avg_efficiency,
                memory_usage_avg,
                bottlenecks: Vec::new(), // Populated later via bottleneck analysis
            };

            category_analysis.insert(category.clone(), analysis);
        }

        category_analysis
    }

    /// Compute efficiency rankings across all benchmarks
    fn compute_efficiency_rankings(&self) -> Vec<EfficiencyRanking> {
        let mut all_results = Vec::new();

        // Collect all results with category averages for normalization
        let category_averages: HashMap<String, f64> = self
            .results
            .iter()
            .map(|(cat, results)| (cat.clone(), results.average_efficiency(cat)))
            .collect();

        for (category, results) in &self.results {
            let category_avg = category_averages
                .get(category)
                .cloned()
                .unwrap_or(1.0)
                .max(1e-9);

            for result in &results.results {
                let relative_performance = result.efficiency / category_avg;

                all_results.push((
                    result.name.clone(),
                    category.clone(),
                    result.efficiency,
                    result.throughput,
                    relative_performance,
                ));
            }
        }

        // Sort by efficiency score (descending)
        all_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        all_results
            .into_iter()
            .enumerate()
            .map(
                |(i, (name, category, efficiency, throughput, relative))| EfficiencyRanking {
                    rank: i + 1,
                    benchmark_name: name,
                    category,
                    efficiency_score: efficiency,
                    throughput,
                    relative_performance: relative,
                },
            )
            .collect()
    }

    /// Analyze performance trends via linear regression
    fn analyze_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();

        for (category, results) in &self.results {
            let category_results = results.get_category_results(category);

            if category_results.len() < 3 {
                continue; // Need at least 3 points for meaningful trend analysis
            }

            // Analyze throughput trend
            let throughputs: Vec<f64> = category_results.iter().map(|r| r.throughput).collect();
            let throughput_trend = self.compute_trend(&throughputs);

            // Analyze efficiency trend
            let efficiencies: Vec<f64> = category_results.iter().map(|r| r.efficiency).collect();
            let efficiency_trend = self.compute_trend(&efficiencies);

            // Combine trends
            let combined_trend = (throughput_trend + efficiency_trend) / 2.0;

            let (trend_type, description) = if combined_trend > 0.1 {
                (
                    TrendType::Improving,
                    "Performance is strictly improving over iterations".to_string(),
                )
            } else if combined_trend < -0.1 {
                (
                    TrendType::Degrading,
                    "Performance is degrading over time (potential leak/throttling)".to_string(),
                )
            } else {
                // Check volatility
                let avg = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
                let volatility = throughputs.iter().map(|t| (t - avg).abs()).sum::<f64>()
                    / throughputs.len() as f64;

                if volatility > avg * 0.2 {
                    (
                        TrendType::Volatile,
                        "Performance shows high volatility (>20% variance)".to_string(),
                    )
                } else {
                    (TrendType::Stable, "Performance is stable".to_string())
                }
            };

            trends.push(PerformanceTrend {
                category: category.clone(),
                trend_type,
                trend_strength: combined_trend,
                confidence: 0.8, // Placeholder: Real confidence requires t-test
                description,
                metric: "Combined Performance".to_string(),
                trend_direction: if combined_trend > 0.0 {
                    TrendDirection::Improving
                } else {
                    TrendDirection::Declining
                },
                slope: combined_trend,
                r_squared: 0.75, // Placeholder
            });
        }

        trends
    }

    /// Compute trend strength (slope normalized by mean)
    fn compute_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * x2_sum - x_sum.powi(2);
        if denominator.abs() < 1e-9 {
            return 0.0;
        }

        let slope = (n * xy_sum - x_sum * y_sum) / denominator;

        // Normalize slope by average value to get percentage-like trend
        let avg_value = y_sum / n;
        slope / avg_value.max(1e-9)
    }

    /// Analyze performance bottlenecks based on collected metrics
    fn analyze_bottlenecks(&self) -> BottleneckAnalysis {
        let mut bottlenecks = Vec::new();

        let all_results: Vec<&BenchmarkResult> =
            self.results.values().flat_map(|r| &r.results).collect();

        if all_results.is_empty() {
            return BottleneckAnalysis {
                primary_bottlenecks: vec![],
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 0.0,
                    cpu_efficiency: 0.0,
                    memory_utilization: 0.0,
                    memory_efficiency: 0.0,
                    cache_efficiency: 0.0,
                    gpu_utilization: 0.0,
                    gpu_efficiency: 0.0,
                    io_utilization: 0.0,
                },
                scaling_analysis: ScalingAnalysis {
                    linear_scaling_score: 0.0,
                    parallel_efficiency: 0.0,
                    optimal_batch_sizes: HashMap::new(),
                },
                optimization_suggestions: vec![],
            };
        }

        // 1. Memory Bottlenecks (> 1GB usage)
        let memory_intensive_benchmarks: Vec<String> = all_results
            .iter()
            .filter(|result| result.memory_usage > 1024 * 1024 * 1024)
            .map(|result| result.name.clone())
            .collect();

        if !memory_intensive_benchmarks.is_empty() {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Memory,
                resource: "Memory".to_string(),
                affected_benchmarks: memory_intensive_benchmarks.clone(),
                severity: 0.7,
                description: "High memory usage (>1GB) detected in several benchmarks".to_string(),
            });
        }

        // 2. Cache Bottlenecks (< 50% Hit Rate)
        let low_cache_benchmarks: Vec<String> = all_results
            .iter()
            .filter(|result| result.cache_hit_rate < 0.5)
            .map(|result| result.name.clone())
            .collect();

        if !low_cache_benchmarks.is_empty() {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Cache,
                resource: "Cache".to_string(),
                affected_benchmarks: low_cache_benchmarks.clone(),
                severity: 0.6,
                description:
                    "Low cache hit rates (<50%) indicating pointer chasing or poor locality"
                        .to_string(),
            });
        }

        // 3. Compute Bottlenecks (< 60% Utilization)
        let low_utilization_benchmarks: Vec<String> = all_results
            .iter()
            .filter(|result| result.device_utilization < 0.6)
            .map(|result| result.name.clone())
            .collect();

        if !low_utilization_benchmarks.is_empty() {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Compute,
                resource: "Compute".to_string(),
                affected_benchmarks: low_utilization_benchmarks.clone(),
                severity: 0.5,
                description:
                    "Low device utilization (<60%) indicates I/O or synchronization binding"
                        .to_string(),
            });
        }

        // Aggregate Metrics
        let avg_cache_hit_rate =
            all_results.iter().map(|r| r.cache_hit_rate).sum::<f64>() / all_results.len() as f64;
        let avg_device_utilization = all_results
            .iter()
            .map(|r| r.device_utilization)
            .sum::<f64>()
            / all_results.len() as f64;

        // Heuristic for memory efficiency
        let avg_mem_eff = all_results
            .iter()
            .map(|r| {
                if r.throughput > 0.0 && r.memory_usage > 0 {
                    r.throughput / r.memory_usage as f64
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / all_results.len() as f64;

        let resource_utilization = ResourceUtilization {
            cpu_utilization: avg_device_utilization,
            cpu_efficiency: avg_device_utilization,
            memory_utilization: 0.75, // Approximation
            memory_efficiency: avg_mem_eff,
            cache_efficiency: avg_cache_hit_rate,
            gpu_utilization: avg_device_utilization,
            gpu_efficiency: avg_device_utilization,
            io_utilization: 0.6, // Approximation
        };

        // Scaling Analysis (Simplified heuristic)
        let scaling_analysis = ScalingAnalysis {
            linear_scaling_score: 0.85,
            parallel_efficiency: avg_device_utilization,
            optimal_batch_sizes: HashMap::new(),
        };

        let mut suggestions = Vec::new();
        if !memory_intensive_benchmarks.is_empty() {
            suggestions.push(
                "Consider increasing batch sizes for memory-intensive operations".to_string(),
            );
        }
        if !low_cache_benchmarks.is_empty() {
            suggestions.push(
                "Optimize cache usage patterns (SoA vs AoS) to improve hit rates".to_string(),
            );
        }
        if !low_utilization_benchmarks.is_empty() {
            suggestions
                .push("Investigate parallel processing or async IO opportunities".to_string());
        }

        BottleneckAnalysis {
            primary_bottlenecks: bottlenecks,
            resource_utilization,
            scaling_analysis,
            optimization_suggestions: suggestions,
        }
    }

    /// Generate performance recommendations based on analysis
    fn generate_recommendations(
        &self,
        category_analysis: &HashMap<String, CategoryAnalysis>,
        bottleneck_analysis: &BottleneckAnalysis,
    ) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        // 1. Address Primary Bottlenecks
        for bottleneck in &bottleneck_analysis.primary_bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::Memory => {
                    recommendations.push(PerformanceRecommendation {
                        priority: RecommendationPriority::High,
                        category: "Memory".to_string(),
                        title: "Optimize Memory Allocations".to_string(),
                        description: "High memory usage detected. Implement memory pooling, use arena allocators, or optimize data structures to reduce footprint. Avoid allocation in hot loops.".to_string(),
                        expected_improvement: 15.0,
                        implementation_effort: ImplementationEffort::Medium,
                    });
                }
                BottleneckType::Cache => {
                    recommendations.push(PerformanceRecommendation {
                        priority: RecommendationPriority::Medium,
                        category: "Cache".to_string(),
                        title: "Improve Data Locality".to_string(),
                        description: "Low cache hit rates detected. Reorganize data structures (e.g., flatten linked lists, use contiguous arrays) and implement software prefetching for sequential access.".to_string(),
                        expected_improvement: 20.0,
                        implementation_effort: ImplementationEffort::High,
                    });
                }
                BottleneckType::Compute => {
                    recommendations.push(PerformanceRecommendation {
                        priority: RecommendationPriority::High,
                        category: "Compute".to_string(),
                        title: "Maximize Compute Saturation".to_string(),
                        description: "Low device utilization. Review algorithms for branch divergence, increase vectorization (SIMD), or offload tasks to GPU if applicable. Check for lock contention.".to_string(),
                        expected_improvement: 30.0,
                        implementation_effort: ImplementationEffort::High,
                    });
                }
                _ => {}
            }
        }

        // 2. Category Specific Recommendations
        for (category, analysis) in category_analysis {
            if analysis.performance_consistency < 0.7 {
                recommendations.push(PerformanceRecommendation {
                    priority: RecommendationPriority::Medium,
                    category: category.clone(),
                    title: "Improve Performance Consistency".to_string(),
                    description: format!(
                        "High variance in {} benchmarks. Investigate garbage collection pauses (if applicable), OS scheduler jitter, or resource contention.",
                        category
                    ),
                    expected_improvement: 10.0,
                    implementation_effort: ImplementationEffort::Low,
                });
            }

            if analysis.average_efficiency < 0.8 {
                recommendations.push(PerformanceRecommendation {
                    priority: RecommendationPriority::High,
                    category: category.clone(),
                    title: "Algorithmic Optimization Required".to_string(),
                    description: format!("Efficiency score for {} is below target (80%). Review algorithmic complexity (Big O) and eliminate redundant operations.", category),
                    expected_improvement: 30.0,
                    implementation_effort: ImplementationEffort::High,
                });
            }
        }

        // Sort by priority (Critical -> Low) then by expected improvement
        recommendations.sort_by(|a, b| {
            let priority_order = |p: &RecommendationPriority| match p {
                RecommendationPriority::Critical => 0,
                RecommendationPriority::High => 1,
                RecommendationPriority::Medium => 2,
                RecommendationPriority::Low => 3,
            };

            priority_order(&a.priority)
                .cmp(&priority_order(&b.priority))
                .then(
                    b.expected_improvement
                        .partial_cmp(&a.expected_improvement)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        recommendations
    }

    /// Add historical analysis for trend tracking
    pub fn add_historical_analysis(&mut self, analysis: PerformanceAnalysis) {
        self.historical_data.push(analysis);

        // Keep only the last 100 analyses to prevent unbounded growth
        if self.historical_data.len() > 100 {
            self.historical_data.remove(0);
        }

        info!(
            "Added historical analysis. Total history: {}",
            self.historical_data.len()
        );
    }

    /// Detect performance regressions against history
    pub fn detect_regressions(&self) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();

        if self.historical_data.len() < 2 {
            return regressions; // Need at least 2 data points
        }

        let current = &self.historical_data[self.historical_data.len() - 1];
        let previous = &self.historical_data[self.historical_data.len() - 2];

        // Compare throughput across categories
        for (category, current_analysis) in &current.category_analysis {
            if let Some(previous_analysis) = previous.category_analysis.get(category) {
                let throughput_change = (current_analysis.average_throughput
                    - previous_analysis.average_throughput)
                    / previous_analysis.average_throughput.max(1e-9)
                    * 100.0;

                if throughput_change < -5.0 {
                    let severity = match throughput_change.abs() {
                        x if x < 5.0 => RegressionSeverity::Minor,
                        x if x < 15.0 => RegressionSeverity::Moderate,
                        x if x < 30.0 => RegressionSeverity::Major,
                        _ => RegressionSeverity::Critical,
                    };

                    regressions.push(PerformanceRegression {
                        regression_type: RegressionType::Throughput,
                        severity,
                        affected_area: "Overall".to_string(),
                        description: format!(
                            "Throughput regression in {}: {:.1}% decrease",
                            category,
                            throughput_change.abs()
                        ),
                        benchmark_name: format!("{}_throughput", category),
                        category: category.clone(),
                        metric: "throughput".to_string(),
                        previous_value: previous_analysis.average_throughput,
                        current_value: current_analysis.average_throughput,
                        regression_percentage: throughput_change.abs(),
                    });
                }

                // Compare efficiency scores
                let efficiency_change = (current_analysis.efficiency_score
                    - previous_analysis.efficiency_score)
                    / previous_analysis.efficiency_score.max(1e-9)
                    * 100.0;

                if efficiency_change < -5.0 {
                    let severity = match efficiency_change.abs() {
                        x if x < 5.0 => RegressionSeverity::Minor,
                        x if x < 15.0 => RegressionSeverity::Moderate,
                        x if x < 30.0 => RegressionSeverity::Major,
                        _ => RegressionSeverity::Critical,
                    };

                    regressions.push(PerformanceRegression {
                        regression_type: RegressionType::Accuracy,
                        severity,
                        affected_area: "Overall".to_string(),
                        description: format!(
                            "Efficiency regression in {}: {:.1}% decrease",
                            category,
                            efficiency_change.abs()
                        ),
                        benchmark_name: format!("{}_efficiency", category),
                        category: category.clone(),
                        metric: "efficiency".to_string(),
                        previous_value: previous_analysis.efficiency_score,
                        current_value: current_analysis.efficiency_score,
                        regression_percentage: efficiency_change.abs(),
                    });
                }
            }
        }

        if !regressions.is_empty() {
            warn!("Detected {} performance regressions", regressions.len());
        }
        regressions
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
