/* src/test_harness.rs */
//! Test Harness with Automatic Report Generation
//!
//! This module provides a robust, asynchronous test harness that automatically generates
//! timestamped reports for all test and benchmark executions in the ArcMoon Suite workspace.
//!
//! ### Capabilities
//! - **Async Execution:** Uses `tokio::process` to run Cargo commands without blocking the reactor.
//! - **Output Capture:** Captures stdout/stderr for analysis while streaming options are available.
//! - **Result Parsing:** Parses standard Cargo test output and Criterion benchmark results into structured data.
//! - **Report Lifecycle:** Manages the creation, storage, and rotation of report directories.
//!
//! ### Architectural Notes
//! This harness acts as a middleware between the user CLI and the underlying Cargo toolchain,
//! injecting observability and analysis into the standard development workflow.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::performance_analysis::{BenchmarkResult, PerformanceAnalyzer};
use crate::report_generator::ReportGenerator;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::env;
use std::fs; // blocking fs is acceptable for setup/teardown of small files
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::process::Command; // Async process execution
use tracing::{debug, error, info, warn};

/// Test execution type
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionType {
    Test,
    Bench,
}

/// Test harness configuration
#[derive(Debug, Clone)]
pub struct TestHarnessConfig {
    /// Root directory for reports (.zExGate-Reports/)
    pub reports_root: PathBuf,
    /// Whether to generate HTML reports
    pub generate_html: bool,
    /// Whether to generate JSON reports
    pub generate_json: bool,
    /// Whether to generate CSV exports
    pub generate_csv: bool,
    /// Whether to capture stdout/stderr
    pub capture_output: bool,
    /// Maximum number of reports to keep per crate
    pub max_reports_per_crate: usize,
}

impl Default for TestHarnessConfig {
    fn default() -> Self {
        Self {
            reports_root: PathBuf::from(".zExGate-Reports"),
            generate_html: true,
            generate_json: true,
            generate_csv: true,
            capture_output: true,
            max_reports_per_crate: 50,
        }
    }
}

/// Test harness for automated report generation
pub struct TestHarness {
    config: TestHarnessConfig,
    workspace_root: PathBuf,
}

impl TestHarness {
    /// Create a new test harness
    pub fn new(config: TestHarnessConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let workspace_root = Self::find_workspace_root()?;

        // Ensure reports directory exists
        let reports_root = workspace_root.join(&config.reports_root);
        if !reports_root.exists() {
            fs::create_dir_all(&reports_root)?;
        }
        fs::create_dir_all(reports_root.join("tests"))?;
        fs::create_dir_all(reports_root.join("benches"))?;

        Ok(Self {
            config,
            workspace_root,
        })
    }

    /// Find the workspace root directory
    fn find_workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let current_dir = env::current_dir()?;
        let mut dir = current_dir.as_path();

        loop {
            if dir.join("Cargo.toml").exists() {
                // Check if this is a workspace by looking for [workspace] section
                let cargo_toml = fs::read_to_string(dir.join("Cargo.toml"))?;
                if cargo_toml.contains("[workspace]") {
                    return Ok(dir.to_path_buf());
                }
            }

            match dir.parent() {
                Some(parent) => dir = parent,
                None => break,
            }
        }

        // Fallback to current directory
        Ok(current_dir)
    }

    /// Generate timestamp in the format mm-dd-yy#NN
    fn generate_timestamp(
        &self,
        crate_name: &str,
        execution_type: ExecutionType,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let now: DateTime<Utc> = Utc::now();
        let date_part = now.format("%m-%d-%y").to_string();

        // Find the next available number for today
        let type_dir = match execution_type {
            ExecutionType::Test => "tests",
            ExecutionType::Bench => "benches",
        };

        let crate_reports_dir = self.config.reports_root.join(type_dir).join(crate_name);

        let mut counter = 1;
        loop {
            let timestamp = format!("{}#{:02}", date_part, counter);
            let report_dir = crate_reports_dir.join(&timestamp);

            if !report_dir.exists() {
                return Ok(timestamp);
            }

            counter += 1;
            if counter > 99 {
                // Fallback to unix timestamp if we somehow exceed 99 runs in a day
                let unix_ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                return Ok(format!("{}#{}", date_part, unix_ts % 10000));
            }
        }
    }

    /// Run tests for a specific crate with report generation
    pub async fn run_crate_tests(
        &self,
        crate_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("🧪 Running tests for crate: {}", crate_name);

        let timestamp = self.generate_timestamp(crate_name, ExecutionType::Test)?;
        let report_dir = self
            .config
            .reports_root
            .join("tests")
            .join(crate_name)
            .join(&timestamp);

        fs::create_dir_all(&report_dir)?;

        let start_time = Instant::now();

        // Run cargo test for the specific crate asynchronously
        let mut cmd = Command::new("cargo");
        cmd.arg("test")
            .arg("--package")
            .arg(crate_name)
            .current_dir(&self.workspace_root);

        if self.config.capture_output {
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
        }

        let output = cmd.output().await?; // Async await
        let duration = start_time.elapsed();

        // Save test output
        if self.config.capture_output {
            fs::write(report_dir.join("stdout.log"), &output.stdout)?;
            fs::write(report_dir.join("stderr.log"), &output.stderr)?;
        }

        // Parse test results from output
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let test_stats = self.parse_test_output(&stdout_str);

        // Create test execution report
        let now = Utc::now();
        let test_report = TestExecutionReport {
            crate_name: crate_name.to_string(),
            execution_type: ExecutionType::Test,
            timestamp: timestamp.clone(),
            duration,
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: stdout_str.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            test_count: test_stats.0,
            passed: test_stats.1,
            failed: test_stats.2,
            ignored: test_stats.3,
            filtered_out: test_stats.4,
            timestamp_formatted: now.format("%Y-%m-%d %H:%M:%S").to_string(),
        };

        // Generate reports
        self.generate_test_reports(&report_dir, &test_report)
            .await?;

        // Clean up old reports
        self.cleanup_old_reports(crate_name, ExecutionType::Test)?;

        if output.status.success() {
            info!("✅ Tests completed successfully for {}", crate_name);
        } else {
            warn!("❌ Tests failed for {}", crate_name);
        }

        Ok(())
    }

    /// Run benchmarks for a specific crate with report generation
    pub async fn run_crate_benchmarks(
        &self,
        crate_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("📊 Running benchmarks for crate: {}", crate_name);

        let timestamp = self.generate_timestamp(crate_name, ExecutionType::Bench)?;
        let report_dir = self
            .config
            .reports_root
            .join("benches")
            .join(crate_name)
            .join(&timestamp);

        fs::create_dir_all(&report_dir)?;

        let start_time = Instant::now();

        // Run cargo bench for the specific crate asynchronously
        let mut cmd = Command::new("cargo");
        cmd.arg("bench")
            .arg("--package")
            .arg(crate_name)
            .current_dir(&self.workspace_root);

        if self.config.capture_output {
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
        }

        let output = cmd.output().await?; // Async await
        let duration = start_time.elapsed();

        // Save benchmark output
        if self.config.capture_output {
            fs::write(report_dir.join("stdout.log"), &output.stdout)?;
            fs::write(report_dir.join("stderr.log"), &output.stderr)?;
        }

        // Parse benchmark results from output
        let benchmark_results = self.parse_benchmark_output(&output.stdout)?;

        // Create benchmark execution report
        let now = Utc::now();
        let bench_report = BenchmarkExecutionReport {
            crate_name: crate_name.to_string(),
            execution_type: ExecutionType::Bench,
            timestamp: timestamp.clone(),
            duration,
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            benchmark_count: benchmark_results.len() as u32,
            benchmark_results,
            timestamp_formatted: now.format("%Y-%m-%d %H:%M:%S").to_string(),
        };

        // Generate reports
        self.generate_benchmark_reports(&report_dir, &bench_report)
            .await?;

        // Clean up old reports
        self.cleanup_old_reports(crate_name, ExecutionType::Bench)?;

        if output.status.success() {
            info!("✅ Benchmarks completed successfully for {}", crate_name);
        } else {
            warn!("❌ Benchmarks failed for {}", crate_name);
        }

        Ok(())
    }

    /// Parse test output to extract test statistics
    /// Looks for the "test result: ok. X passed; Y failed; ..." line
    fn parse_test_output(&self, output: &str) -> (u32, u32, u32, u32, u32) {
        let mut test_count = 0;
        let mut passed = 0;
        let mut failed = 0;
        let mut ignored = 0;
        let mut filtered_out = 0;

        for line in output.lines() {
            if line.trim().starts_with("test result:") {
                // Format: "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"
                let parts: Vec<&str> = line.split_whitespace().collect();

                // Helper closure to parse numbers safely from preceding token
                let parse_prev = |idx: usize, vec: &[&str]| -> u32 {
                    if idx > 0 {
                        vec[idx - 1].parse::<u32>().unwrap_or(0)
                    } else {
                        0
                    }
                };

                for (i, part) in parts.iter().enumerate() {
                    // We match on the semicolon-suffixed words or specific keywords
                    if part.starts_with("passed") {
                        let count = parse_prev(i, &parts);
                        passed += count; // Accumulate if multiple output sections found
                        test_count += count;
                    } else if part.starts_with("failed") {
                        let count = parse_prev(i, &parts);
                        failed += count;
                        test_count += count;
                    } else if part.starts_with("ignored") {
                        ignored += parse_prev(i, &parts);
                    } else if part.starts_with("filtered") {
                        filtered_out += parse_prev(i, &parts);
                    }
                }
                // While theoretically there could be multiple result lines in a workspace run,
                // for a single-package run usually the last one is the summary.
                // We break here to capture the main result block.
                // If "cargo test" outputs multiple blocks (e.g. unit tests, integration tests, doc tests),
                // strictly speaking we should sum them.
                // For this implementation, we continue to accumulate if multiple "test result:" lines exist.
            }
        }

        (test_count, passed, failed, ignored, filtered_out)
    }

    /// Parse benchmark output to extract performance metrics
    fn parse_benchmark_output(
        &self,
        output: &[u8],
    ) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let output_str = String::from_utf8_lossy(output);
        let mut results = Vec::new();

        // Parse criterion benchmark output format
        for line in output_str.lines() {
            // Standard Criterion output line detection
            if line.contains("time:") && line.contains("[") && line.contains("]") {
                // Example: "benchmark_name        time:   [1.2345 us 1.2567 us 1.2789 us]"
                if let Some(benchmark_result) = self.parse_criterion_line(line) {
                    results.push(benchmark_result);
                }
            }
        }

        // If no criterion results found, create a placeholder to ensure report generation doesn't fail empty
        if results.is_empty() && output_str.contains("running") {
            // This might happen if benchmarks failed to run or output format is unexpected
            // We deliberately do not add a placeholder here to avoid skewing data with 0s.
            // The analyzer handles empty result sets gracefully.
        }

        Ok(results)
    }

    /// Parse a single criterion benchmark line
    fn parse_criterion_line(&self, line: &str) -> Option<BenchmarkResult> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 4 {
            return None;
        }

        // Heuristic: The benchmark name is usually the first token(s) before "time:"
        // But sometimes names have spaces. We look for "time:" index.
        let time_idx = parts.iter().position(|&r| r == "time:")?;
        if time_idx == 0 || time_idx + 2 >= parts.len() {
            return None;
        }

        let name = parts[0..time_idx].join(" ");

        // Criterion outputs: [lower_bound estimate upper_bound]
        // We want the estimate (middle value)
        // Format: [val unit val unit val unit]
        // Indices relative to time_idx:
        // time: [ lb_val lb_unit est_val est_unit ub_val ub_unit ]
        // idx:  +1 +2     +3      +4      +5       +6     +7

        // We look for the middle value.
        // Note: Criterion output spacing can be variable. We look for the '[' content.
        let bracket_content_start = line.find('[')?;
        let bracket_content_end = line.find(']')?;
        if bracket_content_start >= bracket_content_end {
            return None;
        }

        let content = &line[bracket_content_start + 1..bracket_content_end];
        let metrics: Vec<&str> = content.split_whitespace().collect();

        // Expecting 3 pairs: "1.2 us", "1.3 us", "1.4 us" -> 6 tokens
        if metrics.len() < 4 {
            return None;
        }

        // Take the middle pair (index 2 and 3) usually, or first pair if simpler output
        let (val_str, unit_str) = if metrics.len() >= 4 {
            (metrics[2], metrics[3])
        } else {
            (metrics[0], metrics[1])
        };

        if let Ok(time_val) = val_str.parse::<f64>() {
            let duration_ns = match unit_str {
                "ps" => (time_val / 1000.0) as u64,
                "ns" => time_val as u64,
                "us" | "µs" => (time_val * 1_000.0) as u64,
                "ms" => (time_val * 1_000_000.0) as u64,
                "s" => (time_val * 1_000_000_000.0) as u64,
                _ => time_val as u64, // Fallback
            };

            let throughput = if duration_ns > 0 {
                1_000_000_000.0 / duration_ns as f64
            } else {
                0.0
            };

            return Some(BenchmarkResult {
                name,
                category: "benchmark".to_string(),
                duration: SerializableDuration::from(Duration::from_nanos(duration_ns)),
                duration_ns,
                throughput,
                efficiency: 1.0, // Criterion doesn't give efficiency, assume baseline 1.0
                efficiency_score: 1.0,
                memory_usage: 0, // External profiler needed for this
                memory_usage_bytes: 0,
                cache_hit_rate: 0.0,
                device_utilization: 0.0,
                metadata: HashMap::new(),
            });
        }

        None
    }

    /// Generate test execution reports
    async fn generate_test_reports(
        &self,
        report_dir: &Path,
        report: &TestExecutionReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Generate JSON report
        if self.config.generate_json {
            let json_content = serde_json::to_string_pretty(report)?;
            fs::write(report_dir.join("test_report.json"), json_content)?;
        }

        // Generate HTML report
        if self.config.generate_html {
            let html_content = self.generate_test_html_report(report);
            fs::write(report_dir.join("test_report.html"), html_content)?;
        }

        // Generate CSV report
        if self.config.generate_csv {
            let csv_content = self.generate_test_csv_report(report);
            fs::write(report_dir.join("test_report.csv"), csv_content)?;
        }

        // Generate summary markdown
        let md_content = self.generate_test_markdown_report(report);
        fs::write(report_dir.join("README.md"), md_content)?;

        Ok(())
    }

    /// Generate benchmark execution reports
    async fn generate_benchmark_reports(
        &self,
        report_dir: &Path,
        report: &BenchmarkExecutionReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create performance analyzer with benchmark results
        let mut analyzer = PerformanceAnalyzer::new();
        for result in &report.benchmark_results {
            analyzer.add_result(result.clone());
        }

        // Generate comprehensive performance report using the existing report generator
        let report_generator = ReportGenerator::new(report_dir)?;
        report_generator
            .generate_report(&analyzer, report.duration)
            .await?;

        // Generate JSON report
        if self.config.generate_json {
            let json_content = serde_json::to_string_pretty(report)?;
            fs::write(report_dir.join("benchmark_report.json"), json_content)?;
        }

        // Generate CSV report
        if self.config.generate_csv {
            let csv_content = self.generate_benchmark_csv_report(report);
            fs::write(report_dir.join("benchmark_report.csv"), csv_content)?;
        }

        // Generate summary markdown
        let md_content = self.generate_benchmark_markdown_report(report);
        fs::write(report_dir.join("BENCHMARK_SUMMARY.md"), md_content)?;

        Ok(())
    }

    /// Generate HTML report for test execution
    fn generate_test_html_report(&self, report: &TestExecutionReport) -> String {
        let status_color = if report.success { "#27ae60" } else { "#e74c3c" };
        let status_text = if report.success {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        };

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {} - {}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0; font-size: 2em; }}
        .status {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin-top: 10px; background: {}; }}
        .content {{ padding: 30px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .info-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .info-label {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 5px; }}
        .info-value {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
        .output-section {{ margin-top: 30px; }}
        .output-section h3 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .output-content {{ background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.9em; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Test Report: {}</h1>
            <div class="status">{}</div>
        </div>
        <div class="content">
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">Execution Time</div>
                    <div class="info-value">{:.2}s</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Timestamp</div>
                    <div class="info-value">{}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Exit Code</div>
                    <div class="info-value">{}</div>
                </div>
            </div>
            
            <div class="output-section">
                <h3>📤 Standard Output</h3>
                <div class="output-content">{}</div>
            </div>
            
            <div class="output-section">
                <h3>⚠️ Standard Error</h3>
                <div class="output-content">{}</div>
            </div>
        </div>
    </div>
</body>
</html>"#,
            report.crate_name,
            report.timestamp,
            status_color,
            report.crate_name,
            status_text,
            report.duration.as_secs_f64(),
            report.timestamp,
            report.exit_code.unwrap_or(-1),
            html_escape(&report.stdout),
            html_escape(&report.stderr)
        )
    }

    /// Generate markdown report for test execution
    fn generate_test_markdown_report(&self, report: &TestExecutionReport) -> String {
        let status_emoji = if report.success { "✅" } else { "❌" };

        format!(
            r#"# Test Report: {}

{} **Status**: {}

## Execution Details

- **Timestamp**: {}
- **Duration**: {:.2}s
- **Exit Code**: {}

## Output Summary

### Standard Output

{}

### Standard Error

{}

---
*Generated by the ArcMoon Suite Test Harness*
"#,
            report.crate_name,
            status_emoji,
            if report.success { "PASSED" } else { "FAILED" },
            report.timestamp,
            report.duration.as_secs_f64(),
            report.exit_code.unwrap_or(-1),
            report.stdout.chars().take(1000).collect::<String>(),
            report.stderr.chars().take(1000).collect::<String>()
        )
    }

    /// Generate CSV report for test execution
    fn generate_test_csv_report(&self, report: &TestExecutionReport) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("timestamp,crate_name,success,duration_ms,test_count,passed,failed,ignored,filtered_out\n");

        // Data row
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            report.timestamp_formatted,
            report.crate_name,
            report.success,
            report.duration.as_millis(),
            report.test_count,
            report.passed,
            report.failed,
            report.ignored,
            report.filtered_out
        ));

        csv
    }

    /// Generate CSV report for benchmark execution
    fn generate_benchmark_csv_report(&self, report: &BenchmarkExecutionReport) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("timestamp,crate_name,success,duration_ms,benchmark_count\n");

        // Data row
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            report.timestamp_formatted,
            report.crate_name,
            report.success,
            report.duration.as_millis(),
            report.benchmark_count
        ));

        csv
    }

    /// Generate markdown report for benchmark execution
    fn generate_benchmark_markdown_report(&self, report: &BenchmarkExecutionReport) -> String {
        let status_emoji = if report.success { "✅" } else { "❌" };

        let mut benchmarks_summary = String::new();
        for result in &report.benchmark_results {
            benchmarks_summary.push_str(&format!(
                "- **{}**: {:.2} ops/sec ({:.2}ms avg)\n",
                result.name,
                result.throughput,
                result.duration_ns as f64 / 1_000_000.0
            ));
        }

        format!(
            r#"# Benchmark Report: {}

{} **Status**: {}

## Execution Details

- **Timestamp**: {}
- **Duration**: {:.2}s
- **Benchmarks Run**: {}

## Performance Summary

{}

## Detailed Results

See `performance_report.html` for comprehensive analysis and visualizations.

---
*Generated by the ArcMoon Suite Test Harness*
"#,
            report.crate_name,
            status_emoji,
            if report.success { "PASSED" } else { "FAILED" },
            report.timestamp,
            report.duration.as_secs_f64(),
            report.benchmark_results.len(),
            benchmarks_summary
        )
    }

    /// Clean up old reports, keeping only the most recent ones
    fn cleanup_old_reports(
        &self,
        crate_name: &str,
        execution_type: ExecutionType,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let type_dir = match execution_type {
            ExecutionType::Test => "tests",
            ExecutionType::Bench => "benches",
        };

        let crate_reports_dir = self.config.reports_root.join(type_dir).join(crate_name);

        if !crate_reports_dir.exists() {
            return Ok(());
        }

        // Get all report directories and sort by creation time
        let mut report_dirs = Vec::new();
        for entry in fs::read_dir(&crate_reports_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && let Ok(metadata) = entry.metadata()
                && let Ok(created) = metadata.created()
            {
                report_dirs.push((entry.path(), created));
            }
        }

        // Sort by creation time (newest first)
        report_dirs.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove old reports if we exceed the limit
        if report_dirs.len() > self.config.max_reports_per_crate {
            for (path, _) in report_dirs.iter().skip(self.config.max_reports_per_crate) {
                debug!("Removing old report directory: {}", path.display());
                if let Err(e) = fs::remove_dir_all(path) {
                    warn!(
                        "Failed to remove old report directory {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Run tests for all crates in the workspace
    pub async fn run_all_tests(&self) -> Result<(), Box<dyn std::error::Error>> {
        let crates = self.discover_workspace_crates()?;

        for crate_name in crates {
            if let Err(e) = self.run_crate_tests(&crate_name).await {
                error!("Failed to run tests for {}: {}", crate_name, e);
            }
        }

        Ok(())
    }

    /// Run benchmarks for all crates in the workspace
    pub async fn run_all_benchmarks(&self) -> Result<(), Box<dyn std::error::Error>> {
        let crates = self.discover_workspace_crates()?;

        for crate_name in crates {
            if let Err(e) = self.run_crate_benchmarks(&crate_name).await {
                error!("Failed to run benchmarks for {}: {}", crate_name, e);
            }
        }

        Ok(())
    }

    /// Discover all crates in the workspace
    fn discover_workspace_crates(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let cargo_toml_path = self.workspace_root.join("Cargo.toml");
        let cargo_toml_content = fs::read_to_string(cargo_toml_path)?;

        // Parse workspace members (simplified parsing)
        let mut crates = Vec::new();
        let mut in_members = false;

        for line in cargo_toml_content.lines() {
            let line = line.trim();
            if line.starts_with("members") {
                in_members = true;
                continue;
            }

            if in_members {
                if line.starts_with(']') {
                    break;
                }

                if line.starts_with('"') && line.ends_with('"') {
                    let crate_path = line.trim_matches('"').trim_end_matches(',');
                    if let Some(crate_name) = crate_path.split('/').next_back() {
                        crates.push(crate_name.to_string());
                    }
                }
            }
        }

        Ok(crates)
    }
}

/// Test execution report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestExecutionReport {
    pub crate_name: String,
    pub execution_type: ExecutionType,
    pub timestamp: String,
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub test_count: u32,
    pub passed: u32,
    pub failed: u32,
    pub ignored: u32,
    pub filtered_out: u32,
    pub timestamp_formatted: String,
}

/// Benchmark execution report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkExecutionReport {
    pub crate_name: String,
    pub execution_type: ExecutionType,
    pub timestamp: String,
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub benchmark_results: Vec<BenchmarkResult>,
    pub benchmark_count: u32,
    pub timestamp_formatted: String,
}

/// HTML escape utility
fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// Implement serde traits for ExecutionType
impl serde::Serialize for ExecutionType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ExecutionType::Test => serializer.serialize_str("test"),
            ExecutionType::Bench => serializer.serialize_str("bench"),
        }
    }
}

impl<'de> serde::Deserialize<'de> for ExecutionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "test" => Ok(ExecutionType::Test),
            "bench" => Ok(ExecutionType::Bench),
            _ => Err(serde::de::Error::custom("Invalid execution type")),
        }
    }
}

// Wrapper type for Duration serialization
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SerializableDuration(f64);

impl From<Duration> for SerializableDuration {
    fn from(duration: Duration) -> Self {
        SerializableDuration(duration.as_secs_f64())
    }
}

impl From<SerializableDuration> for Duration {
    fn from(duration: SerializableDuration) -> Self {
        Duration::from_secs_f64(duration.0)
    }
}

/// Duration serialization module
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}
