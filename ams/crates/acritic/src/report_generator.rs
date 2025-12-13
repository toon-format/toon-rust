/* arcmoon-suite/src/report_generator.rs */
//! Performance Report Generation
//!
//! This module provides comprehensive report generation capabilities for benchmark
//! results, including HTML reports, JSON exports, and performance visualizations.
//!
//! ### Capabilities
//! - **Multi-Format Export:** Generates HTML, JSON, CSV, and Markdown reports simultaneously.
//! - **Visual Analytics:** HTML reports include CSS-styled metric cards and efficiency rankings.
//! - **Data Persistence:** Raw JSON exports allow for historical tracking and external regression analysis.
//!
//! ### Architectural Notes
//! Designed to be run at the end of a benchmark suite execution. It consumes
//! `PerformanceAnalysis` structs and renders them into human and machine-readable formats.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::performance_analysis::{PerformanceAnalysis, PerformanceAnalyzer};
use chrono::{DateTime, Utc};
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

/// Report generator for all performance benchmarks
pub struct ReportGenerator {
    output_dir: PathBuf,
    timestamp: String,
    unix_timestamp: u64,
}

impl ReportGenerator {
    /// Create a new report generator instance
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let output_dir = output_dir.as_ref().to_path_buf();

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir)?;

        // Generate timestamp for this report session
        let now: DateTime<Utc> = Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S").to_string();

        // Record initialization timestamp for precise unix epoch tracking
        let unix_timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        info!(
            "ReportGenerator initialized at unix timestamp: {}",
            unix_timestamp
        );

        Ok(Self {
            output_dir,
            timestamp,
            unix_timestamp,
        })
    }

    /// Generate comprehensive performance report
    pub async fn generate_report(
        &self,
        analyzer: &PerformanceAnalyzer,
        total_duration: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("📊 Generating comprehensive performance report");

        let analysis = analyzer.analyze();

        // Create timestamped report directory
        let report_dir = self
            .output_dir
            .join(format!("arcmoon_report_{}", self.timestamp));
        fs::create_dir_all(&report_dir)?;

        // Generate different report formats
        self.generate_json_report(&report_dir, &analysis).await?;
        self.generate_html_report(&report_dir, &analysis, total_duration)
            .await?;
        self.generate_csv_export(&report_dir, &analysis).await?;
        self.generate_markdown_summary(&report_dir, &analysis)
            .await?;

        // Generate visualizations data
        self.generate_visualization_data(&report_dir, &analysis)
            .await?;

        info!(
            "✅ Report generated successfully at: {}",
            report_dir.display()
        );

        Ok(())
    }

    /// Generate JSON report with full analysis data
    async fn generate_json_report(
        &self,
        report_dir: &Path,
        analysis: &PerformanceAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Generating JSON report");

        // Create comprehensive report with metadata including unix_timestamp
        let report_with_metadata = serde_json::json!({
            "metadata": {
                "generated_at": self.timestamp,
                "unix_timestamp": self.unix_timestamp,
                "generator_version": env!("CARGO_PKG_VERSION"),
                "arcmoon_version": "0.1.0"
            },
            "performance_analysis": analysis
        });

        let json_path = report_dir.join("performance_analysis.json");
        let json_content = serde_json::to_string_pretty(&report_with_metadata)?;

        if let Err(e) = fs::write(json_path, json_content) {
            error!("Failed to write JSON report to disk: {}", e);
            return Err(e.into());
        }

        Ok(())
    }

    /// Generate HTML report with styled elements
    async fn generate_html_report(
        &self,
        report_dir: &Path,
        analysis: &PerformanceAnalysis,
        total_duration: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Generating HTML report");

        let html_content = self.create_html_content(analysis, total_duration);
        let html_path = report_dir.join("performance_report.html");

        fs::write(html_path, html_content)?;

        Ok(())
    }

    /// Create HTML content for the performance report
    fn create_html_content(
        &self,
        analysis: &PerformanceAnalysis,
        total_duration: Duration,
    ) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArcMoon Performance Report - {}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.8;
            font-size: 1.1em;
        }}
        .content {{
            padding: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .metric-card.efficiency {{
            border-left-color: #2ecc71;
        }}
        .metric-card.throughput {{
            border-left-color: #e74c3c;
        }}
        .metric-card.duration {{
            border-left-color: #f39c12;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .category-analysis {{
            display: grid;
            gap: 20px;
        }}
        .category-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }}
        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .category-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .category-score {{
            background: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .category-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.8em;
        }}
        .rankings {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        .ranking-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .ranking-item:last-child {{
            border-bottom: none;
        }}
        .rank {{
            background: #3498db;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        .rank.gold {{
            background: #f1c40f;
        }}
        .rank.silver {{
            background: #95a5a6;
        }}
        .rank.bronze {{
            background: #e67e22;
        }}
        .benchmark-info {{
            flex-grow: 1;
            margin-left: 15px;
        }}
        .benchmark-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .benchmark-category {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .efficiency-score {{
            font-weight: bold;
            color: #27ae60;
        }}
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
        }}
        .recommendation {{
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #f39c12;
        }}
        .recommendation.high {{
            border-left-color: #e74c3c;
        }}
        .recommendation.medium {{
            border-left-color: #f39c12;
        }}
        .recommendation.low {{
            border-left-color: #2ecc71;
        }}
        .recommendation-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .recommendation-description {{
            color: #7f8c8d;
            margin-bottom: 10px;
        }}
        .recommendation-improvement {{
            color: #27ae60;
            font-weight: bold;
        }}
        .footer {{
            background: #ecf0f1;
            padding: 20px;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ArcMoon Performance Evaluation Report</h1>
            <div class="subtitle">Generated on {} • Total Duration: {:.2}s</div>
        </div>
        
        <div class="content">
            <!-- Performance Summary -->
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="summary-grid">
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Total Benchmarks</div>
                    </div>
                    <div class="metric-card throughput">
                        <div class="metric-value">{:.0}</div>
                        <div class="metric-label">Avg Throughput (ops/sec)</div>
                    </div>
                    <div class="metric-card efficiency">
                        <div class="metric-value">{:.1}%</div>
                        <div class="metric-label">Avg Efficiency</div>
                    </div>
                    <div class="metric-card duration">
                        <div class="metric-value">{:.1}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                </div>
            </div>
            
            <!-- Category Analysis -->
            <div class="section">
                <h2>🔍 Category Analysis</h2>
                <div class="category-analysis">
                    {}
                </div>
            </div>
            
            <!-- Efficiency Rankings -->
            <div class="section">
                <h2>🏆 Top Performance Rankings</h2>
                <div class="rankings">
                    {}
                </div>
            </div>
            
            <!-- Recommendations -->
            <div class="section">
                <h2>💡 Performance Recommendations</h2>
                <div class="recommendations">
                    {}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>ArcMoon's Testing & Performance Evaluation Suite © 2025 ArcMoon Studios</p>
            <p>Report generated with ❤️ by the ArcMoon's Automated Benchmarking Ecosystem</p>
        </div>
    </div>
</body>
</html>"#,
            self.timestamp,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            total_duration.as_secs_f64(),
            analysis.summary.total_benchmarks,
            analysis.summary.average_throughput,
            analysis.summary.average_efficiency * 100.0,
            analysis.summary.overall_score,
            self.generate_category_cards(&analysis.category_analysis),
            self.generate_ranking_items(&analysis.efficiency_rankings),
            self.generate_recommendation_items(&analysis.recommendations)
        )
    }

    /// Generate HTML for category analysis cards
    fn generate_category_cards(
        &self,
        categories: &std::collections::HashMap<
            String,
            super::performance_analysis::CategoryAnalysis,
        >,
    ) -> String {
        categories
            .values()
            .map(|analysis| {
                format!(
                    r#"
                <div class="category-card">
                    <div class="category-header">
                        <div class="category-name">{}</div>
                        <div class="category-score">{:.1}%</div>
                    </div>
                    <div class="category-stats">
                        <div class="stat">
                            <div class="stat-value">{}</div>
                            <div class="stat-label">Benchmarks</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{:.0}</div>
                            <div class="stat-label">Avg Throughput</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{:.1}%</div>
                            <div class="stat-label">Consistency</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{}</div>
                            <div class="stat-label">Best Benchmark</div>
                        </div>
                    </div>
                </div>
            "#,
                    analysis.category,
                    analysis.average_efficiency * 100.0,
                    analysis.benchmark_count,
                    analysis.average_throughput,
                    analysis.performance_consistency * 100.0,
                    analysis.best_benchmark
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate HTML for efficiency ranking items
    fn generate_ranking_items(
        &self,
        rankings: &[super::performance_analysis::EfficiencyRanking],
    ) -> String {
        rankings
            .iter()
            .take(10)
            .map(|ranking| {
                let rank_class = match ranking.rank {
                    1 => "gold",
                    2 => "silver",
                    3 => "bronze",
                    _ => "",
                };

                format!(
                    r#"
                <div class="ranking-item">
                    <div class="rank {}">#{}</div>
                    <div class="benchmark-info">
                        <div class="benchmark-name">{}</div>
                        <div class="benchmark-category">{}</div>
                    </div>
                    <div class="efficiency-score">{:.1}%</div>
                </div>
            "#,
                    rank_class,
                    ranking.rank,
                    ranking.benchmark_name,
                    ranking.category,
                    ranking.efficiency_score * 100.0
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate HTML for recommendation items
    fn generate_recommendation_items(
        &self,
        recommendations: &[super::performance_analysis::PerformanceRecommendation],
    ) -> String {
        recommendations
            .iter()
            .take(5)
            .map(|rec| {
                let priority_class = match rec.priority {
                    super::performance_analysis::RecommendationPriority::Critical => "critical",
                    super::performance_analysis::RecommendationPriority::High => "high",
                    super::performance_analysis::RecommendationPriority::Medium => "medium",
                    super::performance_analysis::RecommendationPriority::Low => "low",
                };

                format!(
                    r#"
                <div class="recommendation {}">
                    <div class="recommendation-title">{}</div>
                    <div class="recommendation-description">{}</div>
                    <div class="recommendation-improvement">Expected improvement: +{:.1}%</div>
                </div>
            "#,
                    priority_class, rec.title, rec.description, rec.expected_improvement
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate CSV export for data analysis
    async fn generate_csv_export(
        &self,
        report_dir: &Path,
        analysis: &PerformanceAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Generating CSV export");

        let csv_path = report_dir.join("benchmark_results.csv");
        let mut csv_content =
            String::from("Rank,Benchmark,Category,Efficiency,Throughput,Relative Performance\n");

        for ranking in &analysis.efficiency_rankings {
            csv_content.push_str(&format!(
                "{},{},{},{:.4},{:.2},{:.4}\n",
                ranking.rank,
                ranking.benchmark_name,
                ranking.category,
                ranking.efficiency_score,
                ranking.throughput,
                ranking.relative_performance
            ));
        }

        fs::write(csv_path, csv_content)?;

        Ok(())
    }

    /// Generate markdown summary
    async fn generate_markdown_summary(
        &self,
        report_dir: &Path,
        analysis: &PerformanceAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Generating Markdown summary");

        let md_content = format!(
            r#"# ArcMoon Performance Evaluation Report

Generated: {}

## 📊 Executive Summary

- **Total Benchmarks**: {}
- **Average Throughput**: {:.0} ops/sec
- **Average Efficiency**: {:.1}%
- **Overall Score**: {:.2}
- **Best Category**: {}
- **Needs Attention**: {}

## 🏆 Top Performers

{}

## 💡 Key Recommendations

{}

## 📈 Category Performance

{}

---
*Report generated by ArcMoon's Testing & Performance Evaluation Suite*
"#,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            analysis.summary.total_benchmarks,
            analysis.summary.average_throughput,
            analysis.summary.average_efficiency * 100.0,
            analysis.summary.overall_score,
            analysis.summary.best_performing_category,
            analysis.summary.worst_performing_category,
            self.generate_top_performers_md(&analysis.efficiency_rankings),
            self.generate_recommendations_md(&analysis.recommendations),
            self.generate_category_performance_md(&analysis.category_analysis)
        );

        let md_path = report_dir.join("README.md");
        fs::write(md_path, md_content)?;

        Ok(())
    }

    /// Generate top performers section for markdown
    fn generate_top_performers_md(
        &self,
        rankings: &[super::performance_analysis::EfficiencyRanking],
    ) -> String {
        let mut content = String::new();

        for (i, ranking) in rankings.iter().take(5).enumerate() {
            let medal = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "🏅",
            };

            content.push_str(&format!(
                "{} **{}** ({}): {:.1}% efficiency\n",
                medal,
                ranking.benchmark_name,
                ranking.category,
                ranking.efficiency_score * 100.0
            ));
        }

        content
    }

    /// Generate recommendations section for markdown
    fn generate_recommendations_md(
        &self,
        recommendations: &[super::performance_analysis::PerformanceRecommendation],
    ) -> String {
        let mut content = String::new();

        for rec in recommendations.iter().take(3) {
            let priority_emoji = match rec.priority {
                super::performance_analysis::RecommendationPriority::Critical => "🚨",
                super::performance_analysis::RecommendationPriority::High => "⚠️",
                super::performance_analysis::RecommendationPriority::Medium => "💡",
                super::performance_analysis::RecommendationPriority::Low => "📝",
            };

            content.push_str(&format!(
                "{} **{}**: {} (Expected: +{:.1}%)\n",
                priority_emoji, rec.title, rec.description, rec.expected_improvement
            ));
        }

        content
    }

    /// Generate category performance section for markdown
    fn generate_category_performance_md(
        &self,
        categories: &std::collections::HashMap<
            String,
            super::performance_analysis::CategoryAnalysis,
        >,
    ) -> String {
        let mut content = String::new();

        for analysis in categories.values() {
            content.push_str(&format!(
                "### {}\n- Benchmarks: {}\n- Avg Efficiency: {:.1}%\n- Consistency: {:.1}%\n- Best: {}\n\n",
                analysis.category,
                analysis.benchmark_count,
                analysis.average_efficiency * 100.0,
                analysis.performance_consistency * 100.0,
                analysis.best_benchmark
            ));
        }

        content
    }

    /// Generate performance charts data for external visualization tools
    async fn generate_visualization_data(
        &self,
        report_dir: &Path,
        analysis: &PerformanceAnalysis,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Generating visualization data");

        // Generates structured JSON data optimized for chart libraries (e.g. Chart.js/Plotly)
        // This file serves as the data source for any frontend visualization components.
        let chart_data = serde_json::json!({
            "efficiency_by_category": analysis.category_analysis.iter().map(|(name, cat)| {
                serde_json::json!({
                    "category": name,
                    "efficiency": cat.average_efficiency,
                    "throughput": cat.average_throughput
                })
            }).collect::<Vec<_>>(),
            "top_benchmarks": analysis.efficiency_rankings.iter().take(10).map(|rank| {
                serde_json::json!({
                    "name": rank.benchmark_name,
                    "efficiency": rank.efficiency_score,
                    "category": rank.category
                })
            }).collect::<Vec<_>>()
        });

        let chart_path = report_dir.join("chart_data.json");
        fs::write(chart_path, serde_json::to_string_pretty(&chart_data)?)?;

        Ok(())
    }
}
