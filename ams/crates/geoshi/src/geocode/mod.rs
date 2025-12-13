/* src/geocode/mod.rs */
//! Codebase crystallography and geometric analysis
//!
//! # CODE GEOMETER MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced codebase analysis through geometric embedding of code quality metrics.
//! Maps language functions and modules onto E8 lattice points for structural analysis,
//! complexity assessment, and refactor recommendation generation.
//!
//! ### Key Capabilities
//! - **Function Metric Extraction:** Language-specific metric analysis.
//! - **Geometric Embedding:** 8D E8 lattice positioning for code quality vectors.
//! - **Dependency Mapping:** Call graph analysis with energy-based relationships.
//! - **Refactor Recommendations:** Geometric distance-based improvement suggestions.
//!
//! ### Technical Features
//! - **agnostic module:** Language-agnostic primitives and metrics.
//! - **rust module:** Rust-specific analysis services.
//! - **Lattice Integration:** Uses existing E8 lattice for multi-function positioning.
//!
//! ### Usage Patterns (Requires analyze feature)
//! ```rust
//! use geoshi::geocode::rust::CrateGeometer;
//!
//! let geometer = CrateGeometer::new();
//! let source = r#"fn example(a: i32, b: &mut Vec<i32>) -> i32 { a + 1 }"#;
//! let (root_idx, vector) = geometer.function_to_lattice(source).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Public modules
pub mod agnostic;
pub mod rust;
#[cfg(feature = "scan-rs")]
pub mod kdtree;
#[cfg(feature = "scan-rs")]
pub mod scanner;

// Re-exports for convenience
pub use agnostic::{FnId, HydronMetrics, HydronMetricsBuilder};
pub use rust::CrateGeometer;
#[cfg(feature = "scan-rs")]
pub use kdtree::KdTree;
#[cfg(feature = "scan-rs")]
pub use scanner::CodebaseScanner;
