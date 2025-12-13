/* src/geocode/scanner.rs */
//! Scans file systems to build geometric topologies of codebases
//!
//! # CODEBASE SCANNER MODULE
//!▫~•◦------------------------------------------------‣
//!
//! This module provides file system traversal and parsing capabilities to
//! convert directory trees of source code into geometric topologies.
//! It bridges the gap between the file system and the E8 lattice.
//!
//! ### Key Capabilities
//! - **Recursive Scanning:** Walks directory trees to find source files.
//! - **AST Parsing:** Uses `syn` to parse Rust source code into functions.
//! - **Metric Extraction:** Delegates to `CrateGeometer` for E8 embedding.
//! - **Topology Construction:** Builds a graph where edges represent geometric similarity.
//!
//! ### Usage
//! ```rust
//! use geoshi::geocode::CodebaseScanner;
//! use std::path::Path;
//!
//! let scanner = CodebaseScanner::new();
//! let topology = scanner.scan_crate(Path::new("./src")).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "scan-rs")]
use crate::{GsaError, GsaResult, geocode::{CrateGeometer, KdTree}, worm::Topology};
#[cfg(feature = "scan-rs")]
use std::collections::HashMap;
#[cfg(feature = "scan-rs")]
use std::fs;
#[cfg(feature = "scan-rs")]
use std::path::Path;

#[cfg(feature = "scan-rs")]
const E8_DIMENSIONS: usize = 8;
#[cfg(feature = "scan-rs")]
const CONNECTION_THRESHOLD: f64 = 0.5;

#[cfg(feature = "scan-rs")]
/// Service for scanning directory trees and building geometric topologies
pub struct CodebaseScanner {
    geometer: CrateGeometer,
}

#[cfg(feature = "scan-rs")]
/// Complete results from scanning a codebase, including metadata
pub struct ScanResult {
    /// The geometric topology of the codebase
    pub topology: Topology,
    /// Map of function names to their E8 coordinate vectors
    pub function_map: HashMap<String, Vec<f64>>,
    /// Total number of functions processed
    pub total_functions: usize,
}

#[cfg(feature = "scan-rs")]
impl Default for CodebaseScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl CodebaseScanner {
    /// Create a new CodebaseScanner
    pub fn new() -> Self {
        Self {
            geometer: CrateGeometer::new(),
        }
    }

    /// Scans a directory and converts the codebase into a geometric topology
    ///
    /// This method discards function names and returns only the pure topology structure
    /// suitable for anonymous agent traversal.
    pub fn scan_crate(&self, path: &Path) -> GsaResult<Topology> {
        let mut functions = Vec::new();
        self.visit_dirs(path, &mut functions)?;
        Ok(self.build_topology(&functions))
    }

    /// Scans a directory and returns complete analysis with function mapping
    ///
    /// This method preserves function names, allowing for targeted analysis and
    /// semantic lookups of specific code artifacts within the geometry.
    pub fn scan_crate_complete(&self, path: &Path) -> GsaResult<ScanResult> {
        let mut functions = Vec::new();
        let mut function_map = HashMap::new();

        self.visit_dirs(path, &mut functions)?;

        // 1. Build the complete function map for lookups
        for (name, vector) in &functions {
            function_map.insert(name.clone(), vector.clone());
        }

        let topology = self.build_topology(&functions);

        Ok(ScanResult {
            topology,
            function_map,
            total_functions: functions.len(),
        })
    }

    /// Recursively walk directories to find Rust source files
    fn visit_dirs(&self, dir: &Path, results: &mut Vec<(String, Vec<f64>)>) -> GsaResult<()> {
        if !dir.exists() || !dir.is_dir() {
            return Err(GsaError::Geometry(format!(
                "Directory not found: {:?}",
                dir
            )));
        }
        if dir.is_dir() {
            for entry in fs::read_dir(dir).map_err(|e| GsaError::Geometry(e.to_string()))? {
                let entry = entry.map_err(|e| GsaError::Geometry(e.to_string()))?;
                let path = entry.path();

                if path.is_dir() {
                    self.visit_dirs(&path, results)?;
                } else if let Some(ext) = path.extension()
                    && ext == "rs"
                {
                    self.process_file(&path, results)?;
                }
            }
        }
        Ok(())
    }

    /// Build a topology using a KD-tree to avoid O(N^2) neighbor scans
    fn build_topology(&self, functions: &[(String, Vec<f64>)]) -> Topology {
        let mut topology = Topology::new(E8_DIMENSIONS);

        for (_name, vector) in functions {
            topology.add_position(vector.clone());
        }

        let indexed_points: Vec<(usize, Vec<f64>)> = functions
            .iter()
            .enumerate()
            .map(|(i, (_name, vec))| (i, vec.clone()))
            .collect();

        let kdtree = KdTree::new(&indexed_points, E8_DIMENSIONS);

        for (i, (_name, vector)) in functions.iter().enumerate() {
            let neighbors = kdtree.radius_search(vector, CONNECTION_THRESHOLD);
            for neighbor_idx in neighbors {
                if i < neighbor_idx {
                    topology.add_connection(vector, &functions[neighbor_idx].1);
                }
            }
        }

        topology
    }

    /// Parse a single Rust file and extract metrics for all contained functions
    fn process_file(&self, path: &Path, results: &mut Vec<(String, Vec<f64>)>) -> GsaResult<()> {
        let content = fs::read_to_string(path).map_err(|e| GsaError::Geometry(e.to_string()))?;

        // Parse the full file using syn
        let file_ast = syn::parse_file(&content).map_err(|e| {
            GsaError::Geometry(format!("Failed to parse {}: {}", path.display(), e))
        })?;

        for item in file_ast.items {
            if let syn::Item::Fn(func) = item {
                let func_name = func.sig.ident.to_string();

                // Directly extract metrics from the parsed ItemFn without re-parsing
                let metrics = self.geometer.extract_metrics(&func);
                let vector = metrics.to_e8_vector();

                // Store as standard Vec<f64> for Topology compatibility
                results.push((func_name, vector.to_vec()));
            }
        }
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    #[cfg(feature = "scan-rs")]
    use super::*;
    #[cfg(feature = "scan-rs")]
    use std::path::Path;

    #[cfg(feature = "scan-rs")]
    #[test]
    fn test_scanner_creation() {
        let _scanner = CodebaseScanner::new();
        // Verification that scanner instantiates correctly
    }

    #[cfg(feature = "scan-rs")]
    #[test]
    fn test_scan_nonexistent_directory() {
        let scanner = CodebaseScanner::new();
        let result = scanner.scan_crate(Path::new("/nonexistent/path/arcmoon/void"));
        assert!(result.is_err());
    }
}
