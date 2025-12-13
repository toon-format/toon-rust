//! # Geoshi: GeoSynth Agent
//!
//! # GEOSHI LIBRARY
//!▫~•◦------------------------------------------------‣
//!
//! Complete Rust implementation of the GeoSynth cognition system, providing
//! advanced error recovery, resilience, and geometric intelligence through
//! sophisticated cognitive architectures built on mathematical geometry.
//!
//! ### Key Capabilities
//! - **Geometric Articulation:** Natural language generation from geometric forms.
//! - **Diffusion Processing:** Advanced smoothing and information propagation.
//! - **Lattice Computation:** Mathematical lattices and Lie group operations.
//! - **Hexagonal Geometry:** Efficient hexagonal lattice representations.
//! - **Pathfinding Intelligence:** Topological navigation and exploration algorithms.
//! - **GPU Acceleration:** CUDA/OpenCL kernels for high-performance computation.
//!
//! ### Architecture Overview
//! Geoshi integrates multiple geometric cognition subsystems to provide robust,
//! self-healing AI capabilities through geometric transformations and spatial
//! reasoning. Each module specializes in different aspects of geometric intelligence.
//!
//! ### Core Systems
//! - **Articulator:** Geometric language generation and communication.
//! - **Diffuser:** Information propagation and smoothing algorithms.
//! - **Georganism:** Living geometric structures with evolutionary behaviors.
//! - **Geosynth:** Synthetic geometry generation and composition.
//! - **HexLattice:** Hexagonal coordinate systems for efficient computation.
//! - **Hydron:** Hydrogen mathematics and resonance modeling.
//! - **Lattice:** Abstract lattice structures and E8 lattice implementation.
//! - **Projector:** Geometric projections and transformations.
//! - **Worm:** Pathfinding and topological exploration systems.
//! - **Xage:** Extended aging computations and temporal cognition.
//!
//! ```rust
//! use geoshi::{Articulator, Diffuser, HexLattice};
//!
//! // Example: Basic geometric articulation
//! let articulator = Articulator::new();
//! // ... create geometric structures ...
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod articulator;
pub mod actuator;
pub mod autofix;
pub mod diffuser;
pub mod geocode;
pub mod georganism;
pub mod geosynth;
pub mod hex;
pub mod hydron;
pub mod kernels;
pub mod lattice;
pub mod projector;
pub mod worm;
pub mod xage;

// Re-export key types for convenient access
pub use articulator::Articulator;
pub use actuator::{ActuationTool, GeometricActuator};
pub use autofix::{AutofixConfig, AutofixEngine, AutofixOutcome, AutofixSuggestion, PlannedEdit};
pub use diffuser::Diffuser;
pub use geocode::CrateGeometer;
pub use georganism::Georganism;
pub use geosynth::{GeoSentinel, GeoSynthion};
pub use hex::HexLattice;
pub use worm::WormPathfinder;
pub use xage::Xage;

/// Core result type for Geoshi operations.
pub type GeoshiResult<T> = Result<T, GeoshiError>;
/// Backwards-compatible alias while migrating from the old Geoshi name.
pub type GsaResult<T> = GeoshiResult<T>;

/// Errors in the Geoshi system.
#[derive(thiserror::Error, Debug)]
pub enum GeoshiError {
    #[error("Lattice error: {0}")]
    Lattice(String),
    #[error("Geometry error: {0}")]
    Geometry(String),
    #[error("Perception error: {0}")]
    Perception(String),
    #[error("Resonance error: {0}")]
    Resonance(String),
}

/// Backwards-compatible alias while migrating from the old Geoshi name.
pub type GsaError = GeoshiError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lattice_creation() {
        let lattice = lattice::E8Lattice::new().unwrap();
        assert_eq!(lattice.n_roots(), 240);
    }

    #[cfg(feature = "scan-rs")]
    #[test]
    fn test_geoshi_self_analysis() {
        use std::path::PathBuf;

        // 1. Initialize the Scanner
        let scanner = geocode::CodebaseScanner::new();

        // 2. Scan Geoshi's own source code with complete analysis
        // Assuming running from crate root
        let src_path = PathBuf::from("src");
        let scan_result = scanner
            .scan_crate_complete(&src_path)
            .expect("Failed to scan self");

        println!("Geoshi Codebase Analysis:");
        println!("  Total Functions: {}", scan_result.total_functions);
        println!("  Topology Size: {} nodes", scan_result.topology.size());
        println!(
            "  Function Map Size: {} entries",
            scan_result.function_map.len()
        );

        // 3. We'll initialize an Xage agent and run the Worm pathfinder after we determine
        // the actual simplest and most complex functions from the function map so we
        // operate on actual nodes present in the topology.

        // 6. Analyze the function map to find extremes
        // Find the "simplest" and "most complex" functions by geometric distance from origin
        let mut simplest_function = String::new();
        let mut most_complex_function = String::new();
        let mut min_distance = f64::INFINITY;
        let mut max_distance = 0.0;

        for (func_name, vector) in &scan_result.function_map {
            let distance_from_origin = vector.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if distance_from_origin < min_distance {
                min_distance = distance_from_origin;
                simplest_function = func_name.clone();
            }

            if distance_from_origin > max_distance {
                max_distance = distance_from_origin;
                most_complex_function = func_name.clone();
            }
        }

        // Verify we found actual functions
        assert!(!simplest_function.is_empty());
        assert!(!most_complex_function.is_empty());
        assert!(min_distance >= 0.0);
        assert!(max_distance >= 0.0);

        println!(
            "Simplest function: {} (distance: {:.4})",
            simplest_function, min_distance
        );
        println!(
            "Most complex function: {} (distance: {:.4})",
            most_complex_function, max_distance
        );

        // 7. Verify basic topology properties
        assert!(scan_result.topology.size() > 0);
        assert!(scan_result.topology.dimensions() == 8);
        assert!(!scan_result.function_map.is_empty());
        assert_eq!(scan_result.function_map.len(), scan_result.total_functions);

        // 8. Initialize an Xage Agent at the simplest function and run the Worm
        // to find a path to the most complex function (these vectors come directly
        // from the scanned function_map, so they are guaranteed to be topology nodes).
        let start_pos_vec = scan_result
            .function_map
            .get(&simplest_function)
            .expect("Simplest function vector not found");
        let complexity_target_vec = scan_result
            .function_map
            .get(&most_complex_function)
            .expect("Most complex function vector not found");

        let _agent = xage::Xage::new("Architect-01".to_string(), start_pos_vec.clone());
        let worm = worm::WormPathfinder::new(scan_result.topology.clone());

        if let Ok(result) = worm.find_path_astar(start_pos_vec, complexity_target_vec) {
            println!("Found path between simplest and most complex functions!");
            println!("Path cost (Geometric Debt): {:.4}", result.cost);
            println!("Path length: {} steps", result.path_length);
            assert!(result.path_length > 0);
        } else {
            println!("No direct path found between simplest and most complex functions.");
            println!("This could indicate the codebase has disconnected regions.");
        }

        println!("Integration test passed: Geoshi can analyze its own geometric structure!");
    }
}
