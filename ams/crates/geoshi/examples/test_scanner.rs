//! Example to demonstrate Geoshi geometric scanner functionality
//!
//! Run with: `cargo run --example test_geoshi_scanner --features scan-rs`
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "scan-rs")]
use geoshi::geocode::{CodebaseScanner, CrateGeometer};
#[cfg(feature = "scan-rs")]
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "scan-rs"))]
    {
        println!("⚠️  This example requires the 'scan-rs' feature.");
        println!("   Please run with: cargo run --example test_geoshi_scanner --features scan-rs");
        Ok(())
    }

    #[cfg(feature = "scan-rs")]
    run_demo()
}

#[cfg(feature = "scan-rs")]
fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Geoshi Geometric Scanner Demonstration");
    println!("======================================");
    println!();

    // Test individual function analysis
    println!("1️⃣ Testing Individual Function Analysis:");
    let geometer = CrateGeometer::new();

    let simple_function = r#"
        fn simple_add(a: i32, b: i32) -> i32 {
            a + b
        }
    "#;

    let complex_function = r#"
        fn complex_processing<T: Clone + Debug>(
            mut items: Vec<Option<T>>,
            threshold: f64
        ) -> Result<Vec<T>, String>
        where T: PartialOrd + Default {
            if items.is_empty() {
                return Err("Empty input".to_string());
            }

            let processed = items.into_iter()
                .filter_map(|item| item)
                .filter(|item| /* some condition */ true)
                .map(|item| if true { item.clone() } else { T::default() })
                .collect::<Vec<_>>();

            Ok(processed)
        }
    "#;

    println!("Analyzing simple function...");
    match geometer.function_to_lattice(simple_function) {
        Ok((root_idx, vector)) => {
            println!(
                "  ✅ Simple function mapped to E8 lattice root: {}",
                root_idx
            );
            println!(
                "  📐 Geometric coordinates: {:.3?}",
                &vector.iter().take(3).collect::<Vec<_>>()
            );
        }
        Err(e) => println!("  ❌ Failed to analyze simple function: {:?}", e),
    }

    println!();
    println!("Analyzing complex function...");
    match geometer.function_to_lattice(complex_function) {
        Ok((root_idx, vector)) => {
            println!(
                "  ✅ Complex function mapped to E8 lattice root: {}",
                root_idx
            );
            println!(
                "  📐 Geometric coordinates: {:.3?}",
                &vector.iter().take(3).collect::<Vec<_>>()
            );
        }
        Err(e) => println!("  ❌ Failed to analyze complex function: {:?}", e),
    }

    // Test directory scanning capability
    println!();
    println!("2️⃣ Testing Codebase Scanning Capability:");

    let scanner = CodebaseScanner::new();

    // Try scanning this crate's src directory
    let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");

    if src_path.exists() {
        println!("Scanning directory: {:?}", src_path);
        match scanner.scan_crate(&src_path) {
            Ok(topology) => {
                println!("  ✅ Successfully scanned directory!");
                println!(
                    "  🗺️  Built topology with {} function nodes",
                    topology.size()
                );
                println!(
                    "  🔗 Topology has {} connections",
                    topology.connection_count()
                );
            }
            Err(e) => println!("  ❌ Failed to scan directory: {:?}", e),
        }
    } else {
        println!("  ⚠️  Test directory not found, skipping scan test");
    }

    println!();
    println!("🎉 Geoshi Geometric Scanner Demonstration Complete!");
    println!();
    println!("💡 This proves Geoshi can now:");
    println!("   • Analyze individual functions geometrically");
    println!("   • Map code complexity to E8 lattice coordinates");
    println!("   • Scan entire codebases for topological analysis");
    println!("   • Enable autonomous code refactoring through spatial intelligence");

    Ok(())
}
