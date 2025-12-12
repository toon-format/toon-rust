//! Integration test for RUNE-ArchetypeEngine bridge
//!
//! Tests the end-to-end flow: RUNE kernel declaration -> ArchetypeEngine compilation -> PTX generation

use rune_format::rune::hydron::eval::Evaluator;
use rune_format::rune::parse;
use std::fs;
use std::path::Path;

#[cfg(feature = "cuda")]
#[test]
fn test_archetype_kernel_compilation() {
    // Define RUNE script with kernel declaration
    let script = r#"
Kernel:MyRowDot := CUDA:Archetype:RowDot(D: 8) and MyData -> MyRowDot
"#;

    // Parse the script
    let stmts = parse(script).expect("Failed to parse RUNE script");

    // Set up evaluator with initial data
    let mut eval = Evaluator::new();
    eval.set_var("MyData", rune_format::rune::hydron::values::Value::Array(vec![
        rune_format::rune::hydron::values::Value::Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        rune_format::rune::hydron::values::Value::Vec8([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    ]));

    // Evaluate the kernel declaration
    let result = eval.eval_stmt(&stmts[0]);
    assert!(result.is_ok(), "Kernel declaration evaluation failed");
    assert_eq!(result.unwrap(), rune_format::rune::hydron::values::Value::Null);

    // Check that PTX file was created
    let cache_dir = Path::new("target").join("rune").join("cache");
    assert!(cache_dir.exists(), "Cache directory should exist");

    // Find PTX files in cache directory
    let ptx_files: Vec<_> = fs::read_dir(&cache_dir)
        .expect("Failed to read cache directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().map_or(false, |ext| ext == "ptx"))
        .collect();

    assert!(!ptx_files.is_empty(), "At least one PTX file should be generated");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_archetype_kernel_compilation_no_cuda() {
    // Define RUNE script with kernel declaration
    let script = r#"
Kernel:MyRowDot := CUDA:Archetype:RowDot(D: 8) and MyData -> MyRowDot
"#;

    // Parse the script
    let stmts = parse(script).expect("Failed to parse RUNE script");

    // Set up evaluator
    let mut eval = Evaluator::new();

    // Evaluate the kernel declaration - should fail without CUDA
    let result = eval.eval_stmt(&stmts[0]);
    assert!(result.is_err(), "Kernel declaration should fail without CUDA feature");

    if let Err(err) = result {
        assert!(err.to_string().contains("CUDA feature not enabled"));
    }
}