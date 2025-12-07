/* benches/rune_parser.rs */
//! Performance benchmarks for the RUNE parser engine.
//!
//! # TOON-RUNE – RUNE Parser Benchmarks
//!▫~•◦----------------------------------‣
//!
//! This benchmark suite measures parsing performance across different RUNE
//! expression types and script sizes using the Criterion benchmarking framework.
//!
//! ### Key Capabilities
//! - **Literal Benchmarks**: Performance of basic literals and identifiers.
//! - **Math Expression Benchmarks**: Arithmetic parsing in math blocks.
//! - **Operator Benchmarks**: Structural, glyph, and path operator performance.
//! - **Script Size Benchmarks**: Scaling performance for small to large RUNE scripts.
//! - **Comparative Analysis**: Memory and CPU usage patterns for different constructs.
//!
//! ### Architectural Notes
//! Benchmarks use criterion for statistical rigor and cover the parser::functions
//! and AST construction. Results help guide optimization priorities and validate
//! performance targets for E8 ecosystem integration.
//!
//! ### Example
//! ```rust
//! use criterion::{criterion_group, criterion_main, Criterion};
//!
//! fn bench_simple(c: &mut Criterion) {
//!     c.bench_function("simple", |b| b.iter(|| parse_rune("42").unwrap()));
//! }
//!
//! criterion_group!(benches, bench_simple);
//! criterion_main!(benches);
//!
//! // Results show baseline performance for optimization targets.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use criterion::{Criterion, criterion_group, criterion_main};
use rune_format::rune::parse_rune;

fn bench_rune_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rune_parser");

    // Simple literal
    group.bench_function("literal", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("42")).unwrap();
        });
    });

    // Variable reference
    group.bench_function("variable", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("x")).unwrap();
        });
    });

    // Math expression in brackets
    group.bench_function("math_expr", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("[a + b * c]")).unwrap();
        });
    });

    // Root declaration
    group.bench_function("root_declaration", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("root: sphere_7")).unwrap();
        });
    });

    // Structural operators
    group.bench_function("structural_ops", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("a -> b <- c")).unwrap();
        });
    });

    // Path access
    group.bench_function("path_access", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("a/b/c")).unwrap();
        });
    });

    // Glyph operators
    group.bench_function("glyph_ops", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("a \\|/ b /|\\ c")).unwrap();
        });
    });

    // Complex math block
    group.bench_function("complex_math", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box("[(a + b) * (c - d) / e]")).unwrap();
        });
    });

    group.finish();
}

fn bench_rune_script_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rune_script_sizes");

    // Small script (10 lines)
    let small_script = r#"root: sphere_7
x -> 1
y -> 2
result := [x + y]
a | b
c -> d
e := f
final -> [result * 2]
"#;

    group.bench_function("small_10_lines", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box(small_script)).unwrap();
        });
    });

    // Medium script (50 lines)
    let mut medium_script = String::from("root: quaternion\n");
    for i in 0..48 {
        medium_script.push_str(&format!("var{} := [{} + {}]\n", i, i, i + 1));
    }
    medium_script.push_str("final -> result\n");

    group.bench_function("medium_50_lines", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box(&medium_script)).unwrap();
        });
    });

    // Large script (200 lines)
    let mut large_script = String::from("root: e8::continuum\n");
    for i in 0..198 {
        large_script.push_str(&format!("compute{} := [a{} * b{} + c{}]\n", i, i, i, i));
    }
    large_script.push_str("output -> final\n");

    group.bench_function("large_200_lines", |b| {
        b.iter(|| {
            parse_rune(std::hint::black_box(&large_script)).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rune_parsing, bench_rune_script_sizes);
criterion_main!(benches);
