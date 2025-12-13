/* benches/perf.rs */
//! Benchmark of syn parsing performance for different error enum styles.
//!
//! This benchmark measures the time taken to parse Rust enum definitions
//! with different attribute syntaxes using syn. It provides a baseline
//! comparison of the AST parsing complexity for yoshi-derive vs thiserror styles.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use criterion::{criterion_group, criterion_main, Criterion};
use std::fmt::Write;
use syn::parse_str;

const VARIANT_COUNTS: &[usize] = &[10, 100, 500];

#[derive(Clone, Copy)]
enum MacroType {
    Yoshi,
    ThisError,
}

/// Generates a massive error enum as a string to be parsed by `syn`.
/// This is our synthetic workload.
fn generate_enum_code(num_variants: usize, macro_type: MacroType) -> String {
    let mut code = String::with_capacity(num_variants * 150);

    let (derive_macro, display_attr, error_attr, from_attr, source_attr) = match macro_type {
        MacroType::Yoshi => (
            "AnyError",
            "anyerror",
            "anyerror",
            "from",
            "source",
        ),
        MacroType::ThisError => (
            "Error",
            "error",
            "error",
            "from",
            "source",
        ),
    };

    // Note: The derive attribute is just for `syn` parsing; we don't use it.
    writeln!(code, "#[derive(Debug, {})]", derive_macro).unwrap();
    writeln!(code, "pub enum BigError {{").unwrap();

    for i in 0..num_variants {
        match i % 4 {
            0 => { // Unit Variant
                writeln!(code, "    #[{}(\"Unit variant {}\")]", display_attr, i).unwrap();
                writeln!(code, "    UnitVariant{},", i).unwrap();
            }
            1 => { // Tuple Variant with Source
                writeln!(code, "    #[{}(\"Tuple variant: {{0}}\")]", display_attr).unwrap();
                writeln!(code, "    TupleVariant{}(#[{}] #[{}] std::io::Error),", i, from_attr, source_attr).unwrap();
            }
            2 => { // Struct Variant
                writeln!(code, "    #[{}(\"Struct variant: id={{id}}\")]", error_attr).unwrap();
                writeln!(code, "    StructVariant{} {{ id: u64, context: String }},", i).unwrap();
            }
            3 => { // Transparent Variant
                let transparent_attr = match macro_type {
                    MacroType::Yoshi => "#[anyerror(transparent)]",
                    MacroType::ThisError => "#[error(transparent)]",
                };
                writeln!(code, "    {}", transparent_attr).unwrap();
                writeln!(code, "    TransparentVariant{}(#[{}] OtherError),", i, from_attr).unwrap();
            }
            _ => unreachable!(),
        }
    }

    writeln!(code, "}}").unwrap();

    code
}

fn benchmark_macros(c: &mut Criterion) {
    let mut group = c.benchmark_group("AST Parsing Latency");
    group.significance_level(0.01).sample_size(100);

    for &count in VARIANT_COUNTS {
        // --- Yoshi Derive Style Parsing ---
        let yoshi_code = generate_enum_code(count, MacroType::Yoshi);
        group.bench_with_input(
            criterion::BenchmarkId::new("yoshi-style-enum", count),
            &yoshi_code,
            |b, code| {
                b.iter(|| {
                    // Benchmark syn parsing of yoshi-style enum with #[anyerror(...)] attributes
                    let result = parse_str::<syn::DeriveInput>(code);
                    let _ = match result {
                        Ok(ast) => ast,
                        Err(e) => panic!("Parse error for yoshi style code: {}\nCode:\n{}", e, code),
                    };
                });
            },
        );

        // --- ThisError Style Parsing ---
        let thiserror_code = generate_enum_code(count, MacroType::ThisError);
        group.bench_with_input(
            criterion::BenchmarkId::new("thiserror-style-enum", count),
            &thiserror_code,
            |b, code| {
                b.iter(|| {
                    // Benchmark syn parsing of thiserror-style enum with #[error(...)] attributes
                    let _: syn::DeriveInput = parse_str(code).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_macros);
criterion_main!(benches);
