# Acritic

Workspace harness for running tests and benchmarks across ArcMoon Suite with structured reporting and reproducible workloads.

## Overview

- Runs workspace tests and Criterion benchmarks from one CLI entry point.
- Generates timestamped reports (HTML from Criterion plus structured logs) under `.zExGate-Reports` by default.
- Provides synthetic ML workloads and math-heavy benchmarks to stress CPU/GPU paths.
- Uses Tokio for async command orchestration and Rayon for parallel dataset generation.

## Quick start

```bash
# Show help
cargo run --package arcmoon-suite --bin acritic -- --help

# Run all benchmarks (HTML reports enabled by Criterion)
cargo run --package arcmoon-suite --bin acritic -- bench

# Run workspace tests
cargo run --package arcmoon-suite --bin acritic -- test
```

## CLI subcommands

- `bench`: run benchmarks; flags include `--category` (all|hydra|ml|quaternion|quantum|training) and `--iterations <count>`.
- `test`: run tests for the workspace or a single crate with `--crate <name>`.
- `all-tests` / `all-benches`: convenience wrappers to execute the full workspace suites.
- Global flags: `--output <path>` (default `.zExGate-Reports`), `--verbose`.

## Features

- `cuda`: enable CUDA-specific benchmarks (requires CUDA toolchain).
- `apex`: alias that enables `cuda`.

## Benchmarks and workloads

- ML workloads: synthetic datasets with controllable noise/distribution for regression/classification/time-series.
- Math kernels: quaternion operations, S³ manifold paths, and E8 lattice coordinate tests.
- Quantum/OmniCore paths: numerical stress tests used by other ArcMoon crates.

## Development

- Format: `cargo fmt --all`
- Lints: `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Tests: `RUSTFLAGS="-D warnings" cargo test --workspace --all-features`
- Benchmarks: `cargo bench --package arcmoon-suite`

## License

Dual-licensed under MIT or Apache-2.0, at your option.
