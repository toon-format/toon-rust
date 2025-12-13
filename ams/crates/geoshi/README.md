# Geoshi (GeoSynth Agent)

![ArcMoon logo](../../assets/ArcMoon.png)

[![crates.io](https://img.shields.io/crates/v/geoshi.svg)](https://crates.io/crates/geoshi)
[![docs.rs](https://docs.rs/geoshi/badge.svg)](https://docs.rs/geoshi)

Geoshi provides geometric cognition, lattice math, and GPU-assisted kernels used across the ArcMoon Suite. It contains a mix of pure-Rust math utilities, higher-level lattice transforms (E8), and optional, feature-gated GPU kernels.

## Components

- Lattice math (including E8), diffusion, projection, and pathfinding modules.
- GPU kernels via CUDA/WebGPU features where available.
- Code analysis utilities under `geocode` for scanning Rust sources.
- Autofix engine that turns cargo JSON diagnostics into machine-applied edits (used by `cargo yo`).

## Usage

- Build: `cargo build --package geoshi --all-features`
- Tests: `cargo test --package geoshi --all-features`
- Enable GPU code paths with `--features gpu` when building (add `cuda` for `cust`/CUDA support where available).
- Autofix (via yoshi `cargo yo`): `cargo yo -- --package <pkg>` to plan/apply machine-safe fixes.

Example (library usage):

```rust
use geoshi::lattice::E8; // (example module path)
let lattice = E8::default();
// perform a transform
let out = lattice.project(&[0.0_f32; 8]);
println!("projected: {:?}", out);
```

## Features

- `gpu` - enable GPU kernels (wgpu/CUDA backends where feature-gated)
- `scan-rs` - enable crate source scanning utilities

## License

Dual-licensed under MIT OR Apache-2.0.

See module-level docs in `src/` for detailed APIs.
