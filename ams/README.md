# ArcMoon Suite

![ArcMoon logo](assets/ArcMoon.png)

Rust workspace with shared benchmarking tools, geometric kernels, identity encoding, and the yoshi recovery stack. Use this file as an index; crate-specific details live beside each crate.

## Workspace build/test

- Build (warnings-as-errors): `RUSTFLAGS="-D warnings" cargo build --workspace --all-features`
- Lint: `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Format: `cargo fmt --all -- --check`
- Tests: `RUSTFLAGS="-D warnings" cargo test --workspace --all-features`

## Crates

- [`crates/ams-gate`](crates/ams-gate/README.md): Minimal NATS server launcher with auto-download and lifecycle management.
- [`crates/acritic`](crates/acritic/README.md): Workspace test/bench harness with reporting and synthetic workloads.
- [`crates/geoshi`](crates/geoshi/README.md): GeoSynth Agent kernels (E8 lattice math, diffusion/projection, GPU paths).
- [`crates/xuid`](crates/xuid/README.md): XUID identity encoding, E8 coordinates, compression, and qdrant integrations.
- [`crates/yoshi`](crates/yoshi/README.md): CLI/TUI host for recovery workflows and the `cargo yo` autofix subcommand (feature-gated).
- [`crates/yoshi-std`](crates/yoshi-std/README.md): Error/telemetry facade used by workspace crates.
- [`crates/yoshi-derive`](crates/yoshi-derive/README.md): Derive macros for yoshi error types.

## Common features

- `yoshi` `cli`/`yoshell`: enable command-line shell UI.
- `yoshi` `nats`: enable NATS-based routing; pairs with `ams-gate` for a local server.
- `xuid` `qdrant`: enable qdrant-specific types.
s
