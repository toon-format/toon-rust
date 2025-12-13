# Yoshi

![Yoshi logo](assets/YoshiLogo.png)

[![crates.io](https://img.shields.io/crates/v/yoshi.svg)](https://crates.io/crates/yoshi)
[![docs.rs](https://docs.rs/yoshi/badge.svg)](https://docs.rs/yoshi)

Yoshi provides error handling, recovery orchestration, and an optional CLI/TUI shell for the ArcMoon Suite. The crate is split across three workspace members: `yoshi` (facade and binary), `yoshi-std` (runtime and utilities), and `yoshi-derive` (derive macros).

## Components

- `yoshi`: facade and optional `yoshell` binary (feature-gated).
- `yoshi-std`: core recovery systems, supervision, and utilities.
- `yoshi-derive`: derive macros for error definitions.
- `cargo-yo`: cargo subcommand that streams `cargo check` diagnostics into Geoshi's autofix engine.

## Usage

- Build all features: `cargo build --package yoshi --all-features`
- Run tests: `cargo test --package yoshi --all-features`
- CLI/TUI: `cargo run --package yoshi --bin yoshell --features yoshell -- --help`
- Autofix CLI: `cargo yo` (requires `autofix-cli` feature; applies machine-safe fixes) or `cargo yo --suggest-only`.

Example (facade usage):

```rust
use yoshi::YoshiFacade;
let facade = YoshiFacade::default();
// call into error creation and recovery helpers
let _ = facade.new_error("example");
```

### Key feature flags

- `cli` / `yoshell`: enable CLI/TUI.
- `nats`: distributed error routing (uses `NATS_URL` if set).
- `http-gateway`, `ml-recovery`: enable corresponding subsystems in `yoshi-std`.
- `autofix` / `autofix-cli`: enable the cargo-yo subcommand backed by Geoshi.

## License

Dual-licensed under MIT OR Apache-2.0.
