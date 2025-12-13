![Yoshi logo](assets/YoshiLogo.png)

[![crates.io](https://img.shields.io/crates/v/yoshi-std.svg)](https://crates.io/crates/yoshi-std)
[![docs.rs](https://docs.rs/yoshi-std/badge.svg)](https://docs.rs/yoshi-std)

# yoshi-std

Superset runtime and utilities for the Yoshi error & recovery ecosystem. Includes supervision, persistence, ML-driven recovery, and metrics logging.

## Quick start

- Build: `cargo build --package yoshi-std --all-features`
- Tests: `cargo test --package yoshi-std --all-features`

Add yoshi-std to your crate:

```toml
[dependencies]
yoshi-std = { path = "../yoshi-std" }
```

Example:

```rust
use yoshi_std::initialize_yoshi;

#[tokio::main]
async fn main() -> yoshi_std::Result<()> {
    // Initialize with default config; metrics & recovery enabling via env/feature flags
    initialize_yoshi(None).await?;
    Ok(())
}
```

## Features
- `ml-recovery` - enable ML-based, inference-driven recovery strategies
- `nats` / `workers-network` - enable distributed worker discovery/messaging
- `cli`, `hot-reload` - enable CLI and live reload in runtime

## Documentation
Full API docs are published at `docs.rs` for each release and the crate exposes robust module-level docs inside the source tree.

## License
Dual-licensed under MIT OR Apache-2.0.
