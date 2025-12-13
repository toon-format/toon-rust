# XUID (Xypher Unique Identity Descriptor)

![ArcMoon logo](../../assets/ArcMoon.png)

[![crates.io](https://img.shields.io/crates/v/xuid.svg)](https://crates.io/crates/xuid)
[![docs.rs](https://docs.rs/xuid/badge.svg)](https://docs.rs/xuid)

XUID provides a geometric identity system for the ArcMoon Suite, implementing the "laws of physics" for the Xypher Sphere. Identity is location in an 8D E8 lattice, where semantic-temporal coordinates prevent collision and enable native geometric similarity.

## Core Concepts

- **Packed Delta Identity (Δ)**: Time (48 bits) and Content Hash (80 bits) packed into a 16-byte signature
- **E8 Quantization (E8Q)**: Every ID has coordinates in the E8 Gosset lattice for geometric clustering
- **Unified XU String Format**: Canonical representation with semantic layering
- **Zero-Allocation Core**: 96-byte `Xuid` struct optimized for performance

## Usage

```rust
use xuid::{Xuid, XuidType, XuidConstruct};

// Create a basic XUID (E8 Quantized, timestamped now)
let id = Xuid::new(b"semantic event data");

// Create a specific type
let exp_id = Xuid::new_e8_with_type(b"strategy", XuidType::Experience);

// Wrap in a Semantic Envelope (The Codex Construct)
let construct = XuidConstruct::from_core(id)
    .with_bug("bug-123")
    .with_hint("apply-patch-01");

println!("Codex ID: {}", construct.to_canonical_string());
// Output: XU:E8Q:123...abc:789...xyz:B=74...:H=61...:ID
```

## Features

- `qdrant` - enable Qdrant vector database integration
- `rayon` - enable parallel processing for batch operations
- `system-monitoring` - enable system resource monitoring
- `file-processing` - enable file-based operations

## Build and Test

- Build: `cargo build --package xuid --all-features`
- Tests: `cargo test --package xuid --all-features`
- Lint: `RUSTFLAGS="-D warnings" cargo clippy --package xuid --all-targets --all-features -- -D warnings`

## License

Dual-licensed under MIT OR Apache-2.0.

See module-level docs in `src/` for detailed APIs.
