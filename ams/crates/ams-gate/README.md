# AMS Gate

![ArcMoon logo](../../assets/ArcMoon.png)

Minimal NATS server launcher for the ArcMoon Suite workspace. It downloads the correct `nats-server` binary on demand, starts it, waits for readiness, and shuts it down cleanly.

## Overview

- Auto-fetches `nats-server` into `target/tools/` if it is not already present.
- Spawns the server with Tokio, polls readiness on the configured address, and returns a handle for shutdown.
- Clean stop via child process kill and wait to avoid orphaned processes.
- Uses `yoshi-std`/`yoshi-derive` for error handling.

## Quick start

```rust
use ams_gate::nats::{NatsConfig, NatsServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NatsConfig::default(); // address: 127.0.0.1:4222
    let (server, _addr) = NatsServer::start(config).await?;

    // ... run tests or a local app that expects NATS ...

    server.stop().await?;
    Ok(())
}
```

## Build and test

- Build: `RUSTFLAGS="-D warnings" cargo build --package ams-gate --all-features`
- Lint: `RUSTFLAGS="-D warnings" cargo clippy --package ams-gate --all-targets --all-features -- -D warnings`
- Tests: `RUSTFLAGS="-D warnings" cargo test --package ams-gate --all-features`

## Configuration

- `NATS_SERVER_BIN` (optional): path to an existing `nats-server` binary to skip auto-download.
- `NATS_ADDR` (via `NatsConfig`): override bind address if needed; defaults to `127.0.0.1:4222`.

## Notes

- No external service setup is required; auto-download pulls official release archives based on OS/CPU.
- Artifacts are cached under `target/tools/` to avoid repeated downloads.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
