/* crates/ams-gate/src/lib.rs */
//! Minimal gateway utilities for running a local NATS instance for the ArcMoon Suite.
//!
//! # ArcMoon Studios – AMS Gate Module
//!▫~•◦-------------------------------------‣
//!
//! Provides a small, self-contained launcher for a local NATS server process to support
//! distributed Yoshi workflows without introducing cyclic dependencies.
//!
//! ### Key Capabilities
//! - **Local NATS launch:** Starts a child `nats-server` process with readiness probing.
//! - **Graceful shutdown:** Terminates the process cleanly on drop.
//! - **Minimal surface:** No extra protocols; just spawn, wait, and stop.
//!
//! ### Example
//! ```no_run
//! use ams_gate::nats::{NatsConfig, NatsServer};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), ams_gate::nats::NatsGateError> {
//! let server = NatsServer::start(NatsConfig::default()).await?;
//! // use server.address() in clients (e.g., NATS_URL)
//! server.stop().await?;
//! # Ok(())
//! # }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod nats;
