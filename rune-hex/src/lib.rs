/* src/lib/lib.rs */
//!▫~•◦-------------------------------‣
//! # Crate root module providing public API for the rune-hex semantic graph system.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-hex to achieve efficient organization
//! and clean public API surface for the hexadecimal semantic graph functionality.
//!
//! ### Key Capabilities
//! - **Module Re-export:** Provides clean access to the hex module's functionality.
//! - **Crate Organization:** Serves as the entry point for the entire rune-hex crate.
//! - **Public API Surface:** Defines what functionality is exposed to users of the crate.
//!
//! ### Architectural Notes
//! This module serves as the functional entry point for the rune-hex crate.
//! It re-exports the hex module which contains the core semantic graph functionality.
//! Result structures adhere to standard Rust module organization patterns.
//!
//! ### Example
//! \```rust
//! use rune_hex::hex;
//!
//! // Access the hex module functionality through the crate root
//! let graph = hex::default_graph();
//! let vertices = graph.find_nearest_vertices(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
//! \```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod hex;
