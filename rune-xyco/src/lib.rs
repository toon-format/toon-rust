/* src/lib.rs */
//!▫~•◦-------------------------------‣
//! # Xypher Codex (XYCO) - Deterministic E8 Semantic Hypergraph
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module implements the Xypher Codex, a deterministic 240-root E8 semantic
//! lattice for tokenless meaning encoding and XUID trajectory tracking.
//!
//! ### Key Capabilities
//! - **E8 Root Registry:** Deterministic generation and assignment of 240 roots.
//! - **Semantic Tiers:** Enforces the Taproot -> Lateral -> Tertiary -> Cross hierarchy.
//! - **Tokenless Operators:** Implements the Spec B Operator families.
//! - **Zero-Copy:** Optimized for HPC with minimal allocations.
//!
//! ### Architectural Notes
//! Designed to serve as the semantic coordinate system for the Rune-Gate and GeoSynth systems.
//!
//! ### Example
//! ```rust
//! use rune_xyco::registry::CodexRegistry;
//!
//! let registry = CodexRegistry::instance();
//! let root = registry.get_root(14).unwrap();
//! assert_eq!(root.tier, rune_xyco::codex::Tier::Taproot);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod codex;
pub mod e8;
pub mod operators;
pub mod registry;

// Re-export core types
pub use codex::{Tier, Domain, CodexRoot};
pub use registry::CodexRegistry;
