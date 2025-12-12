/* src/lib.rs */
//!▫~•◦-------------------------------‣
//! # Rune-GSV: Graph Semantic Value Storage System
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gsv to achieve efficient graph-based data storage and querying system.
//!
//! ### Key Capabilities
//! - **Modular Architecture:** Provides core modules for addressing, slot management, storage, and querying.
//! - **Public API:** Exposes key data structures and functionality for external integration.
//! - **Prelude Support:** Offers convenient import of commonly used types and functions.
//!
//! ### Architectural Notes
//! This module serves as the main entry point for the rune-gsv crate.
//! It integrates with modules such as `addressing`, `slot`, `store`, `query`, `persistence`, and `builtins`.
//! The prelude module provides ergonomic access to core functionality.
//!
//! ### Example
//! ```rust
//! use crate::rune_gsv::prelude::*;
//!
//! let address = WeylSemanticAddress::from_text_intent("example").unwrap();
//! let mut store = QuantizedContinuum::default();
//! let slot = SGLRuneSlot::default();
//!
//! // Core types are now available for use.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod addressing;
pub mod builtins;
pub mod persistence;
pub mod query;
pub mod slot;
pub mod store;

pub use crate::addressing::WeylSemanticAddress;
pub use crate::slot::{ExecutionTrace, RankingData, SGLRuneSlot, SemanticGraph};
pub use crate::slot::{Node, Frame};
pub use crate::store::QuantizedContinuum;
pub use crate::persistence::{write_rune_atomic, read_rune};
pub use crate::builtins::{asv_store, asv_get, asv_query};
pub use crate::query::count_head_matches;

pub mod prelude {
    pub use crate::{QuantizedContinuum, SGLRuneSlot, WeylSemanticAddress};
}
