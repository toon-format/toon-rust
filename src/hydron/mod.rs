/* src/rune/hydron/mod.rs */
//!▫~•◦-------------------------------‣
//! # Hydron - E8 Geometric Mathematics Engine for RUNE.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module serves as the root for the Hydron engine, the mathematical core
//! of the RUNE evaluation system. It integrates and re-exports functionality from
//! the `hydron-core` crate and provides the RUNE-specific value system and evaluator.
//!
//! ## Key Capabilities
//! - **Module Organization**: Declares the sub-modules for the value system (`values`),
//!   evaluator (`eval`), perception engine (`perception`), topology (`topology`), and
//!   CUDA bridge (`cuda`).
//! - **Conditional Compilation**: Uses feature flags (`hydron`, `cuda`) to conditionally
//!   include geometric types and accelerator logic.
//! - **Public API Export**: Re-exports essential types like `Value`, `EvalContext`, and
//!   the various geometric layers from `hydron-core` to create a unified API.
//!
//! ### Architectural Notes
//! The separation between `rune-xero` and `hydron-core` allows the pure mathematical
//! implementations to live in a dependency-free crate, while this module provides
//! the application-specific integration and runtime system (`Value`, `EvalContext`).
//!
//! #### Example
//! ```rust
//! // Consumers can import all necessary Hydron types from this module's root.
//! use rune_xero::rune::hydron::{Value, Gf8, SphericalLayer};
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod eval;
pub mod perception;
pub mod topology;
pub mod values;

// Re-export the hydron-core crate's math modules when the feature is enabled.
// This keeps the Rune crate's public API stable while the actual math is
// implemented in the `hydron-core` crate.
#[cfg(feature = "hydron")]
pub use hydron_core::{
    FisherLayer, Gf8, Gf8Tensor, HyperbolicLayer, LorentzianCausalLayer, LorentzianLayer,
    PersistencePair, QuaternionOps, SpacetimePoint, SphericalLayer, SymplecticLayer,
    TopologicalLayer, intrinsics_for_f32_width,
};

// When hydron feature is disabled, keep local types for the evaluator and values as-is.
// Re-export the runtime value types so code can always import `crate::rune::hydron::Value`.
pub use crate::hydron::values::{EvalContext, EvalError, Octonion, Value};
