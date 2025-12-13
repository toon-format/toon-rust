/* e8/gf8/src/math/mod.rs */
//! Core geometric and lattice math utilities for `Gf8` and E₈-style operations.
//!
//! # e8 Primitives – Math Module
//!▫~•◦-------------------------------------‣
//!
//! This module provides higher-level math utilities built on top of the `Gf8`
//! gf8 type, including:
//!
//! - `gf8_ops`: Geometric operations on `Gf8` such as cosine similarity, angle
//!   computation, and (spherical) interpolation.
//! - `lattice`: E₈-inspired lattice quantization utilities and shell projections
//!   around the `Gf8` representation.
//! - `rotation`: Simple 8×8 orthogonal operators (`Gf8Rotation`) for rotating
//!   `Gf8` vectors in ℝ⁸.
//!
//! The intent is to keep `Gf8` itself as a small, focused gf8, while this
//! module houses reusable, higher-level math building blocks for e8-powered
//! systems.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod gf8_ops;
pub mod lattice;
pub mod rotation;

// Re-export the most commonly used items for convenience.
pub use gf8_ops::*;
pub use lattice::*;
pub use rotation::*;
