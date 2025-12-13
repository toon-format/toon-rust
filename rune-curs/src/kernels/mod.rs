/* src/kernels/mod.rs */
//!▫~•◦-------------------------------‣
//! # CUDA kernel modules.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-curs to organize and expose CUDA kernel functionality.
//!
//! ### Key Capabilities
//! - **Module Organization:** Centralizes CUDA kernel module declarations.
//! - **Kernel Access:** Provides access to embedded PTX kernels and CUDA utilities.
//! - **Feature Management:** Manages feature-gated kernel functionality.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `cuda` and `dom_r`.
//! Result structures adhere to the module system and are compatible
//! with the system's kernel management pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_curs::kernels::{cuda, dom_r};
//!
//! // Access kernel modules
//! let ptx = dom_r::DOMR_PTX;
//! // The kernel modules can now be used for GPU operations.
//! ```

/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod cuda;
pub mod dom_r;
