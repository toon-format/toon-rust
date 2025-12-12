/* src/mod.rs */
//!▫~•◦-------------------------------‣
//! # CUDA/RUST bridge for the RUNE ecosystem.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-curs to provide the low-level substrate for compiling and executing CUDA kernels.
//!
//! ### Key Capabilities
//! - **Archetype Engine:** Compiles templated CUDA kernels on demand.
//! - **Device Memory Management:** Safe wrappers for CUDA device context and memory buffers.
//! - **Feature-Gated Design:** Ensures clean builds for non-CUDA targets.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `archetypes`, `buffers`, and `kernels`.
//! Result structures adhere to the Error trait and are compatible
//! with the system's error handling pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_curs::{CudaError, CudaResult};
//!
//! let result: CudaResult<()> = some_cuda_operation();
//! match result {
//!     Ok(_) => println!("CUDA operation successful"),
//!     Err(e) => eprintln!("CUDA error: {}", e),
//! }
//! ```

/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use thiserror::Error;

/// The primary error type for all operations within the `rune-curs` crate.
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("The `cuda` feature is not enabled in this build.")]
    NotEnabled,
    #[error("This CUDA kernel or feature is not yet implemented.")]
    NotImplemented,
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("A CUDA driver API call failed: {0}")]
    Driver(String),
    #[error("A CUDA kernel execution failed: {0}")]
    Kernel(String),
}

/// A specialized `Result` type for `rune-curs` operations.
pub type CudaResult<T> = Result<T, CudaError>;

// Declare all modules, properly gated by the `cuda` feature.
// This ensures a clean build for non-CUDA targets.

#[cfg(feature = "cuda")]
pub mod archetypes;

#[cfg(feature = "cuda")]
pub mod buffers;

#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub mod runtime;

#[cfg(feature = "cuda")]
pub mod types;