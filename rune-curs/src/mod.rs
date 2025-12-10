//! `rune-curs`: The CUDA/RUST bridge for the RUNE ecosystem.
//!
//! This crate provides the low-level substrate for compiling and executing
//! CUDA kernels, orchestrated by the `hydron` evaluation engine. It is
//! feature-gated to ensure that projects without a CUDA toolchain can
//! build without issue.
//!
//! The primary capabilities include:
//! - A generative **Archetype Engine** for compiling templated CUDA kernels.
//! - Safe wrappers for CUDA device context, memory buffers, and module loading.
//! - A clear FFI boundary that is not exposed to the end user.

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