/* src/runtime/runtime.rs */
//!▫~•◦-------------------------------‣
//! # CUDA context and module loading.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-curs to manage CUDA context initialization and module loading.
//!
//! ### Key Capabilities
//! - **Context Management:** Ensures CUDA context is properly initialized for the current thread.
//! - **Device Access:** Manages access to CUDA devices with proper error handling.
//! - **Feature-Gated Implementation:** Provides conditional compilation for CUDA support.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `archetypes` and `lib`.
//! Result structures adhere to the CudaResult type and are compatible
//! with the system's error handling pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_curs::runtime::{ensure_context, CudaError};
//!
//! let result = ensure_context();
//! match result {
//!     Ok(_) => println!("CUDA context initialized"),
//!     Err(e) => eprintln!("Context error: {}", e),
//! }
//! ```

/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::{CudaError, CudaResult};

#[cfg(feature = "cuda")]
use cust::{context::Context, context::ContextFlags, device::Device};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
static CTX: Mutex<Option<Arc<Context>>> = Mutex::new(None);

#[cfg(feature = "cuda")]
/// Ensures a CUDA context is initialized for the current thread.
///
/// # Errors
/// Returns an error if CUDA initialization, device access, or context creation fails.
pub fn ensure_context() -> CudaResult<()> {
    let mut ctx_guard = CTX
        .lock()
        .map_err(|_| CudaError::Driver("Mutex poison".to_string()))?;
    if ctx_guard.is_none() {
        cust::init(cust::CudaFlags::empty()).map_err(|e| CudaError::Driver(e.to_string()))?;
        let device = Device::get_device(0).map_err(|e| CudaError::Driver(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaError::Driver(e.to_string()))?;
        context
            .set_flags(ContextFlags::SCHED_AUTO | ContextFlags::MAP_HOST)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        *ctx_guard = Some(Arc::new(context));
    }
    drop(ctx_guard);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn ensure_context() -> CudaResult<()> {
    Err(CudaError::NotEnabled)
}
