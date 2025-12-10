//! CUDA context and module loading (feature-gated).

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
