//! CUDA/RUNE bridge crate (curs).
//! Provides device runtime wrappers and kernel entry points.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA feature not enabled")]
    NotEnabled,
    #[error("CUDA kernel or feature not implemented")]
    NotImplemented,
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("CUDA driver error: {0}")]
    Driver(String),
    #[error("CUDA kernel error: {0}")]
    Kernel(String),
}

pub type CudaResult<T> = Result<T, CudaError>;

#[cfg(feature = "cuda")]
pub mod archetypes;
pub mod buffers;
pub mod kernels;
pub mod runtime;
pub mod types;

/// Compute `DomR` scores on GPU: scores[x] = `Σ_o` energy[o] * dot8(coords[o], coords[x]).
/// - `energy` length must equal `coords.len()`
/// - `coords` is flattened and sent to device; kernel computes one score per root.
///
/// Returns the scores as a Vec<f32>.
///
/// # Errors
/// Returns an error if input dimensions don't match, CUDA initialization fails,
/// or GPU kernel execution encounters problems.
#[cfg(feature = "cuda")]
pub fn domr_scores_gpu(energy: &[f32], coords: &[[f32; 8]]) -> CudaResult<Vec<f32>> {
    use cust::prelude::*;

    let n = energy.len();
    if n == 0 || coords.len() != n {
        return Err(CudaError::InvalidOperation(format!(
            "Energy len {} must equal coords len {}",
            energy.len(),
            coords.len()
        )));
    }
    // Init CUDA context
    let _ctx = cust::quick_init().map_err(|e| CudaError::Driver(e.to_string()))?;

    // Load embedded PTX
    let module = Module::from_ptx(DOMR_PTX, &[]).map_err(|e| CudaError::Kernel(e.to_string()))?;
    let func = module
        .get_function("domr_kernel")
        .map_err(|e| CudaError::Kernel(e.to_string()))?;

    let flat_coords: Vec<f32> = coords.iter().flat_map(|r| r.iter().copied()).collect();
    let d_energy =
        DeviceBuffer::from_slice(energy).map_err(|e| CudaError::Driver(e.to_string()))?;
    let d_coords =
        DeviceBuffer::from_slice(&flat_coords).map_err(|e| CudaError::Driver(e.to_string()))?;
    let d_scores = unsafe { DeviceBuffer::<f32>::uninitialized(n) }
        .map_err(|e| CudaError::Driver(e.to_string()))?;

    let block = 128u32;
    #[allow(clippy::manual_div_ceil)]
    #[allow(clippy::cast_possible_truncation)]
    let grid = ((n as u32) + block - 1) / block;
    let stream =
        Stream::new(StreamFlags::DEFAULT, None).map_err(|e| CudaError::Driver(e.to_string()))?;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    unsafe {
        launch!(
            func<<<grid, block, 0, stream>>>(
                d_energy.as_device_ptr(),
                d_coords.as_device_ptr(),
                d_scores.as_device_ptr(),
                n as i32
            )
        )
        .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }
    stream
        .synchronize()
        .map_err(|e| CudaError::Driver(e.to_string()))?;

    let mut scores = vec![0.0f32; n];
    d_scores
        .copy_to(&mut scores)
        .map_err(|e| CudaError::Driver(e.to_string()))?;
    Ok(scores)
}

/// Fallback when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
pub fn domr_scores_gpu(_energy: &[f32], _coords: &[[f32; 8]]) -> CudaResult<Vec<f32>> {
    Err(CudaError::NotEnabled)
}

// Embedded PTX for domr_kernel
#[cfg(feature = "cuda")]
const DOMR_PTX: &str = r"
.version 6.0
.target sm_30
.address_size 64

.visible .entry domr_kernel(
    .param .u64 energy,
    .param .u64 coords,
    .param .u64 scores,
    .param .u32 n
) {
    .reg .pred p;
    .reg .f32 acc, dot, cx0,cx1,cx2,cx3,cx4,cx5,cx6,cx7, co;
    .reg .s32 idx, nval, o;
    .reg .u64 eptr, cptr, sptr, base;

    ld.param.u64 eptr, [energy];
    ld.param.u64 cptr, [coords];
    ld.param.u64 sptr, [scores];
    ld.param.u32 nval, [n];
    mov.u32 idx, %tid.x;
    mad.lo.s32 idx, %ctaid.x, %ntid.x, idx;
    setp.ge.s32 p, idx, nval;
    @p ret;

    // load coords[idx]
    mul.wide.s32 base, idx, 32;
    add.s64 base, cptr, base;
    ld.global.f32 cx0, [base+0];
    ld.global.f32 cx1, [base+4];
    ld.global.f32 cx2, [base+8];
    ld.global.f32 cx3, [base+12];
    ld.global.f32 cx4, [base+16];
    ld.global.f32 cx5, [base+20];
    ld.global.f32 cx6, [base+24];
    ld.global.f32 cx7, [base+28];

    mov.f32 acc, 0f00000000;
    mov.s32 o, 0;
L_loop:
    setp.ge.s32 p, o, nval;
    @p bra L_end;
    mul.wide.s32 base, o, 32;
    add.s64 base, cptr, base;
    ld.global.f32 dot, [base+0];
    mul.f32 dot, dot, cx0;
    ld.global.f32 co, [base+4];
    fma.rn.f32 dot, co, cx1, dot;
    ld.global.f32 co, [base+8];
    fma.rn.f32 dot, co, cx2, dot;
    ld.global.f32 co, [base+12];
    fma.rn.f32 dot, co, cx3, dot;
    ld.global.f32 co, [base+16];
    fma.rn.f32 dot, co, cx4, dot;
    ld.global.f32 co, [base+20];
    fma.rn.f32 dot, co, cx5, dot;
    ld.global.f32 co, [base+24];
    fma.rn.f32 dot, co, cx6, dot;
    ld.global.f32 co, [base+28];
    fma.rn.f32 dot, co, cx7, dot;

    ld.global.f32 co, [eptr + o*4];
    fma.rn.f32 acc, co, dot, acc;
    add.s32 o, o, 1;
    bra L_loop;
L_end:
    mul.wide.s32 base, idx, 4;
    add.s64 base, sptr, base;
    st.global.f32 [base], acc;
    ret;
}
";
