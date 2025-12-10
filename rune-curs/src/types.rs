//! Device-side data layout descriptors (feature-gated).

use super::buffers::CudaBuffer;
use super::{CudaError, CudaResult};

pub struct CudaDomRData {
    pub energy: CudaBuffer<f32>,
    pub coords: CudaBuffer<f32>, // flattened AoS
    pub scores: CudaBuffer<f32>,
    pub n: usize,
}

impl CudaDomRData {
    /// Creates new CUDA data structures from host memory.
    ///
    /// # Errors
    /// Returns an error if data dimensions don't match or CUDA memory operations fail.
    #[cfg(feature = "cuda")]
    pub fn new(energy_host: &[f32], coords_host: &[[f32; 8]]) -> CudaResult<Self> {
        let n = energy_host.len();
        if coords_host.len() != n {
            return Err(CudaError::InvalidOperation(
                "coords length must match energy length".into(),
            ));
        }
        let energy = CudaBuffer::from_host(energy_host)?;
        let flat: Vec<f32> = coords_host.iter().flat_map(|r| r.iter().copied()).collect();
        let coords = CudaBuffer::from_host(&flat)?;
        let scores = CudaBuffer::new(n)?;
        Ok(Self {
            energy,
            coords,
            scores,
            n,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_energy_host: &[f32], _coords_host: &[[f32; 8]]) -> CudaResult<Self> {
        Err(CudaError::NotEnabled)
    }
}
