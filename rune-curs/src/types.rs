/* src/types/types.rs */
//!▫~•◦-------------------------------‣
//! # Device-side data layout descriptors.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-curs to manage device-side data structures for CUDA operations.
//!
//! ### Key Capabilities
//! - **Data Structure Management:** Creates and manages CUDA data structures from host memory.
//! - **Memory Validation:** Ensures data dimensions match and handles memory operations safely.
//! - **Feature-Gated Implementation:** Provides conditional compilation for CUDA support.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `buffers` and `kernels`.
//! Result structures adhere to the CudaResult type and are compatible
//! with the system's memory management pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_curs::types::{CudaDomRData, new};
//!
//! let energy_host = vec![1.0f32; 240];
//! let coords_host = vec![[0.0f32; 8]; 240];
//! let cuda_data = CudaDomRData::new(&energy_host, &coords_host)?;
//! // The 'cuda_data' can now be used for GPU computations.
//! ```

/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

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
