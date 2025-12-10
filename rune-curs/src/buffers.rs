//! Typed device buffers (feature-gated).

use super::{CudaError, CudaResult};

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};

#[cfg(feature = "cuda")]
pub struct CudaBuffer<T: DeviceCopy> {
    inner: DeviceBuffer<T>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceCopy> CudaBuffer<T> {
    /// Creates a new uninitialized CUDA buffer of the specified length.
    ///
    /// # Errors
    /// Returns an error if CUDA memory allocation fails.
    pub fn new(len: usize) -> CudaResult<Self> {
        let inner = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Creates a CUDA buffer from a host slice, copying data to the device.
    ///
    /// # Errors
    /// Returns an error if memory allocation or data transfer fails.
    pub fn from_host(slice: &[T]) -> CudaResult<Self> {
        let inner =
            DeviceBuffer::from_slice(slice).map_err(|e| CudaError::Driver(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Copies data from the device buffer back to the host slice.
    ///
    /// # Errors
    /// Returns an error if the data transfer fails.
    pub fn to_host(&self, slice: &mut [T]) -> CudaResult<()> {
        self.inner
            .copy_to(slice)
            .map_err(|e| CudaError::Driver(e.to_string()))
    }

    /// Returns the number of elements in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the buffer contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the device pointer for this buffer.
    #[must_use]
    pub fn as_device_ptr(&self) -> cust::memory::DevicePointer<T> {
        self.inner.as_device_ptr()
    }
}

#[cfg(not(feature = "cuda"))]
pub struct CudaBuffer<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "cuda"))]
impl<T> CudaBuffer<T> {
    pub fn new(_len: usize) -> CudaResult<Self> {
        Err(CudaError::NotEnabled)
    }

    pub fn from_host(_slice: &[T]) -> CudaResult<Self> {
        Err(CudaError::NotEnabled)
    }

    pub fn to_host(&self, _slice: &mut [T]) -> CudaResult<()> {
        Err(CudaError::NotEnabled)
    }

    pub fn len(&self) -> usize {
        0
    }
    pub fn is_empty(&self) -> bool {
        true
    }
}
