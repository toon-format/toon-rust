//! rune-zero: Binary zero-copy serializer utilities (fory + mmap + append).
//!
//! This keeps the compact binary path separate from the readable RUNE text pipeline.

use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use yoshi::{Result, yoshi};

/// Encode any fory-serializable value into a byte Vec.
pub fn to_bytes<T: fory::ser::Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    fory::ser::to_vec(value).map_err(|e| yoshi!("fory serialize error: {}", e))
}

/// Decode a value from a byte slice.
pub fn from_bytes<T: fory::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    fory::de::from_slice(bytes).map_err(|e| yoshi!("fory deserialize error: {}", e))
}

/// Write an encoded payload to a file (truncate + write).
pub fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| yoshi!("open write error: {}", e))?;
    f.write_all(bytes)
        .map_err(|e| yoshi!("file write error: {}", e))?;
    Ok(())
}

/// Memory-mapped view for zero-copy reads.
pub struct ZeroView {
    mmap: Mmap,
}

impl ZeroView {
    /// Map a file into memory (read-only).
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| yoshi!("open mmap error: {}", e))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| yoshi!("mmap error: {}", e))?;
        Ok(Self { mmap })
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Decode a value directly from the mapped bytes.
    pub fn decode<T: fory::de::DeserializeOwned>(&self) -> Result<T> {
        from_bytes(self.as_bytes())
    }
}

/// Append-optimized writer for simple log-like usage.
pub struct ZeroLog {
    file: File,
}

impl ZeroLog {
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)
            .map_err(|e| yoshi!("open log error: {}", e))?;
        Ok(Self { file })
    }

    /// Append raw bytes and return offset.
    pub fn append(&mut self, bytes: &[u8]) -> Result<u64> {
        let offset = self
            .file
            .seek(SeekFrom::End(0))
            .map_err(|e| yoshi!("seek error: {}", e))?;
        self.file
            .write_all(bytes)
            .map_err(|e| yoshi!("append error: {}", e))?;
        Ok(offset)
    }
}
