use crc32fast::Hasher;
use fory::ser::to_vec;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use yoshi::{Result, yoshi};

use crate::slot::SGLRuneSlot;

const LOG_FILE: &str = ".rune.log";
const SNAP_FILE: &str = ".rune";

#[repr(C)]
struct EntryHeader {
    len: u64,
    crc: u32,
}

pub struct AppendLog {
    path: PathBuf,
    file: File,
}

impl AppendLog {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(LOG_FILE);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(|e| yoshi!("open log error: {}", e))?;
        Ok(Self { path, file })
    }

    /// Append a slot to the WAL. Returns the offset where it was written.
    pub fn append(&mut self, slot: &SGLRuneSlot) -> Result<u64> {
        let payload = to_vec(slot).map_err(|e| yoshi!("fory serialize error: {}", e))?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let crc = hasher.finalize();
        let header = EntryHeader {
            len: payload.len() as u64,
            crc,
        };

        let offset = self.file.seek(SeekFrom::End(0)).map_err(|e| yoshi!("seek error: {}", e))?;
        self.file
            .write_all(&header.len.to_le_bytes())
            .and_then(|_| self.file.write_all(&header.crc.to_le_bytes()))
            .and_then(|_| self.file.write_all(&payload))
            .map_err(|e| yoshi!("append error: {}", e))?;
        Ok(offset)
    }
}

pub struct MappedStore {
    mmap: Mmap,
    len: usize,
}

impl MappedStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(LOG_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| yoshi!("open mmap error: {}", e))?;
        let len = file.metadata().map_err(|e| yoshi!("metadata error: {}", e))?.len() as usize;
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| yoshi!("mmap error: {}", e))? };
        Ok(Self { mmap, len })
    }

    pub fn iter<'a>(&'a self) -> LogIter<'a> {
        LogIter {
            buf: &self.mmap[..self.len],
            pos: 0,
        }
    }

    pub fn get_at(&self, offset: u64) -> Result<SGLRuneSlot> {
        let mut pos = offset as usize;
        let buf = &self.mmap[..self.len];
        if pos + 12 > buf.len() {
            return Err(yoshi!("offset out of bounds"));
        }
        let len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let crc = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let end = pos + len;
        if end > buf.len() {
            return Err(yoshi!("corrupt log entry"));
        }
        let payload = &buf[pos..end];
        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != crc {
            return Err(yoshi!("crc mismatch"));
        }
        fory::de::from_slice(payload).map_err(|e| yoshi!("fory deserialize error: {}", e))
    }
}

pub struct LogIter<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for LogIter<'a> {
    type Item = (u64, SGLRuneSlot);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 12 > self.buf.len() {
            return None;
        }
        let start = self.pos as u64;
        let len = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap()) as usize;
        self.pos += 8;
        let crc = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        let end = self.pos + len;
        if end > self.buf.len() {
            return None;
        }
        let payload = &self.buf[self.pos..end];
        self.pos = end;

        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != crc {
            return None;
        }
        match fory::de::from_slice(payload) {
            Ok(slot) => Some((start, slot)),
            Err(_) => None,
        }
    }
}

/// Compact the WAL into a snapshot and reset the log.
pub fn compact(dir: &Path) -> Result<()> {
    let reader = MappedStore::open(dir)?;
    let mut latest: HashMap<Vec<u8>, SGLRuneSlot> = HashMap::new();
    for (_, slot) in reader.iter() {
        latest.insert(slot.address.clone(), slot);
    }

    let snap_path = dir.join(SNAP_FILE);
    let mut snap = File::create(&snap_path).map_err(|e| yoshi!("snapshot create error: {}", e))?;
    for slot in latest.values() {
        let bytes = to_vec(slot).map_err(|e| yoshi!("fory serialize error: {}", e))?;
        snap.write_all(&bytes)
            .map_err(|e| yoshi!("snapshot write error: {}", e))?;
    }

    // Truncate log after compaction
    let _ = File::create(dir.join(LOG_FILE)).map_err(|e| yoshi!("truncate log error: {}", e))?;
    Ok(())
}
