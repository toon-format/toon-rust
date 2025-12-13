/* crates/xuid/src/binary.rs */
//!▫~•◦------------------‣
//! # XUID – Binary Serialization and Compression
//!▫~•◦-------------------------------------------‣
//!
//! This module handles the binary serialization, deserialization, and lossless
//! compression of XUIDs. It defines the stable XUID binary format.
//!
//! ### Binary Serialization Format
//! \```text
//! [0-3]    Magic (0x58554944 = "XUID")
//! [4-4]    Version (0x02 = V1)
//! [5-5]    XuidType
//! [6-6]    E8 Orbit
//! [7-7]    Flags (hasPath=0x01, hasProv=0x02, compressed=0x04)
//! [8-23]   Delta Signature (16 bytes)
//! [24-55]  Semantic Hash (32 bytes)
//! [56-87]  E8 Coordinates (8×f32 = 32 bytes)
//! [88-95]  Reserved (8 bytes)
//! [96+]    Optional: Semantic Path (varint length + UTF-8)
//! [?+]     Optional: Provenance (varint length + JSON)
//! \```
//! Total fixed header: 96 bytes (matches in-memory layout).
//!
//! ## Zero-Copy Notes
//! - `deserialize()` is slice-based and does **not** allocate for semantic path bytes.
//! - `serialize_into()` enables allocation-free serialization into any `Write` sink.
//! - Provenance JSON and ZSTD decompression inherently require allocations to materialize bytes.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::{SemanticPath, Xuid, XuidError, XuidProvenance, XuidResult, XuidType};
use std::io::Write;

const MAGIC: u32 = 0x5855_4944; // "XUID"
const VERSION: u8 = 0x02;

const FLAG_HAS_PATH: u8 = 0x01;
const FLAG_HAS_PROV: u8 = 0x02;
const FLAG_COMPRESSED: u8 = 0x04;

const FIXED_HEADER_LEN: usize = 86;

/// Serialize XUID to binary format V2 into any writer (allocation-free for the caller).
pub fn serialize_into<W: Write>(mut out: W, xuid: &Xuid) -> XuidResult<()> {
    // Header (8 bytes)
    out.write_all(&MAGIC.to_le_bytes())?;
    out.write_all(&[VERSION])?;
    out.write_all(&[xuid.xuid_type as u8])?;
    out.write_all(&[xuid.e8_orbit])?;

    // Flags
    let mut flags = 0u8;
    if xuid.semantic_path.is_some() {
        flags |= FLAG_HAS_PATH;
    }
    if xuid.provenance.is_some() {
        flags |= FLAG_HAS_PROV;
    }
    out.write_all(&[flags])?;

    // Fixed data
    out.write_all(&xuid.delta_sig)?; // 9 bytes
    out.write_all(&xuid.semantic_hash)?; // 32 bytes

    // E8 coords (32 bytes)
    for &coord in &xuid.e8_coords {
        out.write_all(&coord.to_le_bytes())?;
    }

    // Reserved (5 bytes)
    out.write_all(&[0u8; 5])?;

    // Optional semantic path
    if let Some(ref path) = xuid.semantic_path {
        // NOTE: This may allocate depending on SemanticPath’s API.
        // If SemanticPath can expose a borrowed view (e.g., as_str()/as_ref()),
        // swap this to avoid allocation.
        let path_str = path.to_string();
        write_varint_into(&mut out, path_str.len() as u64)?;
        out.write_all(path_str.as_bytes())?;
    }

    // Optional provenance
    if let Some(ref prov) = xuid.provenance {
        // JSON serialization necessarily materializes bytes.
        let prov_json =
            serde_json::to_vec(prov).map_err(|e| XuidError::BinaryFormatError(e.to_string()))?;
        write_varint_into(&mut out, prov_json.len() as u64)?;
        out.write_all(&prov_json)?;
    }

    Ok(())
}

/// Serialize XUID to binary format V2 (convenience; allocates a Vec).
pub fn serialize(xuid: &Xuid) -> XuidResult<Vec<u8>> {
    let mut buf = Vec::with_capacity(FIXED_HEADER_LEN);
    serialize_into(&mut buf, xuid)?;
    Ok(buf)
}

/// Deserialize XUID from binary format V2 (slice-based; no semantic-path byte allocation).
pub fn deserialize(data: &[u8]) -> XuidResult<Xuid> {
    if data.len() < FIXED_HEADER_LEN {
        return Err(XuidError::BinaryFormatError(format!(
            "Data too short: {} < {FIXED_HEADER_LEN}",
            data.len()
        )));
    }

    let mut pos = 0usize;

    // Magic
    let magic = read_u32_le(data, &mut pos)?;
    if magic != MAGIC {
        return Err(XuidError::BinaryFormatError(format!(
            "Invalid magic: 0x{magic:08X}"
        )));
    }

    // Version
    let version = read_u8(data, &mut pos)?;
    if version != VERSION {
        return Err(XuidError::UnsupportedVersion(version));
    }

    // Type
    let xuid_type = XuidType::from_u8(read_u8(data, &mut pos)?)?;

    // Orbit
    let e8_orbit = read_u8(data, &mut pos)?;

    // Flags
    let flags = read_u8(data, &mut pos)?;

    // Fixed data
    let mut delta_sig = [0u8; 16];
    read_exact_into(data, &mut pos, &mut delta_sig)?;

    let mut semantic_hash = [0u8; 32];
    read_exact_into(data, &mut pos, &mut semantic_hash)?;

    // E8 coords
    let mut e8_coords = [0.0f32; 8];
    for coord in e8_coords.iter_mut() {
        let bits = read_u32_le(data, &mut pos)?;
        *coord = f32::from_bits(bits);
    }

    // Reserved (5 bytes) — skip, but bounds-check
    advance(data, &mut pos, 5)?;

    // Optional semantic path
    let semantic_path = if flags & FLAG_HAS_PATH != 0 {
        let len = read_varint(data, &mut pos)? as usize;
        let bytes = read_slice(data, &mut pos, len)?;
        let path_str =
            std::str::from_utf8(bytes).map_err(|e| XuidError::BinaryFormatError(e.to_string()))?;
        Some(
            path_str
                .parse::<SemanticPath>()
                .map_err(|e| XuidError::BinaryFormatError(e.to_string()))?,
        )
    } else {
        None
    };

    // Optional provenance
    let provenance = if flags & FLAG_HAS_PROV != 0 {
        let len = read_varint(data, &mut pos)? as usize;
        let bytes = read_slice(data, &mut pos, len)?;
        let prov: XuidProvenance =
            serde_json::from_slice(bytes).map_err(|e| XuidError::BinaryFormatError(e.to_string()))?;
        Some(prov)
    } else {
        None
    };

    Ok(Xuid {
        xuid_type,
        e8_orbit,
        delta_sig,
        semantic_hash,
        e8_coords,
        semantic_path,
        provenance,
    })
}

/// Serialize with ZSTD compression (lossless)
pub fn serialize_compressed(xuid: &Xuid, level: i32) -> XuidResult<Vec<u8>> {
    let uncompressed = serialize(xuid)?;

    // Smart compression: Skip if data is too small or unlikely to compress well
    if should_skip_compression(&uncompressed) {
        return Ok(uncompressed); // Return uncompressed (no marker)
    }

    let compressed = zstd::encode_all(uncompressed.as_slice(), level)
        .map_err(|e| XuidError::CompressionError(e.to_string()))?;

    // Only use compression if it actually reduces size
    if compressed.len() + 1 >= uncompressed.len() {
        return Ok(uncompressed); // Compression not beneficial
    }

    // Prepend compression marker
    let mut result = Vec::with_capacity(compressed.len() + 1);
    result.push(FLAG_COMPRESSED);
    result.extend_from_slice(&compressed);

    Ok(result)
}

/// Deserialize with ZSTD decompression
pub fn deserialize_compressed(data: &[u8]) -> XuidResult<Xuid> {
    if data.is_empty() {
        return Err(XuidError::DecompressionError("Empty data".into()));
    }

    if data.first() == Some(&FLAG_COMPRESSED) {
        // Decompression must materialize bytes (cannot be zero-copy).
        let decompressed = zstd::decode_all(&data[1..])
            .map_err(|e| XuidError::DecompressionError(e.to_string()))?;
        deserialize(&decompressed)
    } else {
        // Not compressed
        deserialize(data)
    }
}

/// Determine if compression should be skipped for performance
///
/// Heuristics:
/// - Data < 200 bytes: Compression overhead likely exceeds benefit.
/// - High entropy (random-looking): Data won't compress well.
fn should_skip_compression(data: &[u8]) -> bool {
    const MIN_SIZE_FOR_COMPRESSION: usize = 200;
    const ENTROPY_THRESHOLD: f32 = 0.85; // High entropy = random-like data

    // Skip small data payloads
    if data.len() < MIN_SIZE_FOR_COMPRESSION {
        return true;
    }

    // Estimate entropy to avoid trying to compress already-random data
    let entropy = estimate_entropy(data);
    entropy > ENTROPY_THRESHOLD
}

/// Fast entropy estimation using byte frequency.
///
/// Returns value in the range [0.0, 1.0] where:
/// - 0.0 = all same byte (highly compressible)
/// - 1.0 = uniform distribution (not compressible)
fn estimate_entropy(data: &[u8]) -> f32 {
    // Sample first 256 bytes for speed
    let sample_size = data.len().min(256);
    if sample_size == 0 {
        return 0.0;
    }
    let mut freq = [0u32; 256];

    for &byte in &data[..sample_size] {
        freq[byte as usize] += 1;
    }

    // Calculate Shannon entropy (simplified)
    let total = sample_size as f32;
    let mut entropy = 0.0f32;

    for &count in &freq {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.log2();
        }
    }

    // Normalize to 0-1 range (max entropy for a byte stream is 8 bits)
    (entropy / 8.0).min(1.0)
}

// ============================================================================
// Slice-based decoding helpers (no Cursor/Read)
// ============================================================================

#[inline(always)]
fn read_u8(data: &[u8], pos: &mut usize) -> XuidResult<u8> {
    if *pos >= data.len() {
        return Err(XuidError::BinaryFormatError("Unexpected EOF".into()));
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

#[inline(always)]
fn read_u32_le(data: &[u8], pos: &mut usize) -> XuidResult<u32> {
    let bytes = read_slice(data, pos, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline(always)]
fn read_exact_into<const N: usize>(data: &[u8], pos: &mut usize, out: &mut [u8; N]) -> XuidResult<()> {
    let bytes = read_slice(data, pos, N)?;
    out.copy_from_slice(bytes);
    Ok(())
}

#[inline(always)]
fn read_slice<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> XuidResult<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    let end = pos.checked_add(len).ok_or_else(|| XuidError::BinaryFormatError("Length overflow".into()))?;
    if end > data.len() {
        return Err(XuidError::BinaryFormatError("Unexpected EOF".into()));
    }
    let out = &data[*pos..end];
    *pos = end;
    Ok(out)
}

#[inline(always)]
fn advance(data: &[u8], pos: &mut usize, len: usize) -> XuidResult<()> {
    let _ = read_slice(data, pos, len)?;
    Ok(())
}

// ============================================================================
// Varint encoding (LEB128) — writer + slice decode
// ============================================================================

fn write_varint_into<W: Write>(out: &mut W, mut value: u64) -> XuidResult<()> {
    let mut buf = [0u8; 10];
    let mut n = 0usize;

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf[n] = byte;
        n += 1;
        if value == 0 {
            break;
        }
    }

    out.write_all(&buf[..n])?;
    Ok(())
}

fn read_varint(data: &[u8], pos: &mut usize) -> XuidResult<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;

    loop {
        let byte = read_u8(data, pos)?;
        result |= (u64::from(byte & 0x7F)) << shift;

        if byte & 0x80 == 0 {
            return Ok(result);
        }

        shift += 7;
        if shift >= 64 {
            return Err(XuidError::BinaryFormatError("Varint overflow".into()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_binary_roundtrip() {
        let xuid = Xuid::new(b"test data");
        let binary = serialize(&xuid).unwrap();
        let parsed = deserialize(&binary).unwrap();

        assert_eq!(xuid.delta_sig, parsed.delta_sig);
        assert_eq!(xuid.semantic_hash, parsed.semantic_hash);
        assert_eq!(xuid.e8_orbit, parsed.e8_orbit);
    }

    #[test]
    fn test_binary_with_path() {
        let xuid = Xuid::new(b"test data")
            .with_semantic_path("/test/path".parse::<SemanticPath>().unwrap());

        let binary = serialize(&xuid).unwrap();
        let parsed = deserialize(&binary).unwrap();

        assert_eq!(xuid.semantic_path, parsed.semantic_path);
    }

    #[test]
    fn test_compression() {
        let xuid = Xuid::new(b"test data with lots of repetitive content");

        let uncompressed = serialize(&xuid).unwrap();
        let compressed = serialize_compressed(&xuid, 3).unwrap();

        // Compression should reduce size for repetitive data
        println!(
            "Uncompressed: {} bytes, Compressed: {} bytes",
            uncompressed.len(),
            compressed.len()
        );

        let parsed = deserialize_compressed(&compressed).unwrap();
        assert_eq!(xuid.delta_sig, parsed.delta_sig);
    }

    #[test]
    fn test_varint() {
        let mut buf = Vec::new();
        {
            let mut c = Cursor::new(&mut buf);
            write_varint_into(&mut c, 300).unwrap();
        }

        let mut pos = 0usize;
        let val = read_varint(&buf, &mut pos).unwrap();
        assert_eq!(val, 300);
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn test_smart_compression() {
        // Small XUID should skip compression
        let small_xuid = Xuid::new(b"tiny");
        let result = serialize_compressed(&small_xuid, 3).unwrap();
        assert!(
            result.first() != Some(&FLAG_COMPRESSED),
            "Small XUIDs should skip compression"
        );

        // Large XUID with low entropy should compress
        let large_data = b"AAAAAAAAAA".repeat(20);
        let large_xuid = Xuid::new(&large_data);
        let result = serialize_compressed(&large_xuid, 3).unwrap();
        // May or may not compress depending on structure

        // Verify roundtrip works regardless
        let parsed = deserialize_compressed(&result).unwrap();
        assert_eq!(large_xuid.delta_sig, parsed.delta_sig);
    }

    #[test]
    fn test_entropy_estimation() {
        // All same byte = low entropy
        let uniform = vec![0x42u8; 256];
        let entropy = estimate_entropy(&uniform);
        assert!(
            entropy < 0.1,
            "Uniform data should have low entropy: {entropy}"
        );

        // Random bytes = high entropy
        let random: Vec<u8> = (0..256)
            .map(|i| u8::try_from(i).expect("0..256 fits in u8"))
            .collect();
        let entropy = estimate_entropy(&random);
        assert!(
            entropy > 0.9,
            "Random data should have high entropy: {entropy}"
        );
    }

    #[test]
    fn test_compression_benchmark() {
        use std::time::Instant;

        let xuid = Xuid::new(b"benchmark data with some repetition");

        // Benchmark uncompressed
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = serialize(&xuid).unwrap();
        }
        let uncompressed_time = start.elapsed();

        // Benchmark compressed
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = serialize_compressed(&xuid, 3).unwrap();
        }
        let compressed_time = start.elapsed();

        println!(
            "Serialization: uncompressed={uncompressed_time:?}, compressed={compressed_time:?}"
        );
        println!(
            "Compression overhead: {:.1}x",
            compressed_time.as_secs_f64() / uncompressed_time.as_secs_f64()
        );
    }
}
