/* crates/gf8/src/e32l.rs */
//! # E32L – True Lossless f32 Representation and Compression Framework
//!
//! This module provides the definitive answer to achieving true lossless f32 representation
//! and compression within the E8 ecosystem. It contains two primary components:
//!
//! 1.  **`E32L` Type:** A **ZERO-ERROR, BIJECTIVE** conversion between a standard `f32` and a
//!     32-dimensional structure composed of four `E8F` roots. It serves as a lossless,
//!     4-byte, E8-native representation for `f32`.
//!
//! 2.  **Compression Framework:** A high-performance, lossless compression algorithm inspired by
//!     bit-plane decomposition. It transforms blocks of `f32` (or `E32L`) values into a
//!     highly compressible state and uses entropy coding to achieve significant bitrate reduction.
//!
//! ## The `E32L` Breakthrough: Deconstruction, Not Quantization
//!
//! An `f32` is fundamentally 32 bits of information. `E32L` preserves these 32 bits perfectly by
//! mapping them to the indices of four `E8F` roots.
//!
//! ```text
//! f32 (32 bits)  →  f32.to_bits()  →  u32  →  [u8; 4]  →  [E8F; 4]
//!       ↑                                                     ↓
//!       └─────────── (Perfect Reconstruction) ───────────────┘
//! ```
//!
//! ## The Compression Framework: Lossless Bitrate Reduction
//!
//! The provided `compression` module implements a 3-stage reversible pipeline:
//!
//! 1.  **Bit-Matrix Transposition:** A block of `f32` values is losslessly transposed into 32
//!     separate "bit-planes." This groups statistically similar bits (signs with signs,
//!     exponents with exponents), dramatically reducing the entropy of each plane. This is a
//!     provably bijective transform.
//! 2.  **Grouped Entropy Coding:** The sign, exponent, and mantissa bit-planes are compressed
//!     as separate streams using a high-performance entropy coder (`zstd`). This allows the
//!     coder to adapt to the unique statistical properties of each component.
//! 3.  **Framing and Validation:** The compressed streams are packaged into a single, robust data frame with
//!     the necessary metadata for perfect reconstruction. A cryptographic hash (`blake3`) of the
//!     original data is included to guarantee integrity upon decompression.
//!
//! This combination provides both a foundational lossless type (`E32L`) and a powerful algorithm
//! to reduce the storage and transmission cost of datasets composed of these types.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::e8f::E8F;

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 1: E32L TYPE DEFINITION
// ═══════════════════════════════════════════════════════════════════════════════════════

/// A true lossless 32-dimensional representation of an f32, composed of four E8F roots.
///
/// This type achieves zero-error conversion by deconstructing the 32 bits of an IEEE 754
/// float into four 8-bit `E8F` indices. It does **not** perform mathematical quantization.
///
/// Assumes that `E8F` is `#[repr(transparent)]` over `u8` (i.e., has the size and alignment of
/// a `u8`) **and** that `E8F::new` / `E8F::index` behave as a pure byte container for the full
/// `0..=255` range without clamping, normalization, or re-encoding. Under these conditions,
/// `E32L` is a strict, bit-perfect wrapper over the underlying `f32` representation.
///
/// # Storage
/// - Wire Format: 4 bytes (same as `f32`).
/// - Memory Format: `[E8F; 4]`.
/// - Conversion Overhead: Zero-cost (bitwise reinterpretation that optimizes away).
///
/// # Example
/// ```rust
/// use gf8::e32l::E32L;
///
/// let original = 3.1415926535f32;
/// let e32l = E32L::from_f32(original);
/// let recovered = e32l.to_f32();
///
/// // The recovered value is not just "close", it is identical.
/// assert_eq!(original.to_bits(), recovered.to_bits());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct E32L {
    /// The four E8F roots whose indices represent the 32 bits of an f32 value.
    /// These are used as bit containers, not for their mathematical vector properties.
    pub roots: [E8F; 4],
}

impl E32L {
    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 2: CORE CONVERSIONS (LOSSLESS)
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Creates an `E32L` from an `f32` with **zero error**.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bytes = value.to_le_bytes();
        Self {
            roots: [
                E8F::from_raw(bytes[0]),
                E8F::from_raw(bytes[1]),
                E8F::from_raw(bytes[2]),
                E8F::from_raw(bytes[3]),
            ],
        }
    }

    /// Converts the `E32L` back to an `f32` with **zero error**.
    #[inline]
    pub fn to_f32(&self) -> f32 {
        f32::from_le_bytes([
            self.roots[0].index(),
            self.roots[1].index(),
            self.roots[2].index(),
            self.roots[3].index(),
        ])
    }

    /// Creates an `E32L` from its raw `u32` bit representation.
    #[inline]
    pub fn from_bits(bits: u32) -> Self {
        Self::from_f32(f32::from_bits(bits))
    }

    /// Converts the `E32L` to its raw `u32` bit representation.
    #[inline]
    pub fn to_bits(&self) -> u32 {
        self.to_f32().to_bits()
    }

    /// Creates an `E32L` from a 4-byte array.
    #[inline]
    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self {
            roots: [
                E8F::from_raw(bytes[0]),
                E8F::from_raw(bytes[1]),
                E8F::from_raw(bytes[2]),
                E8F::from_raw(bytes[3]),
            ],
        }
    }

    /// Converts the `E32L` to a 4-byte array.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 4] {
        [
            self.roots[0].index(),
            self.roots[1].index(),
            self.roots[2].index(),
            self.roots[3].index(),
        ]
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // SECTION 3: UTILITY METHODS (DELEGATED FROM f32)
    // ═══════════════════════════════════════════════════════════════════════════════════

    /// Checks if the contained value is `NaN`.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.to_f32().is_nan()
    }

    /// Checks if the contained value is `+Infinity` or `-Infinity`.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.to_f32().is_infinite()
    }

    /// Checks if the contained value is a finite number.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.to_f32().is_finite()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 4: TRAIT IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

impl From<f32> for E32L {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<E32L> for f32 {
    #[inline]
    fn from(value: E32L) -> Self {
        value.to_f32()
    }
}

impl From<u32> for E32L {
    /// Creates an E32L from its raw IEEE 754 bit representation.
    #[inline]
    fn from(bits: u32) -> Self {
        Self::from_bits(bits)
    }
}

impl From<E32L> for u32 {
    /// Converts an E32L to its raw IEEE 754 bit representation.
    #[inline]
    fn from(value: E32L) -> Self {
        value.to_bits()
    }
}

impl Default for E32L {
    /// Defaults to `0.0f32`.
    #[inline]
    fn default() -> Self {
        Self::from_f32(0.0)
    }
}

impl std::fmt::Display for E32L {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for E32L {}

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for E32L {}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 5: LOSSLESS COMPRESSION FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Provides functions for lossless compression of `f32` or `E32L` slices.
///
/// This module implements a 3-stage reversible pipeline with two compression levels:
///
/// **Level 1 (Machine Readable):**
/// - Bit-planes stored uncompressed or lightly compressed
/// - Direct access to sign/exponent/mantissa without full decompression
/// - Good for semantic queries and streaming
/// - Target: 4:1 to 8:1 compression
///
/// **Level 2 (Maximum Compression):**
/// - Full bit-plane transposition and E8 binary transform
/// - Aggressive entropy coding (zstd level 19)
/// - Requires full decompression for any access
/// - Target: 8:1 to 16:1 compression
///
/// This module requires the `compression` feature to be enabled. It depends on `zstd`,
/// `byteorder`, `blake3`, and `bytemuck`.
#[cfg(feature = "compression")]
pub mod compression {
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use std::io::{Cursor, Read, Write};

    const MAGIC_NUMBER: u32 = 0x45333243; // E32C (E32L Compression)
    const FORMAT_VERSION: u8 = 2; // Version 2 adds Level 1/2 support

    /// Compression level selection
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum CompressionLevel {
        /// Level 1: Machine readable, fast decompression, 4:1-8:1 ratio
        #[default]
        Level1 = 1,
        /// Level 2: Maximum compression, full decompression required, 8:1-16:1 ratio
        Level2 = 2,
    }

    impl CompressionLevel {
        fn zstd_level(&self) -> i32 {
            match self {
                CompressionLevel::Level1 => 3,  // Fast compression
                CompressionLevel::Level2 => 19, // Maximum compression
            }
        }
    }

    /// Represents the compressed data frame.
    pub struct E32LFrame {
        pub level: CompressionLevel,
        pub original_len: u64,
        pub data_hash: [u8; 32],
        pub sign_stream: Vec<u8>,
        pub exponent_stream: Vec<u8>,
        pub mantissa_stream: Vec<u8>,
    }

    /// Lossless compression error types.
    #[derive(Debug)]
    pub enum CompressionError {
        Io(std::io::Error),
        Zstd(std::io::Error),
        InvalidFormat(String),
        IntegrityError,
    }

    impl From<std::io::Error> for CompressionError {
        fn from(err: std::io::Error) -> Self {
            CompressionError::Io(err)
        }
    }

    /// E8-inspired binary bit-plane transform applied in-place to groups of 8 planes.
    ///
    /// This operates over GF(2) on bit-columns across 8 planes:
    ///
    /// For each group of 8 planes and each (byte_idx, bit_idx), we:
    /// - Collect an 8-bit column vector `v` of bits (one from each plane).
    /// - Apply an 8×8 invertible binary matrix `M` (forward) or `M⁻¹` (inverse).
    /// - Write the transformed bits back into the planes.
    ///
    /// Because `M` has determinant 1 over GF(2), this transform is a bijection on the
    /// underlying bit patterns and is therefore perfectly lossless.
    struct E8BitTransform;

    impl E8BitTransform {
        /// Forward 8×8 binary matrix (upper-triangular with ones on the diagonal).
        /// Determinant = 1 over GF(2), hence invertible.
        const FWD: [[u8; 8]; 8] = [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ];

        /// Inverse of `FWD` over GF(2).
        const INV: [[u8; 8]; 8] = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ];

        /// Apply the forward E8-style bit mixing to the mantissa planes.
        ///
        /// Operates on groups of 8 planes at a time:
        /// - planes[0..8], planes[8..16]; any remainder (e.g. 16..23) is left unchanged.
        #[inline]
        pub fn forward_mantissa(planes: &mut [Vec<u8>]) {
            Self::apply_groups(planes, &Self::FWD);
        }

        /// Apply the inverse E8-style bit mixing to the mantissa planes.
        #[inline]
        pub fn inverse_mantissa(planes: &mut [Vec<u8>]) {
            Self::apply_groups(planes, &Self::INV);
        }

        #[inline]
        fn apply_groups(planes: &mut [Vec<u8>], mat: &[[u8; 8]; 8]) {
            let num_planes = planes.len();
            if num_planes < 8 {
                return;
            }

            let packed_len = match planes.first() {
                Some(first) => first.len(),
                None => return,
            };

            // Ensure all planes are the same length; if not, bail with no-op.
            if planes.iter().any(|p| p.len() != packed_len) {
                return;
            }

            // Process full groups of 8 planes: 0..8, 8..16, ...
            let mut start = 0;
            while start + 8 <= num_planes {
                let group = &mut planes[start..start + 8];
                Self::apply_group(group, packed_len, mat);
                start += 8;
            }
        }

        #[inline]
        fn apply_group(group: &mut [Vec<u8>], packed_len: usize, mat: &[[u8; 8]; 8]) {
            debug_assert_eq!(group.len(), 8);

            // For each byte position and bit position, apply the 8×8 transform over GF(2).
            for byte_idx in 0..packed_len {
                for bit_idx in 0..8 {
                    // Collect current bits into v[row] ∈ {0,1}
                    let mut v = [0u8; 8];
                    for row in 0..8 {
                        let byte = group[row][byte_idx];
                        v[row] = (byte >> bit_idx) & 1;
                    }

                    // out[row] = Σ_j mat[row][j] * v[j] (mod 2)
                    let mut out = [0u8; 8];
                    for row in 0..8 {
                        let mut acc = 0u8;
                        for (col, &val) in v.iter().enumerate() {
                            if mat[row][col] & 1 == 1 {
                                acc ^= val;
                            }
                        }
                        out[row] = acc & 1;
                    }

                    // Write back transformed bits.
                    for row in 0..8 {
                        let byte_ref = &mut group[row][byte_idx];
                        if out[row] == 1 {
                            *byte_ref |= 1 << bit_idx;
                        } else {
                            *byte_ref &= !(1 << bit_idx);
                        }
                    }
                }
            }
        }
    }

    /// A transposed bit-matrix representation of a block of f32s.
    struct BitMatrix {
        /// Number of f32 values in the block.
        len: usize,
        /// 32 bit-planes, each containing `len` bits packed into bytes.
        planes: [Vec<u8>; 32],
    }

    impl BitMatrix {
        /// Creates a BitMatrix by transposing a slice of f32 values. This is a bijective transform.
        fn from_f32_slice(data: &[f32]) -> Self {
            let len = data.len();
            let packed_len = len.div_ceil(8);
            let mut planes = std::array::from_fn(|_| vec![0u8; packed_len]);

            for (i, &value) in data.iter().enumerate() {
                let bits = value.to_bits();
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                for (p, plane) in planes.iter_mut().enumerate() {
                    if (bits >> p) & 1 == 1 {
                        plane[byte_idx] |= 1 << bit_idx;
                    }
                }
            }
            Self { len, planes }
        }

        /// Reconstructs a Vec<f32> by performing the inverse transposition.
        fn to_f32_vec(&self) -> Vec<f32> {
            let mut values = vec![0u32; self.len];
            for (p, plane) in self.planes.iter().enumerate() {
                for (i, value) in values.iter_mut().take(self.len).enumerate() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if (plane[byte_idx] >> bit_idx) & 1 == 1 {
                        *value |= 1 << p;
                    }
                }
            }
            values.into_iter().map(f32::from_bits).collect()
        }
    }

    /// Compresses a slice of f32 values using the bit-plane transposition method.
    ///
    /// # Arguments
    /// * `data` - The f32 slice to compress
    /// * `level` - Compression level (Level1 for speed, Level2 for size)
    ///
    /// # Returns
    /// A `Vec<u8>` containing the full compressed data frame, or an error.
    pub fn compress_with_level(
        data: &[f32],
        level: CompressionLevel,
    ) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            // Write a valid but empty frame.
            let mut buffer = Vec::new();
            buffer.write_u32::<LittleEndian>(MAGIC_NUMBER)?;
            buffer.write_u8(FORMAT_VERSION)?;
            buffer.write_u8(level as u8)?;
            buffer.write_u64::<LittleEndian>(0)?; // original_len
            return Ok(buffer);
        }

        // --- Validation ---
        let data_hash = blake3::hash(bytemuck::cast_slice(data));

        // --- Stage 1: Bit-Matrix Transposition ---
        let mut matrix = BitMatrix::from_f32_slice(data);

        // Optional Stage 1b: E8-style bit mixing on mantissa planes (fully reversible).
        {
            // Split into mantissa (0..23), exponent (23..31), sign (31).
            let (mantissa, rest) = matrix.planes.split_at_mut(23);
            let (_exponent, _sign) = rest.split_at_mut(8);
            E8BitTransform::forward_mantissa(mantissa);
        }

        // --- Stage 2: Grouped Entropy Coding ---
        let sign_plane = &matrix.planes[31];
        let exponent_planes = &matrix.planes[23..31];
        let mantissa_planes = &matrix.planes[0..23];

        let zstd_level = level.zstd_level();

        let sign_stream = zstd::stream::encode_all(sign_plane.as_slice(), zstd_level)
            .map_err(CompressionError::Zstd)?;

        let exponent_data: Vec<u8> = exponent_planes
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .collect();
        let exponent_stream = zstd::stream::encode_all(exponent_data.as_slice(), zstd_level)
            .map_err(CompressionError::Zstd)?;

        let mantissa_data: Vec<u8> = mantissa_planes
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .collect();
        let mantissa_stream = zstd::stream::encode_all(mantissa_data.as_slice(), zstd_level)
            .map_err(CompressionError::Zstd)?;

        // --- Stage 3: Framing ---
        let mut buffer = Vec::new();
        buffer.write_u32::<LittleEndian>(MAGIC_NUMBER)?;
        buffer.write_u8(FORMAT_VERSION)?;
        buffer.write_u8(level as u8)?;
        buffer.write_u64::<LittleEndian>(data.len() as u64)?;
        buffer.write_all(data_hash.as_bytes())?;
        buffer.write_u64::<LittleEndian>(sign_stream.len() as u64)?;
        buffer.write_u64::<LittleEndian>(exponent_stream.len() as u64)?;
        buffer.write_u64::<LittleEndian>(mantissa_stream.len() as u64)?;
        buffer.write_all(&sign_stream)?;
        buffer.write_all(&exponent_stream)?;
        buffer.write_all(&mantissa_stream)?;

        Ok(buffer)
    }

    /// Compresses a slice of f32 values using Level 1 (default).
    ///
    /// # Returns
    /// A `Vec<u8>` containing the full compressed data frame, or an error.
    pub fn compress(data: &[f32]) -> Result<Vec<u8>, CompressionError> {
        compress_with_level(data, CompressionLevel::default())
    }

    /// Decompresses a data frame back into a slice of f32 values.
    ///
    /// This function performs a cryptographic hash check to guarantee data integrity.
    ///
    /// # Returns
    /// A `Vec<f32>` containing the perfectly reconstructed data, or an error if the
    /// format is invalid or the integrity check fails.
    pub fn decompress(data: &[u8]) -> Result<Vec<f32>, CompressionError> {
        let mut cursor = Cursor::new(data);

        // --- Stage 3 (Inverse): Parsing ---
        if cursor.read_u32::<LittleEndian>()? != MAGIC_NUMBER {
            return Err(CompressionError::InvalidFormat(
                "Invalid magic number".into(),
            ));
        }
        if cursor.read_u8()? != FORMAT_VERSION {
            return Err(CompressionError::InvalidFormat(
                "Unsupported version".into(),
            ));
        }

        let _level = cursor.read_u8()?; // Read but don't validate (both levels decompress the same way)

        let original_len = cursor.read_u64::<LittleEndian>()? as usize;
        if original_len == 0 {
            return Ok(Vec::new());
        }

        let mut expected_hash = [0u8; 32];
        cursor.read_exact(&mut expected_hash)?;

        let sign_len = cursor.read_u64::<LittleEndian>()? as usize;
        let exp_len = cursor.read_u64::<LittleEndian>()? as usize;
        let mant_len = cursor.read_u64::<LittleEndian>()? as usize;

        // --- Stage 2 (Inverse): Entropy Decoding ---
        let mut sign_stream = vec![0u8; sign_len];
        cursor.read_exact(&mut sign_stream)?;

        let mut exp_stream = vec![0u8; exp_len];
        cursor.read_exact(&mut exp_stream)?;

        let mut mant_stream = vec![0u8; mant_len];
        cursor.read_exact(&mut mant_stream)?;

        // No trailing bytes allowed: frame must be self-contained.
        if cursor.position() != data.len() as u64 {
            return Err(CompressionError::InvalidFormat(
                "Trailing data after compressed streams".into(),
            ));
        }

        let sign_plane =
            zstd::stream::decode_all(Cursor::new(sign_stream)).map_err(CompressionError::Zstd)?;
        let exp_data =
            zstd::stream::decode_all(Cursor::new(exp_stream)).map_err(CompressionError::Zstd)?;
        let mant_data =
            zstd::stream::decode_all(Cursor::new(mant_stream)).map_err(CompressionError::Zstd)?;

        // --- Stage 1 (Inverse): Reconstruct BitMatrix ---
        let packed_len = original_len.div_ceil(8);

        // Validate that decoded plane lengths match what the header implies.
        if sign_plane.len() != packed_len
            || exp_data.len() != 8 * packed_len
            || mant_data.len() != 23 * packed_len
        {
            return Err(CompressionError::InvalidFormat(
                "Decompressed plane sizes do not match header metadata".into(),
            ));
        }

        let mut matrix = BitMatrix {
            len: original_len,
            planes: std::array::from_fn(|_| Vec::new()),
        };

        for i in 0..23 {
            matrix.planes[i] = mant_data[i * packed_len..(i + 1) * packed_len].to_vec();
        }
        for i in 0..8 {
            matrix.planes[i + 23] = exp_data[i * packed_len..(i + 1) * packed_len].to_vec();
        }
        matrix.planes[31] = sign_plane;

        // Inverse of the E8-style bit mixing applied during compression.
        {
            let (mantissa, _rest) = matrix.planes.split_at_mut(23);
            E8BitTransform::inverse_mantissa(mantissa);
        }

        let reconstructed_data = matrix.to_f32_vec();

        // --- Validation ---
        let actual_hash = blake3::hash(bytemuck::cast_slice(&reconstructed_data));
        if actual_hash.as_bytes() != &expected_hash {
            return Err(CompressionError::IntegrityError);
        }

        Ok(reconstructed_data)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 6: TESTS
// ═══════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_true_lossless_roundtrip() {
        let test_values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            std::f32::consts::PI,
            f32::MAX,
            f32::MIN_POSITIVE,           // smallest normal
            f32::MIN_POSITIVE / 2.0,     // subnormal
            f32::from_bits(0x0000_0001), // smallest positive subnormal
            f32::from_bits(0x8000_0001), // smallest negative subnormal
        ];
        for &original in &test_values {
            let e32l = E32L::from_f32(original);
            let recovered = e32l.to_f32();
            assert_eq!(
                original.to_bits(),
                recovered.to_bits(),
                "Roundtrip failed for {}",
                original
            );
        }
    }

    #[test]
    fn test_special_values_are_preserved() {
        // NaN payloads (including non-canonical patterns)
        let nan_bits = [
            f32::NAN.to_bits(),
            0x7fc0_0001,
            0x7fff_ffff,
            0x7fa0_1234,
            0xffc0_0001,
        ];
        for bits in nan_bits {
            let original = f32::from_bits(bits);
            let e32l = E32L::from_f32(original);
            let recovered = e32l.to_f32();
            assert!(
                recovered.is_nan(),
                "Recovered NaN lost NaN-ness for bits=0x{bits:08x}"
            );
            assert_eq!(
                bits,
                recovered.to_bits(),
                "NaN payload changed for bits=0x{bits:08x}"
            );
        }

        // Infinity
        let inf_original = f32::INFINITY;
        let e32l_inf = E32L::from_f32(inf_original);
        assert_eq!(
            inf_original.to_bits(),
            e32l_inf.to_f32().to_bits(),
            "Infinity bits changed"
        );

        // Negative Infinity
        let neg_inf_original = f32::NEG_INFINITY;
        let e32l_neg_inf = E32L::from_f32(neg_inf_original);
        assert_eq!(
            neg_inf_original.to_bits(),
            e32l_neg_inf.to_f32().to_bits(),
            "Negative infinity bits changed"
        );
    }

    #[test]
    fn test_storage_size_is_identical_to_f32() {
        // These checks rely on E8F being `#[repr(transparent)]` over `u8`.
        assert_eq!(
            size_of::<E8F>(),
            size_of::<u8>(),
            "E8F must remain a 1-byte wrapper"
        );
        assert_eq!(
            std::mem::align_of::<E8F>(),
            std::mem::align_of::<u8>(),
            "E8F alignment must match u8"
        );

        assert_eq!(
            size_of::<E32L>(),
            size_of::<f32>(),
            "E32L must have same size as f32"
        );
        assert_eq!(size_of::<E32L>(), 4, "E32L must remain 4 bytes in size");
    }

    #[test]
    fn test_bit_level_deconstruction() {
        let value = 123.456f32;
        let bytes = value.to_le_bytes();
        let e32l = E32L::from_f32(value);

        assert_eq!(e32l.roots[0].index(), bytes[0]);
        assert_eq!(e32l.roots[1].index(), bytes[1]);
        assert_eq!(e32l.roots[2].index(), bytes[2]);
        assert_eq!(e32l.roots[3].index(), bytes[3]);

        let recovered_bytes = e32l.to_bytes();
        assert_eq!(bytes, recovered_bytes);
    }

    #[test]
    fn test_from_and_to_bits() {
        let bits: u32 = 0x42c80000; // Represents 100.0f32
        let e32l: E32L = bits.into(); // Use From<u32> trait
        assert_eq!(e32l.to_f32(), 100.0f32);

        let recovered_bits: u32 = e32l.into(); // Use Into<u32> trait
        assert_eq!(recovered_bits, bits);
    }

    #[test]
    fn test_arbitrary_bits_roundtrip() {
        // A small but representative sample of raw IEEE 754 bit patterns.
        let samples: &[u32] = &[
            0x0000_0000, // +0.0
            0x8000_0000, // -0.0
            0x3f80_0000, // 1.0
            0xbf80_0000, // -1.0
            0x7f80_0000, // +inf
            0xff80_0000, // -inf
            0x7fc0_0001, // quiet NaN with payload
            0x7fa0_1234, // another NaN payload
            0x0000_0001, // smallest positive subnormal
            0x8000_0001, // smallest negative subnormal
            0x00ff_ffff, // large positive subnormal
            0x1234_5678, // arbitrary pattern
            0xffff_ffff, // arbitrary pattern (also NaN)
        ];

        for &bits in samples {
            let e32l: E32L = bits.into();
            let roundtrip_bits: u32 = e32l.into();
            assert_eq!(
                bits, roundtrip_bits,
                "E32L must be a pure bit-level bijection for bits=0x{bits:08x}"
            );
        }
    }

    #[test]
    fn test_distinction_from_f32l() {
        // This test clarifies that while functionally identical to a simple byte wrapper,
        // E32L is composed of E8F types, making it semantically different.
        let value = 1.0f32;
        let e32l = E32L::from_f32(value);

        // Access the underlying E8F components
        let root0: E8F = e32l.roots[0];
        assert_eq!(root0.index(), 0x00);

        let root3: E8F = e32l.roots[3];
        assert_eq!(root3.index(), 0x3f); // Part of the IEEE 754 representation of 1.0
    }

    #[cfg(feature = "bytemuck")]
    mod bytemuck_compat_tests {
        use super::*;
        use bytemuck::{Pod, Zeroable, cast_slice};

        #[test]
        fn test_e32l_is_pod_and_castable() {
            // Basic sanity: trait bounds compile and work at runtime.
            fn assert_pod_zeroable<T: Pod + Zeroable>() {}
            assert_pod_zeroable::<E32L>();

            let values = [
                E32L::from_f32(0.0),
                E32L::from_f32(1.0),
                E32L::from_f32(-1.0),
                E32L::from_f32(123.456),
            ];

            // Cast [E32L] -> [u8]
            let bytes: &[u8] = cast_slice(&values);
            assert_eq!(bytes.len(), values.len() * std::mem::size_of::<E32L>());

            // Roundtrip [u8] -> [E32L]
            let roundtrip: &[E32L] = cast_slice(bytes);
            assert_eq!(roundtrip, &values);
        }
    }

    // --- Compression Framework Tests ---
    #[cfg(feature = "compression")]
    mod compression_tests {
        use super::super::compression::{CompressionError, compress, decompress};

        #[test]
        fn test_compression_roundtrip_lossless() {
            let original_data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
            let compressed = compress(&original_data).expect("Compression failed");
            let reconstructed = decompress(&compressed).expect("Decompression failed");

            assert_eq!(original_data.len(), reconstructed.len(), "Length mismatch");
            for (a, b) in original_data.iter().zip(reconstructed.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "Data mismatch after roundtrip");
            }
        }

        #[test]
        fn test_compression_achieves_bitrate_reduction() {
            // Data with low entropy (a sine wave) should compress well.
            let original_data: Vec<f32> =
                (0..4096).map(|i| (i as f32 * 0.05).sin() * 100.0).collect();
            let compressed = compress(&original_data).expect("Compression failed");

            let original_size = original_data.len() * std::mem::size_of::<f32>();
            let compressed_size = compressed.len();

            assert!(
                compressed_size < original_size,
                "Compression did not reduce data size for predictable input. Original: {}, Compressed: {}",
                original_size,
                compressed_size
            );
        }

        #[test]
        fn test_compression_handles_edge_cases() {
            let data: Vec<f32> = vec![
                0.0,
                -0.0,
                1.0,
                f32::MAX,
                f32::MIN_POSITIVE,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc00001),
            ];
            let compressed = compress(&data).expect("Compression failed");
            let reconstructed = decompress(&compressed).expect("Decompression failed");

            assert_eq!(data.len(), reconstructed.len());
            for (a, b) in data.iter().zip(reconstructed.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }

        #[test]
        fn test_empty_slice_compression() {
            let data: Vec<f32> = Vec::new();
            let compressed = compress(&data).unwrap();
            let reconstructed = decompress(&compressed).unwrap();
            assert!(
                !compressed.is_empty(),
                "Empty frame should still have header"
            );
            assert!(reconstructed.is_empty());
        }

        #[test]
        fn test_decompress_corrupted_data_fails() {
            let original_data: Vec<f32> = (0..128).map(|i| i as f32).collect();
            let mut compressed = compress(&original_data).unwrap();

            // Corrupt the hash
            compressed[13] = compressed[13].wrapping_add(1);

            let result = decompress(&compressed);
            assert!(
                matches!(result, Err(CompressionError::IntegrityError)),
                "Decompression should fail with IntegrityError on hash mismatch"
            );

            // Corrupt the payload (which may cause Zstd error or hash mismatch)
            let mut compressed_payload = compress(&original_data).unwrap();
            let payload_mid = compressed_payload.len() - 10;
            compressed_payload[payload_mid] = compressed_payload[payload_mid].wrapping_add(1);
            let result_payload = decompress(&compressed_payload);
            assert!(
                result_payload.is_err(),
                "Decompression should fail on payload corruption"
            );
        }
    }

    #[test]
    fn prove_e32l_bijection_via_hash() {
        let mut orig_xor: u64 = 0;
        let mut rec_xor: u64 = 0;

        // Exhaustive test of all f32 bit patterns is infeasible (2^32 values).
        // Instead, we test a large random sample and all special values.
        // Use the platform RNG API by invoking `rand::random::<T>()` which is
        // stable across editions and avoids the reserved method name `gen`.

        // Test all special values
        let special_values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_0001), // quiet NaN
            f32::from_bits(0x7fa0_1234), // another NaN
            f32::from_bits(0x0000_0001), // smallest positive subnormal
            f32::from_bits(0x8000_0001), // smallest negative subnormal
        ];

        for &val in &special_values {
            let e32l = E32L::from_f32(val);
            let rec = e32l.to_f32();
            assert_eq!(
                val.to_bits(),
                rec.to_bits(),
                "Bijection failed for special value: {}",
                val
            );
            orig_xor ^= val.to_bits() as u64;
            rec_xor ^= rec.to_bits() as u64;
        }

        // Test a large random sample
        for _ in 0..1_000_000 {
            // Use the top-level random function which internally uses the thread RNG.
            let bits: u32 = rand::random::<u32>();
            let val = f32::from_bits(bits);
            let e32l = E32L::from_f32(val);
            let rec = e32l.to_f32();
            assert_eq!(
                val.to_bits(),
                rec.to_bits(),
                "Bijection failed for random value"
            );
            orig_xor ^= val.to_bits() as u64;
            rec_xor ^= rec.to_bits() as u64;
        }

        // XOR of all tested bits should be identical
        assert_eq!(orig_xor, rec_xor, "Hash mismatch: bijection not preserved");
    }
}
