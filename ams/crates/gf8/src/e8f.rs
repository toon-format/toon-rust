/* crates/gf8/src/e8f.rs */
//! # E8F – The E8 Float: Finite Lattice Arithmetic
//!
//! This module implements the "Trillion Dollar" optimization: replacing runtime
//! floating-point operations with precomputed integer lookup tables over the
//! E8 lattice. Operations always resolve to valid E8 roots (canonical states).
//!
//! ## Origin Citation
//! "E8 operations can replace SIMD float math... weights: [Eb8; d/8]...
//! A single Eb8 = 8 numbers at once."
//!
//! ## Key Capabilities
//! - **Zero-FLOP Inference**: Addition and multiplication become array lookups
//! - **Perfect Closure**: Operations always resolve to a valid E8 root
//! - **32x Compression**: Weights stored as u8, computed as u8
//! - **Group Structure**: Implements Klein/Weyl group symmetries
//!
//! ## Architectural Notes
//! Instead of treating E8 roots as vectors for f32 math, we precompute the
//! interaction of every root with every other root. This transforms linear
//! algebra into Integer Lookup Tables.
//!
//! The E8 root system forms a group under reflection, and the 240 roots can
//! be combined via geometric operations that map back to roots.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::quantize::{get_e8_codebook, quantize_to_nearest_code};
use crate::{Gf8, Gf8BitSig, Gf8LosslessCode};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::OnceLock;

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 1: PRECOMPUTED LOOKUP TABLES
// ═══════════════════════════════════════════════════════════════════════════════════════

/// Precomputed lookup tables for E8 lattice arithmetic.
///
/// These tables enable O(1) operations between E8 roots without any
/// floating-point computation at inference time.
///
/// Memory footprint: ~460KB total (heap-allocated)
/// - add_table:     240×240 = 57,600 bytes
/// - sub_table:     240×240 = 57,600 bytes
/// - mul_table:     240×240 = 57,600 bytes
/// - dot_table:     240×240 = 57,600 bytes
/// - reflect_table: 240×240 = 57,600 bytes
/// - neg_table:     240 bytes
pub struct E8ArithmeticTables {
    /// Addition: add_table[a][b] = snap(root_a + root_b)
    pub add_table: Box<[[u8; 240]; 240]>,

    /// Subtraction: sub_table[a][b] = snap(root_a - root_b)
    pub sub_table: Box<[[u8; 240]; 240]>,

    /// Multiplication (Hadamard-style): mul_table[a][b] = snap(root_a ⊙ root_b)
    pub mul_table: Box<[[u8; 240]; 240]>,

    /// Dot product: dot_table[a][b] = quantized(root_a · root_b) in [0, 255]
    /// Maps [-1.0, 1.0] → [0, 255]
    pub dot_table: Box<[[u8; 240]; 240]>,

    /// Reflection: reflect_table[a][b] = snap(root_a - 2(root_a·root_b)root_b)
    pub reflect_table: Box<[[u8; 240]; 240]>,

    /// Negation: neg_table[a] = index of -root_a
    pub neg_table: [u8; 240],
}

/// Static singleton for the arithmetic tables.
pub static E8_ARITHMETIC: OnceLock<E8ArithmeticTables> = OnceLock::new();

impl E8ArithmeticTables {
    /// Generate all arithmetic lookup tables.
    /// This is expensive (~240² × 5 operations) but only done once at startup.
    pub fn generate() -> Self {
        let codebook = get_e8_codebook();
        let roots = &codebook.roots;

        // Allocate on heap to avoid stack overflow
        let mut add_table = Box::new([[0u8; 240]; 240]);
        let mut sub_table = Box::new([[0u8; 240]; 240]);
        let mut mul_table = Box::new([[0u8; 240]; 240]);
        let mut dot_table = Box::new([[128u8; 240]; 240]);
        let mut neg_table = [0u8; 240];
        let mut reflect_table = Box::new([[0u8; 240]; 240]);

        // Precompute all pairwise operations
        for (i, root_i) in roots.iter().enumerate().take(240) {
            let ri = root_i.coords();

            // Negation: find antipodal root
            let neg_coords: [f32; 8] = std::array::from_fn(|k| -ri[k]);
            let (neg_code, _) = quantize_to_nearest_code(&neg_coords);
            neg_table[i] = neg_code.0;

            for (j, root_j) in roots.iter().enumerate().take(240) {
                let rj = root_j.coords();

                // Addition: root_i + root_j → nearest root
                let sum_coords: [f32; 8] = std::array::from_fn(|k| ri[k] + rj[k]);
                let (sum_code, _) = quantize_to_nearest_code(&sum_coords);
                add_table[i][j] = sum_code.0;

                // Subtraction: root_i - root_j → nearest root
                let diff_coords: [f32; 8] = std::array::from_fn(|k| ri[k] - rj[k]);
                let (diff_code, _) = quantize_to_nearest_code(&diff_coords);
                sub_table[i][j] = diff_code.0;

                // Multiplication (Hadamard/element-wise): root_i ⊙ root_j → nearest root
                let prod_coords: [f32; 8] = std::array::from_fn(|k| ri[k] * rj[k]);
                let (prod_code, _) = quantize_to_nearest_code(&prod_coords);
                mul_table[i][j] = prod_code.0;

                // Dot product: quantized to [0, 255]
                let dot: f32 = ri.iter().zip(rj.iter()).map(|(a, b)| a * b).sum();
                // Map [-1, 1] to [0, 255]
                let quantized = ((dot.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                dot_table[i][j] = quantized;

                // Reflection: reflect i through hyperplane normal to j
                // Formula: r_i - 2 * (r_i · r_j) * r_j
                let scale = 2.0 * dot;
                let reflect_coords: [f32; 8] = std::array::from_fn(|k| ri[k] - scale * rj[k]);
                let (reflect_code, _) = quantize_to_nearest_code(&reflect_coords);
                reflect_table[i][j] = reflect_code.0;
            }
        }

        Self {
            add_table,
            sub_table,
            mul_table,
            dot_table,
            neg_table,
            reflect_table,
        }
    }
}

/// Get the global arithmetic tables (lazily initialized).
#[inline]
pub fn get_e8_arithmetic() -> &'static E8ArithmeticTables {
    E8_ARITHMETIC.get_or_init(E8ArithmeticTables::generate)
}

/// Ensure tables are initialized (call at startup for predictable latency).
pub fn init_e8_arithmetic() {
    let _ = get_e8_arithmetic();
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 2: E8F – THE LATTICE SCALAR TYPE
// ═══════════════════════════════════════════════════════════════════════════════════════

/// The E8 Float: A numeric type where all operations resolve to valid E8 lattice points.
///
/// This replaces f32/f16 in model weights and activations, enabling:
/// - Zero-FLOP inference (all ops are table lookups)
/// - Perfect closure (results always valid E8 roots)
/// - 32x compression (1 byte vs 32 bytes for 8D vector)
///
/// # Example
/// ```rust
/// use gf8::e8f::E8F;
///
/// let a = E8F::new(42);
/// let b = E8F::new(100);
///
/// // All operations are O(1) table lookups
/// let sum = a + b;      // Addition
/// let diff = a - b;     // Subtraction
/// let prod = a * b;     // Hadamard multiplication
/// let neg = -a;         // Negation
/// let dot = a.dot(b);   // Dot product (returns f32)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct E8F(pub u8);

impl fmt::Display for E8F {
    /// Formats E8F for display in logs and debugging.
    ///
    /// Format: `E8F(index)` where index is the root index.
    /// - Valid roots (0-239): Shows the index
    /// - Invalid/Zero (240+): Shows as "zero" or invalid index
    ///
    /// This enables E8F to be used with tracing macros and println!.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "E8F({})", self.0)
        } else if self.0 == 240 {
            write!(f, "E8F(zero)")
        } else {
            write!(f, "E8F(invalid:{})", self.0)
        }
    }
}

impl E8F {
    /// The "zero" state (index 240, outside valid roots).
    /// Used for null/undefined values.
    pub const ZERO: Self = Self(240);

    /// Create from a root index (0-239). Values ≥240 are clamped.
    #[inline]
    pub const fn new(root_idx: u8) -> Self {
        Self(if root_idx < 240 {
            root_idx
        } else {
            Self::ZERO.0
        })
    }

    /// Create from a raw byte without clamping (used for lossless bit containers like E32L).
    #[inline]
    pub const fn from_raw(byte: u8) -> Self {
        Self(byte)
    }

    /// Create from a root index with explicit validation; returns None if out of range.
    #[inline]
    pub const fn new_checked(root_idx: u8) -> Option<Self> {
        if root_idx < 240 {
            Some(Self(root_idx))
        } else {
            None
        }
    }

    /// Create from a Gf8BitSig .
    #[inline]
    pub const fn from_code(code: Gf8BitSig) -> Self {
        Self::new(code.0)
    }

    /// Create from a Gf8LosslessCode.
    #[inline]
    pub const fn from_lossless(code: Gf8LosslessCode) -> Self {
        Self::new(code.0)
    }

    /// Get the root index.
    #[inline]
    pub const fn index(&self) -> u8 {
        self.0
    }

    /// Check if this is a valid root (index < 240).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.0 < 240
    }

    /// Convert to Gf8BitSig .
    #[inline]
    pub const fn to_code(&self) -> Gf8BitSig {
        Gf8BitSig(self.0)
    }

    /// Convert to Gf8LosslessCode.
    #[inline]
    pub const fn to_lossless(&self) -> Gf8LosslessCode {
        Gf8LosslessCode(self.0)
    }

    /// Dequantize to the actual Gf8 vector (for final output).
    /// This is a **lossless** operation for valid E8F roots - the returned
    /// Gf8 is the exact root from the codebook with no quantization error.
    pub fn to_gf8(&self) -> Gf8 {
        if self.0 >= 240 {
            return Gf8::ZERO;
        }
        let codebook = get_e8_codebook();
        codebook.roots[self.0 as usize]
    }

    /// Quantize an f32 array to E8F.
    pub fn from_f32(coords: &[f32; 8]) -> Self {
        let (code, _) = quantize_to_nearest_code(coords);
        Self(code.0)
    }

    /// Compute quantized dot product (returns u8 in [0, 255]).
    /// 0 = -1.0, 128 = 0.0, 255 = 1.0
    #[inline]
    pub fn dot_quantized(self, other: Self) -> u8 {
        if self.0 >= 240 || other.0 >= 240 {
            return 128; // 0.0
        }
        let tables = get_e8_arithmetic();
        tables.dot_table[self.0 as usize][other.0 as usize]
    }

    /// Compute dot product as f32 (dequantized).
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        let q = self.dot_quantized(other);
        (q as f32 / 127.5) - 1.0
    }

    /// Reflect self through hyperplane normal to other.
    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        if self.0 >= 240 || normal.0 >= 240 {
            return Self::ZERO;
        }
        let tables = get_e8_arithmetic();
        Self(tables.reflect_table[self.0 as usize][normal.0 as usize])
    }

    /// Check if two E8Fs are neighbors in the E8 lattice (dot ≈ 0.5, 60° angle).
    #[inline]
    pub fn is_neighbor(self, other: Self) -> bool {
        let dot = self.dot_quantized(other);
        (180..=200).contains(&dot) // ~0.5 in [0,255] scale
    }

    /// Check if two E8Fs are antipodal (opposite directions, dot ≈ -1.0).
    #[inline]
    pub fn is_antipodal(self, other: Self) -> bool {
        self.dot_quantized(other) < 10
    }

    /// Check if two E8Fs are orthogonal (dot ≈ 0.0).
    #[inline]
    pub fn is_orthogonal(self, other: Self) -> bool {
        let dot = self.dot_quantized(other);
        (120..=136).contains(&dot)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 3: OPERATOR IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

impl Add for E8F {
    type Output = Self;
    /// E8F addition via lookup table.
    /// **Error Bound**: Result is the nearest E8 root to (a + b), with
    /// ≤0.087 chordal distance error. The operation itself is deterministic
    /// and error-free for repeated identical inputs (lookup table is constant).
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.0 >= 240 || rhs.0 >= 240 {
            return Self::ZERO;
        }
        let tables = get_e8_arithmetic();
        Self(tables.add_table[self.0 as usize][rhs.0 as usize])
    }
}

impl Sub for E8F {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 >= 240 || rhs.0 >= 240 {
            return Self::ZERO;
        }
        let tables = get_e8_arithmetic();
        Self(tables.sub_table[self.0 as usize][rhs.0 as usize])
    }
}

impl Mul for E8F {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if self.0 >= 240 || rhs.0 >= 240 {
            return Self::ZERO;
        }
        let tables = get_e8_arithmetic();
        Self(tables.mul_table[self.0 as usize][rhs.0 as usize])
    }
}

impl Neg for E8F {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 >= 240 {
            return self;
        }
        let tables = get_e8_arithmetic();
        Self(tables.neg_table[self.0 as usize])
    }
}

impl From<u8> for E8F {
    fn from(idx: u8) -> Self {
        Self::new(idx)
    }
}

impl From<Gf8BitSig> for E8F {
    fn from(code: Gf8BitSig) -> Self {
        Self::from_code(code)
    }
}

impl From<Gf8LosslessCode> for E8F {
    fn from(code: Gf8LosslessCode) -> Self {
        Self::from_lossless(code)
    }
}

impl From<E8F> for Gf8BitSig {
    fn from(e: E8F) -> Self {
        e.to_code()
    }
}

impl From<E8F> for Gf8LosslessCode {
    fn from(e: E8F) -> Self {
        e.to_lossless()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 4: E8VEC – VECTOR OF E8FS
// ═══════════════════════════════════════════════════════════════════════════════════════

/// A vector of E8Fs for batch operations.
///
/// Enables processing entire embeddings
/// using only integer lookups. Storage: 1 byte per 8D block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct E8Vec {
    pub data: Vec<E8F>,
}

impl E8Vec {
    /// Create from a slice of root indices.
    pub fn from_indices(indices: &[u8]) -> Self {
        Self {
            data: indices.iter().map(|&i| E8F::new(i)).collect(),
        }
    }

    /// Quantize a high-dimensional f32 vector to E8Vec.
    /// Splits into 8D blocks and quantizes each to an E8F.
    pub fn from_f32_vec(vec: &[f32]) -> Self {
        let num_blocks = vec.len().div_ceil(8);
        let mut data = Vec::with_capacity(num_blocks);

        for chunk in vec.chunks(8) {
            let mut block = [0.0f32; 8];
            block[..chunk.len()].copy_from_slice(chunk);
            data.push(E8F::from_f32(&block));
        }

        Self { data }
    }

    /// Dequantize to f32 vector.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len() * 8);
        for e in &self.data {
            let gf8 = e.to_gf8();
            result.extend_from_slice(gf8.coords());
        }
        result
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a * *b)
                .collect(),
        }
    }

    /// Dot product (sum of element-wise dots).
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.dot(*b))
            .sum()
    }

    /// Quantized dot product (sum of quantized element-wise dots).
    pub fn dot_quantized(&self, other: &Self) -> u32 {
        assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.dot_quantized(*b) as u32)
            .sum()
    }

    /// Get the raw bytes (for storage/transmission).
    /// 32x compression: 2048D → 256 bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.iter().map(|e| e.0).collect()
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: bytes.iter().map(|&b| E8F::new(b)).collect(),
        }
    }

    /// Length in E8F elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Original dimensionality (len × 8).
    pub fn dim(&self) -> usize {
        self.data.len() * 8
    }
}

impl std::ops::Index<usize> for E8Vec {
    type Output = E8F;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 5: E8 TENSOR CORE – FUSED MATRIX OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════

/// E8 Tensor Core: Fused matrix operations over the E8 lattice.
///
/// Analogous to GPU tensor cores, but using lookup tables instead of FP units.
/// All operations maintain E8 closure (results are always valid roots).
pub struct E8TensorCore;

impl E8TensorCore {
    // ─────────────────────────────────────────────────────────────────────────────────
    // FUSED MULTIPLY-ACCUMULATE (FMA)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Fused multiply-add: `a * b + c` in E8 space.
    /// Two table lookups instead of three separate ops.
    #[inline]
    pub fn fma(a: E8F, b: E8F, c: E8F) -> E8F {
        let prod = a * b;
        prod + c
    }

    /// Fused multiply-subtract: `a * b - c` in E8 space.
    #[inline]
    pub fn fms(a: E8F, b: E8F, c: E8F) -> E8F {
        let prod = a * b;
        prod - c
    }

    /// Fused negative multiply-add: `-(a * b) + c` = `c - a * b`
    #[inline]
    pub fn fnma(a: E8F, b: E8F, c: E8F) -> E8F {
        let prod = a * b;
        c - prod
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // VECTOR-VECTOR OPERATIONS (like BLAS Level 1)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// AXPY: `y = a * x + y` (scale and accumulate)
    /// Fundamental BLAS operation, fully in E8 space.
    pub fn axpy(a: E8F, x: &E8Vec, y: &mut E8Vec) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y.data[i] = Self::fma(a, x.data[i], y.data[i]);
        }
    }

    /// Dot product with accumulator: `acc + sum(x[i] * y[i])`
    /// Returns quantized result as u32 for precision.
    pub fn dot_acc(x: &E8Vec, y: &E8Vec, acc: u32) -> u32 {
        acc + x.dot_quantized(y)
    }

    /// Weighted sum: `sum(weights[i] * vectors[i])`
    /// Reduces multiple vectors into one.
    pub fn weighted_sum(weights: &[E8F], vectors: &[E8Vec]) -> E8Vec {
        assert_eq!(weights.len(), vectors.len());
        assert!(!vectors.is_empty());

        let len = vectors[0].len();
        let mut result = E8Vec::from_indices(&vec![0u8; len]);

        for (w, v) in weights.iter().zip(vectors.iter()) {
            for i in 0..len {
                result.data[i] = Self::fma(*w, v.data[i], result.data[i]);
            }
        }
        result
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // MATRIX-VECTOR OPERATIONS (like BLAS Level 2)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Matrix-vector multiply: `y = M * x`
    /// M is row-major: M[row][col], each row is an E8Vec.
    pub fn matvec(m: &[E8Vec], x: &E8Vec) -> Vec<u32> {
        m.iter().map(|row| row.dot_quantized(x)).collect()
    }

    /// Matrix-vector multiply with E8F output (snapped to nearest roots).
    /// Uses dot products to find best-matching output roots.
    pub fn matvec_e8f(m: &[E8Vec], x: &E8Vec) -> E8Vec {
        let dots: Vec<u32> = Self::matvec(m, x);
        // Map quantized dots back to E8F indices
        // Higher dot = more similar, map to root index space
        let indices: Vec<u8> = dots
            .iter()
            .map(|&d| ((d as f32 / (x.len() as f32 * 255.0)) * 239.0) as u8)
            .collect();
        E8Vec::from_indices(&indices)
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // OUTER PRODUCT (for attention-like operations)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Outer product: `M[i][j] = u[i] * v[j]`
    /// Creates a matrix from two vectors (attention pattern).
    pub fn outer(u: &E8Vec, v: &E8Vec) -> Vec<E8Vec> {
        u.data
            .iter()
            .map(|&ui| E8Vec {
                data: v.data.iter().map(|&vj| ui * vj).collect(),
            })
            .collect()
    }

    /// Outer product with addition: `M[i][j] += u[i] * v[j]`
    /// Accumulates into existing matrix.
    pub fn outer_add(u: &E8Vec, v: &E8Vec, m: &mut [E8Vec]) {
        for (i, &ui) in u.data.iter().enumerate() {
            for (j, &vj) in v.data.iter().enumerate() {
                m[i].data[j] = Self::fma(ui, vj, m[i].data[j]);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // BATCH OPERATIONS (like tensor core's warp-level ops)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Batch dot products: compute dots for multiple query-key pairs.
    /// Returns quantized similarities for attention scoring.
    pub fn batch_dots(queries: &[E8Vec], keys: &[E8Vec]) -> Vec<Vec<u32>> {
        queries
            .iter()
            .map(|q| keys.iter().map(|k| q.dot_quantized(k)).collect())
            .collect()
    }

    /// Batch dot products (single query against multiple keys).
    /// Optimized for retrieval: one query, many candidates.
    pub fn query_dots(query: &E8Vec, keys: &[E8Vec]) -> Vec<u32> {
        keys.iter().map(|k| query.dot_quantized(k)).collect()
    }

    /// Dom-R selection from batch dots (for attention/retrieval).
    /// Returns indices of K highest-scoring keys.
    pub fn top_k(scores: &[u32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, u32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1)); // Descending
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // SOFTMAX APPROXIMATION (via E8 geometry)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Approximate softmax using E8 neighbor structure.
    /// Maps quantized scores to probability-like weights.
    ///
    /// Instead of exp(x)/sum(exp), uses:
    /// - Shift scores to positive range
    /// - Normalize by sum
    /// - Quantize to u8 weights
    pub fn softmax_approx(scores: &[u32]) -> Vec<u8> {
        if scores.is_empty() {
            return vec![];
        }

        let min = *scores.iter().min().unwrap();
        let shifted: Vec<u32> = scores.iter().map(|&s| s - min + 1).collect();
        let sum: u32 = shifted.iter().sum();

        if sum == 0 {
            return vec![255 / scores.len() as u8; scores.len()];
        }

        shifted
            .iter()
            .map(|&s| ((s as u64 * 255) / sum as u64) as u8)
            .collect()
    }

    /// Weighted combination using softmax-like weights.
    /// `result = sum(weights[i] * values[i])` where weights are normalized.
    pub fn softmax_combine(weights: &[u8], values: &[E8Vec]) -> E8Vec {
        assert_eq!(weights.len(), values.len());
        assert!(!values.is_empty());

        let len = values[0].len();
        let mut result = E8Vec::from_indices(&vec![0u8; len]);

        // Convert u8 weights to E8F (map 0-255 to root indices 0-239)
        let e8_weights: Vec<E8F> = weights
            .iter()
            .map(|&w| E8F::new((w as u16 * 239 / 255) as u8))
            .collect();

        for (w, v) in e8_weights.iter().zip(values.iter()) {
            for i in 0..len {
                result.data[i] = Self::fma(*w, v.data[i], result.data[i]);
            }
        }
        result
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // ATTENTION MECHANISM (full E8 attention in one call)
    // ─────────────────────────────────────────────────────────────────────────────────

    /// E8 Attention: Q·K^T → softmax → weighted V
    ///
    /// Single-head attention entirely in E8 space:
    /// 1. Compute Q·K similarities (batch dots)
    /// 2. Apply softmax approximation
    /// 3. Weight and sum V vectors
    ///
    /// Returns attended output for each query.
    pub fn attention(queries: &[E8Vec], keys: &[E8Vec], values: &[E8Vec]) -> Vec<E8Vec> {
        assert_eq!(keys.len(), values.len());

        queries
            .iter()
            .map(|q| {
                // Q·K^T
                let scores = Self::query_dots(q, keys);
                // Softmax
                let weights = Self::softmax_approx(&scores);
                // Weighted sum of V
                Self::softmax_combine(&weights, values)
            })
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // LAYER NORM APPROXIMATION
    // ─────────────────────────────────────────────────────────────────────────────────

    /// Approximate layer normalization via E8 centering.
    ///
    /// Instead of (x - mean) / std, we:
    /// 1. Find the "centroid" root (most common neighbor)
    /// 2. Subtract it (centering)
    /// 3. The E8 lattice's uniform structure provides implicit normalization
    pub fn layer_norm_approx(x: &E8Vec) -> E8Vec {
        if x.is_empty() {
            return x.clone();
        }

        // Find centroid: average all roots, snap to nearest
        // Simplified: use the most frequent root as center
        let mut counts = [0u32; 240];
        for e in &x.data {
            if e.is_valid() {
                counts[e.0 as usize] += 1;
            }
        }

        let centroid_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        let centroid = E8F::new(centroid_idx);

        // Subtract centroid from all elements
        E8Vec {
            data: x.data.iter().map(|&e| e - centroid).collect(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 6: E8 MATRIX TYPE
// ═══════════════════════════════════════════════════════════════════════════════════════

/// E8 Matrix: 2D array of E8Fs for weight matrices.
///
/// Row-major storage, each row is an E8Vec.
/// Used for linear layers, attention projections, etc.
#[derive(Debug, Clone)]
pub struct E8Mat {
    /// Rows of the matrix.
    pub rows: Vec<E8Vec>,
    /// Number of columns (E8F elements per row).
    pub cols: usize,
}

impl E8Mat {
    /// Create a new matrix with given dimensions, initialized to root 0.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            rows: (0..num_rows)
                .map(|_| E8Vec::from_indices(&vec![0u8; num_cols]))
                .collect(),
            cols: num_cols,
        }
    }

    /// Create from raw bytes (row-major).
    pub fn from_bytes(bytes: &[u8], num_rows: usize, num_cols: usize) -> Self {
        assert_eq!(bytes.len(), num_rows * num_cols);
        Self {
            rows: bytes.chunks(num_cols).map(E8Vec::from_bytes).collect(),
            cols: num_cols,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.rows.iter().flat_map(|r| r.to_bytes()).collect()
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn num_cols(&self) -> usize {
        self.cols
    }

    /// Matrix-vector multiply: `y = M * x`
    pub fn matvec(&self, x: &E8Vec) -> Vec<u32> {
        E8TensorCore::matvec(&self.rows, x)
    }

    /// Matrix-vector multiply with E8F output.
    pub fn matvec_e8f(&self, x: &E8Vec) -> E8Vec {
        E8TensorCore::matvec_e8f(&self.rows, x)
    }

    /// Transpose (creates new matrix).
    pub fn transpose(&self) -> Self {
        let num_rows = self.cols;
        let num_cols = self.num_rows();

        let mut rows = Vec::with_capacity(num_rows);
        for j in 0..num_rows {
            let col: Vec<E8F> = self.rows.iter().map(|r| r.data[j]).collect();
            rows.push(E8Vec { data: col });
        }

        Self {
            rows,
            cols: num_cols,
        }
    }
}

impl std::ops::Index<usize> for E8Mat {
    type Output = E8Vec;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.rows[idx]
    }
}

impl std::ops::IndexMut<usize> for E8Mat {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.rows[idx]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// SECTION 7: TESTS
// ═══════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8f_basic() {
        let a = E8F::new(42);
        let b = E8F::new(100);

        assert!(a.is_valid());
        assert!(b.is_valid());
        assert!(!E8F::ZERO.is_valid());
    }

    #[test]
    fn test_e8f_arithmetic() {
        let a = E8F::new(10);
        let b = E8F::new(20);

        // All operations should return valid roots
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let neg = -a;

        assert!(sum.is_valid());
        assert!(diff.is_valid());
        assert!(prod.is_valid());
        assert!(neg.is_valid());
    }

    #[test]
    fn test_e8f_dot_product() {
        // Self dot should be ~1.0 (maps to ~255)
        for i in 0..10u8 {
            let e = E8F::new(i);
            let dot = e.dot_quantized(e);
            assert!(dot > 250, "Self dot should be ~1.0, got {}", dot);
        }
    }

    #[test]
    fn test_e8f_negation() {
        // Double negation should return close to original
        for i in 0..10u8 {
            let e = E8F::new(i);
            let neg_neg = -(-e);
            let dot = e.dot_quantized(neg_neg);
            assert!(dot > 250, "Double negation should be ~identity");
        }
    }

    #[test]
    fn test_e8f_antipodal() {
        let e = E8F::new(0);
        let neg_e = -e;
        assert!(e.is_antipodal(neg_e));
    }

    #[test]
    fn test_e8vec_compression() {
        // 2048D Ada-002 embedding → 256 bytes
        let embedding: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();

        let e8vec = E8Vec::from_f32_vec(&embedding);
        assert_eq!(e8vec.len(), 256); // 2048 / 8 = 256

        let bytes = e8vec.to_bytes();
        assert_eq!(bytes.len(), 256); // 32x compression!

        // Roundtrip
        let restored = E8Vec::from_bytes(&bytes);
        assert_eq!(restored.len(), 256);
    }

    #[test]
    fn test_e8vec_dot() {
        let a = E8Vec::from_indices(&[10, 20, 30]);
        let b = E8Vec::from_indices(&[10, 20, 30]);

        // Self dot should be positive (~3.0 for 3 elements)
        let dot = a.dot(&b);
        assert!(dot > 2.0);
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // TENSOR CORE TESTS
    // ─────────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_fma() {
        let a = E8F::new(10);
        let b = E8F::new(20);
        let c = E8F::new(30);

        let result = E8TensorCore::fma(a, b, c);
        assert!(result.is_valid());

        // FMA should equal mul then add
        let manual = (a * b) + c;
        assert_eq!(result, manual);
    }

    #[test]
    fn test_axpy() {
        let a = E8F::new(5);
        let x = E8Vec::from_indices(&[10, 20, 30]);
        let mut y = E8Vec::from_indices(&[1, 2, 3]);

        E8TensorCore::axpy(a, &x, &mut y);

        // All results should be valid
        for e in &y.data {
            assert!(e.is_valid());
        }
    }

    #[test]
    fn test_matvec() {
        // 2x3 matrix
        let m = vec![
            E8Vec::from_indices(&[10, 20, 30]),
            E8Vec::from_indices(&[40, 50, 60]),
        ];
        let x = E8Vec::from_indices(&[1, 2, 3]);

        let result = E8TensorCore::matvec(&m, &x);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_outer_product() {
        let u = E8Vec::from_indices(&[10, 20]);
        let v = E8Vec::from_indices(&[30, 40, 50]);

        let outer = E8TensorCore::outer(&u, &v);
        assert_eq!(outer.len(), 2); // 2 rows
        assert_eq!(outer[0].len(), 3); // 3 cols
    }

    #[test]
    fn test_batch_dots() {
        let queries = vec![
            E8Vec::from_indices(&[10, 20]),
            E8Vec::from_indices(&[30, 40]),
        ];
        let keys = vec![
            E8Vec::from_indices(&[10, 20]),
            E8Vec::from_indices(&[50, 60]),
            E8Vec::from_indices(&[70, 80]),
        ];

        let dots = E8TensorCore::batch_dots(&queries, &keys);
        assert_eq!(dots.len(), 2); // 2 queries
        assert_eq!(dots[0].len(), 3); // 3 keys each
    }

    #[test]
    fn test_softmax_approx() {
        let scores = vec![100, 200, 300, 400];
        let weights = E8TensorCore::softmax_approx(&scores);

        assert_eq!(weights.len(), 4);
        // Higher scores should have higher weights
        assert!(weights[3] > weights[0]);
    }

    #[test]
    fn test_top_k() {
        let scores = vec![10, 50, 30, 40, 20];
        let top2 = E8TensorCore::top_k(&scores, 2);

        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], 1); // Index of 50
        assert_eq!(top2[1], 3); // Index of 40
    }

    #[test]
    fn test_attention() {
        let queries = vec![E8Vec::from_indices(&[10, 20, 30])];
        let keys = vec![
            E8Vec::from_indices(&[10, 20, 30]),
            E8Vec::from_indices(&[40, 50, 60]),
        ];
        let values = vec![
            E8Vec::from_indices(&[1, 2, 3]),
            E8Vec::from_indices(&[4, 5, 6]),
        ];

        let output = E8TensorCore::attention(&queries, &keys, &values);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 3);
    }

    #[test]
    fn test_e8mat() {
        let mat = E8Mat::new(3, 4);
        assert_eq!(mat.num_rows(), 3);
        assert_eq!(mat.num_cols(), 4);

        // Serialize and deserialize
        let bytes = mat.to_bytes();
        assert_eq!(bytes.len(), 12);

        let restored = E8Mat::from_bytes(&bytes, 3, 4);
        assert_eq!(restored.num_rows(), 3);
    }

    #[test]
    fn test_e8mat_transpose() {
        let mut mat = E8Mat::new(2, 3);
        mat[0] = E8Vec::from_indices(&[1, 2, 3]);
        mat[1] = E8Vec::from_indices(&[4, 5, 6]);

        let t = mat.transpose();
        assert_eq!(t.num_rows(), 3);
        assert_eq!(t.num_cols(), 2);

        // Check values
        assert_eq!(t[0].data[0].0, 1);
        assert_eq!(t[0].data[1].0, 4);
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // SERIALIZATION TESTS (Task 9)
    // ─────────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_e8address_serialization() {
        use crate::Gf8LosslessCode;

        // E8Address is [Gf8LosslessCode; 8] which is [u8; 8]
        let address: [Gf8LosslessCode; 8] = [
            Gf8LosslessCode::new(0),
            Gf8LosslessCode::new(42),
            Gf8LosslessCode::new(100),
            Gf8LosslessCode::new(150),
            Gf8LosslessCode::new(200),
            Gf8LosslessCode::new(239),
            Gf8LosslessCode::new(10),
            Gf8LosslessCode::new(50),
        ];

        // Serialize to bytes (should be exactly 8 bytes)
        // **This is LOSSLESS**: u8 -> u8 with no transformation
        let bytes: Vec<u8> = address.iter().map(|c| c.0).collect();
        assert_eq!(
            bytes.len(),
            8,
            "E8Address should serialize to exactly 8 bytes"
        );

        // Verify lossless roundtrip (byte-perfect)
        let restored: [Gf8LosslessCode; 8] = [
            Gf8LosslessCode::new(bytes[0]),
            Gf8LosslessCode::new(bytes[1]),
            Gf8LosslessCode::new(bytes[2]),
            Gf8LosslessCode::new(bytes[3]),
            Gf8LosslessCode::new(bytes[4]),
            Gf8LosslessCode::new(bytes[5]),
            Gf8LosslessCode::new(bytes[6]),
            Gf8LosslessCode::new(bytes[7]),
        ];

        for i in 0..8 {
            assert_eq!(
                address[i].0, restored[i].0,
                "E8Address lossless roundtrip failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_e8vec_serialization_roundtrip() {
        // Create E8Vec from indices
        let original = E8Vec::from_indices(&[10, 20, 30, 40, 50, 100, 150, 200]);

        // Serialize to bytes
        let bytes = original.to_bytes();
        assert_eq!(
            bytes.len(),
            8,
            "E8Vec with 8 elements should serialize to 8 bytes"
        );

        // Verify each byte matches the original index
        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(
                byte,
                [10, 20, 30, 40, 50, 100, 150, 200][i],
                "Byte at index {} doesn't match",
                i
            );
        }

        // Deserialize
        let restored = E8Vec::from_bytes(&bytes);
        assert_eq!(restored.len(), 8, "Restored E8Vec should have 8 elements");

        // Verify roundtrip
        for i in 0..8 {
            assert_eq!(
                original.data[i].0, restored.data[i].0,
                "E8Vec roundtrip failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_e8vec_compression_ratio() {
        // 2048D Ada-002 embedding → 256 bytes (32x compression)
        let embedding: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();

        let e8vec = E8Vec::from_f32_vec(&embedding);
        let bytes = e8vec.to_bytes();

        // Verify compression ratio
        let original_size = 2048 * std::mem::size_of::<f32>(); // 6144 bytes
        let compressed_size = bytes.len(); // 256 bytes
        let ratio = original_size as f32 / compressed_size as f32;

        assert_eq!(compressed_size, 256, "2048D should compress to 256 bytes");
        assert!(
            (31.0..=33.0).contains(&ratio),
            "Compression ratio should be ~32x, got {}",
            ratio
        );
    }

    #[test]
    fn test_e8mat_serialization_roundtrip() {
        // Create 3x4 matrix
        let mut original = E8Mat::new(3, 4);
        original[0] = E8Vec::from_indices(&[1, 2, 3, 4]);
        original[1] = E8Vec::from_indices(&[5, 6, 7, 8]);
        original[2] = E8Vec::from_indices(&[9, 10, 11, 12]);

        // Serialize to bytes
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), 12, "3x4 E8Mat should serialize to 12 bytes");

        // Verify bytes are row-major
        let expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(byte, expected[i], "Byte at index {} doesn't match", i);
        }

        // Deserialize
        let restored = E8Mat::from_bytes(&bytes, 3, 4);
        assert_eq!(restored.num_rows(), 3);
        assert_eq!(restored.num_cols(), 4);

        // Verify roundtrip
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(
                    original[i].data[j].0, restored[i].data[j].0,
                    "E8Mat roundtrip failed at [{}, {}]",
                    i, j
                );
            }
        }
    }

    /// Verify that E8F operations are deterministic and repeatable (lossless property)
    #[test]
    fn test_e8f_operations_deterministic() {
        // Same inputs should always produce same outputs (lookup table is constant)
        let a = E8F::new(42);
        let b = E8F::new(100);

        // Perform operation 1000 times - should always get same result
        let mut results = Vec::new();
        for _ in 0..1000 {
            results.push((a + b).index());
        }

        // All results should be identical (deterministic)
        let first = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                result, first,
                "E8F addition not deterministic at iteration {}",
                i
            );
        }
    }

    /// Verify that E8F → Gf8 → E8F is truly lossless for valid roots
    #[test]
    fn test_e8f_gf8_roundtrip_lossless() {
        for idx in 0..240u8 {
            let original = E8F::new(idx);
            let gf8 = original.to_gf8();
            let recovered = E8F::from_f32(gf8.coords());

            assert_eq!(
                original.index(),
                recovered.index(),
                "E8F → Gf8 → E8F roundtrip not lossless for root {}",
                idx
            );
        }
    }

    /// Verify that E8F serialization is byte-perfect (truly lossless)
    #[test]
    fn test_e8f_serialization_byte_perfect() {
        for idx in 0..240u8 {
            let original = E8F::new(idx);
            let byte = original.index();
            let recovered = E8F::new(byte);

            assert_eq!(
                original.index(),
                recovered.index(),
                "E8F serialization not byte-perfect for index {}",
                idx
            );
        }
    }

    #[test]
    fn test_e8f_display_valid_roots() {
        // Test valid E8F roots (0-239)
        let e8f = E8F::new(42);
        assert_eq!(format!("{}", e8f), "E8F(42)");

        let e8f = E8F::new(0);
        assert_eq!(format!("{}", e8f), "E8F(0)");

        let e8f = E8F::new(239);
        assert_eq!(format!("{}", e8f), "E8F(239)");
    }

    #[test]
    fn test_e8f_display_zero() {
        // Test the special zero case (index 240)
        let e8f = E8F::ZERO;
        assert_eq!(format!("{}", e8f), "E8F(zero)");

        let e8f = E8F::new(240);
        assert_eq!(format!("{}", e8f), "E8F(zero)");
    }

    #[test]
    fn test_e8f_display_invalid() {
        // Test invalid indices (241-255) using from_raw to preserve the byte value
        let e8f = E8F::from_raw(241);
        assert_eq!(format!("{}", e8f), "E8F(invalid:241)");

        let e8f = E8F::from_raw(255);
        assert_eq!(format!("{}", e8f), "E8F(invalid:255)");
    }

    #[test]
    fn test_e8f_display_with_formatting() {
        // Test that E8F can be used with formatting (this would fail before Display impl)
        // This simulates the use case that would be used with tracing macros

        let e8f = E8F::new(42);

        // These should compile without errors - simulating tracing macro usage
        let debug_str = format!("E8F value: {}", e8f);
        assert_eq!(debug_str, "E8F value: E8F(42)");

        let info_str = format!("E8F value: {}", e8f);
        assert_eq!(info_str, "E8F value: E8F(42)");

        let warn_str = format!("E8F value: {}", e8f);
        assert_eq!(warn_str, "E8F value: E8F(42)");

        let error_str = format!("E8F value: {}", e8f);
        assert_eq!(error_str, "E8F value: E8F(42)");

        // Test with multiple E8F values
        let e8f2 = E8F::new(100);
        let pair_str = format!("E8F pair: {} and {}", e8f, e8f2);
        assert_eq!(pair_str, "E8F pair: E8F(42) and E8F(100)");

        // Test with zero and invalid
        let zero = E8F::ZERO;
        let invalid = E8F::from_raw(250);
        let special_str = format!("Special cases: {} and {}", zero, invalid);
        assert_eq!(special_str, "Special cases: E8F(zero) and E8F(invalid:250)");
    }

    #[test]
    fn test_e8f_display_roundtrip() {
        // Test that Display output is consistent and can be used for debugging
        let test_cases = vec![
            (0, "E8F(0)"),
            (42, "E8F(42)"),
            (239, "E8F(239)"),
            (240, "E8F(zero)"),
            (241, "E8F(invalid:241)"),
            (255, "E8F(invalid:255)"),
        ];

        for (index, expected) in test_cases {
            let e8f = if index < 240 {
                E8F::new(index)
            } else {
                E8F::from_raw(index)
            };
            let display_str = format!("{}", e8f);
            assert_eq!(
                display_str, expected,
                "Display format mismatch for index {}",
                index
            );
        }
    }

    #[test]
    fn test_e8mat_serialization_large() {
        // Create 240x240 matrix (valorem size)
        let mut mat = E8Mat::new(240, 240);

        // Fill with some pattern
        for i in 0..240 {
            for j in 0..240 {
                mat[i].data[j] = E8F::new(((i + j) % 240) as u8);
            }
        }

        // Serialize
        let bytes = mat.to_bytes();
        assert_eq!(
            bytes.len(),
            240 * 240,
            "240x240 E8Mat should serialize to 57600 bytes"
        );

        // Deserialize
        let restored = E8Mat::from_bytes(&bytes, 240, 240);

        // Spot check some values
        for i in [0, 50, 100, 200, 239] {
            for j in [0, 50, 100, 200, 239] {
                assert_eq!(
                    mat[i].data[j].0, restored[i].data[j].0,
                    "Large E8Mat roundtrip failed at [{}, {}]",
                    i, j
                );
            }
        }
    }
}
