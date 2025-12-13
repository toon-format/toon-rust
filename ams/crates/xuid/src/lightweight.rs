/* arcmoon-suite/crates/xuid/src/lightweight.rs */
//! **Lightweight XUID Utilities**
//!
//! Ultra-lean, stack-optimized semantic utilities for Tier-0 prefiltering.
//! Migrated from HoloSphere for integration with XUID.
//!
//! ### Key Capabilities
//! - **Minimal footprint:** `no_std` capable, zero external dependencies for core functions.
//! - **Fast prefilter:** Deterministic ring expansion, stack-only buffers, branch-light cosine.
//! - **Semantic richness:** 256-bit hash, orbit class, and 8D E8 embeddings for cheap re-rank.
//!
//! ### Integration with XUID
//! These utilities complement the main XUID implementation by providing:
//! - Fast cosine distance computation for E8 coordinates
//! - Ring expansion for hash-based neighborhood lookups
//! - Stack-friendly top-N keeping for prefiltering
//! - Optional bucket-based hash lookups
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios    ◦    Proprietary & Confidential    ◦    Author: Lord Xyn    ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::{collections::BTreeMap, vec::Vec};

/// 256-bit opaque semantic hash, stable and uniformly distributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash256(pub [u8; 32]);

/// Cosine distance (1 - cosine similarity) on 8D E8 embeddings.
///
/// Uses manual loop unrolling for instruction-level parallelism. Numerically stable
/// via sqrt and epsilon clamping. Returns a value in [0, 2] (lower = more similar).
///
/// # Example
/// ```
/// use xuid::lightweight::cosine8;
/// let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let b = [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let dist = cosine8(&a, &b);
/// assert!(dist < 0.2, "Nearby vectors should be close");
/// ```
#[inline]
pub fn cosine8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut ab = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;

    // Manual unroll for ILP: 8 iterations, each computing dot, sum-of-squares-a, sum-of-squares-b.
    ab += a[0] * b[0];
    aa += a[0] * a[0];
    bb += b[0] * b[0];
    ab += a[1] * b[1];
    aa += a[1] * a[1];
    bb += b[1] * b[1];
    ab += a[2] * b[2];
    aa += a[2] * a[2];
    bb += b[2] * b[2];
    ab += a[3] * b[3];
    aa += a[3] * a[3];
    bb += b[3] * b[3];
    ab += a[4] * b[4];
    aa += a[4] * a[4];
    bb += b[4] * b[4];
    ab += a[5] * b[5];
    aa += a[5] * a[5];
    bb += b[5] * b[5];
    ab += a[6] * b[6];
    aa += a[6] * a[6];
    bb += b[6] * b[6];
    ab += a[7] * b[7];
    aa += a[7] * a[7];
    bb += b[7] * b[7];

    let denom = (aa * bb).sqrt().max(1e-8);
    let cosine_sim = ab / denom;
    // Convert cosine similarity [-1, 1] to distance [0, 2]
    // cosine=1 (identical) -> dist=0
    // cosine=0 (orthogonal) -> dist=1
    // cosine=-1 (opposite) -> dist=2
    1.0 - cosine_sim
}

/// Cheap pre-ranking combining E8 cosine distance + orbit proximity penalty.
///
/// Lower scores indicate better matches for Tier-0 prefiltering. Orbit deltas are
/// penalized lightly (clamped at 5.0) to allow queries to bridge nearby orbits.
///
/// # Example
/// ```
/// use xuid::{Xuid, lightweight::cheap_rank_coords};
/// let q_coords = [0.1; 8];
/// let q_orbit = 42u8;
/// let c_coords = [0.1; 8];
/// let c_orbit = 41u8;
/// let score = cheap_rank_coords(&q_coords, q_orbit, &c_coords, c_orbit);
/// assert!(score <= 0.15, "Very similar vectors should score low");
/// ```
#[inline]
pub fn cheap_rank_coords(
    q_coords: &[f32; 8],
    q_orbit: u8,
    c_coords: &[f32; 8],
    c_orbit: u8,
) -> f32 {
    let cos = cosine8(q_coords, c_coords);
    let od = (q_orbit as i16 - c_orbit as i16).unsigned_abs() as f32;
    cos + 0.05 * od.min(5.0)
}

/// Flip a single bit in-place within a 256-bit hash using big-endian bit order per byte.
///
/// # Arguments
/// * `hash` – mutable 32-byte array.
/// * `bit_index` – 0..=255, where 0 = MSB of byte 0.
#[inline]
fn flip_bit_be(hash: &mut [u8; 32], bit_index: u16) {
    let byte = (bit_index / 8) as usize;
    let off = (bit_index % 8) as u8;
    let mask = 0x80u8 >> off;
    hash[byte] ^= mask;
}

/// Deterministic, bounded ring expansion over high-order bits of a 256-bit hash.
///
/// Strategy:
/// - Include the original hash first.
/// - Generate all Hamming-1 flips within `hi_window` bits (default 64).
/// - Add a subset of Hamming-2 pairs (stride-2 to keep count bounded).
/// - Returns the count of hashes written; does not exceed `dst.len()`.
///
/// Use this to pre-generate candidate neighborhoods for bucket-based lookups.
///
/// # Arguments
/// * `orig` – The original 256-bit hash to expand.
/// * `dst` – Mutable destination buffer.
/// * `hi_window` – Bit window to explore (typically 32..=128); clamped at 256.
///
/// # Returns
/// Number of hashes written to `dst` (at least 1, at most `dst.len()`).
///
/// # Example
/// ```
/// use xuid::lightweight::{Hash256, ring_expand};
/// let orig = Hash256([0u8; 32]);
/// let mut variants = [[0u8; 32]; 75];
/// let count = ring_expand(&orig, &mut variants, 64);
/// assert!(count >= 1 && count <= 75);
/// ```
#[inline]
pub fn ring_expand(orig: &Hash256, dst: &mut [[u8; 32]], hi_window: u8) -> usize {
    let hi = (hi_window as u16).min(256);
    let mut count = 0;

    // Include the original hash first.
    if count < dst.len() {
        dst[count] = orig.0;
        count += 1;
    }

    // Hamming-1: flip each bit in the high-order window.
    let base = orig.0;
    let mut i = 0u16;
    while i < hi && count < dst.len() {
        let mut h = base;
        flip_bit_be(&mut h, i);
        dst[count] = h;
        count += 1;
        i += 1;
    }

    // Hamming-2: deterministic stride to bound count. Pairs (a, a+1), (a+3, a+4), etc.
    if count < dst.len() {
        let step = 2u16;
        let mut a = 0u16;
        while a + 1 < hi && count < dst.len() {
            let b = a + 1;
            let mut h2 = base;
            flip_bit_be(&mut h2, a);
            flip_bit_be(&mut h2, b);
            dst[count] = h2;
            count += 1;
            a += step;
        }
    }

    count
}

/// Stack-friendly top-N keeper: inserts scores into a small fixed buffer, evicting worst on overflow.
///
/// Intended for Tier-0 prefiltering where keeping the best N candidates (on the stack)
/// avoids heap allocation. For typical N ≤ 16, this is faster than heap-based selection.
///
/// # Arguments
/// * `scores` – Mutable buffer of (id, score) tuples; must be sized to hold at least N items.
/// * `len` – Current fill count (0..N); updated in-place.
/// * `id` – Row identifier to insert.
/// * `score` – Score to insert (lower is better).
///
/// If `len < N`, the new item is appended and `len` is incremented.
/// If `len == N`, the item with the worst (highest) score is evicted and replaced if the new score is better.
///
/// # Example
/// ```
/// use xuid::lightweight::keep_top_n;
/// let mut scores = [(0u32, 0.0f32); 16];
/// let mut len = 0usize;
///
/// for (i, s) in [(1, 0.5), (2, 0.3), (3, 0.1), (4, 0.9)].iter() {
///     keep_top_n::<16>(&mut scores, &mut len, *i as u32, *s);
/// }
/// assert_eq!(len, 4);
/// ```
#[inline]
pub fn keep_top_n<const N: usize>(scores: &mut [(u32, f32)], len: &mut usize, id: u32, score: f32) {
    if *len < N {
        scores[*len] = (id, score);
        *len += 1;
    } else {
        let mut worst = 0usize;
        let mut worst_val = scores[0].1;
        let mut i = 1usize;
        while i < N {
            if scores[i].1 > worst_val {
                worst = i;
                worst_val = scores[i].1;
            }
            i += 1;
        }
        if score < worst_val {
            scores[worst] = (id, score);
        }
    }
}

/// Optional: Tiny associative map from Hash256 to row lists (requires `alloc` or `std` feature).
///
/// Provides O(log N) lookups for exact hash matches within a ring-expanded neighborhood.
/// Rows are stored as small vectors; dedup is done externally via `dedup_trunc`.
#[cfg(any(feature = "alloc", feature = "std"))]
pub struct XuidBuckets {
    map: BTreeMap<[u8; 32], Vec<u32>>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl XuidBuckets {
    /// Create a new empty bucket map.
    #[inline]
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Insert a row identifier under a given hash bucket.
    #[inline]
    pub fn insert(&mut self, hash: [u8; 32], row: u32) {
        self.map.entry(hash).or_default().push(row);
    }

    /// Probe all variant hashes, collecting row IDs into `out` up to capacity `cap`.
    #[inline]
    pub fn probe_variants(&self, variants: &[[u8; 32]], out: &mut Vec<u32>, cap: usize) {
        for v in variants {
            if let Some(rows) = self.map.get(v) {
                out.extend_from_slice(rows);
                if out.len() >= cap {
                    break;
                }
            }
        }
    }

    /// Deduplicate and truncate row list in-place (unsorted; stable sort not required).
    #[inline]
    pub fn dedup_trunc(&self, rows: &mut Vec<u32>, keep: usize) {
        rows.sort_unstable();
        rows.dedup();
        if rows.len() > keep {
            rows.truncate(keep);
        }
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl Default for XuidBuckets {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine8_identical() {
        let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = cosine8(&v, &v);
        assert!(
            (dist - 0.0).abs() < 1e-6,
            "Identical vectors should have zero distance"
        );
    }

    #[test]
    fn test_cosine8_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = cosine8(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "Orthogonal vectors should have distance ~1.0"
        );
    }

    #[test]
    fn test_cosine8_opposite() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = cosine8(&a, &b);
        assert!(
            (dist - 2.0).abs() < 1e-5,
            "Opposite vectors should have distance ~2.0"
        );
    }

    #[test]
    fn test_cheap_rank_coords() {
        let q_coords = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let c_coords = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let score = cheap_rank_coords(&q_coords, 42, &c_coords, 42);
        assert!(
            (0.0..=2.0).contains(&score),
            "Score must be in [0, 2] range, got {}",
            score
        );
    }

    #[test]
    fn test_cheap_rank_orbit_penalty() {
        let coords = [0.1; 8];
        let score_same = cheap_rank_coords(&coords, 42, &coords, 42);
        let score_diff = cheap_rank_coords(&coords, 42, &coords, 50);

        assert!(
            score_diff > score_same,
            "Different orbit should have higher (worse) score"
        );
    }

    #[test]
    fn test_flip_bit_be() {
        let mut hash = [0u8; 32];
        flip_bit_be(&mut hash, 0); // flip MSB of byte 0
        assert_eq!(hash[0], 0x80, "MSB flip of byte 0 should yield 0x80");

        let mut hash = [0u8; 32];
        flip_bit_be(&mut hash, 7); // flip LSB of byte 0
        assert_eq!(hash[0], 0x01, "LSB flip of byte 0 should yield 0x01");

        let mut hash = [0u8; 32];
        flip_bit_be(&mut hash, 8); // flip MSB of byte 1
        assert_eq!(hash[1], 0x80, "MSB flip of byte 1 should yield 0x80");
    }

    #[test]
    fn test_ring_expand_includes_original() {
        let orig = Hash256([42u8; 32]);
        let mut variants = [[0u8; 32]; 100];
        let count = ring_expand(&orig, &mut variants, 64);

        assert!(count >= 1, "Must include at least original");
        assert_eq!(variants[0], orig.0, "First variant must be original");
    }

    #[test]
    fn test_ring_expand_hamming1() {
        let orig = Hash256([0u8; 32]);
        let mut variants = [[0u8; 32]; 100];
        let count = ring_expand(&orig, &mut variants, 16);

        // Original + all Hamming-1 flips in window + some Hamming-2
        assert!(
            count >= 17,
            "Must include original + at least 16 Hamming-1 variants"
        );
        assert!(count <= 100, "Must not exceed buffer");
    }

    #[test]
    fn test_ring_expand_respects_buffer_bounds() {
        let orig = Hash256([5u8; 32]);
        let mut variants = [[0u8; 32]; 10];
        let count = ring_expand(&orig, &mut variants, 255);

        assert_eq!(count, 10, "Must not exceed buffer size");
        assert!(count <= variants.len(), "Count must respect buffer bounds");
    }

    #[test]
    fn test_keep_top_n_basic() {
        let mut scores = [(0u32, 0.0f32); 4];
        let mut len = 0usize;

        keep_top_n::<4>(&mut scores, &mut len, 1, 0.5);
        assert_eq!(len, 1);

        keep_top_n::<4>(&mut scores, &mut len, 2, 0.3);
        assert_eq!(len, 2);

        keep_top_n::<4>(&mut scores, &mut len, 3, 0.8);
        assert_eq!(len, 3);

        keep_top_n::<4>(&mut scores, &mut len, 4, 0.1);
        assert_eq!(len, 4);
    }

    #[test]
    fn test_keep_top_n_overflow_replaces_worst() {
        let mut scores = [(0u32, 0.0f32); 3];
        let mut len = 0usize;

        keep_top_n::<3>(&mut scores, &mut len, 1, 0.5);
        keep_top_n::<3>(&mut scores, &mut len, 2, 0.8);
        keep_top_n::<3>(&mut scores, &mut len, 3, 0.3);
        assert_eq!(len, 3);

        // All scores: 0.5, 0.8, 0.3. Worst is 0.8 at index 1.
        keep_top_n::<3>(&mut scores, &mut len, 4, 0.6);
        assert_eq!(len, 3);

        // Index 1 should have been replaced with (4, 0.6).
        let found = scores[..len]
            .iter()
            .any(|&(id, s)| id == 4 && (s - 0.6).abs() < 1e-6);
        assert!(found, "New better score should have evicted worst");
    }

    #[test]
    fn test_keep_top_n_ignores_worse_score() {
        let mut scores = [(0u32, 0.0f32); 2];
        let mut len = 0usize;

        keep_top_n::<2>(&mut scores, &mut len, 1, 0.3);
        keep_top_n::<2>(&mut scores, &mut len, 2, 0.5);
        assert_eq!(len, 2);

        // Buffer full; new score 0.9 is worse than max (0.5), so it is ignored.
        keep_top_n::<2>(&mut scores, &mut len, 3, 0.9);
        assert_eq!(len, 2);

        let found = scores[..len].iter().any(|&(id, _)| id == 3);
        assert!(!found, "Worse score should be rejected");
    }

    #[cfg(any(feature = "alloc", feature = "std"))]
    #[test]
    fn test_xuid_buckets_insert_and_probe() {
        let mut buckets = XuidBuckets::new();
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];

        buckets.insert(hash1, 100);
        buckets.insert(hash1, 101);
        buckets.insert(hash2, 200);

        let mut out = vec![];
        buckets.probe_variants(&[hash1], &mut out, 10);
        assert_eq!(out.len(), 2);
        assert!(out.contains(&100));
        assert!(out.contains(&101));
    }

    #[cfg(any(feature = "alloc", feature = "std"))]
    #[test]
    fn test_xuid_buckets_dedup_trunc() {
        let buckets = XuidBuckets::new();
        let mut rows = vec![1, 2, 2, 3, 3, 3, 4, 5, 5];

        buckets.dedup_trunc(&mut rows, 3);

        assert_eq!(rows.len(), 3);
        // After sort + dedup, should be [1, 2, 3], then truncated to first 3.
        assert_eq!(rows, vec![1, 2, 3]);
    }
}
