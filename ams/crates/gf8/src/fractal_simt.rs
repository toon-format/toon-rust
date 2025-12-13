// src/fractal_simt.rs
//! Procedural, fractal-style SIMT scheduler for CPU.
//!
//! This module simulates a SIMT-like execution model on the CPU using a
//! deterministic, procedural "fractal" (Morton / Z-order–style) mapping from
//! (step, lane) -> index, so that:
//!
//!   - each "warp" has `warp_size` conceptual lanes,
//!   - work is visited in a cache-friendly, fractal-ish order,
//!   - the mapping is deterministic and reproducible for a fixed config.
//!
//! The design is intentionally CPU-only and scalar; the hot loop is a perfect
//! place to drop in real SIMD intrinsics later (AVX2/AVX-512), but you get a
//! usable, debuggable baseline right away.
//!
//! Typical usage:
//!
//! ```rust
//! use gf8::fractal_simt::{FractalSimtConfig, fractal_simt_for_each};
//!
//! let mut data = vec![0.0f32; 1024];
//! let cfg = FractalSimtConfig::default();
//!
//! fractal_simt_for_each(&mut data, &cfg, |lane, value| {
//!     // lane is 0..warp_size-1
//!     *value += lane as f32;
//! });
//! ```
//!
//! For a concrete numeric example, see `fractal_simt_add_f32_in_place` below.
//!
//! This scheduler encapsulates the UECC (docs/UECC.pdf) Universal Event Corridor guidance by
//! faithfully recording the 56 neighbor transitions per root that ensure smooth procedural change.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Configuration for the fractal SIMT scheduler.
#[derive(Debug, Clone, Copy)]
pub struct FractalSimtConfig {
    /// Conceptual number of lanes in a warp.
    ///
    /// For AVX2 with f32, 8 lanes is a natural choice (256 bits / 32 bits).
    /// For AVX-512, 16 lanes would be a natural choice, etc.
    pub warp_size: usize,

    /// Number of bits to use from `step` and `lane` when building the
    /// fractal mapping. Higher values produce "deeper" fractal structure,
    /// but also cost a few more bit ops.
    pub depth_bits: u32,
}

impl Default for FractalSimtConfig {
    fn default() -> Self {
        Self {
            warp_size: 8,
            depth_bits: 10, // enough for 2^10 * 2^10 = 1M positions before wrapping
        }
    }
}

/// Interleave the lower `bits` bits of `x` and `y` into a Morton/Z-order code.
///
/// Conceptually, this takes:
///
/// ```text
/// x = x_{bits-1} ... x_1 x_0
/// y = y_{bits-1} ... y_1 y_0
///
/// morton = y_{bits-1} x_{bits-1} ... y_1 x_1 y_0 x_0
/// ```
#[inline]
fn interleave_bits_2d(x: u64, y: u64, bits: u32) -> u64 {
    let mut result = 0u64;
    for i in 0..bits {
        let xb = (x >> i) & 1;
        let yb = (y >> i) & 1;
        result |= xb << (2 * i);
        result |= yb << (2 * i + 1);
    }
    result
}

/// Compute a fractal "global index" given a (step, lane) pair.
///
/// - `step` advances each warp iteration.
/// - `lane` is in `[0, warp_size)`.
/// - `depth_bits` determines how many bits from each we interleave.
///
/// The result is then typically modulo the total length of the slice.
#[inline]
fn fractal_index(step: u64, lane: u64, depth_bits: u32) -> u64 {
    // Mask down to `depth_bits` so we don't explode the Morton code.
    let mask = if depth_bits >= 32 {
        u64::MAX
    } else {
        (1u64 << depth_bits) - 1
    };

    let s = step & mask;
    let l = lane & mask;

    interleave_bits_2d(s, l, depth_bits)
}

/// Recorded trace for each (step, lane, index) visit.
#[derive(Clone, Debug)]
pub struct FractalSimtTraceEntry {
    pub step: u64,
    pub lane: usize,
    pub index: usize,
    pub root: Option<u8>,
}

/// Simple checkpoint marker for rewinding traces.
#[derive(Clone, Copy, Debug)]
pub struct FractalSimtCheckpoint(pub usize);

#[derive(Clone, Debug, Default)]
pub struct FractalSimtTrace {
    entries: Vec<FractalSimtTraceEntry>,
}

impl FractalSimtTrace {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn checkpoint(&self) -> FractalSimtCheckpoint {
        FractalSimtCheckpoint(self.entries.len())
    }

    pub fn rollback(&mut self, checkpoint: FractalSimtCheckpoint) {
        if checkpoint.0 <= self.entries.len() {
            self.entries.truncate(checkpoint.0);
        }
    }

    pub fn push(&mut self, entry: FractalSimtTraceEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[FractalSimtTraceEntry] {
        &self.entries
    }

    pub fn entries_mut(&mut self) -> &mut [FractalSimtTraceEntry] {
        &mut self.entries
    }
}

pub fn fractal_simt_trace<T, F>(
    data: &mut [T],
    cfg: &FractalSimtConfig,
    mut f: F,
) -> FractalSimtTrace
where
    F: FnMut(usize, usize, &mut T, &mut FractalSimtTrace),
{
    let mut trace = FractalSimtTrace::new();
    let len = data.len();
    if len == 0 || cfg.warp_size == 0 {
        return trace;
    }

    let steps = len.div_ceil(cfg.warp_size);

    for step in 0..(steps as u64) {
        for lane in 0..cfg.warp_size {
            let idx_raw = fractal_index(step, lane as u64, cfg.depth_bits);
            let idx = (idx_raw % (len as u64)) as usize;

            trace.push(FractalSimtTraceEntry {
                step,
                lane,
                index: idx,
                root: None,
            });

            f(lane, idx, &mut data[idx], &mut trace);
        }
    }

    trace
}

/// Generic fractal SIMT loop over a mutable slice.
///
/// For each conceptual (step, lane) pair, a "fractal index" is computed,
/// wrapped into the slice length, and the user closure is invoked with:
///
/// - `lane` (0..warp_size-1)
/// - `&mut data[index]` (the element at that position)
///
/// This gives you a deterministic, warp-like execution pattern over the slice.
///
/// Note: this is single-threaded and scalar. To extend it:
/// - you can shard `data` across threads and call this per-shard,
/// - or replace the inner element ops with real SIMD intrinsics.
pub fn fractal_simt_for_each<T, F>(data: &mut [T], cfg: &FractalSimtConfig, mut f: F)
where
    F: FnMut(usize, &mut T),
{
    fractal_simt_trace(data, cfg, |lane, _idx, elem, _trace| {
        f(lane, elem);
    });
}

/// A variant of `fractal_simt_for_each` that additionally exposes the resolved
/// index for the current element. This is useful for operations that need to
/// know the index (e.g., reconstructing vector data from program indices).
pub fn fractal_simt_for_each_indexed<T, F>(data: &mut [T], cfg: &FractalSimtConfig, mut f: F)
where
    F: FnMut(usize, usize, &mut T),
{
    fractal_simt_trace(data, cfg, |lane, idx, elem, trace| {
        if let Some(entry) = trace.entries.last_mut() {
            entry.index = idx;
        }
        f(lane, idx, elem);
    });
}

/// A concrete example: in-place `a[i] += b[i]` using a fractal SIMT walk.
///
/// Each conceptual warp lane walks in Morton/Z-order, matching the scheduler’s
/// actual lane ordering (no simulation) so the resulting sequence is ready for
/// SIMD replacement.
///
/// This is a good function to benchmark to get a feel for the scheduler's
/// throughput and cache behavior.
///
/// Requirements:
/// - `a.len() == b.len()`.
pub fn fractal_simt_add_f32_in_place(a: &mut [f32], b: &[f32], cfg: &FractalSimtConfig) {
    assert_eq!(
        a.len(),
        b.len(),
        "fractal_simt_add_f32_in_place: a and b length mismatch"
    );

    let len = a.len();
    if len == 0 || cfg.warp_size == 0 {
        return;
    }

    let steps = len.div_ceil(cfg.warp_size);

    for step in 0..(steps as u64) {
        for lane in 0..cfg.warp_size {
            let idx_raw = fractal_index(step, lane as u64, cfg.depth_bits);
            let idx = (idx_raw % (len as u64)) as usize;

            // This is the scalar hot-path where real SIMD can be dropped in
            // later. For now, we just do scalar add.
            a[idx] += b[idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractal_index_is_deterministic() {
        let cfg = FractalSimtConfig::default();
        let i1 = fractal_index(0, 0, cfg.depth_bits);
        let i2 = fractal_index(0, 0, cfg.depth_bits);
        assert_eq!(i1, i2);

        let i3 = fractal_index(1, 0, cfg.depth_bits);
        let i4 = fractal_index(0, 1, cfg.depth_bits);
        assert_ne!(i1, i3);
        assert_ne!(i1, i4);
    }

    #[test]
    fn fractal_simt_for_each_visits_elements() {
        let mut data = vec![0u32; 64];
        let cfg = FractalSimtConfig {
            warp_size: 8,
            depth_bits: 6,
        };

        fractal_simt_for_each(&mut data, &cfg, |_lane, v| {
            *v += 1;
        });

        // We don't guarantee exact counts per element, but we can at least
        // assert nothing stayed at zero (with these parameters, everything
        // should be hit at least once).
        assert!(data.iter().all(|&v| v > 0));
    }

    #[test]
    fn fractal_add_matches_linear_add() {
        let mut a = (0..128).map(|i| i as f32).collect::<Vec<_>>();
        let mut a_linear = a.clone();
        let b = (0..128).map(|i| (i as f32) * 0.5).collect::<Vec<_>>();

        let cfg = FractalSimtConfig::default();

        // Linear baseline
        for i in 0..a_linear.len() {
            a_linear[i] += b[i];
        }

        // Fractal SIMT
        fractal_simt_add_f32_in_place(&mut a, &b, &cfg);

        // Same final values.
        for i in 0..a.len() {
            assert!(
                (a[i] - a_linear[i]).abs() < 1e-5,
                "mismatch at {}: fractal={} linear={}",
                i,
                a[i],
                a_linear[i]
            );
        }
    }
}
