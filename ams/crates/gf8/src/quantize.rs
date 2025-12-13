/* e8/gf8/src/quantize.rs */
//! Functions for quantizing float vectors into the discrete E8 lattice manifold.
//!
//! # E8 Primitives – Gf8 Quantization Module
//!▫~•◦------------------------------------------‣
//!
//! This module implements the canonical mapping of arbitrary 8D vectors onto the
//! true 240-root E8 lattice. Unlike previous approximations which only used the
//! 128 even-parity roots, this module generates the full Gosset 4_21 polytope vertices:
//!
//! 1. **D8 Subset (112 roots):** Permutations of `(±1, ±1, 0^6)`.
//! 2. **Spinor Subset (128 roots):** `(±0.5)^8` with even number of minus signs.
//!
//! ### Key Capabilities
//! - **Full E8 Codebook:** Lazily generates the 240 canonical unit-normalized roots.
//! - **Discrete Quantization:** Snaps arbitrary vectors to the nearest of the 240 roots.
//! - **Code Mapping:** Provides bidirectional mapping between `Gf8BitSig ` (u8) and geometric roots.
//!
//! ### Architectural Notes
//! This defines the "Static Cartography" of the system. All semantic meaning in the
//! higher-level E8DB is keyed off the indices generated here. The search space is
//! fixed, finite, and maximally symmetric.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{Gf8, Gf8BitSig, Gf8Tensor};
use std::sync::OnceLock;

/// A struct holding the pre-computed 240-root codebook and adjacency graph.
///
/// This ensures the geometry is calculated exactly once and remains immutable.
pub struct E8Codebook {
    /// The 240 canonical `Gf8` direction vectors (normalized).
    /// Index matches `Gf8BitSig `.
    pub roots: [Gf8; 240],
    /// Precomputed neighbors for each root.
    /// Each entry contains the indices (0..239) of the nearest roots.
    /// E8 roots typically have 56 nearest neighbors (kissing number).
    pub adjacency: [[u8; 56]; 240],
}

/// A static, lazily-initialized instance of the full E8 codebook.
pub static E8_CODEBOOK: OnceLock<E8Codebook> = OnceLock::new();

/// Generates the canonical E8 roots and their adjacency graph.
///
/// The E8 root system consists of:
/// - 112 roots from the D8 system: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0).
/// - 128 roots from the Spinor system: (±0.5, ±0.5, ..., ±0.5) with even number of minus signs.
///
/// All roots have squared length 2 in the standard lattice definition.
/// We normalize them to unit length for `Gf8` representation.
fn generate_e8_roots() -> E8Codebook {
    let mut roots = Vec::with_capacity(240);
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt(); // Normalization factor since lattice norm is sqrt(2)

    // 1. Generate D8 Roots (112 roots)
    // Permutations of two non-zero entries with values ±1.
    for i in 0..8 {
        for j in (i + 1)..8 {
            for &s1 in &[1.0, -1.0] {
                for &s2 in &[1.0, -1.0] {
                    let mut v = [0.0f32; 8];
                    v[i] = s1 * inv_sqrt_2;
                    v[j] = s2 * inv_sqrt_2;
                    roots.push(Gf8::from_coords(v));
                }
            }
        }
    }

    // 2. Generate Spinor Roots (128 roots)
    // (±0.5)^8 with even number of minus signs.
    // Since we normalize, ±0.5 becomes ±0.5 * (1/sqrt(2)).
    // Actually, vector (±0.5...)*8 has sq_len = 8 * 0.25 = 2.
    // So normalization is again * inv_sqrt_2.
    // Effective coord is ±0.5 * 0.7071...
    for i in 0..256u16 {
        // Only even parity of bits (even number of 1s, where 1 represents a minus sign)
        if i.count_ones() % 2 == 0 {
            let mut v = [0.0f32; 8];
            for (bit, val) in v.iter_mut().enumerate() {
                let is_neg = (i >> bit) & 1 == 1;
                // Unnormalized: 0.5. Normalized: 0.5 / sqrt(2) = 0.35355...
                *val = if is_neg { -0.5 } else { 0.5 };
            }
            // Manual normalization to ensure precision consistency
            // The geometric constructor handles re-normalization, but we pass raw coords.
            // Gf8::from_coords will normalize it.
            roots.push(Gf8::from_coords(v));
        }
    }

    assert_eq!(
        roots.len(),
        240,
        "E8 generation failed to produce exactly 240 roots"
    );

    // Convert vector to fixed array
    let mut roots_array = [Gf8::ZERO; 240];
    for (i, root) in roots.into_iter().enumerate() {
        roots_array[i] = root;
    }

    // 3. Generate Adjacency Graph (Nearest Neighbors)
    // For each root, find the 56 closest other roots.
    // In E8, neighbors have dot product 0.5 (angle 60 degrees).
    let mut adjacency = [[0u8; 56]; 240];

    for i in 0..240 {
        let mut neighbors: Vec<(usize, f32)> = (0..240)
            .filter(|&j| i != j)
            .map(|j| {
                // Use dot product as similarity metric
                (
                    j,
                    roots_array[i].dot(roots_array[j].as_slice().try_into().unwrap()),
                )
            })
            .collect();

        // Sort by similarity descending (highest dot product = closest)
        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top 56
        for (k, &(j, _)) in neighbors.iter().enumerate().take(56) {
            adjacency[i][k] = j as u8;
        }
    }

    E8Codebook {
        roots: roots_array,
        adjacency,
    }
}

/// Accessor for the singleton codebook.
pub fn get_e8_codebook() -> &'static E8Codebook {
    E8_CODEBOOK.get_or_init(generate_e8_roots)
}

/// Accessor for a root's neighbors.
/// Returns a slice of root indices.
pub fn get_root_neighbors(root_idx: u8) -> &'static [u8] {
    let cb = get_e8_codebook();
    if (root_idx as usize) < 240 {
        &cb.adjacency[root_idx as usize]
    } else {
        &[]
    }
}

/// Normalizes an arbitrary 8D float vector into a `Gf8` on the continuous unit sphere.
#[inline]
pub fn normalize_to_gf8(v: &[f32; 8]) -> Gf8 {
    Gf8::from_coords(*v)
}

/// Quantizes an arbitrary 8D float vector to the nearest canonical E8 root.
///
/// Returns the `Gf8BitSig ` (index 0..239) and the canonical `Gf8` vector.
///
/// # Complexity
/// Currently performs a linear scan (240 dot products). For 240 items, this is
/// extremely fast due to cache locality and SIMD autovectorization, often faster
/// than hierarchical lookups for this specific size.
pub fn quantize_to_nearest_code(v: &[f32; 8]) -> (Gf8BitSig, Gf8) {
    let codebook = get_e8_codebook();

    // Normalize input first to ensure valid cosine similarity comparison
    // (though dot product works for ranking if we assume roots are unit length)
    let target = Gf8::from_coords(*v);

    let mut best_idx = 0;
    let mut max_dot = f32::NEG_INFINITY;

    // Iterate through all 240 roots
    for (i, root) in codebook.roots.iter().enumerate() {
        let dot = target.dot(root.as_slice().try_into().unwrap());
        if dot > max_dot {
            max_dot = dot;
            best_idx = i;
        }
    }

    // Safety: best_idx is guaranteed to be < 240 by the loop
    (Gf8BitSig(best_idx as u8), codebook.roots[best_idx])
}

/// Convenience wrapper to return just the quantized vector.
#[inline]
pub fn quantize_to_gf8(v: &[f32; 8]) -> Gf8 {
    let (_, gf) = quantize_to_nearest_code(v);
    gf
}

/// Dequantizes a `Gf8` back into a standard 8D float vector.
#[inline]
pub fn dequantize_to_vec(gf: &Gf8) -> [f32; 8] {
    *gf.coords()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_generation() {
        let cb = get_e8_codebook();
        assert_eq!(cb.roots.len(), 240);

        // Verify all roots are unit length
        for root in &cb.roots {
            assert!((root.norm2() - 1.0).abs() < 1e-5);
        }

        // Verify a known D8 root exists (e.g., (1, 1, 0...) normalized)
        // (1,1,0...) -> norm sqrt(2). Normalized: (0.707, 0.707, 0...)
        let d8_target = std::f32::consts::FRAC_1_SQRT_2;
        let d8_exists = cb.roots.iter().any(|r| {
            r.coords()[0].abs() > d8_target - 0.01 && r.coords()[1].abs() > d8_target - 0.01
        });
        assert!(d8_exists, "D8 subset roots should be present");

        // Verify a known Spinor root exists (0.5 normalized -> 0.3535)
        let spinor_target = std::f32::consts::FRAC_1_SQRT_2 / 2.0;
        let spinor_exists = cb.roots.iter().any(|r| {
            r.coords()
                .iter()
                .all(|&c| (c.abs() - spinor_target).abs() < 0.01)
        });
        assert!(spinor_exists, "Spinor subset roots should be present");
    }

    #[test]
    fn test_quantization_fidelity() {
        let cb = get_e8_codebook();
        // Take a root, perturb it slightly, ensure it snaps back
        let root = cb.roots[42];
        let mut perturbed = *root.coords();
        perturbed[0] += 0.1; // Small nudge

        let (code, snapped) = quantize_to_nearest_code(&perturbed);

        // Should snap back to index 42
        assert_eq!(code.0, 42);
        assert!((snapped.dot(root.coords()) - 1.0).abs() < 1e-5);
    }

    // ============================================================================
    // Task 2.1: Audit E8 Codebook for Correctness
    // ============================================================================

    #[test]
    fn test_codebook_has_exactly_240_roots() {
        let cb = get_e8_codebook();
        assert_eq!(
            cb.roots.len(),
            240,
            "E8 codebook must have exactly 240 roots (112 D8 + 128 Spinor)"
        );
    }

    #[test]
    fn test_all_roots_unit_normalized() {
        let cb = get_e8_codebook();
        for (i, root) in cb.roots.iter().enumerate() {
            let norm2 = root.norm2();
            assert!(
                (norm2 - 1.0).abs() < 1e-5,
                "Root {} has norm² = {}, expected 1.0 ± 1e-5",
                i,
                norm2
            );
        }
    }

    #[test]
    fn test_adjacency_has_56_neighbors_per_root() {
        let cb = get_e8_codebook();
        for (i, neighbors) in cb.adjacency.iter().enumerate() {
            // Each root should have exactly 56 neighbors
            assert_eq!(
                neighbors.len(),
                56,
                "Root {} has {} neighbors, expected 56",
                i,
                neighbors.len()
            );

            // All neighbor indices should be valid (0..239)
            for &neighbor_idx in neighbors {
                assert!(
                    (neighbor_idx as usize) < 240,
                    "Root {} has invalid neighbor index {}",
                    i,
                    neighbor_idx
                );
                // Neighbor should not be self
                assert_ne!(
                    neighbor_idx as usize, i,
                    "Root {} lists itself as a neighbor",
                    i
                );
            }
        }
    }

    // ============================================================================
    // Task 2.2: Codebook Validation Tests
    // ============================================================================

    #[test]
    fn test_d8_roots_have_two_nonzero_coords() {
        let cb = get_e8_codebook();
        let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
        let d8_target = inv_sqrt_2;
        let tolerance = 1e-5;

        // D8 roots are the first 112 roots (indices 0..112)
        // They should have exactly 2 non-zero coordinates with values ±1/√2
        for i in 0..112 {
            let root = cb.roots[i];
            let coords = root.coords();

            // Count non-zero coordinates
            let nonzero_count = coords.iter().filter(|&&c| c.abs() > tolerance).count();

            assert_eq!(
                nonzero_count, 2,
                "D8 root {} has {} non-zero coords, expected 2",
                i, nonzero_count
            );

            // Verify non-zero coords have magnitude ±1/√2
            for &coord in coords {
                if coord.abs() > tolerance {
                    assert!(
                        (coord.abs() - d8_target).abs() < tolerance,
                        "D8 root {} has coord {} with magnitude {}, expected ±{}",
                        i,
                        coord,
                        coord.abs(),
                        d8_target
                    );
                }
            }
        }
    }

    #[test]
    fn test_spinor_roots_have_all_coords_with_even_parity() {
        let cb = get_e8_codebook();
        let spinor_target = 0.5 / 2.0f32.sqrt(); // ±0.5/√2
        let tolerance = 1e-5;

        // Spinor roots are indices 112..240
        for i in 112..240 {
            let root = cb.roots[i];
            let coords = root.coords();

            // All coordinates should be non-zero and have magnitude ±0.5/√2
            let mut neg_count = 0;
            for &coord in coords {
                assert!(
                    coord.abs() > tolerance,
                    "Spinor root {} has zero coordinate, all should be ±0.5/√2",
                    i
                );
                assert!(
                    (coord.abs() - spinor_target).abs() < tolerance,
                    "Spinor root {} has coord {} with magnitude {}, expected ±{}",
                    i,
                    coord,
                    coord.abs(),
                    spinor_target
                );
                if coord < 0.0 {
                    neg_count += 1;
                }
            }

            // Even parity: even number of minus signs
            assert_eq!(
                neg_count % 2,
                0,
                "Spinor root {} has {} negative coords (odd), expected even parity",
                i,
                neg_count
            );
        }
    }

    #[test]
    fn test_neighbor_symmetry() {
        let cb = get_e8_codebook();

        // For each root, verify that if A neighbors B, then B neighbors A
        for i in 0..240 {
            for &neighbor_idx in &cb.adjacency[i] {
                let neighbor_idx = neighbor_idx as usize;

                // Check if i is in the neighbor list of neighbor_idx
                let is_reciprocal = cb.adjacency[neighbor_idx].iter().any(|&n| n as usize == i);

                assert!(
                    is_reciprocal,
                    "Neighbor symmetry broken: {} neighbors {}, but {} does not neighbor {}",
                    i, neighbor_idx, neighbor_idx, i
                );
            }
        }
    }

    #[test]
    fn test_neighbor_dot_products_are_consistent() {
        let cb = get_e8_codebook();

        // In E8, neighbors should have consistent dot products (approximately 0.5 for 60° angle)
        // This verifies the adjacency graph is geometrically meaningful
        let mut dot_products = Vec::new();

        for i in 0..240 {
            for &neighbor_idx in &cb.adjacency[i] {
                let neighbor_idx = neighbor_idx as usize;
                let dot = cb.roots[i].dot(cb.roots[neighbor_idx].coords());
                dot_products.push(dot);
            }
        }

        // All dot products should be positive (neighbors are in same hemisphere)
        for &dot in &dot_products {
            assert!(dot > 0.0, "Neighbor dot product {} is not positive", dot);
        }

        // Compute mean and verify consistency
        let mean_dot: f32 = dot_products.iter().sum::<f32>() / dot_products.len() as f32;
        let variance: f32 = dot_products
            .iter()
            .map(|&d| (d - mean_dot).powi(2))
            .sum::<f32>()
            / dot_products.len() as f32;
        let std_dev = variance.sqrt();

        // Neighbors should have relatively consistent dot products
        // (low variance indicates a well-formed adjacency graph)
        assert!(
            std_dev < 0.2,
            "Neighbor dot products have high variance (std_dev = {}), adjacency may be malformed",
            std_dev
        );
    }
}
