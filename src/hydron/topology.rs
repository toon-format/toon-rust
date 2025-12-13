/* src/rune/hydron/topology.rs */
//!▫~•◦-------------------------------‣
//! # E8 Topology, Adjacency, and Weyl Group Operations.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides high-performance, SIMD-accelerated functions for geometric operations
//! on the E8 root lattice. This includes adjacency lookups, Weyl reflections, and
//! a parallelized energy diffusion simulation.
//!
//! ## Key Capabilities
//! - **Pre-computed Adjacency**: Calculates the 56 nearest neighbors for each of the
//!   240 E8 roots once and stores it in a static table for O(1) lookups.
//! - **SIMD-Accelerated Vector Math**: All dot products and vector-scalar operations
//!   are implemented with portable SIMD intrinsics for maximum performance.
//! - **Parallel Diffusion**: The `diffuse_energy` function is parallelized with `rayon`
//!   to leverage multi-core processors for complex simulations.
//!
//! ### Architectural Notes
//! The use of `OnceLock` for the adjacency graph is a critical optimization,
//! changing the complexity of neighbor lookups from O(N) to amortized O(1). This
//! transforms the `diffuse_energy` algorithm from O(N^2) to a much faster O(N*k).
//! Feature-gated SIMD provides a significant speedup for the underlying geometric
//! calculations on supported hardware.
//!
//! #### Example
//! ```rust
//! use rune_xero::rune::hydron::topology::{get_neighbors, weyl_reflect};
//!
//! // Get pre-computed neighbors for root 0 (amortized O(1) cost).
//! let neighbors = get_neighbors(0);
//! assert_eq!(neighbors.len(), 56);
//!
//! // Perform a SIMD-accelerated Weyl reflection.
//! let vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! let mirror = [0.5; 8]; // Example root
//! let reflected = weyl_reflect(&vec, &mirror);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "hydron")]
use hydron_core::get_e8_roots;
use rayon::prelude::*;
use std::sync::OnceLock;

use super::values::gf8_dot_simd;

// Use portable SIMD if the feature is enabled.
#[cfg(feature = "simd")]
use std::simd::{f32x8, SimdFloat};

#[cfg(feature = "hydron")]
#[doc = "Statically computes the E8 adjacency graph once and caches it."]
static E8_ADJACENCY_GRAPH: OnceLock<Vec<Vec<u8>>> = OnceLock::new();

#[cfg(feature = "hydron")]
#[doc = "Computes and returns the full 240x56 adjacency graph for the E8 root system."]
fn get_adjacency_graph() -> &'static Vec<Vec<u8>> {
    E8_ADJACENCY_GRAPH.get_or_init(|| {
        let roots = get_e8_roots();
        (0..roots.len())
            .into_par_iter()
            .map(|i| {
                let target = roots[i];
                let mut neighbors = Vec::with_capacity(56);
                for (j, root) in roots.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let dot = gf8_dot_simd(&target, root);
                    // Neighbors in the E8 root system have a dot product of +/- 0.5.
                    if (dot - 0.5).abs() < 1e-4 || (dot + 0.5).abs() < 1e-4 {
                        neighbors.push(j as u8);
                    }
                }
                neighbors
            })
            .collect()
    })
}

/// Returns a pre-computed list of indices of the 56 nearest neighbors for a given root.
/// This is an amortized O(1) operation.
#[cfg(feature = "hydron")]
pub fn get_neighbors(root_idx: usize) -> &'static [u8] {
    let graph = get_adjacency_graph();
    graph.get(root_idx).map_or(&[], |v| v.as_slice())
}

#[cfg(not(feature = "hydron"))]
pub fn get_neighbors(_root_idx: usize) -> &'static [u8] {
    &[]
}

/// Performs a SIMD-accelerated Weyl reflection of `vec` across the hyperplane orthogonal to `mirror_root`.
/// Formula (for unit roots): v' = v - 2 * <v, r> * r
#[cfg(feature = "hydron")]
pub fn weyl_reflect(vec: &[f32; 8], mirror_root: &[f32; 8]) -> [f32; 8] {
    let dot = gf8_dot_simd(vec, mirror_root);
    let scale = -2.0 * dot;

    #[cfg(feature = "simd")]
    {
        let v_simd = f32x8::from_array(*vec);
        let m_simd = f32x8::from_array(*mirror_root);
        let scale_simd = f32x8::splat(scale);
        // Fused multiply-add is ideal here: v + scale * m
        (m_simd.mul_add(scale_simd, v_simd)).to_array()
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = vec[i] + scale * mirror_root[i];
        }
        result
    }
}

#[cfg(not(feature = "hydron"))]
pub fn weyl_reflect(vec: &[f32; 8], _mirror_root: &[f32; 8]) -> [f32; 8] {
    *vec // No-op fallback
}

/// Diffuses energy over the E8 lattice in parallel.
/// Each root adds a fraction of its energy to its 56 neighbors.
#[cfg(feature = "hydron")]
pub fn diffuse_energy(energy: &[f32; 240], diffusion_rate: f32) -> [f32; 240] {
    // Each parallel iteration calculates its own "row" of contributions
    let partial_contributions: Vec<Vec<(usize, f32)>> = (0..240)
        .into_par_iter()
        .map(|i| {
            let e = energy[i];
            let mut current_root_contributions = Vec::new();
            if e > 1e-6 { // Only diffuse if energy is significant
                let neighbors = get_neighbors(i);
                if !neighbors.is_empty() {
                    let flow_per_neighbor = (e * diffusion_rate) / neighbors.len() as f32;
                    for &n_idx in neighbors {
                        // Store (target_index, contribution_amount)
                        current_root_contributions.push((n_idx as usize, flow_per_neighbor));
                    }
                }
            }
            current_root_contributions
        })
        .collect(); // Collect all partial contributions

    let mut new_field = *energy; // Start with existing energy

    // Sequentially apply all collected contributions
    for root_contributions in partial_contributions {
        for (target_idx, amount) in root_contributions {
            new_field[target_idx] += amount;
        }
    }

    new_field
}

#[cfg(not(feature = "hydron"))]
pub fn diffuse_energy(energy: &[f32; 240], _diffusion_rate: f32) -> [f32; 240] {
    *energy // No-op fallback
}