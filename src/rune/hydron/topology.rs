//! E8 Topology & Weyl Group Operations.
//!
//! # Hydron – Topology Module
//! ▫~•◦------------------------‣
//!
//! Provides adjacency (kissing) relations, Weyl reflections, and simple diffusion over
//! the E8 root lattice. Uses the static root table from hydron-core.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hydron_core::get_e8_roots;

/// Returns the indices of the 56 nearest neighbors in the E8 lattice.
/// Roots are neighbors if their dot product is approximately 0.5 (60 degrees).
pub fn get_neighbors(root_idx: usize) -> Vec<u8> {
    let roots = get_e8_roots();
    if root_idx >= roots.len() {
        return vec![];
    }
    let target = roots[root_idx];
    let mut neighbors = Vec::with_capacity(56);

    for (i, root) in roots.iter().enumerate() {
        if i == root_idx {
            continue;
        }
        let dot: f32 = target.iter().zip(root.iter()).map(|(a, b)| a * b).sum();
        if (dot - 0.5).abs() < 1e-4 {
            neighbors.push(i as u8);
        }
    }
    neighbors
}

/// Performs a Weyl reflection of a vector `v` across the hyperplane orthogonal to root `r`.
/// Formula (unit roots): v' = v - 2 * <v, r> * r
pub fn weyl_reflect(vec: &[f32; 8], mirror_root: &[f32; 8]) -> [f32; 8] {
    let dot: f32 = vec.iter().zip(mirror_root.iter()).map(|(a, b)| a * b).sum();
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = vec[i] - 2.0 * dot * mirror_root[i];
    }
    result
}

/// Diffuse energy over the E8 lattice (swarm/attention style).
/// Keeps source energy and adds a fraction to neighbors.
pub fn diffuse_energy(energy: &[f32; 240], diffusion_rate: f32) -> [f32; 240] {
    let _roots = get_e8_roots();
    let mut new_field = *energy;

    for i in 0..240 {
        let e = energy[i];
        if e <= 1e-6 {
            continue;
        }
        let neighbors = get_neighbors(i);
        if neighbors.is_empty() {
            continue;
        }
        let flow = e * diffusion_rate;
        let flow_per = flow / neighbors.len() as f32;
        for n in neighbors {
            new_field[n as usize] += flow_per;
        }
    }

    new_field
}
