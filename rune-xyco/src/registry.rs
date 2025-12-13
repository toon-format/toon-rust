/* src/registry.rs */
//!▫~•◦-------------------------------‣
//! # Codex Registry Implementation
//!▫~•◦-------------------------------------------------------------------‣
//! Implements the Spec C deterministic assignment of roots to tiers.

use crate::codex::{CodexRoot, Tier};
use crate::e8::{self, generate_canonical_roots, simple_roots, dot, add, neg, snap_to_root};
use std::cmp::Ordering;
use std::collections::HashSet;
use once_cell::sync::Lazy;

pub struct CodexRegistry {
    roots: Vec<CodexRoot>,
}

impl CodexRegistry {
    /// Returns the singleton instance of the registry.
    pub fn instance() -> &'static CodexRegistry {
        static REGISTRY: Lazy<CodexRegistry> = Lazy::new(|| CodexRegistry::generate());
        &REGISTRY
    }

    pub fn get_root(&self, id: u16) -> Option<&CodexRoot> {
        self.roots.get(id as usize)
    }

    /// Generates the registry according to Spec C.
    fn generate() -> Self {
        let all_roots = generate_canonical_roots();
        let simple_roots = simple_roots();
        let mut registry_roots: Vec<Option<CodexRoot>> = (0..240).map(|_| None).collect();
        let mut used_vectors = HashSet::new();

        // Helper to check if a vector is roughly equal to another (float issues)
        // Since we generate exact floats (0.5, 1.0, -1.0), strict equality might work if bits match,
        // but let's use a "key" of integers for the set.
        fn vec_key(v: &[f32; 8]) -> [i32; 8] {
            let mut k = [0; 8];
            for i in 0..8 {
                k[i] = (v[i] * 2.0).round() as i32;
            }
            k
        }

        // 1. Assign Taproots (Tier 0)
        // Pos 14 in each domain d gets simple_roots[d]
        for d in 0..8 {
            let root_id = (d * 30 + 14) as u16;
            let vector = simple_roots[d];

            registry_roots[root_id as usize] = Some(CodexRoot::new(
                root_id,
                vector,
                Tier::Taproot,
                0xFFFFFFFF, // All families allowed for now, refine later
            ));
            used_vectors.insert(vec_key(&vector));
        }

        // 2. Assign Laterals (Tier 1)
        // Pos 4 (+) and Pos 24 (-)
        for d in 0..8 {
            let tap_vector = simple_roots[d];

            // Lateral A (+)
            let target_a = tap_vector; // Start searching near +alpha
            let lat_a_vector = find_nearest_unused(target_a, &all_roots, &used_vectors);
            let lat_a_id = (d * 30 + 4) as u16;
            registry_roots[lat_a_id as usize] = Some(CodexRoot::new(lat_a_id, lat_a_vector, Tier::Lateral, 0xFFFFFFFF));
            used_vectors.insert(vec_key(&lat_a_vector));

            // Lateral B (-)
            let target_b = neg(&tap_vector);
            let lat_b_vector = find_nearest_unused(target_b, &all_roots, &used_vectors);
            let lat_b_id = (d * 30 + 24) as u16;
            registry_roots[lat_b_id as usize] = Some(CodexRoot::new(lat_b_id, lat_b_vector, Tier::Lateral, 0xFFFFFFFF));
            used_vectors.insert(vec_key(&lat_b_vector));
        }

        // 3. Assign Tertiaries (Tier 2)
        // Pos 8, 18, 22. Derived from Dynkin neighbors.
        let dynkin_edges = vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7)
        ];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; 8];
        for (u, v) in dynkin_edges {
            adj[u].push(v);
            adj[v].push(u);
        }

        for d in 0..8 {
            let neighbors = &adj[d];
            // Pick neighbors deterministically (smallest indices)
            let mut sorted_neighbors = neighbors.clone();
            sorted_neighbors.sort();

            // We need 3 tertiaries.
            // Strategy:
            // T0 = alpha_d + alpha_n0
            // T1 = alpha_d + alpha_n1 (if exists, else wrap)
            // T2 = alpha_d + alpha_n0 + alpha_n1

            let alpha_d = simple_roots[d];
            let n0 = sorted_neighbors.get(0).copied().unwrap_or((d + 1) % 8); // Fallback should not happen for E8
            let n1 = sorted_neighbors.get(1).copied().unwrap_or((d + 2) % 8);

            let alpha_n0 = simple_roots[n0];
            let alpha_n1 = simple_roots[n1];

            let targets = [
                add(&alpha_d, &alpha_n0),
                add(&alpha_d, &alpha_n1),
                add(&add(&alpha_d, &alpha_n0), &alpha_n1),
            ];

            let positions = [8, 18, 22];
            for (i, &pos) in positions.iter().enumerate() {
                let target = targets[i];
                let snapped = snap_to_root(&target, &all_roots); // Ensure it's a valid root

                // If snapped is used, find nearest unused
                let final_vector = if used_vectors.contains(&vec_key(&snapped)) {
                    find_nearest_unused(snapped, &all_roots, &used_vectors)
                } else {
                    snapped
                };

                let tid = (d * 30 + pos) as u16;
                registry_roots[tid as usize] = Some(CodexRoot::new(tid, final_vector, Tier::Tertiary, 0xFFFFFFFF));
                used_vectors.insert(vec_key(&final_vector));
            }
        }

        // 4. Assign Crosses (Tier 3)
        // Remaining 24 slots per domain.
        // Collect remaining unused roots
        let mut unused_roots: Vec<[f32; 8]> = all_roots.iter()
            .filter(|r| !used_vectors.contains(&vec_key(r)))
            .cloned()
            .collect();

        assert_eq!(unused_roots.len(), 192);

        for d in 0..8 {
            let tap_vector = simple_roots[d];
            // Sort unused roots by closeness to taproot
            unused_roots.sort_by(|a, b| {
                let da = dot(a, &tap_vector);
                let db = dot(b, &tap_vector);
                // Higher dot product = closer. Sort descending.
                db.partial_cmp(&da).unwrap_or(Ordering::Equal)
            });

            // Take top 24
            let best_24: Vec<[f32; 8]> = unused_roots.drain(0..24).collect();

            // Assign to slots 0..29 excluding reserved
            let mut slot_idx = 0;
            let reserved = [4, 8, 14, 18, 22, 24];

            for vector in best_24.into_iter() {
                // Find next non-reserved slot
                while reserved.contains(&slot_idx) {
                    slot_idx += 1;
                }

                let rid = (d * 30 + slot_idx) as u16;
                registry_roots[rid as usize] = Some(CodexRoot::new(rid, vector, Tier::Cross, 0xFFFFFFFF));
                used_vectors.insert(vec_key(&vector));
                slot_idx += 1;
            }
        }

        CodexRegistry {
            roots: registry_roots.into_iter().map(|opt| opt.unwrap()).collect(),
        }
    }
}

fn find_nearest_unused(target: [f32; 8], all: &[[f32; 8]], used: &HashSet<[i32; 8]>) -> [f32; 8] {
    let mut best_dist = f32::MAX;
    let mut best_root = [0.0; 8];

    // Helper to key
    fn k(v: &[f32; 8]) -> [i32; 8] {
        let mut arr = [0; 8];
        for i in 0..8 { arr[i] = (v[i] * 2.0).round() as i32; }
        arr
    }

    for root in all {
        if used.contains(&k(root)) {
            continue;
        }
        let d = e8::distance_sq(&target, root);
        if d < best_dist {
            best_dist = d;
            best_root = *root;
        }
    }
    best_root
}
