/* e8/gf8/src/resonance_router.rs */
/***
 * @file E8 Resonance Router – Swarm Attention over E8 Lattice
 * @packageDocumentation
 *
 * @remarks
 * # E8 Primitives – Resonance Routing Module
 * ▫~•◦------------------------------------------------‣
 *
 * This module implements the dynamic "resonance" layer on top of the static E8
 * lattice geometry defined in `quantize.rs`. It consumes multi-head E8 root
 * activations (e.g., from `HoloSphereBridge::lift_to_address`) and diffuses
 * their energy over the 240-root codebook using the precomputed adjacency
 * graph (56 neighbors per root).
 *
 * ### Key Capabilities
 * - **Multi-Head Swarm Attention:** Accumulates energy from multiple heads.
 * - **Neighbor Diffusion:** Direct + scaled energy to 56 lattice neighbors.
 * - **Resonance Ranking:** Produces a sorted list of the most "resonant" roots.
 *
 * ### Architectural Notes
 * This module is the dynamic counterpart to `quantize.rs`:
 *
 * - `quantize.rs`         = Static Cartography (roots + adjacency).
 * - `resonance_router.rs` = Dynamic Swarm Attention (energy on that graph).
 *
 * Higher-level systems (E8-GC, HoloSphere, ANN benchmarks) should depend on
 * this module when they need recall-optimized bucket selection rather than
 * single-root hard assignments.
 *
 *▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•-----------------------------------------------------------------------------------‣

use crate::Gf8BitSig;
use crate::quantize::get_root_neighbors;

/// Configuration parameters for resonance routing.
#[derive(Debug, Clone, Copy)]
pub struct ResonanceConfig {
    /// Weight applied to the direct root activation.
    pub direct_weight: f32,
    /// Factor applied to neighbors relative to the direct weight.
    /// For the Python sim: typically 0.5.
    pub diffusion_factor: f32,
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            direct_weight: 1.0,
            diffusion_factor: 0.5,
        }
    }
}

/// A single head activation over the E8 codebook.
#[derive(Debug, Clone, Copy)]
pub struct HeadActivation {
    /// The E8 root index (0..239).
    pub code: Gf8BitSig,
    /// The activation strength for this head (e.g., similarity score).
    pub score: f32,
}

/// A ranked resonance result over the E8 codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResonanceResult {
    pub code: Gf8BitSig,
    pub energy: u32,
}

/// Core accumulation primitive: given a set of head activations, accumulate
/// resonance energy over the 240 roots and return the full energy array.
///
/// The returned array is indexed by `Gf8BitSig .0` (0..239).
/// Energy is accumulated as u32 (no floating-point drift).
pub fn accumulate_resonance(heads: &[HeadActivation], cfg: ResonanceConfig) -> [u32; 240] {
    let mut energy = [0u32; 240];

    for head in heads {
        let idx = head.code.0 as usize;
        if idx >= 240 {
            // Defensive guard; should never happen in practice.
            continue;
        }

        let score = head.score;

        // Direct contribution (convert f32 score to u32 energy).
        let direct_energy = (score * cfg.direct_weight * 1000.0) as u32;
        energy[idx] = energy[idx].saturating_add(direct_energy);

        // Neighbor diffusion using the canonical adjacency graph.
        let neighbors = get_root_neighbors(head.code.0);
        let neighbor_energy = (score * cfg.direct_weight * cfg.diffusion_factor * 1000.0) as u32;

        for &nbr in neighbors {
            let n_idx = nbr as usize;
            if n_idx < 240 {
                energy[n_idx] = energy[n_idx].saturating_add(neighbor_energy);
            }
        }
    }

    energy
}

/// Compute the dom-R resonant roots given a list of heads.
/// Returns up to K roots sorted by descending energy.
pub fn top_k_resonant_roots(
    heads: &[HeadActivation],
    cfg: ResonanceConfig,
    k: usize,
) -> Vec<ResonanceResult> {
    let energy = accumulate_resonance(heads, cfg);

    // Collect (idx, energy) pairs.
    let mut items: Vec<(usize, u32)> = energy
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e > 0)
        .map(|(i, &e)| (i, e))
        .collect();

    // Sort by energy desc.
    items.sort_by(|a, b| b.1.cmp(&a.1));

    items
        .into_iter()
        .take(k.min(240))
        .map(|(i, e)| ResonanceResult {
            code: Gf8BitSig(i as u8),
            energy: e,
        })
        .collect()
}

/// Convenience helper: build `HeadActivation`s from raw `(u8, f32)` pairs.
pub fn heads_from_raw_pairs(pairs: &[(u8, f32)]) -> Vec<HeadActivation> {
    pairs
        .iter()
        .map(|&(code, score)| HeadActivation {
            code: Gf8BitSig(code),
            score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_energy_dominates_with_no_diffusion() {
        let heads = vec![HeadActivation {
            code: Gf8BitSig(42),
            score: 1.0,
        }];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.0,
        };

        let energy = accumulate_resonance(&heads, cfg);
        assert!(energy[42] > 0);

        // With no diffusion, only the direct root should be active.
        for (i, &e) in energy.iter().enumerate() {
            if i == 42 {
                continue;
            }
            assert_eq!(e, 0);
        }
    }

    #[test]
    fn diffusion_activates_neighbors() {
        let heads = vec![HeadActivation {
            code: Gf8BitSig(10),
            score: 2.0,
        }];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.5,
        };

        let energy = accumulate_resonance(&heads, cfg);
        assert!(energy[10] > 0);

        let neighbors = get_root_neighbors(10);
        assert!(!neighbors.is_empty());

        // At least one neighbor should receive energy from diffusion.
        let neighbor_has_energy = neighbors.iter().any(|&n| energy[n as usize] > 0);
        assert!(neighbor_has_energy);
    }

    #[test]
    fn top_k_resonant_roots_is_sorted() {
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(1),
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(2),
                score: 0.5,
            },
        ];

        let cfg = ResonanceConfig::default();
        let results = top_k_resonant_roots(&heads, cfg, 8);

        assert!(!results.is_empty());
        for w in results.windows(2) {
            assert!(w[0].energy >= w[1].energy);
        }
    }

    // ============================================================================
    // Task 3.2: Add Resonance Routing Tests
    // ============================================================================

    #[test]
    fn test_direct_energy_accumulation_no_diffusion() {
        // Test direct energy accumulation with diffusion_factor = 0
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(5),
                score: 2.0,
            },
            HeadActivation {
                code: Gf8BitSig(10),
                score: 1.5,
            },
        ];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.0,
        };

        let energy = accumulate_resonance(&heads, cfg);

        // Expected: root 5 gets 2.0 * 1.0 * 1000 = 2000
        // Expected: root 10 gets 1.5 * 1.0 * 1000 = 1500
        assert_eq!(energy[5], 2000);
        assert_eq!(energy[10], 1500);

        // All other roots should have zero energy
        for (i, &e) in energy.iter().enumerate() {
            if i != 5 && i != 10 {
                assert_eq!(e, 0, "Root {} should have zero energy", i);
            }
        }
    }

    #[test]
    fn test_neighbor_diffusion_with_factor() {
        // Test neighbor diffusion with diffusion_factor = 0.5
        let heads = vec![HeadActivation {
            code: Gf8BitSig(0),
            score: 1.0,
        }];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.5,
        };

        let energy = accumulate_resonance(&heads, cfg);

        // Direct root should get 1.0 * 1.0 * 1000 = 1000
        assert_eq!(energy[0], 1000);

        // Get neighbors of root 0
        let neighbors = get_root_neighbors(0);
        assert_eq!(
            neighbors.len(),
            56,
            "Root 0 should have exactly 56 neighbors"
        );

        // Each neighbor should get 1.0 * 1.0 * 0.5 * 1000 = 500
        for &nbr in neighbors {
            let nbr_idx = nbr as usize;
            assert_eq!(
                energy[nbr_idx], 500,
                "Neighbor {} should have energy 500",
                nbr_idx
            );
        }

        // Verify that exactly 57 roots have energy (1 direct + 56 neighbors)
        let active_roots = energy.iter().filter(|&&e| e > 0).count();
        assert_eq!(active_roots, 57, "Should have exactly 57 active roots");
    }

    #[test]
    fn test_top_k_resonant_roots_returns_sorted_results() {
        // Test that top_k_resonant_roots returns results sorted by energy descending
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(0),
                score: 3.0,
            },
            HeadActivation {
                code: Gf8BitSig(1),
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(2),
                score: 2.0,
            },
        ];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.0,
        };

        let results = top_k_resonant_roots(&heads, cfg, 10);

        // Should have at least 3 results (the 3 direct roots)
        assert!(results.len() >= 3);

        // Verify sorted order (descending by energy)
        for i in 0..results.len() - 1 {
            assert!(
                results[i].energy >= results[i + 1].energy,
                "Results should be sorted by energy descending"
            );
        }

        // First result should be root 0 with energy 3000
        assert_eq!(results[0].code.0, 0);
        assert_eq!(results[0].energy, 3000);

        // Second result should be root 2 with energy 2000
        assert_eq!(results[1].code.0, 2);
        assert_eq!(results[1].energy, 2000);

        // Third result should be root 1 with energy 1000
        assert_eq!(results[2].code.0, 1);
        assert_eq!(results[2].energy, 1000);
    }

    #[test]
    fn test_top_k_respects_k_limit() {
        // Test that top_k_resonant_roots respects the k parameter
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(0),
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(1),
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(2),
                score: 1.0,
            },
        ];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.0,
        };

        let results_k3 = top_k_resonant_roots(&heads, cfg, 3);
        assert_eq!(results_k3.len(), 3);

        let results_k1 = top_k_resonant_roots(&heads, cfg, 1);
        assert_eq!(results_k1.len(), 1);

        let results_k100 = top_k_resonant_roots(&heads, cfg, 100);
        assert_eq!(results_k100.len(), 3); // Only 3 active roots
    }

    #[test]
    fn test_determinism_same_input_same_output() {
        // Test determinism: same input should always produce same output
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(42),
                score: 1.5,
            },
            HeadActivation {
                code: Gf8BitSig(100),
                score: 0.8,
            },
        ];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.5,
        };

        // Run accumulate_resonance multiple times
        let energy1 = accumulate_resonance(&heads, cfg);
        let energy2 = accumulate_resonance(&heads, cfg);
        let energy3 = accumulate_resonance(&heads, cfg);

        // All results should be identical
        assert_eq!(energy1, energy2);
        assert_eq!(energy2, energy3);

        // Run top_k_resonant_roots multiple times
        let results1 = top_k_resonant_roots(&heads, cfg, 16);
        let results2 = top_k_resonant_roots(&heads, cfg, 16);
        let results3 = top_k_resonant_roots(&heads, cfg, 16);

        // All results should be identical
        assert_eq!(results1, results2);
        assert_eq!(results2, results3);
    }

    #[test]
    fn test_empty_heads_produces_zero_energy() {
        // Test that empty heads produce zero energy everywhere
        let heads: Vec<HeadActivation> = vec![];

        let cfg = ResonanceConfig::default();
        let energy = accumulate_resonance(&heads, cfg);

        // All energy should be zero
        for &e in energy.iter() {
            assert_eq!(e, 0);
        }

        // top_k should return empty
        let results = top_k_resonant_roots(&heads, cfg, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_multiple_heads_same_root() {
        // Test that multiple heads activating the same root accumulate energy
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(50),
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(50),
                score: 2.0,
            },
        ];

        let cfg = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.0,
        };

        let energy = accumulate_resonance(&heads, cfg);

        // Root 50 should accumulate both scores: (1.0 + 2.0) * 1000 = 3000
        assert_eq!(energy[50], 3000);
    }

    #[test]
    fn test_diffusion_factor_scaling() {
        // Test that diffusion_factor correctly scales neighbor energy
        let heads = vec![HeadActivation {
            code: Gf8BitSig(0),
            score: 1.0,
        }];

        // Test with diffusion_factor = 0.25
        let cfg_low = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.25,
        };

        let energy_low = accumulate_resonance(&heads, cfg_low);
        let neighbors = get_root_neighbors(0);

        // Each neighbor should get 1.0 * 1.0 * 0.25 * 1000 = 250
        for &nbr in neighbors {
            assert_eq!(energy_low[nbr as usize], 250);
        }

        // Test with diffusion_factor = 0.75
        let cfg_high = ResonanceConfig {
            direct_weight: 1.0,
            diffusion_factor: 0.75,
        };

        let energy_high = accumulate_resonance(&heads, cfg_high);

        // Each neighbor should get 1.0 * 1.0 * 0.75 * 1000 = 750
        for &nbr in neighbors {
            assert_eq!(energy_high[nbr as usize], 750);
        }
    }

    #[test]
    fn test_direct_weight_scaling() {
        // Test that direct_weight correctly scales energy
        let heads = vec![HeadActivation {
            code: Gf8BitSig(0),
            score: 1.0,
        }];

        // Test with direct_weight = 2.0
        let cfg = ResonanceConfig {
            direct_weight: 2.0,
            diffusion_factor: 0.0,
        };

        let energy = accumulate_resonance(&heads, cfg);

        // Root 0 should get 1.0 * 2.0 * 1000 = 2000
        assert_eq!(energy[0], 2000);
    }

    #[test]
    fn test_invalid_root_index_ignored() {
        // Test that invalid root indices (>= 240) are safely ignored
        let heads = vec![
            HeadActivation {
                code: Gf8BitSig(250), // Invalid: >= 240
                score: 1.0,
            },
            HeadActivation {
                code: Gf8BitSig(0), // Valid
                score: 1.0,
            },
        ];

        let cfg = ResonanceConfig::default();
        let energy = accumulate_resonance(&heads, cfg);

        // Only root 0 and its neighbors should have energy
        // Invalid root 250 should be ignored
        assert!(energy[0] > 0);
        let active_count = energy.iter().filter(|&&e| e > 0).count();
        assert!(active_count > 0);
    }
}
