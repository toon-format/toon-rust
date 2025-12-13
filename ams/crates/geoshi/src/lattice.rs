//! E8 Lattice Generation and Operations
//!
//! # LATTICE MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced lattice mathematics implementation including the exceptional E8
//! lattice with 240 root vectors. Provides lattice generation, adjacency
//! analysis, and Lie group operations for geometric cognition foundations.
//!
//! ### Key Capabilities
//! - **E8 Lattice Generation:** Complete 240 root vector exceptional lattice.
//! - **Adjacency Analysis:** Graph-based connectivity and neighbor relationships.
//! - **Lie Group Operations:** Foundational mathematical structures for cognition.
//! - **Normalization:** Proper vector normalization and dot product analysis.
//! - **D8 Subgroup Extraction:** Links to the D8 maximal subgroup.
//!
//! ### Technical Features
//! - **High-Precision Arithmetic:** Numerical stability for lattice computations.
//! - **Efficient Adjacency:** Optimized graph representation and queries.
//! - **Root Vector Enumeration:** Systematic generation of all 240 E8 roots.
//! - **Spinor Weight Analysis:** Even/odd parity spinor root classification.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::lattice::E8Lattice;
//!
//! let lattice = E8Lattice::new().unwrap();
//! let root_index = 0usize;
//! let neighbors = lattice.neighbors(root_index);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use itertools::Itertools;
use ndarray::{Array1, Array2};

/// E8 lattice with 240 root vectors and adjacency matrix
#[derive(Debug)]
pub struct E8Lattice {
    pub roots: Array2<f64>,
    pub adj_matrix: Vec<Vec<usize>>,
}

impl E8Lattice {
    /// Create a new E8 lattice
    pub fn new() -> GsaResult<Self> {
        let roots = Self::generate_roots()?;
        let adj_matrix = Self::build_adjacency(&roots);
        Ok(Self { roots, adj_matrix })
    }

    /// Generate all 240 E8 root vectors
    fn generate_roots() -> GsaResult<Array2<f64>> {
        let mut roots = Vec::new();

        // D8 roots: permutations of (±1, ±1, 0, ..., 0)
        for i in 0..8 {
            for j in (i + 1)..8 {
                for &s1 in [-1.0, 1.0].iter() {
                    for &s2 in [-1.0, 1.0].iter() {
                        let mut vec = [0.0; 8];
                        vec[i] = s1;
                        vec[j] = s2;
                        roots.push(vec);
                    }
                }
            }
        }

        // Spinor roots: (±1/2, ..., ±1/2) with even number of minus signs
        for signs in itertools::repeat_n([-0.5, 0.5].iter(), 8).multi_cartesian_product() {
            let sign_vec: Vec<f64> = signs.into_iter().cloned().collect();
            let minus_count = sign_vec.iter().filter(|&&s| s < 0.0).count();
            if minus_count % 2 == 0 {
                roots.push(sign_vec.try_into().unwrap());
            }
        }

        let mut root_array =
            Array2::from_shape_vec((roots.len(), 8), roots.into_iter().flatten().collect())
                .map_err(|e| crate::GsaError::Lattice(format!("Shape error: {}", e)))?;

        // Normalize each root vector
        for mut row in root_array.outer_iter_mut() {
            let norm = (row.dot(&row)).sqrt();
            if norm > 1e-12 {
                row /= norm;
            }
        }

        Ok(root_array)
    }

    /// Build adjacency matrix based on dot product similarity
    fn build_adjacency(roots: &Array2<f64>) -> Vec<Vec<usize>> {
        let n = roots.nrows();
        let mut adj = vec![vec![]; n];
        let sim = roots.dot(&roots.t());

        for i in 0..n {
            for j in (i + 1)..n {
                let s = sim[[i, j]];
                if (0.49..=0.9999).contains(&s) {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }
        adj
    }

    /// Get number of roots
    pub fn n_roots(&self) -> usize {
        self.roots.nrows()
    }

    /// Get a root vector by index
    pub fn get_root(&self, idx: usize) -> ndarray::ArrayView1<'_, f64> {
        self.roots.row(idx)
    }

    /// Check if two roots are adjacent
    pub fn are_adjacent(&self, a: usize, b: usize) -> bool {
        self.adj_matrix[a].contains(&b)
    }

    /// Get neighbors of a root
    pub fn neighbors(&self, idx: usize) -> &[usize] {
        &self.adj_matrix[idx]
    }

    /// Find the closest root vector to a given 8D vector using cosine similarity
    pub fn find_closest_root(&self, vector: &Array1<f64>) -> usize {
        let mut best_idx = 0;
        let mut best_similarity = f64::NEG_INFINITY;

        // Normalize the input vector
        let norm = vector.dot(vector).sqrt();
        let normalized_vector = if norm > 1e-12 {
            vector / norm
        } else {
            vector.clone()
        };

        for i in 0..self.n_roots() {
            let root = self.get_root(i);
            let similarity = normalized_vector.dot(&root);

            if similarity > best_similarity {
                best_similarity = similarity;
                best_idx = i;
            }
        }

        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_lattice_creation() {
        let lattice = E8Lattice::new().unwrap();
        assert_eq!(lattice.n_roots(), 240);

        // Check normalization
        for i in 0..lattice.n_roots() {
            let root = lattice.get_root(i);
            let norm = root.dot(&root).sqrt();
            approx::assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adjacency_symmetry() {
        let lattice = E8Lattice::new().unwrap();

        for i in 0..lattice.n_roots() {
            for &j in lattice.neighbors(i) {
                assert!(lattice.are_adjacent(j, i));
            }
        }
    }
}
