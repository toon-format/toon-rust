//! Topological Analysis Layer - Persistent Homology
//!
//! Implements persistent homology for analyzing topological features in point clouds.
//! Computes Betti numbers (β₀, β₁, β₂) representing:
//! - β₀: Connected components
//! - β₁: Loops/cycles
//! - β₂: Voids/cavities
//!
//! Topological features are invariant under continuous deformation,
//! making them robust descriptors of data structure.
//!
//! Key operations:
//! - Topological signature generation
//! - Persistence diagram computation
//! - Filtration-based analysis
//! - Betti number tracking
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashSet;

/// Persistence diagram entry: (birth, death) for a topological feature
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize,
}

impl PersistencePair {
    /// Lifetime of the topological feature
    pub fn persistence(&self) -> f32 {
        self.death - self.birth
    }
}

/// Topological layer for persistent homology
pub struct TopologicalLayer {
    /// Persistence diagrams by dimension (0, 1, 2)
    pub diagrams: [Vec<PersistencePair>; 3],

    /// Current Betti numbers [β₀, β₁, β₂]
    pub betti: [u32; 3],

    /// Point cloud data for simplicial complex construction
    points: Vec<[f32; 8]>,
}

impl TopologicalLayer {
    /// Create new topological layer
    pub fn new() -> Self {
        Self {
            diagrams: [Vec::new(), Vec::new(), Vec::new()],
            betti: [1, 0, 0], // Start with single connected component
            points: Vec::new(),
        }
    }

    /// Add point to point cloud
    pub fn add_point(&mut self, point: [f32; 8]) {
        self.points.push(point);
    }

    /// Clear all points
    pub fn clear_points(&mut self) {
        self.points.clear();
    }

    /// Compute distance matrix for point cloud
    fn distance_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.points.len();
        let mut distances = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&self.points[i], &self.points[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Build Vietoris-Rips filtration up to given radius
    /// Returns edges that appear at each threshold
    fn vietoris_rips_filtration(
        &self,
        max_radius: f32,
        steps: usize,
    ) -> Vec<(f32, Vec<(usize, usize)>)> {
        let n = self.points.len();
        if n < 2 {
            return Vec::new();
        }

        let distances = self.distance_matrix();
        let mut filtration = Vec::new();

        let step_size = max_radius / steps as f32;

        for step in 0..=steps {
            let threshold = step as f32 * step_size;
            let mut edges = Vec::new();

            for i in 0..n {
                for j in (i + 1)..n {
                    if distances[i][j] <= threshold {
                        edges.push((i, j));
                    }
                }
            }

            filtration.push((threshold, edges));
        }

        filtration
    }

    /// Compute Betti numbers using Union-Find for β₀
    pub fn compute_betti_numbers(&mut self, max_radius: f32, steps: usize) {
        let n = self.points.len();

        if n == 0 {
            self.betti = [0, 0, 0];
            return;
        }

        let filtration = self.vietoris_rips_filtration(max_radius, steps);

        // Initialize β₀ = number of points (all disconnected)
        let mut uf = UnionFind::new(n);
        let mut beta0_history = Vec::new();

        // Track β₀ through filtration
        for (threshold, edges) in &filtration {
            for &(i, j) in edges {
                uf.union(i, j);
            }
            beta0_history.push((*threshold, uf.count_components()));
        }

        // Current β₀ is final component count
        self.betti[0] = uf.count_components() as u32;

        // Estimate β₁ (loops) from Euler characteristic
        // χ = β₀ - β₁ + β₂
        // For simplicial complexes: χ = V - E + F
        if let Some((_, final_edges)) = filtration.last() {
            let v = n as i32;
            let e = final_edges.len() as i32;

            // Count triangles (2-simplices) for better β₁ estimation
            let mut triangle_count = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        // Check if edges (i,j), (j,k), (k,i) all exist
                        let has_ij = final_edges.contains(&(i, j));
                        let has_jk = final_edges.contains(&(j, k));
                        let has_ki = final_edges.contains(&(i, k));

                        if has_ij && has_jk && has_ki {
                            triangle_count += 1;
                        }
                    }
                }
            }

            let f = triangle_count;
            let chi = v - e + f;
            let beta1_estimate = (self.betti[0] as i32 - chi).max(0);
            self.betti[1] = beta1_estimate as u32;

            // β₂ (voids) estimation using Euler characteristic
            // For closed surfaces: χ = 2 - 2g where g is genus
            // For 3D complexes: β₂ relates to enclosed voids

            // Count tetrahedra (3-simplices) for void detection
            let mut tetrahedron_count = 0;
            if n >= 4 {
                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            for l in (k + 1)..n {
                                // Check if all 6 edges and 4 faces exist
                                let edges_exist = final_edges.contains(&(i, j))
                                    && final_edges.contains(&(i, k))
                                    && final_edges.contains(&(i, l))
                                    && final_edges.contains(&(j, k))
                                    && final_edges.contains(&(j, l))
                                    && final_edges.contains(&(k, l));

                                if edges_exist {
                                    tetrahedron_count += 1;
                                }
                            }
                        }
                    }
                }
            }

            // χ = β₀ - β₁ + β₂ for 2D surfaces
            // For 3D: χ = V - E + F - T (where T is tetrahedra)
            // β₂ = χ - β₀ + β₁
            if tetrahedron_count > 0 {
                let chi_3d = v - e + f - tetrahedron_count;
                let beta2_estimate = (chi_3d - self.betti[0] as i32 + self.betti[1] as i32).max(0);
                self.betti[2] = beta2_estimate as u32;
            } else {
                // No 3D structure detected, β₂ = 0
                self.betti[2] = 0;
            }
        }
    }

    /// Generate topological signature (hash of Betti numbers and persistence)
    pub fn signature(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.betti[0].hash(&mut hasher);
        self.betti[1].hash(&mut hasher);
        self.betti[2].hash(&mut hasher);

        // Include persistence information
        for dim in 0..3 {
            for pair in &self.diagrams[dim] {
                let persistence = (pair.persistence() * 1000.0) as u64;
                persistence.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Compute persistence diagram for dimension 0 (connected components)
    pub fn compute_persistence_diagram_dim0(&mut self, max_radius: f32, steps: usize) {
        let n = self.points.len();
        if n < 2 {
            return;
        }

        let filtration = self.vietoris_rips_filtration(max_radius, steps);
        let mut uf = UnionFind::new(n);
        let birth_times = vec![0.0f32; n];
        let mut alive = vec![true; n];

        self.diagrams[0].clear();

        for (threshold, edges) in &filtration {
            for &(i, j) in edges {
                let root_i = uf.find(i);
                let root_j = uf.find(j);

                if root_i != root_j {
                    // Merge components: older component survives
                    let (_survivor, victim) = if birth_times[root_i] <= birth_times[root_j] {
                        (root_i, root_j)
                    } else {
                        (root_j, root_i)
                    };

                    if alive[victim] {
                        // Record death of victim component
                        self.diagrams[0].push(PersistencePair {
                            birth: birth_times[victim],
                            death: *threshold,
                            dimension: 0,
                        });
                        alive[victim] = false;
                    }

                    uf.union(root_i, root_j);
                }
            }
        }

        // Surviving components have infinite death time (we use max_radius as proxy)
        for i in 0..n {
            let root = uf.find(i);
            if alive[root] && uf.is_root(i) {
                self.diagrams[0].push(PersistencePair {
                    birth: birth_times[root],
                    death: f32::INFINITY,
                    dimension: 0,
                });
            }
        }
    }

    /// Get persistence pairs with lifetime above threshold
    pub fn significant_features(&self, min_persistence: f32) -> Vec<PersistencePair> {
        let mut features = Vec::new();

        for dim in 0..3 {
            for &pair in &self.diagrams[dim] {
                if pair.persistence() >= min_persistence {
                    features.push(pair);
                }
            }
        }

        features.sort_by(|a, b| b.persistence().partial_cmp(&a.persistence()).unwrap());
        features
    }

    /// Compute total persistence (sum of all feature lifetimes)
    pub fn total_persistence(&self) -> f32 {
        let mut total = 0.0f32;

        for dim in 0..3 {
            for pair in &self.diagrams[dim] {
                let pers = pair.persistence();
                if pers.is_finite() {
                    total += pers;
                }
            }
        }

        total
    }

    /// Check if two point clouds are topologically similar
    pub fn is_similar(&self, other: &TopologicalLayer, tolerance: u32) -> bool {
        for i in 0..3 {
            if self.betti[i].abs_diff(other.betti[i]) > tolerance {
                return false;
            }
        }
        true
    }
}

impl Default for TopologicalLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Union-Find data structure for connected components
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn count_components(&mut self) -> usize {
        let mut roots = HashSet::new();
        for i in 0..self.parent.len() {
            roots.insert(self.find(i));
        }
        roots.len()
    }

    fn is_root(&self, x: usize) -> bool {
        self.parent[x] == x
    }
}

/// Compute Euclidean distance between two 8D points
fn euclidean_distance(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    #[cfg(feature = "simd")]
    {
        use super::gf8::{gf8_norm2_simd, gf8_sub_simd};
        let diff = gf8_sub_simd(a, b);
        gf8_norm2_simd(&diff).sqrt()
    }
    #[cfg(not(feature = "simd"))]
    {
        let sum_sq: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_layer_creation() {
        let layer = TopologicalLayer::new();
        assert_eq!(layer.betti, [1, 0, 0]);
    }

    #[test]
    fn test_add_points() {
        let mut layer = TopologicalLayer::new();
        layer.add_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(layer.points.len(), 2);
    }

    #[test]
    fn test_betti_numbers_disconnected() {
        let mut layer = TopologicalLayer::new();

        // Add two widely separated points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // With small radius, should have 2 components
        layer.compute_betti_numbers(1.0, 10);
        assert_eq!(layer.betti[0], 2);
    }

    #[test]
    fn test_betti_numbers_connected() {
        let mut layer = TopologicalLayer::new();

        // Add two close points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // With large radius, should have 1 component
        layer.compute_betti_numbers(10.0, 10);
        assert_eq!(layer.betti[0], 1);
    }

    #[test]
    fn test_signature() {
        let layer1 = TopologicalLayer::new();
        let layer2 = TopologicalLayer::new();

        // Same Betti numbers should give same signature
        assert_eq!(layer1.signature(), layer2.signature());
    }

    #[test]
    fn test_persistence_diagram() {
        let mut layer = TopologicalLayer::new();

        // Add triangle of points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        layer.compute_persistence_diagram_dim0(5.0, 20);

        // Should have persistence pairs
        assert!(!layer.diagrams[0].is_empty());
    }

    #[test]
    fn test_significant_features() {
        let mut layer = TopologicalLayer::new();

        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 2.0,
            dimension: 0,
        });
        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 0.1,
            dimension: 0,
        });

        let features = layer.significant_features(1.0);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].persistence(), 2.0);
    }

    #[test]
    fn test_total_persistence() {
        let mut layer = TopologicalLayer::new();

        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 2.0,
            dimension: 0,
        });
        layer.diagrams[1].push(PersistencePair {
            birth: 1.0,
            death: 3.0,
            dimension: 1,
        });

        let total = layer.total_persistence();
        assert_eq!(total, 4.0);
    }

    #[test]
    fn test_is_similar() {
        let mut layer1 = TopologicalLayer::new();
        layer1.betti = [2, 1, 0];

        let mut layer2 = TopologicalLayer::new();
        layer2.betti = [2, 1, 0];

        assert!(layer1.is_similar(&layer2, 0));

        let mut layer3 = TopologicalLayer::new();
        layer3.betti = [5, 1, 0];

        assert!(!layer1.is_similar(&layer3, 1));
    }
}
