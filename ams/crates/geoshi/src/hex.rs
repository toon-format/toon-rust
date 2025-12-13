//! Hexagonal lattice operations with geometric computations
//!
//! # HEX MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Comprehensive hexagonal grid mathematics implementing axial and cube coordinate
//! systems with efficient neighbor finding, distance calculations, and pathfinding
//! algorithms specifically optimized for hexagonal geometry.
//!
//! ### Key Capabilities
//! - **Coordinate Systems:** Axial (q,r) and cube (x,y,z) coordinate representations.
//! - **Geometric Operations:** Distance calculations, neighbor finding, and ring generation.
//! - **Lattice Construction:** Configurable hexagonal lattices with radius-based generation.
//! - **Pathfinding Integration:** A* algorithm optimized for hexagonal movement costs.
//! - **Boundary Analysis:** Automatic boundary detection and topological operations.
//!
//! ### Technical Features
//! - **Hexagonal Distance:** Manifold-aware distance metrics for hexagonal spaces.
//! - **Memory Efficiency:** HashSet-based coordinate storage with fast lookups.
//! - **Coordinate Conversion:** Seamless axial ↔ cube coordinate transformations.
//! - **Ring-Based Operations:** Efficient ring generation for cellular automata.
//! - **A* Optimization:** Hexagonal movement costs and heuristic optimization.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::hex::{HexLattice, AxialCoord};
//!
//! let lattice = HexLattice::new(3).unwrap();
//! let start_coord = AxialCoord::new(0, 0);
//! let goal_coord = AxialCoord::new(2, -1);
//! let path = lattice.find_path(start_coord, goal_coord);
//! let neighbors = start_coord.neighbors_within(2);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use std::collections::HashSet;

/// Hexagonal coordinate in axial system (q, r)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AxialCoord {
    pub q: i32,
    pub r: i32,
}

impl AxialCoord {
    /// Create a new axial coordinate
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Convert to cube coordinates
    pub fn to_cube(&self) -> CubeCoord {
        let x = self.q;
        let z = self.r;
        let y = -x - z;
        CubeCoord { x, y, z }
    }

    /// Get distance to another axial coordinate
    pub fn distance(&self, other: &AxialCoord) -> i32 {
        let cube_self = self.to_cube();
        let cube_other = other.to_cube();
        cube_self.distance(&cube_other)
    }

    /// Get all neighbors within given distance
    pub fn neighbors_within(&self, distance: i32) -> Vec<AxialCoord> {
        let mut neighbors = Vec::new();
        for dq in -distance..=distance {
            let r1 = (-distance).max(dq - distance);
            let r2 = distance.min(dq + distance);
            for dr in r1..=r2 {
                let coord = AxialCoord::new(self.q + dq, self.r + dr);
                if coord != *self {
                    neighbors.push(coord);
                }
            }
        }
        neighbors
    }
}

/// Hexagonal coordinate in cube system (x, y, z)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CubeCoord {
    /// Create a new cube coordinate
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert to axial coordinates
    pub fn to_axial(&self) -> AxialCoord {
        AxialCoord::new(self.x, self.z)
    }

    /// Get distance to another cube coordinate
    pub fn distance(&self, other: &CubeCoord) -> i32 {
        ((self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()) / 2
    }

    /// Get the six neighboring coordinates
    pub fn neighbors(&self) -> [CubeCoord; 6] {
        [
            CubeCoord::new(self.x + 1, self.y - 1, self.z),
            CubeCoord::new(self.x + 1, self.y, self.z - 1),
            CubeCoord::new(self.x, self.y + 1, self.z - 1),
            CubeCoord::new(self.x - 1, self.y + 1, self.z),
            CubeCoord::new(self.x - 1, self.y, self.z + 1),
            CubeCoord::new(self.x, self.y - 1, self.z + 1),
        ]
    }

    /// Get coordinates in a ring around this position
    pub fn ring(&self, radius: i32) -> Vec<CubeCoord> {
        if radius == 0 {
            return vec![*self];
        }

        let mut results = Vec::new();
        // Start at cube coordinate when stepping from direction 4 scaled by radius
        // direction 4 corresponds to (-1, 0, +1)
        let mut coord = CubeCoord::new(self.x - radius, self.y, self.z + radius);

        for direction in 0..6 {
            for _ in 0..radius {
                results.push(coord);
                coord = coord.neighbor_direction(direction);
            }
        }

        results
    }

    /// Get neighbor in a specific direction (0-5)
    fn neighbor_direction(&self, direction: i32) -> CubeCoord {
        match direction % 6 {
            0 => CubeCoord::new(self.x + 1, self.y - 1, self.z),
            1 => CubeCoord::new(self.x + 1, self.y, self.z - 1),
            2 => CubeCoord::new(self.x, self.y + 1, self.z - 1),
            3 => CubeCoord::new(self.x - 1, self.y + 1, self.z),
            4 => CubeCoord::new(self.x - 1, self.y, self.z + 1),
            5 => CubeCoord::new(self.x, self.y - 1, self.z + 1),
            _ => unreachable!(),
        }
    }
}

/// Hexagonal lattice with operations
pub struct HexLattice {
    pub radius: i32,
    pub coords: HashSet<AxialCoord>,
}

impl HexLattice {
    /// Create a new hexagonal lattice with given radius
    pub fn new(radius: i32) -> GsaResult<Self> {
        if radius < 0 {
            return Err(crate::GsaError::Geometry(
                "Radius must be non-negative".to_string(),
            ));
        }

        let mut coords = HashSet::new();
        for q in -radius..=radius {
            let r1 = (-radius).max(-q - radius);
            let r2 = radius.min(-q + radius);
            for r in r1..=r2 {
                coords.insert(AxialCoord::new(q, r));
            }
        }

        Ok(Self { radius, coords })
    }

    /// Check if coordinate is within the lattice
    pub fn contains(&self, coord: &AxialCoord) -> bool {
        self.coords.contains(coord)
    }

    /// Get all coordinates in the lattice
    pub fn all_coords(&self) -> Vec<AxialCoord> {
        self.coords.iter().cloned().collect()
    }

    /// Find shortest path between two coordinates using A*
    pub fn find_path(&self, start: AxialCoord, goal: AxialCoord) -> Option<Vec<AxialCoord>> {
        if !self.contains(&start) || !self.contains(&goal) {
            return None;
        }

        use std::collections::BinaryHeap;

        #[derive(Clone, PartialEq, Eq)]
        struct Node {
            coord: AxialCoord,
            cost: i32,
            estimate: i32,
        }

        impl Ord for Node {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                (self.cost + self.estimate)
                    .cmp(&(other.cost + other.estimate))
                    .reverse()
            }
        }

        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut open_set = BinaryHeap::new();
        let mut came_from = std::collections::HashMap::new();
        let mut g_score = std::collections::HashMap::new();

        open_set.push(Node {
            coord: start,
            cost: 0,
            estimate: start.distance(&goal),
        });
        g_score.insert(start, 0);

        while let Some(current) = open_set.pop() {
            if current.coord == goal {
                // Reconstruct path
                let mut path = vec![goal];
                let mut current_coord = goal;
                while current_coord != start {
                    current_coord = came_from[&current_coord];
                    path.push(current_coord);
                }
                path.reverse();
                return Some(path);
            }

            let cube_current = current.coord.to_cube();
            for neighbor_cube in cube_current.neighbors() {
                let neighbor = neighbor_cube.to_axial();
                if !self.contains(&neighbor) {
                    continue;
                }

                let tentative_g_score = g_score[&current.coord] + 1;
                if tentative_g_score < *g_score.get(&neighbor).unwrap_or(&i32::MAX) {
                    came_from.insert(neighbor, current.coord);
                    g_score.insert(neighbor, tentative_g_score);

                    open_set.push(Node {
                        coord: neighbor,
                        cost: tentative_g_score,
                        estimate: neighbor.distance(&goal),
                    });
                }
            }
        }

        None
    }

    /// Calculate geometric center of the lattice
    pub fn center(&self) -> AxialCoord {
        AxialCoord::new(0, 0)
    }

    /// Get coordinates on the boundary of the lattice
    pub fn boundary(&self) -> Vec<AxialCoord> {
        let mut boundary = Vec::new();
        for coord in &self.coords {
            let cube = coord.to_cube();
            let neighbors = cube.neighbors();
            let has_missing_neighbor = neighbors.iter().any(|n| !self.contains(&n.to_axial()));

            if has_missing_neighbor {
                boundary.push(*coord);
            }
        }
        boundary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axial_to_cube_conversion() {
        let axial = AxialCoord::new(1, 2);
        let cube = axial.to_cube();
        assert_eq!(cube.x, 1);
        assert_eq!(cube.y, -3);
        assert_eq!(cube.z, 2);

        let back_to_axial = cube.to_axial();
        assert_eq!(back_to_axial, axial);
    }

    #[test]
    fn test_distance_calculation() {
        let a = AxialCoord::new(0, 0);
        let b = AxialCoord::new(3, -2);
        assert_eq!(a.distance(&b), 3);
        assert_eq!(b.distance(&a), 3);
    }

    #[test]
    fn test_hex_lattice_creation() {
        let lattice = HexLattice::new(2).unwrap();
        assert_eq!(lattice.radius, 2);

        // Radius 2 should have 19 hexes (1 + 6 + 12)
        assert_eq!(lattice.coords.len(), 19);

        // Check that center is included
        assert!(lattice.contains(&AxialCoord::new(0, 0)));
    }

    #[test]
    fn test_cube_neighbors() {
        let center = CubeCoord::new(0, 0, 0);
        let neighbors = center.neighbors();

        // Should have exactly 6 neighbors
        assert_eq!(neighbors.len(), 6);

        // All neighbors should be distance 1 from center
        for neighbor in &neighbors {
            assert_eq!(center.distance(neighbor), 1);
        }
    }

    #[test]
    fn test_ring_generation() {
        let center = CubeCoord::new(0, 0, 0);

        // Ring 0 should just be the center
        let ring0 = center.ring(0);
        println!("DEBUG ring0: len = {}: {:?}", ring0.len(), ring0);
        assert_eq!(ring0.len(), 1);
        assert_eq!(ring0[0], center);

        // Ring 1 should have 6 hexes
        let ring1 = center.ring(1);
        println!("DEBUG ring1: len = {}: {:?}", ring1.len(), ring1);
        assert_eq!(ring1.len(), 6);

        // All should be distance 1 from center
        for coord in &ring1 {
            assert_eq!(center.distance(coord), 1);
        }
    }

    #[test]
    fn test_pathfinding() {
        let lattice = HexLattice::new(5).unwrap();

        let start = AxialCoord::new(0, 0);
        let goal = AxialCoord::new(2, 1);

        let path = lattice.find_path(start, goal);
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));

        // Path should be connected
        for i in 0..path.len() - 1 {
            assert_eq!(path[i].distance(&path[i + 1]), 1);
        }
    }

    #[test]
    fn test_boundary_detection() {
        let lattice = HexLattice::new(1).unwrap();
        let boundary = lattice.boundary();

        // Radius 1 boundary should have 6 hexes
        assert_eq!(boundary.len(), 6);

        // All boundary hexes should be distance 1 from center
        let center = lattice.center();
        for coord in &boundary {
            assert_eq!(center.distance(coord), 1);
        }
    }
}
