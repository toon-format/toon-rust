//! Pathfinding and topological operations through geometric spaces
//!
//! # WORMPATH MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Advanced worm-like pathfinding algorithms for geometric topology navigation,
//! implementing A* search, breadth-first exploration, and multi-agent swarm
//! intelligence through complex geometric cognition spaces.
//!
//! ### Key Capabilities
//! - **A* Pathfinding:** Optimal path computation with heuristic guidance.
//! - **Breadth-First Search:** Guaranteed shortest path in unweighted topologies.
//! - **Swarm Exploration:** Multi-agent cooperative topology mapping.
//! - **Arbitrary Topologies:** Grid, hexagonal, and custom graph structures.
//! - **Cost Function Integration:** Configurable path costs and traversal metrics.
//!
//! ### Technical Features
//! - **Heuristic Optimization:** Euclidean distance and generic heuristics.
//! - **Exploration Budgeting:** Configurable exploration depth limits.
//! - **Graph Representations:** Adjacency matrix and topology abstractions.
//! - **Coverage Analysis:** Exploration efficiency and completeness metrics.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::worm::{WormPathfinder, Topology};
//!
//! let topology = Topology::grid(10, 10);
//! let finder = WormPathfinder::new(topology);
//! let start = vec![0.0, 0.0];
//! let goal = vec![9.0, 9.0];
//! let path = finder.find_path_astar(&start, &goal).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Type alias for heuristic function to reduce complexity
type HeuristicFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;

/// Type alias for cost function to reduce complexity
type CostFunction = Box<dyn Fn(&[f64], &[f64]) -> f64>;

/// Pathfinding result containing the path and metadata
#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Vec<Vec<f64>>,
    pub cost: f64,
    pub nodes_explored: usize,
    pub path_length: usize,
}

/// Worm pathfinder with topological awareness
pub struct WormPathfinder {
    pub topology: Topology,
    pub heuristic: HeuristicFn,
    pub cost_function: CostFunction,
}

impl WormPathfinder {
    /// Create a new worm pathfinder
    pub fn new(topology: Topology) -> Self {
        Self {
            topology,
            heuristic: Box::new(euclidean_distance),
            cost_function: Box::new(euclidean_distance),
        }
    }

    /// Find path using A* algorithm
    pub fn find_path_astar(&self, start: &[f64], goal: &[f64]) -> GsaResult<PathResult> {
        if !self.topology.contains(start) || !self.topology.contains(goal) {
            return Err(crate::GsaError::Geometry(
                "Start or goal not in topology".to_string(),
            ));
        }

        let mut open_set = BinaryHeap::new();
        let mut came_from = HashMap::new();
        let mut g_score = HashMap::new();
        let mut f_score = HashMap::new();
        let mut closed_set = HashSet::new();

        let start_key = self.position_to_key(start);
        let goal_key = self.position_to_key(goal);

        open_set.push(Reverse(Node {
            position: start.to_vec(),
            f_score: (self.heuristic)(start, goal),
        }));

        g_score.insert(start_key.clone(), 0.0);
        f_score.insert(start_key.clone(), (self.heuristic)(start, goal));

        let mut nodes_explored = 0;

        while let Some(Reverse(current)) = open_set.pop() {
            nodes_explored += 1;
            let current_key = self.position_to_key(&current.position);

            if current_key == goal_key {
                // Reconstruct path
                let path = self.reconstruct_path(&came_from, &current.position, start);
                let path_length = path.len();
                let cost = g_score[&current_key];
                return Ok(PathResult {
                    path,
                    cost,
                    nodes_explored,
                    path_length,
                });
            }

            closed_set.insert(current_key.clone());

            // Explore neighbors
            for neighbor in self.topology.get_neighbors(&current.position) {
                let neighbor_key = self.position_to_key(&neighbor);

                if closed_set.contains(&neighbor_key) {
                    continue;
                }

                let tentative_g_score =
                    g_score[&current_key] + (self.cost_function)(&current.position, &neighbor);

                if tentative_g_score < *g_score.get(&neighbor_key).unwrap_or(&f64::INFINITY) {
                    came_from.insert(neighbor_key.clone(), current.position.clone());
                    g_score.insert(neighbor_key.clone(), tentative_g_score);
                    let f_score_value = tentative_g_score + (self.heuristic)(&neighbor, goal);
                    f_score.insert(neighbor_key.clone(), f_score_value);

                    // Remove old entry if it exists and insert new one
                    open_set.retain(|Reverse(n)| self.position_to_key(&n.position) != neighbor_key);
                    open_set.push(Reverse(Node {
                        position: neighbor,
                        f_score: f_score_value,
                    }));
                }
            }
        }

        Err(crate::GsaError::Geometry("No path found".to_string()))
    }

    /// Find path using breadth-first search
    pub fn find_path_bfs(&self, start: &[f64], goal: &[f64]) -> GsaResult<PathResult> {
        if !self.topology.contains(start) || !self.topology.contains(goal) {
            return Err(crate::GsaError::Geometry(
                "Start or goal not in topology".to_string(),
            ));
        }

        let mut queue = VecDeque::new();
        let mut came_from = HashMap::new();
        let mut visited = HashSet::new();

        let start_key = self.position_to_key(start);
        let goal_key = self.position_to_key(goal);

        queue.push_back(start.to_vec());
        visited.insert(start_key.clone());
        came_from.insert(start_key, start.to_vec()); // Self-reference for start

        let mut nodes_explored = 0;

        while let Some(current) = queue.pop_front() {
            nodes_explored += 1;
            let current_key = self.position_to_key(&current);

            if current_key == goal_key {
                let path = self.reconstruct_path(&came_from, &current, start);
                let path_length = path.len();
                return Ok(PathResult {
                    path,
                    cost: path_length as f64, // BFS uses step count as cost
                    nodes_explored,
                    path_length,
                });
            }

            for neighbor in self.topology.get_neighbors(&current) {
                let neighbor_key = self.position_to_key(&neighbor);

                if !visited.contains(&neighbor_key) {
                    visited.insert(neighbor_key.clone());
                    queue.push_back(neighbor.clone());
                    came_from.insert(neighbor_key, current.clone());
                }
            }
        }

        Err(crate::GsaError::Geometry("No path found".to_string()))
    }

    /// Explore topology with worm-like movement
    pub fn explore_topology(&self, start: &[f64], exploration_budget: usize) -> ExplorationResult {
        let mut visited = HashSet::new();
        let mut path = vec![start.to_vec()];
        let mut current = start.to_vec();
        let mut total_cost = 0.0;

        visited.insert(self.position_to_key(start));

        for _ in 0..exploration_budget {
            let neighbors = self.topology.get_neighbors(&current);
            let unvisited: Vec<_> = neighbors
                .into_iter()
                .filter(|n| !visited.contains(&self.position_to_key(n)))
                .collect();

            if unvisited.is_empty() {
                break; // No more places to explore
            }

            // Choose next position (could be made more sophisticated)
            let next = &unvisited[0];
            let step_cost = (self.cost_function)(&current, next);

            path.push(next.clone());
            visited.insert(self.position_to_key(next));
            total_cost += step_cost;
            current = next.clone();
        }

        ExplorationResult {
            path,
            visited_positions: visited.len(),
            total_cost,
            coverage_ratio: visited.len() as f64 / self.topology.size() as f64,
        }
    }

    /// Helper to convert position to hashable key
    fn position_to_key(&self, position: &[f64]) -> String {
        position
            .iter()
            .map(|&x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Reconstruct path from came_from map
    fn reconstruct_path(
        &self,
        came_from: &HashMap<String, Vec<f64>>,
        current: &[f64],
        start: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut path = vec![current.to_vec()];
        let mut current_pos = current.to_vec();

        while current_pos != start {
            let key = self.position_to_key(&current_pos);
            if let Some(prev) = came_from.get(&key) {
                path.push(prev.clone());
                current_pos = prev.clone();
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

/// Node for A* priority queue
#[derive(Debug, Clone, PartialEq)]
struct Node {
    position: Vec<f64>,
    f_score: f64,
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.f_score
            .partial_cmp(&other.f_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Topology representation
#[derive(Debug, Clone)]
pub struct Topology {
    pub positions: Vec<Vec<f64>>,
    pub connections: HashMap<String, Vec<String>>, // Position key -> neighbor keys
    pub dimensions: usize,
}

impl Topology {
    /// Create empty topology
    pub fn new(dimensions: usize) -> Self {
        Self {
            positions: Vec::new(),
            connections: HashMap::new(),
            dimensions,
        }
    }

    /// Add position to topology
    pub fn add_position(&mut self, position: Vec<f64>) {
        if position.len() != self.dimensions {
            return; // Invalid dimension
        }
        self.positions.push(position);
        // Ensure the connections map contains an entry for this position
        let key = self.position_to_key(self.positions.last().unwrap());
        self.connections.entry(key).or_default();
    }

    /// Add connection between positions
    pub fn add_connection(&mut self, pos1: &[f64], pos2: &[f64]) {
        let key1 = self.position_to_key(pos1);
        let key2 = self.position_to_key(pos2);
        // Ensure both endpoints have entries in connections map
        self.connections.entry(key1.clone()).or_default();
        self.connections.entry(key2.clone()).or_default();

        // Add neighbor links only if not already present
        if !self
            .connections
            .get(&key1)
            .map(|v| v.contains(&key2))
            .unwrap_or(false)
        {
            self.connections.get_mut(&key1).unwrap().push(key2.clone());
        }

        if !self
            .connections
            .get(&key2)
            .map(|v| v.contains(&key1))
            .unwrap_or(false)
        {
            self.connections.get_mut(&key2).unwrap().push(key1);
        }
    }

    /// Check if position exists in topology
    pub fn contains(&self, position: &[f64]) -> bool {
        let key = self.position_to_key(position);
        self.connections.contains_key(&key)
    }

    /// Get neighbors of a position
    pub fn get_neighbors(&self, position: &[f64]) -> Vec<Vec<f64>> {
        let key = self.position_to_key(position);
        self.connections
            .get(&key)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|k| self.key_to_position(k))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all positions
    pub fn get_positions(&self) -> &[Vec<f64>] {
        &self.positions
    }

    /// Get topology size
    pub fn size(&self) -> usize {
        self.connections.len()
    }

    /// Get the number of independent connections (edges) in the topology
    pub fn connection_count(&self) -> usize {
        // connections map stores bi-directional entries, so divide the sum by 2
        let total_entries: usize = self.connections.values().map(|v| v.len()).sum();
        total_entries / 2
    }

    /// Accessor for topology dimensionality
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Create grid topology
    pub fn grid(width: usize, height: usize) -> Self {
        let mut topology = Self::new(2);

        // Add all positions
        for y in 0..height {
            for x in 0..width {
                topology.add_position(vec![x as f64, y as f64]);
            }
        }

        // Add connections (4-way connectivity)
        for y in 0..height {
            for x in 0..width {
                let current = vec![x as f64, y as f64];

                // Right neighbor
                if x + 1 < width {
                    topology.add_connection(&current, &[x as f64 + 1.0, y as f64]);
                }

                // Down neighbor
                if y + 1 < height {
                    topology.add_connection(&current, &[x as f64, y as f64 + 1.0]);
                }

                // Left neighbor
                if x > 0 {
                    topology.add_connection(&current, &[x as f64 - 1.0, y as f64]);
                }

                // Up neighbor
                if y > 0 {
                    topology.add_connection(&current, &[x as f64, y as f64 - 1.0]);
                }
            }
        }

        topology
    }

    /// Create hexagonal topology
    pub fn hexagonal(radius: i32) -> Self {
        use crate::hex::HexLattice;

        let hex_lattice = HexLattice::new(radius).unwrap();
        let mut topology = Self::new(2);

        // Convert axial coordinates to 2D positions
        for coord in hex_lattice.all_coords() {
            let _cube = coord.to_cube();
            // Convert cube to 2D using axial coordinates
            topology.add_position(vec![coord.q as f64, coord.r as f64]);
        }

        // Add connections based on hex adjacency
        for coord in hex_lattice.all_coords() {
            let pos1 = vec![coord.q as f64, coord.r as f64];
            let cube = coord.to_cube();

            for neighbor_cube in cube.neighbors() {
                let neighbor_axial = neighbor_cube.to_axial();
                let pos2 = vec![neighbor_axial.q as f64, neighbor_axial.r as f64];

                if topology.contains(&pos2) {
                    topology.add_connection(&pos1, &pos2);
                }
            }
        }

        topology
    }

    /// Helper to convert position to key
    fn position_to_key(&self, position: &[f64]) -> String {
        position
            .iter()
            .map(|&x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Helper to convert key back to position
    fn key_to_position(&self, key: &str) -> Option<Vec<f64>> {
        key.split(',')
            .map(|s| s.parse::<f64>().ok())
            .collect::<Option<Vec<_>>>()
    }
}

/// Result of topology exploration
#[derive(Debug, Clone)]
pub struct ExplorationResult {
    pub path: Vec<Vec<f64>>,
    pub visited_positions: usize,
    pub total_cost: f64,
    pub coverage_ratio: f64,
}

/// Worm swarm for multi-agent exploration
pub struct WormSwarm {
    pub worms: Vec<WormPathfinder>,
    pub communication_range: f64,
}

impl WormSwarm {
    /// Create new worm swarm
    pub fn new(worm_count: usize, topology: Topology, communication_range: f64) -> Self {
        let worms = (0..worm_count)
            .map(|_| WormPathfinder::new(topology.clone()))
            .collect();

        Self {
            worms,
            communication_range,
        }
    }

    /// Collaborative exploration
    pub fn collaborative_explore(&mut self, starts: Vec<Vec<f64>>, budget: usize) -> SwarmResult {
        let mut total_path_lengths = 0;
        let mut total_coverage = HashSet::new();
        let mut all_paths = Vec::new();

        for (i, start) in starts.into_iter().enumerate() {
            if let Some(worm) = self.worms.get_mut(i) {
                let exploration = worm.explore_topology(&start, budget);
                total_path_lengths += exploration.path.len();
                total_coverage.extend(exploration.path.iter().map(|p| worm.position_to_key(p)));
                all_paths.push(exploration.path);
            }
        }

        SwarmResult {
            total_path_lengths,
            unique_positions_covered: total_coverage.len(),
            coverage_efficiency: total_coverage.len() as f64 / total_path_lengths as f64,
            individual_paths: all_paths,
        }
    }
}

/// Result of swarm exploration
#[derive(Debug, Clone)]
pub struct SwarmResult {
    pub total_path_lengths: usize,
    pub unique_positions_covered: usize,
    pub coverage_efficiency: f64,
    pub individual_paths: Vec<Vec<Vec<f64>>>,
}

/// Utility function for Euclidean distance
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_topology_creation() {
        let topology = Topology::grid(3, 3);

        // Should have 9 positions
        assert_eq!(topology.size(), 9);

        // Center position should have 4 neighbors
        let center = vec![1.0, 1.0];
        assert!(topology.contains(&center));
        assert_eq!(topology.get_neighbors(&center).len(), 4);
    }

    #[test]
    fn test_hexagonal_topology_creation() {
        let topology = Topology::hexagonal(2);

        // Radius 2 should have 19 positions
        assert_eq!(topology.size(), 19);

        // Positions should exist
        assert!(topology.contains(&[0.0, 0.0]));
    }

    #[test]
    fn test_bfs_pathfinding() {
        let topology = Topology::grid(5, 5);
        let pathfinder = WormPathfinder::new(topology);

        let start = vec![0.0, 0.0];
        let goal = vec![4.0, 4.0];

        let result = pathfinder.find_path_bfs(&start, &goal).unwrap();

        // Should find a path
        assert!(!result.path.is_empty());
        assert_eq!(result.path.first(), Some(&start));
        assert_eq!(result.path.last(), Some(&goal));

        // Path should be connected
        for i in 0..result.path.len() - 1 {
            let dist = euclidean_distance(&result.path[i], &result.path[i + 1]);
            assert!((dist - 1.0).abs() < 1e-10); // Grid steps are 1.0
        }
    }

    #[test]
    fn test_astar_pathfinding() {
        let topology = Topology::grid(5, 5);
        let pathfinder = WormPathfinder::new(topology);

        let start = vec![0.0, 0.0];
        let goal = vec![4.0, 4.0];

        let result = pathfinder.find_path_astar(&start, &goal).unwrap();

        // Should find a path
        assert!(!result.path.is_empty());
        assert_eq!(result.path.first(), Some(&start));
        assert_eq!(result.path.last(), Some(&goal));
    }

    #[test]
    fn test_topology_exploration() {
        let topology = Topology::grid(4, 4);
        let pathfinder = WormPathfinder::new(topology);

        let start = vec![0.0, 0.0];
        let result = pathfinder.explore_topology(&start, 10);

        // Should have explored some positions
        assert!(!result.path.is_empty());
        assert!(result.visited_positions > 1);
        assert!(result.coverage_ratio > 0.0);
    }

    #[test]
    fn test_worm_swarm() {
        let topology = Topology::grid(5, 5);
        let mut swarm = WormSwarm::new(2, topology, 2.0);

        let starts = vec![vec![0.0, 0.0], vec![4.0, 4.0]];
        let result = swarm.collaborative_explore(starts, 5);

        // Should have explored positions
        assert!(result.total_path_lengths > 0);
        assert!(result.unique_positions_covered > 0);
        assert_eq!(result.individual_paths.len(), 2);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let distance = euclidean_distance(&a, &b);
        assert!((distance - 5.0).abs() < 1e-10);
    }
}
