//! Lorentzian Geometry Layer - Spacetime Metrics and Geodesics
//!
//! Implements Lorentzian (pseudo-Riemannian) geometry with signature (-,+,+,+,+,+,+,+).
//! Core mathematical operations:
//! - Minkowski metric: ds² = -dt² + dx₁² + dx₂² + ... + dx₇²
//! - Proper time along timelike worldlines
//! - Geodesic computations
//! - Light cone structure (null surfaces)
//! - Lorentz transformations and boosts
//! - Causal relationships (timelike, spacelike, lightlike separation)
//!
//! Extension: Causal DAG for event ordering (optional, game-specific logic)
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet, VecDeque};

/// Spacetime point in 7+1 dimensional Lorentzian manifold
#[derive(Clone, Debug, PartialEq)]
pub struct SpacetimePoint {
    /// Coordinates [t, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
    pub coords: [f64; 8],
}

impl SpacetimePoint {
    /// Create new spacetime point
    pub fn new(coords: [f64; 8]) -> Self {
        Self { coords }
    }

    /// Time coordinate
    pub fn time(&self) -> f64 {
        self.coords[0]
    }

    /// Spatial coordinates as slice
    pub fn spatial(&self) -> &[f64] {
        &self.coords[1..8]
    }

    /// Compute Minkowski metric interval to another point
    /// ds² = -dt² + Σ(dxᵢ²)
    pub fn minkowski_interval(&self, other: &SpacetimePoint) -> f64 {
        let dt = self.coords[0] - other.coords[0];
        let spatial_dist_sq: f64 = (1..8)
            .map(|i| {
                let dx = self.coords[i] - other.coords[i];
                dx * dx
            })
            .sum();

        -dt * dt + spatial_dist_sq
    }

    /// Compute proper time (for timelike curves)
    /// τ = √(-ds²) if ds² < 0
    pub fn proper_time(&self, other: &SpacetimePoint) -> Option<f64> {
        let interval = self.minkowski_interval(other);
        if interval < 0.0 {
            Some((-interval).sqrt())
        } else {
            None // Not timelike separated
        }
    }

    /// Check if two points are timelike separated (ds² < 0)
    pub fn is_timelike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other) < 0.0
    }

    /// Check if two points are spacelike separated (ds² > 0)
    pub fn is_spacelike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other) > 0.0
    }

    /// Check if two points are lightlike/null separated (ds² = 0)
    pub fn is_lightlike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other).abs() < 1e-10
    }

    /// Determine causal relationship based on interval and time ordering
    pub fn causal_relation(&self, other: &SpacetimePoint) -> CausalRelation {
        let interval = self.minkowski_interval(other);
        let dt = self.coords[0] - other.coords[0];

        if interval < -1e-10 {
            if dt > 0.0 {
                CausalRelation::Future
            } else {
                CausalRelation::Past
            }
        } else if interval > 1e-10 {
            CausalRelation::Spacelike
        } else {
            // Lightlike
            if dt > 0.0 {
                CausalRelation::LightlikeFuture
            } else if dt < 0.0 {
                CausalRelation::LightlikePast
            } else {
                CausalRelation::Coincident
            }
        }
    }
}

/// Causal relationship between two spacetime points
#[derive(Clone, Debug, PartialEq)]
pub enum CausalRelation {
    /// Timelike future (this event is in the causal future)
    Future,
    /// Timelike past (this event is in the causal past)
    Past,
    /// Spacelike separated (no causal relationship)
    Spacelike,
    /// Null/lightlike future
    LightlikeFuture,
    /// Null/lightlike past
    LightlikePast,
    /// Same event
    Coincident,
}

/// Worldline - timelike curve through spacetime
#[derive(Clone, Debug)]
pub struct Worldline {
    /// Points along the worldline (must be timelike connected)
    pub points: Vec<SpacetimePoint>,
}

impl Worldline {
    /// Create new worldline
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add point to worldline (validates timelike connection)
    pub fn add_point(&mut self, point: SpacetimePoint) -> Result<(), &'static str> {
        if let Some(last) = self.points.last() {
            if !last.is_timelike(&point) || point.time() <= last.time() {
                return Err("Point must be timelike future-connected to previous point");
            }
        }
        self.points.push(point);
        Ok(())
    }

    /// Compute total proper time along worldline
    pub fn proper_time(&self) -> f64 {
        let mut tau = 0.0;
        for i in 1..self.points.len() {
            if let Some(dtau) = self.points[i - 1].proper_time(&self.points[i]) {
                tau += dtau;
            }
        }
        tau
    }

    /// Get velocity 4-vector at index (forward difference)
    pub fn four_velocity(&self, index: usize) -> Option<[f64; 8]> {
        if index + 1 >= self.points.len() {
            return None;
        }

        let p1 = &self.points[index];
        let p2 = &self.points[index + 1];

        let dtau = p1.proper_time(p2)?;
        let mut v = [0.0; 8];

        for i in 0..8 {
            v[i] = (p2.coords[i] - p1.coords[i]) / dtau;
        }

        Some(v)
    }
}

impl Default for Worldline {
    fn default() -> Self {
        Self::new()
    }
}

/// Lorentzian geometry layer
pub struct LorentzianLayer {
    /// Worldlines tracked in this layer
    pub worldlines: Vec<Worldline>,

    /// Metric signature (-1, +1, +1, +1, +1, +1, +1, +1)
    pub signature: [i8; 8],
}

impl LorentzianLayer {
    /// Create new Lorentzian layer with standard signature
    pub fn new() -> Self {
        Self {
            worldlines: Vec::new(),
            signature: [-1, 1, 1, 1, 1, 1, 1, 1],
        }
    }

    /// Add worldline to layer
    pub fn add_worldline(&mut self, worldline: Worldline) {
        self.worldlines.push(worldline);
    }

    /// Compute geodesic distance between two points (proper time for timelike)
    pub fn geodesic_distance(&self, p1: &SpacetimePoint, p2: &SpacetimePoint) -> Option<f64> {
        p1.proper_time(p2)
    }

    /// Check if point is in past light cone of another
    pub fn in_past_light_cone(&self, point: &SpacetimePoint, reference: &SpacetimePoint) -> bool {
        let interval = point.minkowski_interval(reference);
        let dt = reference.time() - point.time();
        interval <= 0.0 && dt > 0.0
    }

    /// Check if point is in future light cone of another
    pub fn in_future_light_cone(&self, point: &SpacetimePoint, reference: &SpacetimePoint) -> bool {
        let interval = point.minkowski_interval(reference);
        let dt = point.time() - reference.time();
        interval <= 0.0 && dt > 0.0
    }

    /// Lorentz boost along x₁ axis
    /// γ = 1/√(1 - v²)
    pub fn lorentz_boost(&self, point: &SpacetimePoint, velocity: f64) -> SpacetimePoint {
        if velocity.abs() >= 1.0 {
            return point.clone(); // Invalid velocity
        }

        let gamma = 1.0 / (1.0 - velocity * velocity).sqrt();
        let mut boosted = point.coords;

        let t = point.coords[0];
        let x = point.coords[1];

        boosted[0] = gamma * (t - velocity * x);
        boosted[1] = gamma * (x - velocity * t);

        SpacetimePoint::new(boosted)
    }
}

impl Default for LorentzianLayer {
    fn default() -> Self {
        Self::new()
    }
}

//====================================================================================
// EXTENSION: CAUSAL DAG (Game-Specific Event Ordering)
//====================================================================================

/// Event types in the causal graph (game/application specific)
#[derive(Clone, Debug, PartialEq)]
pub enum EventType {
    /// Movement between E8 roots
    Move { from_root: usize, to_root: usize },

    /// Combat encounter
    Combat { monster_id: String, outcome: bool },

    /// Death event
    Death { cause: String },

    /// Concept emergence
    Emergence { concept_id: u64 },

    /// Goal transition
    GoalSwitch { old_root: usize, new_root: usize },

    /// Healing event
    Heal { amount: f32 },

    /// Trap trigger
    Trap { damage: f32 },

    /// Generic action
    Action { description: String },
}

/// Causal node representing an event in spacetime (extends SpacetimePoint)
#[derive(Clone, Debug)]
pub struct CausalNode {
    /// Spacetime location
    pub location: SpacetimePoint,

    /// Unique event identifier
    pub event_id: u64,

    /// Spatial position (E8 root index, application-specific)
    pub e8_root: usize,

    /// Event type (application-specific)
    pub event_type: EventType,
}

impl CausalNode {
    /// Create new causal node
    pub fn new(
        event_id: u64,
        location: SpacetimePoint,
        e8_root: usize,
        event_type: EventType,
    ) -> Self {
        Self {
            location,
            event_id,
            e8_root,
            event_type,
        }
    }

    /// Compute spacetime interval to another event (delegates to SpacetimePoint)
    pub fn spacetime_interval(&self, other: &CausalNode) -> f64 {
        self.location.minkowski_interval(&other.location)
    }

    /// Check if this event is in the causal future of another
    pub fn is_causally_after(&self, other: &CausalNode) -> bool {
        matches!(
            self.location.causal_relation(&other.location),
            CausalRelation::Future | CausalRelation::LightlikeFuture
        )
    }

    /// Check if this event is spacelike separated from another
    pub fn is_spacelike_separated(&self, other: &CausalNode) -> bool {
        self.location.is_spacelike(&other.location)
    }
}

/// Causal directed acyclic graph (DAG) - Extension for event ordering
#[derive(Clone, Debug)]
pub struct CausalDAG {
    /// All events in the causal history
    pub nodes: Vec<CausalNode>,

    /// Causal edges: (cause_id, effect_id)
    pub edges: Vec<(u64, u64)>,

    /// Adjacency list for efficient traversal: event_id -> [effect_ids]
    adjacency: HashMap<u64, Vec<u64>>,

    /// Reverse adjacency: event_id -> [cause_ids]
    reverse_adjacency: HashMap<u64, Vec<u64>>,
}

impl CausalDAG {
    /// Create empty causal DAG
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Add event with causal dependencies
    pub fn add_event(&mut self, node: CausalNode, causes: &[u64]) -> u64 {
        let event_id = node.event_id;

        // Add node
        self.nodes.push(node);

        // Add edges
        for &cause_id in causes {
            self.edges.push((cause_id, event_id));

            // Update adjacency lists
            self.adjacency.entry(cause_id).or_default().push(event_id);

            self.reverse_adjacency
                .entry(event_id)
                .or_default()
                .push(cause_id);
        }

        event_id
    }

    /// Get past light cone (all causal ancestors)
    pub fn past_light_cone(&self, event_id: u64) -> Vec<u64> {
        let mut past = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(event_id);
        visited.insert(event_id);

        while let Some(current) = queue.pop_front() {
            if let Some(causes) = self.reverse_adjacency.get(&current) {
                for &cause_id in causes {
                    if !visited.contains(&cause_id) {
                        past.push(cause_id);
                        queue.push_back(cause_id);
                        visited.insert(cause_id);
                    }
                }
            }
        }

        past
    }

    /// Get future light cone (all causal descendants)
    pub fn future_light_cone(&self, event_id: u64) -> Vec<u64> {
        let mut future = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(event_id);
        visited.insert(event_id);

        while let Some(current) = queue.pop_front() {
            if let Some(effects) = self.adjacency.get(&current) {
                for &effect_id in effects {
                    if !visited.contains(&effect_id) {
                        future.push(effect_id);
                        queue.push_back(effect_id);
                        visited.insert(effect_id);
                    }
                }
            }
        }

        future
    }

    /// Check if event A is in causal past of event B
    pub fn is_causal_past(&self, a: u64, b: u64) -> bool {
        self.past_light_cone(b).contains(&a)
    }

    /// Check if event A is in causal future of event B
    pub fn is_causal_future(&self, a: u64, b: u64) -> bool {
        self.future_light_cone(b).contains(&a)
    }

    /// Find node by event ID
    pub fn get_node(&self, event_id: u64) -> Option<&CausalNode> {
        self.nodes.iter().find(|n| n.event_id == event_id)
    }

    /// Verify causal consistency (no cycles)
    pub fn verify_consistency(&self) -> bool {
        // Use topological sort to detect cycles
        let mut in_degree: HashMap<u64, usize> = HashMap::new();

        // Initialize in-degrees
        for node in &self.nodes {
            in_degree.insert(node.event_id, 0);
        }

        for &(_, effect) in &self.edges {
            *in_degree.get_mut(&effect).unwrap() += 1;
        }

        // Find all nodes with in-degree 0
        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted_count = 0;

        while let Some(node_id) = queue.pop_front() {
            sorted_count += 1;

            if let Some(effects) = self.adjacency.get(&node_id) {
                for &effect_id in effects {
                    let deg = in_degree.get_mut(&effect_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(effect_id);
                    }
                }
            }
        }

        // If all nodes were processed, no cycles exist
        sorted_count == self.nodes.len()
    }

    /// Get topological ordering of events
    pub fn topological_order(&self) -> Option<Vec<u64>> {
        if !self.verify_consistency() {
            return None;
        }

        let mut in_degree: HashMap<u64, usize> = HashMap::new();

        for node in &self.nodes {
            in_degree.insert(node.event_id, 0);
        }

        for &(_, effect) in &self.edges {
            *in_degree.get_mut(&effect).unwrap() += 1;
        }

        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);

            if let Some(effects) = self.adjacency.get(&node_id) {
                for &effect_id in effects {
                    let deg = in_degree.get_mut(&effect_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(effect_id);
                    }
                }
            }
        }

        Some(order)
    }
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined layer with both Lorentzian geometry and causal DAG extension
pub struct LorentzianCausalLayer {
    /// Core Lorentzian geometry
    pub geometry: LorentzianLayer,

    /// Causal DAG extension
    pub dag: CausalDAG,

    /// Current proper time
    pub proper_time: f64,

    /// Next event ID
    next_event_id: u64,
}

impl LorentzianCausalLayer {
    /// Create new combined layer
    pub fn new() -> Self {
        Self {
            geometry: LorentzianLayer::new(),
            dag: CausalDAG::new(),
            proper_time: 0.0,
            next_event_id: 0,
        }
    }

    /// Add event to causal graph
    pub fn add_event(&mut self, e8_root: usize, event_type: EventType, causes: &[u64]) -> u64 {
        let event_id = self.next_event_id;
        self.next_event_id += 1;
        self.proper_time += 1.0;

        // Create spacetime coordinates [t, x1, ..., x7]
        let mut coordinates = [0.0f64; 8];
        coordinates[0] = self.proper_time;
        // Spatial coordinates encode E8 root (simplified mapping)
        for (i, coord) in coordinates.iter_mut().enumerate().skip(1) {
            *coord = ((e8_root >> i) & 1) as f64;
        }

        let location = SpacetimePoint::new(coordinates);
        let node = CausalNode::new(event_id, location, e8_root, event_type);

        self.dag.add_event(node, causes);
        event_id
    }

    /// Get past light cone of an event
    pub fn past_light_cone(&self, event_id: u64) -> Vec<u64> {
        self.dag.past_light_cone(event_id)
    }

    /// Get future light cone of an event
    pub fn future_light_cone(&self, event_id: u64) -> Vec<u64> {
        self.dag.future_light_cone(event_id)
    }

    /// Check causal ordering
    pub fn is_causal_past(&self, a: u64, b: u64) -> bool {
        self.dag.is_causal_past(a, b)
    }

    /// Check causal future
    pub fn is_causal_future(&self, a: u64, b: u64) -> bool {
        self.dag.is_causal_future(a, b)
    }

    /// Verify causal consistency
    pub fn verify_consistency(&self) -> bool {
        self.dag.verify_consistency()
    }

    /// Get event by ID
    pub fn get_event(&self, event_id: u64) -> Option<&CausalNode> {
        self.dag.get_node(event_id)
    }
}

impl Default for LorentzianCausalLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Core Lorentzian geometry tests
    #[test]
    fn test_minkowski_interval() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let interval = p1.minkowski_interval(&p2);
        // ds² = -dt² + dx² = -(1)² + (1)² = 0 (lightlike)
        assert_eq!(interval, 0.0);
    }

    #[test]
    fn test_timelike_separation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(p1.is_timelike(&p2));
        assert_eq!(p1.proper_time(&p2), Some((3.0_f64).sqrt()));
    }

    #[test]
    fn test_spacelike_separation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(p1.is_spacelike(&p2));
        assert_eq!(p1.proper_time(&p2), None);
    }

    #[test]
    fn test_causal_relation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(p2.causal_relation(&p1), CausalRelation::Future);
        assert_eq!(p1.causal_relation(&p2), CausalRelation::Past);
    }

    #[test]
    fn test_worldline() {
        let mut wl = Worldline::new();

        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p3 = SpacetimePoint::new([4.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(wl.add_point(p1).is_ok());
        assert!(wl.add_point(p2).is_ok());
        assert!(wl.add_point(p3).is_ok());

        assert!(wl.proper_time() > 0.0);
    }

    #[test]
    fn test_lorentz_boost() {
        let layer = LorentzianLayer::new();
        let p = SpacetimePoint::new([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = 0.5;

        let boosted = layer.lorentz_boost(&p, v);

        // Check that time and x1 are transformed
        assert_ne!(boosted.coords[0], p.coords[0]);
        assert_ne!(boosted.coords[1], p.coords[1]);
        // Other coordinates unchanged
        assert_eq!(boosted.coords[2], p.coords[2]);
    }

    // Extension: Causal DAG tests
    #[test]
    fn test_lorentzian_causal_layer_creation() {
        let layer = LorentzianCausalLayer::new();
        assert_eq!(layer.proper_time, 0.0);
        assert_eq!(layer.next_event_id, 0);
    }

    #[test]
    fn test_add_event() {
        let mut layer = LorentzianCausalLayer::new();

        let event1 = layer.add_event(
            0,
            EventType::Action {
                description: "Start".to_string(),
            },
            &[],
        );
        assert_eq!(event1, 0);
        assert_eq!(layer.proper_time, 1.0);

        let event2 = layer.add_event(
            1,
            EventType::Move {
                from_root: 0,
                to_root: 1,
            },
            &[event1],
        );
        assert_eq!(event2, 1);
        assert_eq!(layer.proper_time, 2.0);
    }

    #[test]
    fn test_causal_ordering() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e2],
        );

        assert!(layer.is_causal_past(e1, e2));
        assert!(layer.is_causal_past(e1, e3));
        assert!(layer.is_causal_past(e2, e3));

        assert!(!layer.is_causal_past(e2, e1));
        assert!(!layer.is_causal_past(e3, e1));
    }

    #[test]
    fn test_light_cones() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "Root".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "Child1".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "Child2".to_string(),
            },
            &[e1],
        );
        let e4 = layer.add_event(
            3,
            EventType::Action {
                description: "Grandchild".to_string(),
            },
            &[e2, e3],
        );

        let past = layer.past_light_cone(e4);
        assert!(past.contains(&e1));
        assert!(past.contains(&e2));
        assert!(past.contains(&e3));

        let future = layer.future_light_cone(e1);
        assert!(future.contains(&e2));
        assert!(future.contains(&e3));
        assert!(future.contains(&e4));
    }

    #[test]
    fn test_causal_consistency() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let _e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e2],
        );

        assert!(layer.verify_consistency());
    }

    #[test]
    fn test_topological_order() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e1],
        );

        let order = layer.dag.topological_order().unwrap();

        // e1 must come before e2 and e3
        let pos1 = order.iter().position(|&x| x == e1).unwrap();
        let pos2 = order.iter().position(|&x| x == e2).unwrap();
        let pos3 = order.iter().position(|&x| x == e3).unwrap();

        assert!(pos1 < pos2);
        assert!(pos1 < pos3);
    }
}
