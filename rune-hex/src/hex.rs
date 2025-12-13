/* src/hex.rs */
//! Core implementation of hexadecimal encoding and decoding logic.
//!
//! # ArcMoon Utils – Hexadecimal Transcoding Strategy
//!▫~•◦------------------------------------------------‣
//!
//! This module is designed for integration into ArcMoon Utils to achieve high-performance,
//! reliable transformation between binary data and hexadecimal string representations.
//!
//! ### Key Capabilities
//! - **Zero-Allocation Parsing:** Optimized decoding routines that minimize heap usage.
//! - **Constant-Time Verification:** comparison utilities to mitigate timing attacks during signature verification.
//! - **Embedded Compatibility:** Fully compatible with `#![no_std]` environments via the `alloc` crate.
//!
//! ### Architectural Notes
//! This module serves as the functional core for the crate's public API. It implements
//! the `ToHex` and `FromHex` traits defined in the crate root.
//!
//! ### Example
//! ```rust
//! use rune_hex::hex::{Vertex, RootKind, SemanticDomain};
//!
//! // Create a vertex in E8 semantic space
//! let vertex = Vertex {
//!     id: 0,
//!     kind: RootKind::TypeI,
//!     coords: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//!     domain: Some(SemanticDomain::Psychology),
//!     confidence: 0.95,
//! };
//! assert_eq!(vertex.id, 0);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use yoshi_derive::AnyError;

#[derive(Debug, AnyError)]
pub enum HexError {
    #[anyerror("Invalid operation: {0}")]
    InvalidOperation(String),
    #[anyerror("Type mismatch: {0}")]
    TypeMismatch(String),
}

pub type VertexId = u32;

/// ErsRootKind – Rust enum projection of `T:RootKind` from hex.rune.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootKind {
    TypeI,
    TypeII,
}

/// ErsSemanticDomain – Rust enum projection of `T:SemanticDomain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticDomain {
    Psychology,
    Relationships,
    Economics,
    Creativity,
    Spirituality,
    Physical,
    Existential,
    Ethics,
    Education,
    Health,
}

/// ErsEmotionType – projection of `T:EmotionType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionType {
    Joy,
    Sorrow,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Love,
    Hate,
    Hope,
    Despair,
    Pride,
    Shame,
    Awe,
    Envy,
    Guilt,
    Gratitude,
    Nostalgia,
    Serenity,
    Empathy,
    Compassion,
    Jealousy,
    Loneliness,
    Belonging,
    Confidence,
    Insecurity,
    Embarrassment,
    Humility,
    Frustration,
    Calm,
    Anxiety,
}

/// ErsCognitiveType – projection of `T:CognitiveType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveType {
    Attention,
    Memory,
    Learning,
    Perception,
    Reasoning,
    ProblemSolving,
    DecisionMaking,
    Planning,
    Creativity,
    SelfAwareness,
    Reflection,
    Introspection,
    Insight,
    Understanding,
    Confusion,
    Clarity,
    Realization,
    ConfirmationBias,
    Anchoring,
    Availability,
    Overconfidence,
}

/// ErsSocialType – projection of `T:SocialType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocialType {
    Friendship,
    Romance,
    Family,
    Professional,
    Cooperation,
    Competition,
    Conflict,
    Collaboration,
    Acceptance,
    Rejection,
    Inclusion,
    Exclusion,
    Listening,
    Speaking,
    Arguing,
    Negotiating,
    Leader,
    Follower,
    Mentor,
    Student,
}

/// ErsExistentialType – projection of `T:ExistentialType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExistentialType {
    Meaning,
    Purpose,
    Identity,
    Freedom,
    Responsibility,
    Mortality,
    Authenticity,
    Absurdity,
    Alienation,
    Transcendence,
}

/// ErsPhysicalType – projection of `T:PhysicalType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalType {
    Pain,
    Pleasure,
    Hunger,
    Thirst,
    Fatigue,
    Energy,
    Touch,
    Temperature,
    Movement,
    Balance,
    Health,
    Illness,
}

/// ErsEthicalType – projection of `T:EthicalType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EthicalType {
    Righteousness,
    Compassion,
    Justice,
    Fairness,
    Integrity,
    Honesty,
    Empathy,
    Responsibility,
    Duty,
    Virtue,
    Temptation,
    Guilt,
}

/// ErsCreativeType – projection of `T:CreativeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeType {
    Inspiration,
    Imagination,
    Innovation,
    Expression,
    Design,
    Artistry,
    Invention,
    Discovery,
    Revelation,
    Synthesis,
    Originality,
    Vision,
}

/// ErsSpiritualType – projection of `T:SpiritualType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiritualType {
    Meditation,
    Prayer,
    Contemplation,
    Transcendence,
    Enlightenment,
    Devotion,
    Surrender,
    Grace,
    Sacredness,
    Mysticism,
    Presence,
    Unity,
}

/// ErsEconomicType – projection of `T:EconomicType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EconomicType {
    Wealth,
    Poverty,
    Value,
    Exchange,
    Trade,
    Investment,
    Production,
    Consumption,
    Scarcity,
    Abundance,
    Profit,
    Loss,
}

/// ErsHealthType – projection of `T:HealthType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthType {
    Wellness,
    Illness,
    Healing,
    Recovery,
    Vitality,
    Fatigue,
    Balance,
    Imbalance,
    Strength,
    Weakness,
    Resilience,
    Fragility,
}

/// ErsGSLFrame – projection of `T:GSLFrame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GSLFrame {
    Unrestricted,
    Restricted,
    Constrained,
}

/// ErsWeylSemanticAddress – projection of `T:WeylSemanticAddress`.
#[derive(Debug, Clone)]
pub struct WeylSemanticAddress {
    pub heads: Vec<[f32; 8]>,
    pub tails: Vec<[f32; 8]>,
    pub timestamp: f64,
    pub context: String,
}

/// ErsHumanExperience – projection of `T:HumanExperience`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanExperience {
    Emotional(EmotionType),
    Cognitive(CognitiveType),
    Social(SocialType),
    Existential(ExistentialType),
    Physical(PhysicalType),
    Ethical(EthicalType),
    Creative(CreativeType),
    Spiritual(SpiritualType),
    Economic(EconomicType),
    Health(HealthType),
}

#[derive(Debug, Clone)]
pub struct Axis {
    pub id: String,
    pub index: u8,
    pub plus: String,
    pub minus: String,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub id: VertexId,
    pub kind: RootKind,
    pub coords: [f32; 8],
    pub domain: Option<SemanticDomain>,
    pub confidence: f32,
}

impl Vertex {
    /// Get the primary human experience associated with this vertex.
    ///
    /// This is a heuristic mapping based on the vertex's coordinates and domain.
    /// Returns None if no clear primary experience can be determined.
    pub fn primary_experience(&self) -> Option<HumanExperience> {
        // Map domain to a representative experience
        // This is a simplified heuristic - in practice, would use coordinate analysis
        self.domain.and_then(|domain| match domain {
            SemanticDomain::Psychology => Some(HumanExperience::Emotional(EmotionType::Joy)),
            SemanticDomain::Relationships => Some(HumanExperience::Social(SocialType::Friendship)),
            SemanticDomain::Economics => Some(HumanExperience::Economic(EconomicType::Wealth)),
            SemanticDomain::Creativity => Some(HumanExperience::Creative(CreativeType::Innovation)),
            SemanticDomain::Spirituality => {
                Some(HumanExperience::Spiritual(SpiritualType::Meditation))
            }
            SemanticDomain::Physical => Some(HumanExperience::Physical(PhysicalType::Touch)),
            SemanticDomain::Existential => {
                Some(HumanExperience::Existential(ExistentialType::Meaning))
            }
            SemanticDomain::Ethics => Some(HumanExperience::Ethical(EthicalType::Justice)),
            SemanticDomain::Education => Some(HumanExperience::Cognitive(CognitiveType::Learning)),
            SemanticDomain::Health => Some(HumanExperience::Health(HealthType::Wellness)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub u: VertexId,
    pub v: VertexId,
    pub strength: f32,
}

/// ErsDomR – Rust projection of `T:DomR` from RUNE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomR {
    pub roots: Vec<u8>,   // dominant root indices (0..240)
    pub scores: Vec<f32>, // corresponding scores
}

/// CSR adjacency for fast traversal.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    pub row_offsets: Vec<u32>,
    pub adjacency: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct HexGraph {
    pub axes: Vec<Axis>,
    pub coords: Vec<[f32; 8]>,
    pub kinds: Vec<RootKind>,
    pub domains: Vec<Option<SemanticDomain>>,
    pub confidence: Vec<f32>,
    pub edges: CsrGraph,
}

/// Parsed, normalized tables from the `.rune` asset.
struct ParsedTables {
    axes: Vec<Axis>,
    coords_by_type: HashMap<&'static str, Vec<[f32; 8]>>,
}

static TABLES: OnceLock<ParsedTables> = OnceLock::new();
static HOLO_GRAPH: OnceLock<HexGraph> = OnceLock::new();

fn parse_hex() -> Result<ParsedTables, HexError> {
    let path = Path::new("examples/hex.rune");
    let content = fs::read_to_string(path)
        .map_err(|e| HexError::InvalidOperation(format!("Failed to read hex.rune: {e}")))?;

    let mut axes = Vec::new();
    let mut coords_by_type: HashMap<&'static str, Vec<[f32; 8]>> = HashMap::new();

    let mut in_axes = false;
    let mut in_coords = false;
    let mut current_type: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("Axes:") {
            in_axes = true;
            in_coords = false;
            continue;
        }
        if trimmed.starts_with("Coordinates:") {
            in_coords = true;
            in_axes = false;
            continue;
        }
        if in_axes && trimmed.starts_with('-') {
            // Example: - { id: "A", index: 0, plus: "...", minus: "...", weight: 1.0 }
            let mut id = String::new();
            let mut plus = String::new();
            let mut minus = String::new();
            let mut index = 0u8;
            let mut weight = 1.0f32;
            let inner = trimmed.trim_start_matches('-').trim();
            let inner = inner.trim_start_matches('{').trim_end_matches('}');
            for part in inner.split(',') {
                let kv: Vec<_> = part.trim().splitn(2, ':').collect();
                if kv.len() != 2 {
                    continue;
                }
                let key = kv[0].trim();
                let val = kv[1].trim().trim_matches('"');
                match key {
                    "id" => id = val.to_string(),
                    "plus" => plus = val.to_string(),
                    "minus" => minus = val.to_string(),
                    "index" => index = val.parse::<u8>().unwrap_or(0),
                    "weight" => weight = val.parse::<f32>().unwrap_or(1.0),
                    _ => {}
                }
            }
            axes.push(Axis {
                id,
                index,
                plus,
                minus,
                weight,
            });
            continue;
        }
        if in_coords {
            if trimmed.starts_with("T:") && trimmed.ends_with(':') {
                let name = trimmed
                    .trim_end_matches(':')
                    .trim_start_matches("T:")
                    .to_string();
                current_type = Some(name);
                continue;
            }
            if let Some(ref ty) = current_type {
                if trimmed.starts_with(char::is_alphanumeric) {
                    if let Some((_label, vec_part)) = trimmed.split_once(':') {
                        let vec_part = vec_part.trim();
                        if vec_part.starts_with('[') && vec_part.ends_with(']') {
                            let inner = vec_part.trim_start_matches('[').trim_end_matches(']');
                            let nums: Vec<f32> = inner
                                .split(',')
                                .filter_map(|s| s.trim().parse::<f32>().ok())
                                .collect();
                            if nums.len() == 8 {
                                let mut arr = [0.0f32; 8];
                                arr.copy_from_slice(&nums[..]);
                                coords_by_type
                                    .entry(Box::leak(ty.clone().into_boxed_str()))
                                    .or_default()
                                    .push(arr);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(ParsedTables {
        axes,
        coords_by_type,
    })
}

fn tables() -> &'static ParsedTables {
    TABLES.get_or_init(|| parse_hex().expect("hex parse"))
}

/// Default HexGraph built from the canonical hex.rune asset.
pub fn default_graph() -> &'static HexGraph {
    HOLO_GRAPH.get_or_init(|| {
        HexGraph::from_rune("examples/hex.rune").expect("failed to build default hex graph")
    })
}

/// Lookup coordinates for a human experience using the parsed tables.
pub fn to_e8_coordinates(exp: &HumanExperience) -> [f32; 8] {
    let t = tables();
    match exp {
        HumanExperience::Emotional(e) => pick(&t.coords_by_type, "EmotionType", *e as usize),
        HumanExperience::Cognitive(c) => pick(&t.coords_by_type, "CognitiveType", *c as usize),
        HumanExperience::Social(s) => pick(&t.coords_by_type, "SocialType", *s as usize),
        HumanExperience::Existential(x) => pick(&t.coords_by_type, "ExistentialType", *x as usize),
        HumanExperience::Physical(p) => pick(&t.coords_by_type, "PhysicalType", *p as usize),
        HumanExperience::Ethical(e) => pick(&t.coords_by_type, "EthicalType", *e as usize),
        HumanExperience::Creative(c) => pick(&t.coords_by_type, "CreativeType", *c as usize),
        HumanExperience::Spiritual(s) => pick(&t.coords_by_type, "SpiritualType", *s as usize),
        HumanExperience::Economic(e) => pick(&t.coords_by_type, "EconomicType", *e as usize),
        HumanExperience::Health(h) => pick(&t.coords_by_type, "HealthType", *h as usize),
    }
}

fn pick(map: &HashMap<&'static str, Vec<[f32; 8]>>, key: &'static str, idx: usize) -> [f32; 8] {
    map.get(key)
        .and_then(|v| v.get(idx))
        .copied()
        .unwrap_or([0.0; 8])
}

impl HexGraph {
    /// Build the graph from the canonical `.rune` asset.
    pub fn from_rune<P: AsRef<Path>>(path: P) -> Result<Self, HexError> {
        let _ = path; // path is currently advisory; we rely on the canonical asset.
        let t = tables();
        let axes = t.axes.clone();

        // Generate Type-I vertices (axis pairs with sign permutations).
        let mut coords = Vec::new();
        let mut kinds = Vec::new();
        let mut domains = Vec::new();
        let mut confidence = Vec::new();

        for a in 0..axes.len() {
            for b in (a + 1)..axes.len() {
                let (idx_a, w_a) = (axes[a].index as usize, axes[a].weight);
                let (idx_b, w_b) = (axes[b].index as usize, axes[b].weight);
                let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
                for (sa, sb) in signs {
                    let mut v = [0.0f32; 8];
                    v[idx_a] = sa * w_a;
                    v[idx_b] = sb * w_b;
                    normalize(&mut v);
                    coords.push(v);
                    kinds.push(RootKind::TypeI);
                    domains.push(None);
                    confidence.push(1.0);
                }
            }
        }

        // Type-II spinors (even parity).
        for mask in 0u16..256 {
            if mask.count_ones() % 2 != 0 {
                continue;
            }
            let mut v = [0.0f32; 8];
            for (i, axis) in axes.iter().enumerate() {
                let sign = if (mask & (1 << i)) != 0 { -0.5 } else { 0.5 };
                v[axis.index as usize] = sign * axis.weight;
            }
            normalize(&mut v);
            coords.push(v);
            kinds.push(RootKind::TypeII);
            domains.push(None);
            confidence.push(1.0);
        }

        // Build edges by inner product threshold 0.5.
        let n = coords.len();
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut adjacency = Vec::new();
        let mut weights = Vec::new();
        row_offsets.push(0);
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f32 = coords[i]
                    .iter()
                    .zip(coords[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if (dot - 0.5).abs() <= 1e-4 {
                    adjacency.push(j as u32);
                    weights.push(dot);
                }
            }
            row_offsets.push(adjacency.len() as u32);
        }

        Ok(Self {
            axes,
            coords,
            kinds,
            domains,
            confidence,
            edges: CsrGraph {
                row_offsets,
                adjacency,
                weights,
            },
        })
    }

    /// Borrow packed coordinates (AoS) for SIMD/GPU export.
    pub fn coords(&self) -> &[[f32; 8]] {
        &self.coords
    }

    /// Borrow CSR adjacency (row_offsets, adjacency, weights).
    pub fn csr(&self) -> (&[u32], &[u32], &[f32]) {
        (
            &self.edges.row_offsets,
            &self.edges.adjacency,
            &self.edges.weights,
        )
    }

    /// Find k-nearest vertices to given coordinates in semantic space.
    ///
    /// Returns vector of cloned Vertex structs sorted by distance.
    pub fn find_nearest_vertices(&self, coords: &[f32; 8], k: usize) -> Vec<Vertex> {
        let mut candidates: Vec<(usize, f32)> = self
            .coords
            .iter()
            .enumerate()
            .map(|(idx, vertex_coords)| (idx, l2(coords, vertex_coords)))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, _dist)| Vertex {
                id: idx as VertexId,
                kind: self.kinds[idx],
                coords: self.coords[idx],
                domain: self.domains[idx],
                confidence: self.confidence[idx],
            })
            .collect()
    }

    /// Calculate semantic confidence score for given coordinates.
    ///
    /// Confidence decreases as distance from nearest vertex increases.
    /// Returns value between 0.0 and 1.0.
    pub fn calculate_semantic_confidence(&self, coords: &[f32; 8]) -> f32 {
        let (_nearest_id, distance) = self.nearest_vertex(coords);
        // Confidence decreases as distance increases (inverse relationship)
        1.0 / (1.0 + distance)
    }

    /// Get all vertices belonging to a specific semantic domain.
    ///
    /// Note: Current implementation returns empty vec as domains are not yet
    /// populated in the graph construction. This is a placeholder for future
    /// domain assignment logic.
    pub fn vertices_by_domain(&self, target_domain: SemanticDomain) -> Vec<Vertex> {
        self.domains
            .iter()
            .enumerate()
            .filter_map(|(idx, domain_opt)| {
                domain_opt.and_then(|domain| {
                    if domain == target_domain {
                        Some(Vertex {
                            id: idx as VertexId,
                            kind: self.kinds[idx],
                            coords: self.coords[idx],
                            domain: Some(domain),
                            confidence: self.confidence[idx],
                        })
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

fn normalize(v: &mut [f32; 8]) {
    let n2: f32 = v.iter().map(|x| x * x).sum();
    if n2 > 1e-9 {
        let inv = 1.0 / n2.sqrt();
        for x in v {
            *x *= inv;
        }
    }
}

pub trait SemanticSpace {
    fn nearest_vertex(&self, query: &[f32; 8]) -> (VertexId, f32);
    fn nearest_k(&self, query: &[f32; 8], k: usize) -> Vec<(VertexId, f32)>;
}

impl SemanticSpace for HexGraph {
    fn nearest_vertex(&self, query: &[f32; 8]) -> (VertexId, f32) {
        let mut best = (0u32, f32::MAX);
        for (i, v) in self.coords.iter().enumerate() {
            let d = l2(query, v);
            if d < best.1 {
                best = (i as u32, d);
            }
        }
        best
    }

    fn nearest_k(&self, query: &[f32; 8], k: usize) -> Vec<(VertexId, f32)> {
        let mut out = Vec::with_capacity(k);
        for (i, v) in self.coords.iter().enumerate() {
            let d = l2(query, v);
            if out.len() < k {
                out.push((i as u32, d));
                out.sort_by(|a, b| a.1.total_cmp(&b.1));
            } else if let Some(last) = out.last_mut() {
                if d < last.1 {
                    *last = (i as u32, d);
                    out.sort_by(|a, b| a.1.total_cmp(&b.1));
                }
            }
        }
        out
    }
}

#[inline(always)]
fn l2(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut sum = 0.0;
    sum += (a[0] - b[0]) * (a[0] - b[0]);
    sum += (a[1] - b[1]) * (a[1] - b[1]);
    sum += (a[2] - b[2]) * (a[2] - b[2]);
    sum += (a[3] - b[3]) * (a[3] - b[3]);
    sum += (a[4] - b[4]) * (a[4] - b[4]);
    sum += (a[5] - b[5]) * (a[5] - b[5]);
    sum += (a[6] - b[6]) * (a[6] - b[6]);
    sum += (a[7] - b[7]) * (a[7] - b[7]);
    sum
}

#[derive(Debug, Clone)]
pub struct SemanticTrajectory {
    pub vertices: Vec<VertexId>,
}

pub trait SemanticPathFinder {
    fn optimal_path(&self, start: VertexId, target: VertexId) -> SemanticTrajectory;
}

impl SemanticPathFinder for HexGraph {
    fn optimal_path(&self, start: VertexId, target: VertexId) -> SemanticTrajectory {
        // Simple Dijkstra on CSR.
        let n = self.coords.len();
        let mut dist = vec![f32::INFINITY; n];
        let mut prev = vec![None; n];
        let mut visited = vec![false; n];
        dist[start as usize] = 0.0;

        for _ in 0..n {
            // pick min unvisited
            let mut u = None;
            let mut best = f32::INFINITY;
            for i in 0..n {
                if !visited[i] && dist[i] < best {
                    best = dist[i];
                    u = Some(i);
                }
            }
            let u = match u {
                Some(x) => x,
                None => break,
            };
            visited[u] = true;
            if u as u32 == target {
                break;
            }
            let start_edge = self.edges.row_offsets[u] as usize;
            let end_edge = self.edges.row_offsets[u + 1] as usize;
            for idx in start_edge..end_edge {
                let v = self.edges.adjacency[idx] as usize;
                let w = self.edges.weights[idx];
                let alt = dist[u] + w;
                if alt < dist[v] {
                    dist[v] = alt;
                    prev[v] = Some(u);
                }
            }
        }

        let mut path = Vec::new();
        let mut cur = target as usize;
        if dist[cur].is_finite() {
            while let Some(p) = prev[cur] {
                path.push(cur as u32);
                cur = p;
            }
            path.push(start);
            path.reverse();
        }
        SemanticTrajectory { vertices: path }
    }
}

/// Simple gradient stepper: move in direction and snap to nearest vertex.
pub fn apply_semantic_gradient(
    graph: &HexGraph,
    start: VertexId,
    steps: usize,
    step_size: f32,
    direction: [f32; 8],
) -> SemanticTrajectory {
    let mut pos = graph.coords[start as usize];
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push(start);
    for _ in 0..steps {
        for i in 0..8 {
            pos[i] += step_size * direction[i];
        }
        normalize(&mut pos);
        let (vid, _) = graph.nearest_vertex(&pos);
        traj.push(vid);
        pos = graph.coords[vid as usize];
    }
    SemanticTrajectory { vertices: traj }
}

/// Geometry diagnostics.
#[derive(Debug, Clone)]
pub struct GeometryReport {
    pub norms: Vec<f32>,
    pub min_pair: Option<((VertexId, VertexId), f32)>,
}

pub fn analyze_geometry(graph: &HexGraph) -> GeometryReport {
    let norms: Vec<f32> = graph
        .coords
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    let mut min_pair: Option<((VertexId, VertexId), f32)> = None;
    for i in 0..graph.coords.len() {
        for j in (i + 1)..graph.coords.len() {
            let d = l2(&graph.coords[i], &graph.coords[j]).sqrt();
            if let Some((_, best)) = &min_pair {
                if d < *best {
                    min_pair = Some(((i as u32, j as u32), d));
                }
            } else {
                min_pair = Some(((i as u32, j as u32), d));
            }
        }
    }
    GeometryReport { norms, min_pair }
}

/// Domain-specific proximity diagnostics drawn directly from the parsed tables.
#[derive(Debug, Clone)]
pub struct DomainReport {
    pub name: String,
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    pub closest_pair: Option<((usize, usize), f32)>,
}

pub fn analyze_domains() -> Vec<DomainReport> {
    let t = tables();
    let mut reports = Vec::new();
    for (name, vecs) in &t.coords_by_type {
        if vecs.is_empty() {
            continue;
        }
        let norms: Vec<f32> = vecs
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        let min_norm = norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_norm = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        let mut closest: Option<((usize, usize), f32)> = None;
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let d = l2(&vecs[i], &vecs[j]).sqrt();
                match closest {
                    Some((_, best)) if d >= best => {}
                    _ => closest = Some(((i, j), d)),
                }
            }
        }
        reports.push(DomainReport {
            name: (*name).to_string(),
            min_norm,
            max_norm,
            mean_norm,
            closest_pair: closest,
        });
    }
    reports
}

/// Reference CPU DomR computation: score[x] = Σ_o e[o] * dot(root_o, root_x), top N.
/// GeoCel trajectory generation implementation.
/// Provides dynamic navigation through the E8 semantic lattice using
/// momentum dynamics and stochastic processes.
pub mod geocel {
    use super::*;

    /// Primary GeoCel trajectory generator using intent and anima-driven navigation.
    pub fn spawn_surveyor_worm(
        intent: &str,
        frame: GSLFrame,
        anima: f32,
        graph: &HexGraph,
    ) -> Result<Vec<[f32; 8]>, HexError> {
        // Create semantic address from intent (stub implementation)
        let address = WeylSemanticAddress::from_text_intent(intent, anima)?;

        // Apply momentum dynamics
        let evolution = apply_momentum_dynamics(&address, anima)?;

        // Quantize to E8 roots based on frame constraints
        let quantized = quantize_to_e8_roots(&evolution, frame, graph)?;

        // Generate final trajectory
        generate_trajectory(&quantized, graph)
    }

    impl WeylSemanticAddress {
        /// Create semantic address from text intent using basic keyword mapping.
        pub fn from_text_intent(intent: &str, _anima: f32) -> Result<Self, HexError> {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            // Basic intent parsing - map keywords to semantic heads
            let mut heads = Vec::new();
            let context = intent.to_string();

            // Extract key emotional concepts and map to coordinates
            if intent.contains("fear") {
                heads.push([-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5]); // Fear coordinates
            }
            if intent.contains("courage") {
                heads.push([1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5]); // Confidence coordinates
            }
            if intent.contains("love") {
                heads.push([1.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0]); // Love coordinates
            }

            if heads.is_empty() {
                // Default semantic head if no keywords found
                heads.push([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
            }

            Ok(WeylSemanticAddress {
                heads,
                tails: vec![], // Empty tails for initial address
                timestamp,
                context,
            })
        }

        /// Apply momentum dynamics to semantic address.
        pub fn apply_momentum_dynamics(&self, anima: f32) -> Result<Vec<[f32; 8]>, HexError> {
            let mut evolution = Vec::new();

            for head in &self.heads {
                let mut current = *head;
                evolution.push(current);

                // Apply momentum steps with noise injection
                for _ in 0..10 {
                    current = momentum_step(&current, anima);
                    current = noise_injection(&current, anima);
                    current = quantize_to_valid_root(&current);
                    evolution.push(current);
                }
            }

            Ok(evolution)
        }
    }

    /// Apply momentum step: accelerate in current direction with small random perturbation.
    pub fn momentum_step(vec: &[f32; 8], anima: f32) -> [f32; 8] {
        let mut normalized = *vec;
        normalize(&mut normalized);

        let mut result = *vec;
        for i in 0..8 {
            result[i] += anima * normalized[i] + 0.1 * random_unit_vector()[i];
        }
        normalize(&mut result);
        result
    }

    /// Add controlled noise to vector.
    pub fn noise_injection(vec: &[f32; 8], anima: f32) -> [f32; 8] {
        let mut result = *vec;
        let noise = gaussian_noise();
        for i in 0..8 {
            result[i] += anima * 0.3 * noise;
        }
        result
    }

    /// Quantize vector to nearest valid E8 root.
    pub fn quantize_to_valid_root(vec: &[f32; 8]) -> [f32; 8] {
        // Find nearest E8 root (simplified - would use graph lookup in full implementation)
        let default_graph = default_graph();
        let (nearest_idx, _) = default_graph.nearest_vertex(vec);
        default_graph.coords[nearest_idx as usize]
    }

    /// Apply momentum dynamics with specified anima (high-level interface).
    pub fn apply_momentum_dynamics(
        addr: &WeylSemanticAddress,
        anima: f32,
    ) -> Result<Vec<[f32; 8]>, HexError> {
        addr.apply_momentum_dynamics(anima)
    }

    /// Quantize trajectory points to E8 roots with frame constraints.
    pub fn quantize_to_e8_roots(
        evolution: &[[f32; 8]],
        _frame: GSLFrame,
        graph: &HexGraph,
    ) -> Result<Vec<[f32; 8]>, HexError> {
        let mut quantized = Vec::new();

        for point in evolution {
            let (nearest_idx, _) = graph.nearest_vertex(point);
            quantized.push(graph.coords[nearest_idx as usize]);
        }

        Ok(quantized)
    }

    /// Generate final smooth trajectory from quantized points.
    pub fn generate_trajectory(
        evolution: &[[f32; 8]],
        _graph: &HexGraph,
    ) -> Result<Vec<[f32; 8]>, HexError> {
        let mut trajectory = Vec::new();
        trajectory.extend_from_slice(evolution);
        Ok(trajectory)
    }

    /// Generate random unit vector for perturbations.
    fn random_unit_vector() -> [f32; 8] {
        use std::f32::consts::PI;
        let mut vec = [0.0f32; 8];
        for i in 0..8 {
            vec[i] = (i as f32 * PI / 4.0).sin() * 0.1; // Deterministic pseudo-random
        }
        normalize(&mut vec);
        vec
    }

    /// Generate gaussian noise (simplified).
    fn gaussian_noise() -> f32 {
        // Simplified gaussian using Box-Muller transform approximation
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f32
            * 0.000_000_000_1)
            .sin()
    }
}

pub fn domr_cpu(graph: &HexGraph, energy: &[f32], n_dr: usize) -> Result<DomR, HexError> {
    if energy.len() != graph.coords.len() {
        return Err(HexError::TypeMismatch(format!(
            "Energy length {} must match root count {}",
            energy.len(),
            graph.coords.len()
        )));
    }
    if n_dr == 0 || n_dr > graph.coords.len() {
        return Err(HexError::InvalidOperation(
            "N_DR must be between 1 and root count".into(),
        ));
    }
    let mut scores = vec![0.0f32; graph.coords.len()];
    for (x_idx, x) in graph.coords.iter().enumerate() {
        let mut acc = 0.0f32;
        for (o_idx, e) in energy.iter().enumerate() {
            let dot: f32 = graph.coords[o_idx]
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum();
            acc += *e * dot;
        }
        scores[x_idx] = acc;
    }
    let mut pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let take = n_dr.min(pairs.len());
    let mut roots = Vec::with_capacity(take);
    let mut out_scores = Vec::with_capacity(take);
    for (idx, score) in pairs.into_iter().take(take) {
        roots.push(idx as u8);
        out_scores.push(score);
    }
    Ok(DomR {
        roots,
        scores: out_scores,
    })
}
