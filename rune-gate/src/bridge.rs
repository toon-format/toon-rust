/* src/bridge.rs */
//!▫~•◦-------------------------------‣
//! # Backend abstraction layer for E8 geometry data access and manipulation.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to provide a unified interface
//! between the Bevy frontend and E8 geometry backends.
//!
//! ### Key Capabilities
//! - **Backend Abstraction:** Defines the E8Backend trait for consistent data access across different implementations.
//! - **E8 Geometry Data:** Provides access to E8 root systems, vertices, and domain information.
//! - **Query Processing:** Supports geometric queries and path finding in E8 space.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `app` and `scene`.
//! The HydronBackend implementation connects to hydron-core for actual E8 geometry data.
//!
//! ### Example
//! ```rust
//! use crate::rune_gate::{E8Backend, HydronBackend, BackendHandle};
//!
//! let backend = HydronBackend::new();
//! let handle = BackendHandle::new(backend);
//! let domains = handle.0.list_domains();
//! let vertices = handle.0.list_vertices();
//! ```

#[cfg(feature = "viewer")]
use bevy::prelude::Resource;
// use hydron_core::gf8::get_e8_roots; // Use Gf8 from gf8 crate via hydron_core
use hydron_core::get_e8_roots;
use serde::{Deserialize, Serialize};

/// Trait that defines the contract between the Bevy frontend and the E8 backend.
pub trait E8Backend: Send + Sync + 'static {
    fn list_domains(&self) -> Vec<DomainSummary>;
    fn get_vertex(&self, id: u32) -> Option<VertexDetail>;
    fn list_vertices(&self) -> Vec<VertexDetail>;
    fn run_query(&self, query: E8Query) -> QueryResult;
    fn get_path(&self, from: u32, to: u32) -> PathResult;
}

/// Handle stored in Bevy world to access the injected backend.
#[cfg_attr(feature = "viewer", derive(Resource))]
pub struct BackendHandle(pub Box<dyn E8Backend>);

impl BackendHandle {
    pub fn new<B: E8Backend>(backend: B) -> Self {
        Self(Box::new(backend))
    }
}

/// Selected vertex resource shared between scene and UI.
#[cfg_attr(feature = "viewer", derive(Resource))]
#[derive(Default, Clone)]
pub struct SelectedVertex(pub Option<VertexDetail>);

/// Adapter that exposes the real Hydron data to the viewer.
/// For now this surfaces the canonical 240 E8 roots with minimal labeling.
#[derive(Clone, Default)]
pub struct HydronBackend;

impl HydronBackend {
    pub fn new() -> Self {
        Self
    }
}

impl E8Backend for HydronBackend {
    fn list_domains(&self) -> Vec<DomainSummary> {
        vec![
            DomainSummary {
                name: "E8 Root".into(),
                count: 112,
            },
            DomainSummary {
                name: "E8 Spinor".into(),
                count: 128,
            },
        ]
    }

    fn get_vertex(&self, id: u32) -> Option<VertexDetail> {
        self.list_vertices().into_iter().find(|v| v.id == id)
    }

    fn list_vertices(&self) -> Vec<VertexDetail> {
        // If get_e8_roots returns empty (placeholder), we should generate them or use gf8 crate function.
        // Assuming get_e8_roots returns Vec<Gf8>.
        let roots = get_e8_roots();
        roots
            .iter()
            .enumerate()
            .map(|(idx, gf)| {
                let kind = if idx < 112 { "TypeI" } else { "TypeII" };
                let domain = if idx < 112 { "E8 Root" } else { "E8 Spinor" };
                // Gf8 has coords() -> [f32; 8]
                let coord: [f32; 8] = *gf.coords();
                VertexDetail {
                    id: idx as u32,
                    label: format!("root-{idx:03}"),
                    domain: domain.into(),
                    kind: kind.into(),
                    coord8d: coord,
                    blurb: "Canonical E8 basis element.".into(),
                    positive_color: None,
                    inverted_color: None,
                    opposite: None,
                }
            })
            .collect()
    }

    fn run_query(&self, _query: E8Query) -> QueryResult {
        QueryResult::Stub("Query execution not yet wired to hydron-core".into())
    }

    fn get_path(&self, from: u32, to: u32) -> PathResult {
        PathResult::Stub(format!("Path request from {from} to {to} not yet wired"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSummary {
    pub name: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexDetail {
    pub id: u32,
    pub label: String,
    pub domain: String,
    pub kind: String,
    pub coord8d: [f32; 8],
    pub blurb: String,
    pub positive_color: Option<String>,
    pub inverted_color: Option<String>,
    pub opposite: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E8Query {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResult {
    Stub(String),
    // Future: real results.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathResult {
    Stub(String),
    // Future: real path data.
}
