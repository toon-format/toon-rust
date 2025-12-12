/* src/slot/slot.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Graph Language Rune Slot Data Structures
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gsv to achieve efficient graph-based data storage and querying system.
//!
//! ### Key Capabilities
//! - **Graph Data Structures:** Provides Node, Frame, and SemanticGraph structures for knowledge representation.
//! - **Execution Tracing:** Supports tracking of execution traces with step-by-step operation recording.
//! - **Ranking Data:** Includes structures for query-based ranking with ELO scoring support.
//! - **Slot Management:** Offers the core SGLRuneSlot structure with address-based storage and flexible payload support.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `addressing` and `store`.
//! Slot structures adhere to the semantic addressing pattern and are compatible
//! with the system's serialization and persistence layers.
//! The payload field supports dynamic JSON data for extensibility.
//!
//! ### Example
//! ```rust
//! use crate::rune_gsv::{SGLRuneSlot, SemanticGraph, Node, Frame};
//!
//! let mut slot = SGLRuneSlot::default();
//! slot.intent = "example intent".to_string();
//! slot.semantic_graph.nodes.push(Node {
//!     id: "node1".to_string(),
//!     kind: "concept".to_string(),
//!     label: "Example Concept".to_string(),
//!     types: vec!["Type1".to_string()],
//!     meta: std::collections::HashMap::new(),
//! });
//!
//! let json_value = slot.to_map();
//! // The slot can now be stored, retrieved, and processed.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use fory::ForyObject;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, ForyObject)]
pub struct Node {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub types: Vec<String>,
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ForyObject)]
pub struct Frame {
    pub id: String,
    pub concept_id: String,
    pub roles: HashMap<String, Vec<String>>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, ForyObject)]
pub struct SemanticGraph {
    pub nodes: Vec<Node>,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, ForyObject)]
pub struct ExecutionTrace {
    pub id: String,
    pub effect_set_id: Option<String>,
    pub steps: Vec<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, ForyObject)]
pub struct RankingData {
    pub query_id: String,
    pub text: String,
    pub candidate_ids: Vec<String>,
    pub elo_scores: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, ForyObject)]
pub struct SGLRuneSlot {
    pub address: Vec<u8>,
    pub intent: String,
    pub created_at: f64,

    #[serde(default)]
    pub semantic_graph: SemanticGraph,
    #[serde(default)]
    pub execution: ExecutionTrace,
    #[serde(default)]
    pub ranking: RankingData,
    /// Free-form payload for dynamic / plugin-driven data; optional and can hold any JSON-compatible info.
    #[serde(default)]
    pub payload: String,
}

impl SGLRuneSlot {
    pub fn to_map(&self) -> serde_json::Value {
        serde_json::json!({
            "address_heads": self.address,
            "intent": self.intent,
            "created_at": self.created_at,
            "semantic_graph": self.semantic_graph,
            "execution": self.execution,
            "ranking": self.ranking,
            "payload": self.payload,
        })
    }

    pub fn set_payload_value(&mut self, value: serde_json::Value) {
        self.payload = value.to_string();
    }

    pub fn payload_value(&self) -> Option<serde_json::Value> {
        serde_json::from_str(&self.payload).ok()
    }
}
