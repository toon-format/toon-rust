use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub types: Vec<String>,
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub id: String,
    pub concept_id: String,
    pub roles: HashMap<String, Vec<String>>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticGraph {
    pub nodes: Vec<Node>,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionTrace {
    pub id: String,
    pub effect_set_id: Option<String>,
    pub steps: Vec<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RankingData {
    pub query_id: String,
    pub text: String,
    pub candidate_ids: Vec<String>,
    pub elo_scores: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(default = "serde_json::Value::default")]
    pub payload: serde_json::Value,
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
}
