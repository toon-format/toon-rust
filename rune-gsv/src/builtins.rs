use anyhow::Result;
use crate::store::QuantizedContinuum;
use crate::slot::{SGLRuneSlot, SemanticGraph, ExecutionTrace, RankingData};
use std::collections::HashMap;
use serde_json::Value;

/// Built-in public API that integrates with hydron evaluator.
/// These functions are sync and return JSON-like results to the evaluator.

pub fn asv_store(store: &mut QuantizedContinuum, intent: &str, bundle: Value) -> Result<String> {
    // Try to convert incoming bundle into SGLRuneSlot; if missing fields, fallback to minimal slot
    let slot: SGLRuneSlot = match serde_json::from_value(bundle.clone()) {
        Ok(s) => s,
        Err(_) => {
            // fallback: build a minimal slot using provided payload and intent
            let mut address: Vec<u8> = Vec::new();
            let created_at = 0.0f64;
            let semantic_graph = SemanticGraph { nodes: Vec::new(), frames: Vec::new() };
            let execution = ExecutionTrace { id: String::new(), effect_set_id: None, steps: Vec::new() };
            let ranking = RankingData { query_id: String::new(), text: String::new(), candidate_ids: Vec::new(), elo_scores: std::collections::HashMap::new() };
            let payload_val = bundle.get("payload").cloned().unwrap_or(serde_json::Value::Null);
            SGLRuneSlot { address, intent: intent.to_string(), created_at, semantic_graph, execution, ranking, payload: payload_val }
        }
    };
    slot
        .address
        .len(); // ensure vector present
    let key = store.store(intent, slot)?;
    Ok(key)
}

pub fn asv_get(store: &QuantizedContinuum, intent: &str) -> Result<Option<Value>> {
    if let Some(slot) = store.retrieve(intent) {
        let v = serde_json::to_value(slot)?;
        Ok(Some(v))
    } else {
        Ok(None)
    }
}

pub fn asv_query(store: &QuantizedContinuum, intent: &str, k: usize) -> Result<Value> {
    let results = store.query_similar_intents(intent, k);
    let v = serde_json::to_value(results)?;
    Ok(v)
}
