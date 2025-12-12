/* src/builtins/builtins.rs */
//!▫~•◦-------------------------------‣
//! # Built-in ASV (Address-Slot-Value) Operations for Rune-GSV integration.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gsv to achieve efficient graph-based data storage and querying system.
//!
//! ### Key Capabilities
//! - **Slot Storage:** Provides ASV storage operations with automatic slot creation and validation.
//! - **Slot Retrieval:** Enables intent-based slot retrieval with JSON serialization support.
//! - **Similarity Querying:** Offers k-nearest-neighbor query functionality for semantic similarity search.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `slot` and `store`.
//! Built-in functions adhere to the hydron evaluator interface and return JSON-compatible results.
//! Result structures are compatible with the system's serialization pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_gsv::{asv_store, asv_get, asv_query, QuantizedContinuum};
//!
//! let mut store = QuantizedContinuum::default();
//! let bundle = serde_json::json!({ "payload": {"note": "test"} });
//! let key = asv_store(&mut store, "find documents", bundle).unwrap();
//! let retrieved = asv_get(&store, "find documents").unwrap();
//! let similar = asv_query(&store, "find documents", 5).unwrap();
//!
//! // The results can be used for further processing in the evaluator.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::slot::{ExecutionTrace, RankingData, SGLRuneSlot, SemanticGraph};
use crate::store::QuantizedContinuum;
use serde_json::Value;
use yoshi::error::Result;

/// Built-in public API that integrates with hydron evaluator.
/// These functions are sync and return JSON-like results to the evaluator.

pub fn asv_store(store: &mut QuantizedContinuum, intent: &str, bundle: Value) -> Result<String> {
    // Try to convert incoming bundle into SGLRuneSlot; if missing fields, fallback to minimal slot
    let slot: SGLRuneSlot = match serde_json::from_value(bundle.clone()) {
        Ok(s) => s,
        Err(_) => {
            // fallback: build a minimal slot using provided payload and intent
            let address: Vec<u8> = Vec::new();
            let created_at = 0.0f64;
            let semantic_graph = SemanticGraph {
                nodes: Vec::new(),
                frames: Vec::new(),
            };
            let execution = ExecutionTrace {
                id: String::new(),
                effect_set_id: None,
                steps: Vec::new(),
            };
            let ranking = RankingData {
                query_id: String::new(),
                text: String::new(),
                candidate_ids: Vec::new(),
                elo_scores: std::collections::HashMap::new(),
            };
            let payload_val = bundle
                .get("payload")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            SGLRuneSlot {
                address,
                intent: intent.to_string(),
                created_at,
                semantic_graph,
                execution,
                ranking,
                payload: payload_val,
            }
        }
    };
    slot.address.len(); // ensure vector present
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
