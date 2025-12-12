use rune_gsv::prelude::*;
use rune_gsv::slot::SGLRuneSlot;
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;

fn main() {
    // Use a temporary directory for example/demo to avoid touching user's home vaults.
    let tmp = tempdir().expect("create temp dir");
    let path = tmp.path().join("gsv_example.rune");
    let mut store = QuantizedContinuum::new(path.to_string_lossy().to_string(), "gsv_example");

    // Construct a small slot with a dynamic payload so examples do not use 'real' data.
    let slot = SGLRuneSlot {
        address: vec![0, 1, 2, 3, 4, 5, 6, 7],
        intent: "example intent".to_string(),
        created_at: 0.0,
        semantic_graph: rune_gsv::slot::SemanticGraph {
            nodes: vec![],
            frames: vec![],
        },
        execution: rune_gsv::slot::ExecutionTrace {
            id: "t1".to_string(),
            effect_set_id: None,
            steps: vec![],
        },
        ranking: rune_gsv::slot::RankingData {
            query_id: "q1".to_string(),
            text: "q".to_string(),
            candidate_ids: vec![],
            elo_scores: HashMap::new(),
        },
        payload: json!({"demo": true, "note": "temporary slot - no sensitive data"}),
    };

    let k = store.store("example intent", slot).unwrap();
    println!("Stored key: {} in path {}", k, store.storage_path);
}
