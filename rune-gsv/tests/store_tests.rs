use rune_gsv::prelude::*;
use rune_gsv::slot::SGLRuneSlot;
use std::fs;

#[test]
fn test_default_path_exists_and_store_load_cycle() {
    // Use a unique vault name to avoid interference
    let vn = "gsv_test_vault";
    let mut store = QuantizedContinuum::with_name(vn);
    let slot = SGLRuneSlot {
        address: vec![1, 2, 3, 4, 5, 6, 7, 8],
        intent: "test intent".to_string(),
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
            elo_scores: std::collections::HashMap::new(),
        },
        payload: serde_json::Value::default(),
    };
    let key = store.store("test intent", slot).unwrap();
    assert!(!key.is_empty());

    // reload into new instance
    let store2 = QuantizedContinuum::with_name(vn);
    let res = store2.retrieve("test intent");
    assert!(res.is_some());

    // cleanup path
    let path = QuantizedContinuum::storage_path_for_name(vn);
    let _ = fs::remove_file(path);
}
