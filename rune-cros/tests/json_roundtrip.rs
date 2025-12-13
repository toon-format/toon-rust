#[test]
fn json_rune_round_trip() {
    use rune_cros::json::*;
    use serde_json::json;

    let input = json!({"name":"Alice","age":30,"nested":{"ok":true}});
    let input_str = serde_json::to_string(&input).unwrap();

    let rune = json_to_rune_string(&input_str).expect("json -> rune");
    let back = rune_string_to_json(&rune).expect("rune -> json");
    let round: serde_json::Value = serde_json::from_str(&back).unwrap();

    assert_eq!(round["name"], json!("Alice"));
    assert_eq!(round["age"], json!(30));
    assert_eq!(round["nested"]["ok"], json!(true));
}
