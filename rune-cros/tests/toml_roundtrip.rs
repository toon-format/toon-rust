#[cfg(feature = "toml")]
#[test]
fn toml_round_trip_rune_encode() {
    use rune_cros::toml::*;
    use rune_format::{decode_default, encode_object};
    use serde_json::json;

    let toml = "name = 'Alice'\nage = 30\n";
    let json_str = toml_to_json_string(toml).expect("toml to json");
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let toon = encode_object(v, &rune_format::types::EncodeOptions::default()).unwrap();
    let final_json: serde_json::Value = decode_default(&toon).unwrap();
    assert_eq!(final_json["name"], json!("Alice"));
    assert_eq!(final_json["age"], json!(30));
}
