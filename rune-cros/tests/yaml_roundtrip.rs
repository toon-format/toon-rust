#[cfg(feature = "yaml")]
#[test]
fn yaml_round_trip_rune_encode() {
    use rune_cros::yaml::*;
    use rune_format::{decode_default, encode_object};
    use serde_json::json;

    let yaml = "---\nname: Alice\nage: 30\n";
    let json_str = yaml_to_json_string(yaml).expect("convert yaml to json");
    // parse JSON to serde_json::Value and encode to RUNE
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    // encode as RUNE
    let toon = encode_object(v, &rune_format::types::EncodeOptions::default()).unwrap();
    // decode back
    let final_json: serde_json::Value = decode_default(&toon).unwrap();
    assert_eq!(final_json["name"], json!("Alice"));
    assert_eq!(final_json["age"], json!(30));
}
