#[cfg(feature = "mpak")]
#[test]
fn mpak_round_trip_json() {
    use rune_cros::mpak::*;
    use serde_json::json;

    let s = json!({"name":"Alice","age":30});
    let json_str = serde_json::to_string(&s).unwrap();
    let bytes = json_string_to_mpak(&json_str).expect("json to mpak");

    let back = mpak_to_json_string(&bytes).expect("mpak to json");
    let v: serde_json::Value = serde_json::from_str(&back).unwrap();
    assert_eq!(v["name"], json!("Alice"));
}
