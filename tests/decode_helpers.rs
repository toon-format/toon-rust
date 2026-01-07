use serde_json::{json, Value};
use toon_format::{decode_no_coerce_with_options, decode_strict_with_options, DecodeOptions};

#[test]
fn test_decode_strict_with_options_forces_strict() {
    let opts = DecodeOptions::new().with_strict(false);
    let result: Result<Value, _> = decode_strict_with_options("items[2]: a", &opts);
    assert!(result.is_err(), "Strict mode should reject length mismatch");
}

#[test]
fn test_decode_no_coerce_with_options_disables_coercion() {
    let opts = DecodeOptions::new().with_coerce_types(true);
    let result: Value = decode_no_coerce_with_options("value: 123", &opts).unwrap();
    assert_eq!(result, json!({"value": "123"}));
}
