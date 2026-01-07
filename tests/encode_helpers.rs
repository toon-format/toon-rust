use indexmap::IndexMap;
use serde_json::json;
use toon_format::{encode_array, encode_object, EncodeOptions, ToonError};
use toon_format::types::{JsonValue, Number};

#[test]
fn test_encode_array_and_object_with_json() {
    let array = json!(["a", "b"]);
    let encoded = encode_array(&array, &EncodeOptions::default()).unwrap();
    assert!(encoded.starts_with("[2]:"));

    let object = json!({"a": 1});
    let encoded = encode_object(&object, &EncodeOptions::default()).unwrap();
    assert!(encoded.contains("a: 1"));
}

#[test]
fn test_encode_array_object_type_mismatch() {
    let err = encode_array(&json!({"a": 1}), &EncodeOptions::default()).unwrap_err();
    match err {
        ToonError::TypeMismatch { expected, found } => {
            assert_eq!(expected, "array");
            assert_eq!(found, "object");
        }
        _ => panic!("Expected TypeMismatch for encode_array"),
    }

    let err = encode_object(&json!(["a", "b"]), &EncodeOptions::default()).unwrap_err();
    match err {
        ToonError::TypeMismatch { expected, found } => {
            assert_eq!(expected, "object");
            assert_eq!(found, "array");
        }
        _ => panic!("Expected TypeMismatch for encode_object"),
    }
}

#[test]
fn test_encode_array_object_with_json_value() {
    let value = JsonValue::Array(vec![JsonValue::Number(Number::from(1))]);
    let encoded = encode_array(value, &EncodeOptions::default()).unwrap();
    assert!(encoded.contains("1"));

    let mut obj = IndexMap::new();
    obj.insert("key".to_string(), JsonValue::Bool(true));
    let value = JsonValue::Object(obj);
    let encoded = encode_object(value, &EncodeOptions::default()).unwrap();
    assert!(encoded.contains("key: true"));
}
