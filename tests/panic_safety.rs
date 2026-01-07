//! Tests to verify panic safety of JsonValue operations

use toon_format::JsonValue;

#[test]
fn test_get_missing_key_returns_none() {
    let obj = JsonValue::Object(Default::default());
    assert!(obj.get("nonexistent").is_none());
}

#[test]
fn test_get_on_non_object_returns_none() {
    let arr = JsonValue::Array(vec![]);
    assert!(arr.get("key").is_none());
}

#[test]
fn test_get_index_out_of_bounds_returns_none() {
    let arr = JsonValue::Array(vec![]);
    assert!(arr.get_index(0).is_none());
}

#[test]
fn test_get_index_on_non_array_returns_none() {
    let obj = JsonValue::Object(Default::default());
    assert!(obj.get_index(0).is_none());
}
