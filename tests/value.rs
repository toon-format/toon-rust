use std::panic::{catch_unwind, AssertUnwindSafe};

use indexmap::IndexMap;
use serde_json::json;
use toon_format::types::{IntoJsonValue, JsonValue, Number};

#[test]
fn test_number_from_f64_rejects_non_finite() {
    assert!(Number::from_f64(f64::NAN).is_none());
    assert!(Number::from_f64(f64::INFINITY).is_none());
    assert!(Number::from_f64(f64::NEG_INFINITY).is_none());
    assert!(Number::from_f64(1.5).is_some());
}

#[test]
fn test_number_integer_checks() {
    let float_int = Number::Float(42.0);
    assert!(float_int.is_i64());
    assert!(float_int.is_u64());

    let float_frac = Number::Float(42.5);
    assert!(!float_frac.is_i64());
    assert!(!float_frac.is_u64());

    let float_max = Number::Float(i64::MAX as f64);
    assert!(!float_max.is_i64());
    assert_eq!(float_max.as_i64(), Some(i64::MAX));

    let float_neg = Number::Float(-1.0);
    assert!(!float_neg.is_u64());
}

#[test]
fn test_number_as_conversions() {
    let too_large = Number::PosInt(i64::MAX as u64 + 1);
    assert_eq!(too_large.as_i64(), None);

    let neg = Number::NegInt(-5);
    assert_eq!(neg.as_u64(), None);

    let float_exact = Number::Float(7.0);
    assert_eq!(float_exact.as_i64(), Some(7));
    assert_eq!(float_exact.as_u64(), Some(7));

    let float_frac = Number::Float(7.25);
    assert_eq!(float_frac.as_i64(), None);
    assert_eq!(float_frac.as_u64(), None);

    let float_nan = Number::Float(f64::NAN);
    assert!(!float_nan.is_integer());
}

#[test]
fn test_number_display_nan() {
    let value = Number::from(f64::NAN);
    assert_eq!(format!("{value}"), "0");
}

#[test]
fn test_json_value_accessors_and_take() {
    let mut obj = IndexMap::new();
    obj.insert("a".to_string(), JsonValue::Number(Number::from(1)));

    let mut value = JsonValue::Object(obj);
    assert!(value.is_object());
    assert_eq!(value.type_name(), "object");
    assert_eq!(value.get("a").and_then(JsonValue::as_i64), Some(1));

    value
        .as_object_mut()
        .unwrap()
        .insert("b".to_string(), JsonValue::String("hi".to_string()));
    assert_eq!(value.get("b").and_then(JsonValue::as_str), Some("hi"));

    let mut arr = JsonValue::Array(vec![JsonValue::Bool(true)]);
    assert!(arr.is_array());
    arr.as_array_mut().unwrap().push(JsonValue::Null);
    assert_eq!(arr.as_array().unwrap().len(), 2);

    let mut taken = JsonValue::String("take".to_string());
    let prior = taken.take();
    assert!(matches!(taken, JsonValue::Null));
    assert_eq!(prior.as_str(), Some("take"));
}

#[test]
fn test_json_value_indexing_success() {
    let mut arr = JsonValue::Array(vec![JsonValue::Number(Number::from(1)), JsonValue::Null]);
    assert_eq!(arr[0].as_i64(), Some(1));
    arr[1] = JsonValue::Bool(true);
    assert_eq!(arr[1].as_bool(), Some(true));

    let mut obj = IndexMap::new();
    obj.insert("key".to_string(), JsonValue::Bool(false));
    let mut value = JsonValue::Object(obj);

    assert_eq!(value["key"].as_bool(), Some(false));
    value["key"] = JsonValue::Bool(true);
    assert_eq!(value["key"].as_bool(), Some(true));

    let owned_key = "key".to_string();
    assert_eq!(value[owned_key].as_bool(), Some(true));
}

#[test]
fn test_json_value_indexing_panics() {
    let value = JsonValue::Null;
    let err = catch_unwind(AssertUnwindSafe(|| {
        let _ = &value["missing"];
    }));
    assert!(err.is_err());

    let empty_array = JsonValue::Array(Vec::new());
    let err = catch_unwind(AssertUnwindSafe(|| {
        let _ = &empty_array[1];
    }));
    assert!(err.is_err());

    let mut not_array = JsonValue::Null;
    let err = catch_unwind(AssertUnwindSafe(|| {
        not_array[0] = JsonValue::Null;
    }));
    assert!(err.is_err());

    let empty_object = JsonValue::Object(IndexMap::new());
    let err = catch_unwind(AssertUnwindSafe(|| {
        let _ = &empty_object["absent"];
    }));
    assert!(err.is_err());
}

#[test]
fn test_json_value_conversions() {
    let json_value = json!({"a": [1, 2], "b": {"c": true}});
    let value = JsonValue::from(json_value.clone());
    let roundtrip: serde_json::Value = value.clone().into();
    assert_eq!(roundtrip, json_value);

    let nan_value = JsonValue::Number(Number::Float(f64::NAN));
    let json_nan: serde_json::Value = nan_value.into();
    assert_eq!(json_nan, json!(null));
}

#[test]
fn test_into_json_value_trait() {
    let json_value = json!({"a": 1});
    let owned = json_value.into_json_value();
    assert_eq!(owned.get("a").and_then(JsonValue::as_i64), Some(1));

    let json_value = json!({"b": true});
    let borrowed = (&json_value).into_json_value();
    assert_eq!(borrowed.get("b").and_then(JsonValue::as_bool), Some(true));

    let value = JsonValue::Bool(false);
    let cloned = value.into_json_value();
    assert!(matches!(cloned, JsonValue::Bool(false)));

    let value = JsonValue::Bool(true);
    let borrowed = (&value).into_json_value();
    assert!(matches!(borrowed, JsonValue::Bool(true)));
}
