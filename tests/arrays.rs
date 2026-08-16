use std::f64;

use serde_json::{
    json,
    Value,
};
use toon_format::{
    decode_default,
    encode_default,
};

#[test]
fn test_tabular_arrays() {
    let cases = vec![
        json!({
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        }),
        json!({
            "products": [
                {"sku": "A1", "name": "Widget", "price": 9.99, "stock": 100},
                {"sku": "B2", "name": "Gadget", "price": 19.99, "stock": 50}
            ]
        }),
        json!({
            "items": [
                {"a": 1, "b": 2, "c": 3}
            ]
        }),
        json!({
            "data": (0..10).map(|i| json!({"id": i, "value": i * 2})).collect::<Vec<_>>()
        }),
    ];

    for case in cases {
        let encoded = encode_default(&case).unwrap();
        assert!(encoded.contains("{"));
        assert!(encoded.contains("}"));
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(case, decoded);
    }
}

#[test]
fn test_mixed_arrays() {
    let data = json!({
        "mixed": [1, "two", true, null, f64::consts::PI]
    });

    let encoded = encode_default(&data).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);
}

#[test]
fn test_empty_values() {
    let cases = vec![
        json!({"array": []}),
        json!({"object": {}}),
        json!({"string": ""}),
        json!({"null": null}),
    ];

    for case in cases {
        let encoded = encode_default(&case).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(case, decoded);
    }
}

#[test]
fn test_large_arrays() {
    let large_array = json!({
        "numbers": (0..1000).collect::<Vec<i32>>()
    });

    let encoded = encode_default(&large_array).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(large_array, decoded);

    let large_tabular = json!({
        "records": (0..500).map(|i| json!({
            "id": i,
            "name": format!("user_{}", i),
            "value": i * 2
        })).collect::<Vec<_>>()
    });

    let encoded = encode_default(&large_tabular).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(large_tabular, decoded);
}

/// Builds `levels` nested `{"a": ...}` objects around a primitive.
fn nested_object(levels: usize) -> Value {
    let mut inner = json!(1);
    for _ in 0..levels {
        inner = json!({ "a": inner });
    }
    inner
}

#[test]
fn test_deeply_nested_uniform_column_is_rejected_not_overflowed() {
    // A uniform nested column is classified recursively into a field group.
    // Past the depth limit the array falls back to list form, where the
    // encoder's own depth check rejects it. Either way the process must not
    // abort on a stack overflow.
    let shallow = json!({ "rows": [nested_object(255), nested_object(255)] });
    let encoded = encode_default(&shallow).unwrap();
    assert!(
        encoded.starts_with("rows[2]{a{a{"),
        "not tabular: {encoded}"
    );

    let deep = json!({ "rows": [nested_object(300), nested_object(300)] });
    let err = encode_default(&deep).expect_err("past the depth limit must be rejected");
    assert!(
        err.to_string()
            .contains("Maximum nesting depth of 256 exceeded"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_deeply_nested_object_is_rejected_not_overflowed() {
    // Encoding normalizes the whole tree before the depth check runs, so this
    // guards a separate recursion from the one above: 1000 levels of plain
    // object nesting must produce an error, not abort the process.
    let err = encode_default(&nested_object(1000)).expect_err("past the depth limit must fail");
    assert!(
        err.to_string()
            .contains("Maximum nesting depth of 256 exceeded"),
        "unexpected error: {err}"
    );
}
