use toon_format::{
    decode_default,
    encode_default,
};
use serde_json::json;

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
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(case, decoded);
    }
}

#[test]
fn test_mixed_arrays() {
    let data = json!({
        "mixed": [1, "two", true, null, 3.14]
    });

    let encoded = encode_default(&data).unwrap();
    let decoded = decode_default(&encoded).unwrap();
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
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(case, decoded);
    }
}

#[test]
fn test_large_arrays() {
    let large_array = json!({
        "numbers": (0..1000).collect::<Vec<i32>>()
    });

    let encoded = encode_default(&large_array).unwrap();
    let decoded = decode_default(&encoded).unwrap();
    assert_eq!(large_array, decoded);

    let large_tabular = json!({
        "records": (0..500).map(|i| json!({
            "id": i,
            "name": format!("user_{}", i),
            "value": i * 2
        })).collect::<Vec<_>>()
    });

    let encoded = encode_default(&large_tabular).unwrap();
    let decoded = decode_default(&encoded).unwrap();
    assert_eq!(large_tabular, decoded);
}
