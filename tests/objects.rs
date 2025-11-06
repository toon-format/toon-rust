use serde_json::json;
use toon_format::{
    decode_default,
    encode_default,
};

#[test]
fn test_special_characters_and_quoting() {
    let cases = vec![
        json!({"value": "true"}),
        json!({"value": "false"}),
        json!({"value": "42"}),
        json!({"value": "3.14"}),
        json!({"value": "hello, world"}),
        json!({"value": "hello|world"}),
        json!({"value": "say \"hello\""}),
        json!({"value": "line1\nline2"}),
        json!({"value": ""}),
        json!({"value": " hello "}),
    ];

    for case in cases {
        let encoded = encode_default(&case).unwrap();
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(case, decoded, "Failed for: {case}");
    }
}

#[test]
fn test_nested_structures() {
    let nested = json!({
        "level1": {
            "level2": {
                "level3": {
                    "value": "deep"
                }
            }
        }
    });

    let encoded = encode_default(&nested).unwrap();
    let decoded = decode_default(&encoded).unwrap();
    assert_eq!(nested, decoded);
}
