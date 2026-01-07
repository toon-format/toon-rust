//! Spec compliance edge cases

use serde_json::json;
use toon_format::types::PathExpansionMode;
use toon_format::{decode, decode_default, DecodeOptions};

#[test]
fn test_keyword_keys_allowed() {
    let input = "true: 1\nfalse: 2\nnull: 3";
    let result: serde_json::Value = decode_default(input).unwrap();
    assert_eq!(result, json!({"true": 1, "false": 2, "null": 3}));
}

#[test]
fn test_nested_array_delimiter_scoping() {
    let input = "outer[2|]: [2]: a,b | [2]: c,d";
    let result: serde_json::Value = decode_default(input).unwrap();
    assert_eq!(result, json!({"outer": [["a", "b"], ["c", "d"]]}));
}

#[test]
fn test_quoted_dotted_field_not_expanded() {
    let input = "rows[1]{\"a.b\"}:\n  1";
    let opts = DecodeOptions::new().with_expand_paths(PathExpansionMode::Safe);
    let result: serde_json::Value = decode(input, &opts).unwrap();
    assert_eq!(result, json!({"rows": [{"a.b": 1}]}));
}

#[test]
fn test_negative_leading_zero_string() {
    let input = "val: -05";
    let result: serde_json::Value = decode_default(input).unwrap();
    assert_eq!(result, json!({"val": "-05"}));
}

#[test]
fn test_unquoted_tab_rejected_in_strict() {
    let input = "val: a\tb";
    let result: Result<serde_json::Value, _> = decode_default(input);
    assert!(result.is_err());
}

#[test]
fn test_multiple_spaces_preserved() {
    let input = "msg: hello  world";
    let result: serde_json::Value = decode_default(input).unwrap();
    assert_eq!(result, json!({"msg": "hello  world"}));
}

#[test]
fn test_coerce_types_toggle() {
    let input = "value: 123\nflag: true\nnone: null";
    let opts = DecodeOptions::new().with_coerce_types(false);
    let result: serde_json::Value = decode(input, &opts).unwrap();
    assert_eq!(
        result,
        json!({"value": "123", "flag": "true", "none": "null"})
    );
}
