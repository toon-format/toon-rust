use serde_json::json;
use toon_format::{decode, DecodeOptions};

#[test]
fn test_negative_array_length_rejected() {
    let input = "items[-1]:";
    let opts = DecodeOptions::new().with_strict(true);
    let result = decode::<serde_json::Value>(input, &opts);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("non-negative"));
}

#[test]
fn test_float_array_length_rejected() {
    let input = "items[3.5]:";
    let opts = DecodeOptions::new().with_strict(true);
    let result = decode::<serde_json::Value>(input, &opts);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("integer"));
}

#[test]
fn test_mixed_delimiters_rejected_in_strict_mode() {
    let input = "items[3]: a,b|c";
    let opts = DecodeOptions::new().with_strict(true);
    let result = decode::<serde_json::Value>(input, &opts);

    assert!(result.is_err());
}

#[test]
fn test_length_mismatch_allowed_in_non_strict_inline() {
    let input = "items[1]: a,b";
    let opts = DecodeOptions::new().with_strict(false);
    let result = decode::<serde_json::Value>(input, &opts).unwrap();

    assert_eq!(result["items"], json!(["a", "b"]));
}

#[test]
fn test_length_mismatch_allowed_in_non_strict_list() {
    let input = "items[1]:\n  - 1\n  - 2";
    let opts = DecodeOptions::new().with_strict(false);
    let result = decode::<serde_json::Value>(input, &opts).unwrap();

    assert_eq!(result["items"], json!([1, 2]));
}

#[test]
fn test_tab_indentation_allowed_in_non_strict_mode() {
    let input = "items[1]:\n\t- 1";
    let opts = DecodeOptions::new().with_strict(false);
    let result = decode::<serde_json::Value>(input, &opts).unwrap();

    assert_eq!(result["items"], json!([1]));
}

#[test]
fn test_unquoted_key_rejected_in_strict_mode() {
    let input = "bad-key: 1";
    let opts = DecodeOptions::new().with_strict(true);
    let result = decode::<serde_json::Value>(input, &opts);

    assert!(result.is_err());
}
