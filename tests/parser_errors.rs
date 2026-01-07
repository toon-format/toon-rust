use serde_json::Value;
use toon_format::{decode, decode_strict, DecodeOptions, Delimiter};

#[test]
fn test_strict_rejects_multiple_root_values() {
    let err = decode_strict::<Value>("hello\nworld").unwrap_err();
    assert!(err
        .to_string()
        .contains("Multiple values at root level are not allowed"));
}

#[test]
fn test_strict_invalid_unquoted_key() {
    let err = decode_strict::<Value>("bad-key: 1").unwrap_err();
    assert!(err.to_string().contains("Invalid unquoted key"));
}

#[test]
fn test_strict_missing_colon_in_object() {
    let err = decode_strict::<Value>("obj:\n  key").unwrap_err();
    assert!(err
        .to_string()
        .contains("Expected ':' after 'key' in object context"));
}

#[test]
fn test_array_header_hash_marker_rejected() {
    let err = decode_strict::<Value>("items[#2]: a,b").unwrap_err();
    assert!(err
        .to_string()
        .contains("Length marker '#' is not supported"));
}

#[test]
fn test_array_header_missing_right_bracket() {
    let err = decode_strict::<Value>("items[1|: a|b").unwrap_err();
    assert!(err.to_string().contains("Expected ']'"));
}

#[test]
fn test_tabular_header_requires_newline() {
    let err = decode_strict::<Value>("items[1]{a}: 1").unwrap_err();
    assert!(err
        .to_string()
        .contains("Expected newline after tabular array header"));
}

#[test]
fn test_tabular_row_missing_delimiter() {
    let err = decode_strict::<Value>("items[1]{a,b}:\n  1 2").unwrap_err();
    assert!(err.to_string().contains("Expected delimiter"));
}

#[test]
fn test_tabular_blank_line_strict() {
    let err = decode_strict::<Value>("items[2]{a}:\n  1\n\n  2").unwrap_err();
    assert!(err
        .to_string()
        .contains("Blank lines are not allowed inside tabular arrays"));
}

#[test]
fn test_inline_array_missing_delimiter_strict() {
    let err = decode_strict::<Value>("items[2]: a b").unwrap_err();
    assert!(err.to_string().contains("Expected delimiter"));
}

#[test]
fn test_list_array_blank_line_strict() {
    let err = decode_strict::<Value>("items[2]:\n  - a\n\n  - b").unwrap_err();
    assert!(err
        .to_string()
        .contains("Blank lines are not allowed inside list arrays"));
}

#[test]
fn test_array_header_delimiter_mismatch() {
    let opts = DecodeOptions::new().with_delimiter(Delimiter::Pipe);
    let err = decode::<Value>("items[2,]: a,b", &opts).unwrap_err();
    assert!(err.to_string().contains("Detected delimiter"));
}
