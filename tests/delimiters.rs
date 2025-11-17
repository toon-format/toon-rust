use serde_json::{
    json,
    Value,
};
use toon_format::{
    decode_default,
    encode,
    encode_default,
    Delimiter,
    EncodeOptions,
};

#[test]
fn test_delimiter_variants() {
    let data = json!({"tags": ["a", "b", "c"]});

    let encoded = encode_default(&data).unwrap();
    assert!(encoded.contains("a,b,c"));
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);

    let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
    let encoded = encode(&data, &opts).unwrap();
    assert!(encoded.contains("a|b|c"));
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);

    let opts = EncodeOptions::new().with_delimiter(Delimiter::Tab);
    let encoded = encode(&data, &opts).unwrap();
    assert!(encoded.contains("a\tb\tc"));
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);
}

#[test]
fn test_delimiter_in_values() {
    let data = json!({"tags": ["a,b", "c|d", "e\tf"]});

    let encoded = encode_default(&data).unwrap();
    assert!(encoded.contains("\"a,b\""));
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);

    let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
    let encoded = encode(&data, &opts).unwrap();
    assert!(encoded.contains("\"c|d\""));
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(data, decoded);
}

#[test]
fn test_non_active_delimiters_in_tabular_arrays() {
    // When comma is the active delimiter, pipe and tab should be treated as regular data
    // Per TOON spec §11: "non-active delimiters MUST NOT cause splits"

    // Test 1: Pipe character in value when comma is active delimiter (default)
    let data = r#"item-list[1]{a,b}:
  ":",|
"#;
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["item-list"][0]["a"], ":");
    assert_eq!(decoded["item-list"][0]["b"], "|");

    // Test 2: Both values quoted
    let data = r#"item-list[1]{a,b}:
  ":","|"
"#;
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["item-list"][0]["a"], ":");
    assert_eq!(decoded["item-list"][0]["b"], "|");

    // Test 3: Tab character in value when comma is active
    let data = "item-list[1]{a,b}:\n  \":\",\t\n";
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["item-list"][0]["a"], ":");
    assert_eq!(decoded["item-list"][0]["b"], "\t");

    // Test 4: Comma in value when pipe is active delimiter - should quote the comma
    let data = r#"item-list[1|]{a|b}:
  ":"|","
"#;
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["item-list"][0]["a"], ":");
    assert_eq!(decoded["item-list"][0]["b"], ",");
}

#[test]
fn test_non_active_delimiters_in_inline_arrays() {
    // Test pipe in inline array when comma is active
    let data = r#"tags[3]: a,|,c"#;
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["tags"], json!(["a", "|", "c"]));

    // Test comma in inline array when pipe is active - comma needs quoting
    let data = "tags[3|]: a|\",\"|c";
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["tags"], json!(["a", ",", "c"]));

    // Test multiple non-active delimiters - pipes when comma is active
    let data = r#"items[4]: |,|,|,"#;
    let decoded: Value = decode_default(&data).unwrap();
    assert_eq!(decoded["items"], json!(["|", "|", "|", ""]));
}

#[test]
fn test_delimiter_mismatch_error() {
    // Per TOON spec §6: delimiter in brackets must match delimiter in braces
    // This should error: pipe in brackets, comma in braces
    let data = r#"item-list[1|]{a,b}:
  ":",|
"#;
    let result: Result<Value, _> = decode_default(&data);
    assert!(result.is_err(), "Mismatched delimiters should error");
}
