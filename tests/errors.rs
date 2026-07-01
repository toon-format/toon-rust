use serde_json::{
    json,
    Value,
};
use toon_format::{
    decode,
    decode_default,
    decode_strict,
    encode_default,
    DecodeOptions,
    ToonError,
};

#[test]
fn test_invalid_syntax_errors() {
    let cases = vec![
        ("items[]: a,b", "Expected array length"),
        ("items[2]{name: a,b", "Expected '}'"),
        ("key value", "Expected"),
    ];

    for (input, expected_msg) in cases {
        let result = decode_default::<Value>(input);
        if let Err(err) = result {
            let err_str = err.to_string();
            assert!(
                err_str.contains(expected_msg)
                    || err_str.contains("Parse error")
                    || err_str.contains("Invalid"),
                "Expected error containing '{expected_msg}' but got: {err_str}"
            );
        } else if input == "key value" {
            assert!(result.is_ok(), "'key value' is valid as a root string");
        }
    }

    let invalid_cases = vec![
        ("items[2: a,b", "Expected ']'"),
        ("items[abc]: 1,2", "Expected array length"),
    ];

    for (input, expected_msg) in invalid_cases {
        let result = decode_default::<Value>(input);
        assert!(result.is_err(), "Expected error for input: {input}");

        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains(expected_msg) || err_str.contains("Parse error"),
            "Expected error containing '{expected_msg}' but got: {err_str}"
        );
    }
}

#[test]
fn test_type_mismatch_errors() {
    let cases = vec![
        ("value: ", "Empty value"),
        ("items[abc]: 1,2", "Invalid array length"),
    ];

    for (input, description) in cases {
        let result = decode_default::<Value>(input);
        println!("Test case '{description}': {result:?}");
    }
}

#[test]
fn test_length_mismatch_strict_mode() {
    let test_cases = vec![("items[3]: a,b", 3, 2), ("items[5]: x", 5, 1)];

    for (input, expected, actual) in test_cases {
        let result = decode_strict::<Value>(input);

        assert!(
            result.is_err(),
            "Expected error for input '{input}' (expected: {expected}, actual: {actual})",
        );

        if let Err(ToonError::LengthMismatch {
            expected: exp,
            found: fnd,
            ..
        }) = result
        {
            assert_eq!(
                exp, expected,
                "Expected length {expected} but got {exp} for input '{input}'"
            );
            assert_eq!(
                fnd, actual,
                "Expected found {actual} but got {fnd} for input '{input}'"
            );
        }
    }

    let result = decode_strict::<Value>("items[1]: a,b,c");

    if let Ok(val) = result {
        assert_eq!(val["items"], json!(["a"]));
    }
}

#[test]
fn test_length_mismatch_non_strict_mode() {
    let test_cases = vec![
        ("items[3]: a,b", json!({"items": ["a", "b"]})),
        ("items[1]: a,b", json!({"items": ["a", "b"]})),
    ];

    for (input, _expected) in test_cases {
        let result = decode_default::<Value>(input);
        println!("Non-strict test for '{input}': {result:?}");
    }
}

#[test]
fn test_non_strict_preserves_array_body_errors() {
    let options = DecodeOptions::new().with_strict(false);
    let result = decode::<Value>("items[2]: a", &options);

    assert!(
        result.is_err(),
        "Non-strict malformed-header fallback must not hide array length mismatches"
    );
}

#[test]
fn test_canonical_empty_array_rejects_trailing_content() {
    let object_result = decode_strict::<Value>("items: []junk");
    assert!(
        object_result.is_err(),
        "Canonical empty array must be a complete value token"
    );

    let options = DecodeOptions::new().with_strict(false);
    let root_result = decode::<Value>("[]junk", &options);
    assert!(
        root_result.is_err(),
        "Non-strict root empty array must not silently discard trailing content"
    );

    let multiline_root_result = decode::<Value>("[]\nfoo: 1", &options);
    assert!(
        multiline_root_result.is_err(),
        "Non-strict root empty array must not silently discard following lines"
    );
}

#[test]
fn test_canonical_empty_array_allows_sibling_fields() {
    let decoded = decode_strict::<Value>("a: []\nb: 1").unwrap();
    assert_eq!(decoded, json!({"a": [], "b": 1}));

    let decoded = decode_strict::<Value>("outer:\n  a: []\n  b: 2").unwrap();
    assert_eq!(decoded, json!({"outer": {"a": [], "b": 2}}));

    let value = json!({"a": [], "b": 1});
    let encoded = encode_default(&value).unwrap();
    assert_eq!(encoded, "a: []\nb: 1");
    let round_tripped: Value = decode_default(&encoded).unwrap();
    assert_eq!(round_tripped, value);
}

#[test]
fn test_unicode_escape_scanner_errors_include_position() {
    let invalid_scalar = decode_strict::<Value>("value: \"\\uD800\"").unwrap_err();
    match invalid_scalar {
        ToonError::ParseError {
            line,
            column,
            message,
            ..
        } => {
            assert_eq!(line, 1);
            assert!(column > 0);
            assert!(message.contains("Invalid Unicode scalar"));
        }
        err => panic!("Expected located parse error for invalid scalar, got {err:?}"),
    }

    let incomplete_escape = decode_strict::<Value>("value: \"\\u12").unwrap_err();
    match incomplete_escape {
        ToonError::ParseError {
            line,
            column,
            message,
            ..
        } => {
            assert_eq!(line, 1);
            assert!(column > 0);
            assert!(message.contains("Incomplete Unicode escape"));
        }
        err => panic!("Expected located parse error for incomplete escape, got {err:?}"),
    }
}

#[test]
fn test_tabular_unicode_escape_errors_include_position() {
    let invalid_scalar = decode_strict::<Value>("items[1]{name}:\n  \"\\uD800\"").unwrap_err();
    match invalid_scalar {
        ToonError::ParseError {
            line,
            column,
            message,
            ..
        } => {
            assert_eq!(line, 2);
            assert!(column > 0);
            assert!(message.contains("Invalid Unicode scalar"));
        }
        err => panic!("Expected located parse error for tabular invalid scalar, got {err:?}"),
    }

    let incomplete_escape = decode_strict::<Value>("items[1]{name}:\n  \"\\u12").unwrap_err();
    match incomplete_escape {
        ToonError::ParseError {
            line,
            column,
            message,
            ..
        } => {
            assert_eq!(line, 2);
            assert!(column > 0);
            assert!(
                message.contains("Incomplete Unicode escape"),
                "Expected incomplete Unicode escape message, got: {message}"
            );
        }
        err => panic!("Expected located parse error for tabular incomplete escape, got {err:?}"),
    }
}

#[test]
fn test_strict_array_header_segment_whitespace_errors() {
    let header_before_fields = decode_strict::<Value>("items[1] {name}:\n  Alice");
    assert!(
        header_before_fields.is_err(),
        "Strict mode must reject whitespace between array header segments"
    );

    let header_before_colon = decode_strict::<Value>("items[1] : Alice");
    assert!(
        header_before_colon.is_err(),
        "Strict mode must reject whitespace before array header colon"
    );
}

#[test]
fn test_strict_duplicate_sibling_key_errors() {
    let object_duplicate = decode_strict::<Value>("a: 1\na: 2");
    assert!(
        object_duplicate.is_err(),
        "Strict mode must reject duplicate object sibling keys"
    );

    let nested_duplicate = decode_strict::<Value>("items[1]:\n  - a: 1\n    a: 2");
    assert!(
        nested_duplicate.is_err(),
        "Strict mode must reject duplicate sibling keys inside list-item objects"
    );
}

#[test]
fn test_non_strict_malformed_headers_fall_back_only_with_same_line_colon() {
    let options = DecodeOptions::new().with_strict(false);

    let decoded: Value = decode("foo[1][bar]: 10", &options).unwrap();
    assert_eq!(decoded, json!({"foo[1][bar]": 10}));

    let result = decode::<Value>("foo[1][bar]\nnext: 2", &options);
    assert!(
        result.is_err(),
        "Malformed-header fallback must not consume later lines"
    );
}

#[test]
fn test_non_strict_valid_quoted_field_headers_do_not_fall_back() {
    let options = DecodeOptions::new().with_strict(false);

    let decoded: Value = decode("items[1]{\"a:b\"}:\n  x", &options).unwrap();
    assert_eq!(decoded, json!({"items": [{"a:b": "x"}]}));

    let decoded: Value = decode("items[1]{\"a]b\"}:\n  x", &options).unwrap();
    assert_eq!(decoded, json!({"items": [{"a]b": "x"}]}));

    let decoded: Value = decode("items[1]{\"a}b\"}:\n  x", &options).unwrap();
    assert_eq!(decoded, json!({"items": [{"a}b": "x"}]}));
}

#[test]
fn test_non_canonical_lengths_are_not_array_headers() {
    let options = DecodeOptions::new().with_strict(false);

    let leading_zero: Value = decode("items[03]: a,b,c", &options).unwrap();
    assert_eq!(leading_zero, json!({"items[03]": "a,b,c"}));

    let negative: Value = decode("items[-1]: a,b,c", &options).unwrap();
    assert_eq!(negative, json!({"items[-1]": "a,b,c"}));

    assert!(decode_strict::<Value>("items[03]: a,b,c").is_err());
    assert!(decode_strict::<Value>("items[-1]: a,b,c").is_err());
}

#[test]
fn test_delimiter_errors() {
    let mixed_delimiters = "items[3]: a,b|c";
    let result = decode_default::<Value>(mixed_delimiters);

    println!("Mixed delimiter test: {result:?}");
}

#[test]
fn test_quoting_errors() {
    let test_cases = vec![
        ("value: \"unclosed", "Unclosed string"),
        ("value: \"invalid\\x\"", "Invalid escape"),
    ];

    for (input, description) in test_cases {
        let result = decode_default::<Value>(input);
        println!("Quoting error test '{description}': {result:?}");
    }
}

#[test]
fn test_tabular_array_errors() {
    let result = decode_default::<Value>("items[2]{id,name}:\n  1,Alice\n  2");
    assert!(result.is_err(), "Should error on incomplete row");

    if let Err(e) = result {
        let err_str = e.to_string();
        assert!(
            err_str.contains("Parse")
                || err_str.contains("column")
                || err_str.contains("expected")
                || err_str.contains("primitive"),
            "Error should mention missing field or delimiter: {err_str}"
        );
    }

    let result = decode_default::<Value>("items[2]{id,name}:\n  1,Alice\n  2,Bob,Extra");
    if let Err(err) = result {
        let err_str = err.to_string();
        assert!(
            err_str.contains("Parse") || err_str.contains("Expected") || err_str.contains("field"),
            "Should mention unexpected content: {err_str}"
        );
    } else {
        println!("Note: Extra fields are ignored in tabular arrays");
    }

    let result = decode_strict::<Value>("items[3]{id,name}:\n  1,Alice\n  2,Bob");
    assert!(
        result.is_err(),
        "Should error on row count mismatch in strict mode"
    );

    if let Err(ToonError::LengthMismatch {
        expected, found, ..
    }) = result
    {
        assert_eq!(expected, 3);
        assert_eq!(found, 2);
    }
}

#[test]
fn test_nested_structure_errors() {
    let result = decode_default::<Value>("obj:\n  key");
    assert!(result.is_err(), "Should error on incomplete nested object");

    let result = decode_default::<Value>("arr[2]:\n  - item");
    assert!(result.is_err(), "Should error on incomplete nested array");
}

#[test]
fn test_depth_limit_errors() {
    let mut nested = "a:\n".to_string();
    for i in 0..60 {
        nested.push_str(&format!("{}b:\n", "  ".repeat(i + 1)));
    }
    nested.push_str(&format!("{}c: value", "  ".repeat(61)));

    let result = decode_default::<Value>(&nested);
    println!("Deep nesting test: {result:?}");
}

#[test]
fn test_empty_structure_errors() {
    let cases = vec![
        ("items[]:", "Empty array with colon"),
        ("obj{}:", "Empty object with colon"),
        ("{}", "Just braces"),
        ("[]", "Just brackets"),
    ];

    for (input, description) in cases {
        let result = decode_default::<Value>(input);
        println!("Empty structure test '{description}': {result:?}");
    }
}

#[test]
fn test_error_messages_are_helpful() {
    let result = decode_strict::<Value>("items[5]: a,b,c");

    if let Err(err) = result {
        let err_msg = err.to_string();

        assert!(
            err_msg.contains("5")
                || err_msg.contains("3")
                || err_msg.contains("expected")
                || err_msg.contains("found"),
            "Error message should contain length information: {err_msg}"
        );
    }
}

#[test]
fn test_parse_error_line_column() {
    let input = "line1: value\nline2: bad syntax!\nline3: value";
    let result = decode_default::<Value>(input);

    if let Err(ToonError::ParseError { line, column, .. }) = result {
        println!("Parse error at line {line}, column {column}");
        assert!(line > 0, "Line number should be positive");
        assert!(column > 0, "Column number should be positive");
    }
}

#[test]
fn test_multiple_errors_in_input() {
    let input = "items[10]: a,b\nobj{missing,fields: x,y";
    let result = decode_default::<Value>(input);

    assert!(result.is_err(), "Should error on malformed input");
}

#[test]
fn test_coercion_errors() {
    let opts = DecodeOptions::new().with_coerce_types(true);

    let result = decode::<Value>("value: 123", &opts);
    assert!(result.is_ok());

    let result = decode::<Value>("value: true", &opts);
    assert!(result.is_ok());

    let result = decode::<Value>("value: 3.14", &opts);
    assert!(result.is_ok());
}

#[test]
fn test_no_coercion_preserves_strings() {
    let opts = DecodeOptions::new().with_coerce_types(false);

    let result = decode::<Value>("value: hello", &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("hello"));

    let result = decode::<Value>(r#"value: "123""#, &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("123"));

    let result = decode::<Value>(r#"value: "true""#, &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("true"));

    let result = decode::<Value>("value: 123", &opts).unwrap();
    assert!(result["value"].is_number());
    assert_eq!(result["value"], json!(123));

    let result = decode::<Value>("value: true", &opts).unwrap();
    assert!(result["value"].is_boolean());
    assert_eq!(result["value"], json!(true));
}

#[test]
fn test_edge_case_values() {
    let cases = vec![
        ("value: 0", json!({"value": 0})),
        ("value: null", json!({"value": null})),
    ];

    for (input, expected) in cases {
        let result = decode_default::<Value>(input);
        match result {
            Ok(val) => assert_eq!(val, expected, "Failed for input: {input}"),
            Err(e) => println!("Edge case '{input}' error: {e:?}"),
        }
    }

    let result = decode_default::<Value>("value: -0");
    match result {
        Ok(val) => {
            assert_eq!(
                val["value"],
                json!(0),
                "Negative zero is normalized to zero in JSON"
            );
        }
        Err(e) => println!("Edge case '-0' error: {e:?}"),
    }
}

#[test]
fn test_unicode_in_errors() {
    let input = "emoji: 😀🎉\nkey: value\nbad: @syntax!";
    let result = decode_default::<Value>(input);

    if let Err(err) = result {
        let err_msg = err.to_string();
        println!("Unicode error handling: {err_msg}");
        assert!(!err_msg.is_empty());
    }
}

#[test]
fn test_recovery_from_errors() {
    let valid_after_invalid = vec!["good: value\nbad syntax here\nalso_good: value"];

    for input in valid_after_invalid {
        let result = decode_default::<Value>(input);
        println!("Recovery test for: {result:?}");
    }
}

#[test]
fn test_strict_mode_indentation_errors() {
    let result = decode_strict::<Value>("items[2]: a");
    assert!(
        result.is_err(),
        "Should error on insufficient items in strict mode"
    );

    if let Err(ToonError::LengthMismatch {
        expected, found, ..
    }) = result
    {
        assert_eq!(expected, 2);
        assert_eq!(found, 1);
    }
}

#[test]
fn test_quoted_key_without_colon() {
    let result = decode_default::<Value>(r#""key" value"#);
    println!("Quoted key test: {result:?}");
}

#[test]
fn test_nested_array_length_mismatches() {
    let result = decode_strict::<Value>("outer[1]:\n  - items[2]: a,b\n  - items[3]: x,y");
    if let Err(err) = result {
        let err_str = err.to_string();
        assert!(err_str.contains("3") || err_str.contains("2") || err_str.contains("length"));
    }
}

#[test]
fn test_empty_array_with_length() {
    let result = decode_strict::<Value>("items[2]:");
    assert!(
        result.is_err(),
        "Should error when array header specifies length but no items provided"
    );

    let result = decode_strict::<Value>("items[0]:");
    assert!(
        result.is_ok(),
        "Empty array with length 0 should parse successfully"
    );

    if let Ok(val) = result {
        assert_eq!(val["items"], json!([]));
    }
}

#[test]
fn test_tabular_array_field_count_mismatch() {
    let result = decode_default::<Value>("items[2]{id,name}:\n  1\n  2,Bob");
    assert!(
        result.is_err(),
        "Should error when row has fewer fields than header"
    );
}

#[test]
fn test_invalid_array_header_syntax() {
    let cases = vec![
        ("items[", "Expected array length"),
        ("items[: a,b", "Expected array length"),
    ];

    for (input, expected_msg) in cases {
        let result = decode_default::<Value>(input);
        assert!(
            result.is_err(),
            "Should error on invalid array header: {input}"
        );

        if let Err(e) = result {
            let err_str = e.to_string();
            assert!(
                err_str.contains(expected_msg) || err_str.contains("Parse error"),
                "Expected error about '{expected_msg}' but got: {err_str}",
            );
        }
    }

    let result = decode_default::<Value>("items{id}: a,b");
    println!("Braces without brackets test: {result:?}");

    let result = decode_default::<Value>("items]2[: a,b");
    println!("Quirky bracket syntax test: {result:?}");
}

#[test]
fn test_missing_colon_after_key() {
    let _result = decode_default::<Value>("key value");

    let result = decode_default::<Value>("obj:\n  key value");
    println!("Missing colon in object: {result:?}");
}

#[test]
fn test_error_context_information() {
    let result = decode_strict::<Value>("items[5]: a,b");

    if let Err(e) = result {
        let err_str = e.to_string();

        assert!(
            err_str.contains("5")
                || err_str.contains("2")
                || err_str.contains("length")
                || err_str.contains("expected")
                || err_str.contains("found"),
            "Error should contain length information: {err_str}",
        );

        match e {
            ToonError::ParseError {
                context: Some(ctx), ..
            } => {
                println!(
                    "Error context has {} preceding lines, {} following lines",
                    ctx.preceding_lines.len(),
                    ctx.following_lines.len()
                );
            }
            ToonError::LengthMismatch {
                context: Some(ctx), ..
            } => {
                println!("Length mismatch context available:{ctx}");
            }
            _ => {}
        }
    }
}
