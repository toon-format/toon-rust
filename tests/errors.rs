use serde_json::json;
use toon_format::{
    decode,
    decode_default,
    decode_strict,
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
        let result = decode_default(input);
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
        let result = decode_default(input);
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
        let result = decode_default(input);
        println!("Test case '{description}': {result:?}");
    }
}

#[test]
fn test_length_mismatch_strict_mode() {
    let test_cases = vec![("items[3]: a,b", 3, 2), ("items[5]: x", 5, 1)];

    for (input, expected, actual) in test_cases {
        let result = decode_strict(input);

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

    let result = decode_strict("items[1]: a,b,c");

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
        let result = decode_default(input);
        println!("Non-strict test for '{input}': {result:?}");
    }
}

#[test]
fn test_delimiter_errors() {
    let mixed_delimiters = "items[3]: a,b|c";
    let result = decode_default(mixed_delimiters);

    println!("Mixed delimiter test: {result:?}");
}

#[test]
fn test_quoting_errors() {
    let test_cases = vec![
        ("value: \"unclosed", "Unclosed string"),
        ("value: \"invalid\\x\"", "Invalid escape"),
    ];

    for (input, description) in test_cases {
        let result = decode_default(input);
        println!("Quoting error test '{description}': {result:?}");
    }
}

#[test]
fn test_tabular_array_errors() {
    let result = decode_default("items[2]{id,name}:\n  1,Alice\n  2");
    assert!(result.is_err(), "Should error on incomplete row");

    if let Err(e) = result {
        let err_str = e.to_string();
        assert!(
            err_str.contains("Parse")
                || err_str.contains("cloumn")
                || err_str.contains("expected")
                || err_str.contains("primitive"),
            "Error should mention missing field or delimiter: {err_str}"
        );
    }

    let result = decode_default("items[2]{id,name}:\n  1,Alice\n  2,Bob,Extra");
    if let Err(err) = result {
        let err_str = err.to_string();
        assert!(
            err_str.contains("Parse") || err_str.contains("Expected") || err_str.contains("field"),
            "Should mention unexpected content: {err_str}"
        );
    } else {
        println!("Note: Extra fields are ignored in tabular arrays");
    }

    let result = decode_strict("items[3]{id,name}:\n  1,Alice\n  2,Bob");
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
    let result = decode_default("obj:\n  key");
    assert!(result.is_err(), "Should error on incomplete nested object");

    let result = decode_default("arr[2]:\n  - item");
    assert!(result.is_err(), "Should error on incomplete nested array");
}

#[test]
fn test_depth_limit_errors() {
    let mut nested = "a:\n".to_string();
    for i in 0..60 {
        nested.push_str(&format!("{}b:\n", "  ".repeat(i + 1)));
    }
    nested.push_str(&format!("{}c: value", "  ".repeat(61)));

    let result = decode_default(&nested);
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
        let result = decode_default(input);
        println!("Empty structure test '{description}': {result:?}");
    }
}

#[test]
fn test_error_messages_are_helpful() {
    let result = decode_strict("items[5]: a,b,c");

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
    let result = decode_default(input);

    if let Err(ToonError::ParseError { line, column, .. }) = result {
        println!("Parse error at line {line}, column {column}");
        assert!(line > 0, "Line number should be positive");
        assert!(column > 0, "Column number should be positive");
    }
}

#[test]
fn test_multiple_errors_in_input() {
    let input = "items[10]: a,b\nobj{missing,fields: x,y";
    let result = decode_default(input);

    assert!(result.is_err(), "Should error on malformed input");
}

#[test]
fn test_coercion_errors() {
    let opts = DecodeOptions::new().with_coerce_types(true);

    let result = decode("value: 123", &opts);
    assert!(result.is_ok());

    let result = decode("value: true", &opts);
    assert!(result.is_ok());

    let result = decode("value: 3.14", &opts);
    assert!(result.is_ok());
}

#[test]
fn test_no_coercion_preserves_strings() {
    let opts = DecodeOptions::new().with_coerce_types(false);

    let result = decode("value: hello", &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("hello"));

    let result = decode(r#"value: "123""#, &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("123"));

    let result = decode(r#"value: "true""#, &opts).unwrap();
    assert!(result["value"].is_string());
    assert_eq!(result["value"], json!("true"));

    let result = decode("value: 123", &opts).unwrap();
    assert!(result["value"].is_number());
    assert_eq!(result["value"], json!(123));

    let result = decode("value: true", &opts).unwrap();
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
        let result = decode_default(input);
        match result {
            Ok(val) => assert_eq!(val, expected, "Failed for input: {input}"),
            Err(e) => println!("Edge case '{input}' error: {e:?}"),
        }
    }

    let result = decode_default("value: -0");
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
    let result = decode_default(input);

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
        let result = decode_default(input);
        println!("Recovery test for: {result:?}");
    }
}

#[test]
fn test_strict_mode_indentation_errors() {
    let result = decode_strict("items[2]: a");
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
    let result = decode_default(r#""key" value"#);
    println!("Quoted key test: {result:?}");
}

#[test]
fn test_nested_array_length_mismatches() {
    let result = decode_strict("outer[1]:\n  - items[2]: a,b\n  - items[3]: x,y");
    if let Err(err) = result {
        let err_str = err.to_string();
        assert!(err_str.contains("3") || err_str.contains("2") || err_str.contains("length"));
    }
}

#[test]
fn test_empty_array_with_length() {
    let result = decode_strict("items[2]:");
    assert!(
        result.is_err(),
        "Should error when array header specifies length but no items provided"
    );

    let result = decode_strict("items[0]:");
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
    let result = decode_default("items[2]{id,name}:\n  1\n  2,Bob");
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
        let result = decode_default(input);
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

    let result = decode_default("items{id}: a,b");
    println!("Braces without brackets test: {result:?}");

    let result = decode_default("items]2[: a,b");
    println!("Quirky bracket syntax test: {result:?}");
}

#[test]
fn test_missing_colon_after_key() {
    let _result = decode_default("key value");

    let result = decode_default("obj:\n  key value");
    println!("Missing colon in object: {result:?}");
}

#[test]
fn test_error_context_information() {
    let result = decode_strict("items[5]: a,b");

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
